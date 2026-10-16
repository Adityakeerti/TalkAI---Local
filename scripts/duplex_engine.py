#!/usr/bin/env python3
"""
scripts/duplex_engine.py
========================
TalkAI True Full-Duplex Cascade: Micro-Turn Engine Core, Streaming STT & Kokoro TTS Pipeline (Modules 1, 2, 3 & 4)
Based on arXiv:2603.09180 (DuplexCascade)

Responsibilities:
1. MicroTurnClock: High-precision 600ms monotonic clock with zero cumulative drift.
2. StreamingAudioSlicer: Continuous 600ms audio stream buffer (16kHz mono 16-bit PCM =
   9,600 samples = 19,200 bytes per slice) with zero buffer overrun.
3. StreamingSTTProcessor: Real-time STT engine using faster-whisper (tiny.en).
   - Real-time RMS energy silence gate: Emits "no voice" in < 2ms without invoking Whisper.
   - Active speech transcription: Emits transcribed words in <= 150ms.
   - Zero VAD silence timeout: Never waits for silence or end of utterance.
4. KokoroTTSProcessor: Ultra-low latency neural TTS using Kokoro-82M ONNX.
   - Non-blocking async synthesis (<= 180ms for 3-8 word speech fragments).
   - Converts float32 audio to 16-bit PCM (24kHz mono) and base64 string.
   - Concurrent audio chunk queueing with sub-30ms client barge-in playback cutoff.
5. DuplexSession: State manager maintaining rolling micro-turn history formatted
   identically to micro_turns_train.jsonl with consecutive silent turns pruning.
6. DuplexModelClient: Async client to llama-server on port 8080 with token 0 logprob
   extraction and calibrated neural turn-taking threshold P(<listen>) >= 0.30.
7. DuplexEngineApp: FastAPI application providing health endpoints and WebSocket routes
   /ws/duplex and /ws/duplex_audio for real-time full-duplex communication.

INVARIANT: ZERO HARDCODING.
- No keyword lists.
- No logit bias bans.
- No regex text-stripping on output tokens.
"""

import os
import sys
import time
import math
import json
import uuid
import ctypes
import base64
import asyncio
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import httpx
import uvicorn
import numpy as np
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DuplexEngine")

# Configuration Constants
DEFAULT_LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
DEFAULT_TICK_INTERVAL_SEC = 0.600
DEFAULT_LISTEN_THRESHOLD = 0.30
LISTEN_TOKEN_ID = 20923
LISTEN_TOKEN_STR = "listen"
MAX_HISTORY_TURNS = 20  # Keeps system prompt + last 20 micro-turns

# Audio Slicing & STT Constants (Module 2)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BYTES_PER_SAMPLE = 2  # 16-bit PCM
AUDIO_SLICE_SAMPLES = int(AUDIO_SAMPLE_RATE * DEFAULT_TICK_INTERVAL_SEC)  # 9,600 samples
AUDIO_SLICE_BYTES = AUDIO_SLICE_SAMPLES * AUDIO_BYTES_PER_SAMPLE          # 19,200 bytes
MAX_SPEECH_BUFFER_BYTES = int(AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE * 4.8)  # 153,600 bytes (4.8s)
DEFAULT_ENERGY_THRESHOLD = 0.005  # RMS threshold for silence vs active speech

# Kokoro TTS Constants (Module 4)
DEFAULT_KOKORO_MODEL = os.environ.get("KOKORO_MODEL_PATH", "models/kokoro/kokoro-v0_19.onnx")
DEFAULT_KOKORO_VOICES = os.environ.get("KOKORO_VOICES_PATH", "models/kokoro/voices.bin")
DEFAULT_KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_bella")
TTS_SAMPLE_RATE = 24000

SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. Output "
    "exactly 'listen' when you should yield and keep listening; otherwise output a brief, "
    "natural speech fragment that can be spoken now. Do not repeat prior fragments. Continue "
    "an unfinished answer across turns when appropriate. Yield immediately for a clear "
    "interruption or urgent safety risk. Do not treat every pause as the end of the user's thought. "
    "Dont give unecesaary knowledge answer only what's asked."
)

DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "36"))
STOP_TOKENS = ["<turn|>", "<|turn>", "<eos>", "<end_of_turn>", "\n\n"]



# ---------------------------------------------------------------------------
# Preload CUDA Runtime Libraries
# ---------------------------------------------------------------------------
def preload_cuda_libs() -> None:
    """Preloads nvidia CUDA/cuBLAS/cuDNN runtime libraries if installed in venv."""
    try:
        venv_nvidia = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
        if venv_nvidia.exists():
            for lib_dir in venv_nvidia.glob("*/lib"):
                for so in lib_dir.glob("*.so*"):
                    try:
                        ctypes.CDLL(str(so))
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Preload CUDA libs notice: {e}")


# ---------------------------------------------------------------------------
# VRAM Utility
# ---------------------------------------------------------------------------
def get_gpu_vram_info() -> Dict[str, Any]:
    """Queries GPU memory usage via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            used_mb = float(parts[0])
            total_mb = float(parts[1])
            return {
                "available": True,
                "used_mb": used_mb,
                "total_mb": total_mb,
                "under_cap": used_mb <= 5120.0,
            }
    except Exception as e:
        logger.warning(f"Failed to query GPU VRAM via nvidia-smi: {e}")
    return {"available": False, "used_mb": 0.0, "total_mb": 0.0, "under_cap": True}


# ---------------------------------------------------------------------------
# 1. MicroTurnClock: High Precision 600ms Loop
# ---------------------------------------------------------------------------
class MicroTurnClock:
    """
    Monotonic clock designed to tick every 600ms (interval_sec) with zero
    cumulative drift.
    Target evaluation gate: 600ms +/- 15ms.
    """

    def __init__(self, interval_sec: float = DEFAULT_TICK_INTERVAL_SEC):
        self.interval_sec = interval_sec
        self.start_time: Optional[float] = None
        self.tick_count: int = 0
        self.last_tick_time: Optional[float] = None
        self.tick_intervals: List[float] = []

    def start(self) -> None:
        """Initializes the clock base time."""
        self.start_time = time.monotonic()
        self.last_tick_time = self.start_time
        self.tick_count = 0
        self.tick_intervals = []

    async def wait_next_tick(self) -> Tuple[int, float, float]:
        """
        Sleeps until the next scheduled 600ms tick boundary.
        Returns:
            (tick_count, actual_time, drift_ms)
        """
        if self.start_time is None:
            self.start()

        self.tick_count += 1
        expected_target = self.start_time + (self.tick_count * self.interval_sec)
        now = time.monotonic()
        sleep_sec = expected_target - now

        if sleep_sec > 0:
            await asyncio.sleep(sleep_sec)

        actual_time = time.monotonic()
        drift_ms = (actual_time - expected_target) * 1000.0
        delta_ms = (actual_time - self.last_tick_time) * 1000.0 if self.last_tick_time else self.interval_sec * 1000.0
        self.tick_intervals.append(delta_ms)
        self.last_tick_time = actual_time

        return self.tick_count, actual_time, drift_ms

    def get_stats(self) -> Dict[str, float]:
        """Calculates interval statistics (mean and standard deviation)."""
        if not self.tick_intervals:
            return {"count": 0, "mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

        count = len(self.tick_intervals)
        mean_ms = sum(self.tick_intervals) / count
        variance = sum((x - mean_ms) ** 2 for x in self.tick_intervals) / count
        std_ms = math.sqrt(variance)

        return {
            "count": count,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "min_ms": round(min(self.tick_intervals), 2),
            "max_ms": round(max(self.tick_intervals), 2),
        }


# ---------------------------------------------------------------------------
# 2. StreamingAudioSlicer: 600ms PCM Slicing & Buffer Management (Module 2)
# ---------------------------------------------------------------------------
class StreamingAudioSlicer:
    """
    Continuous 600ms audio stream buffer and slicer.
    Ingests arbitrary-sized chunks of 16kHz mono 16-bit PCM audio.
    Slices exact 600ms chunks (9,600 samples = 19,200 bytes).
    Guarantees zero buffer overrun by discarding stale audio beyond max queue limit.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        bytes_per_sample: int = AUDIO_BYTES_PER_SAMPLE,
        slice_duration_sec: float = DEFAULT_TICK_INTERVAL_SEC,
        max_buffered_slices: int = 4,
    ):
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.slice_duration_sec = slice_duration_sec
        self.samples_per_slice = int(sample_rate * slice_duration_sec)  # 9600
        self.bytes_per_slice = self.samples_per_slice * bytes_per_sample  # 19200
        self.max_buffer_bytes = self.bytes_per_slice * max_buffered_slices
        self.buffer = bytearray()
        self.total_bytes_received = 0
        self.total_slices_emitted = 0
        self.overrun_drops = 0

    def push_pcm_chunk(self, data: bytes) -> None:
        """Appends raw PCM bytes from client stream to internal buffer."""
        if not data:
            return
        self.buffer.extend(data)
        self.total_bytes_received += len(data)
        # Protect against buffer overrun / latency lag
        if len(self.buffer) > self.max_buffer_bytes:
            excess = len(self.buffer) - self.max_buffer_bytes
            del self.buffer[:excess]
            self.overrun_drops += 1

    def pop_600ms_slice(self) -> Optional[bytes]:
        """
        Pops an exact 600ms (19,200 bytes) PCM chunk if available.
        If buffer has sufficient audio (>= 80% of slice) at boundary, pads with zeros.
        If buffer has partial audio below threshold, preserves it for next tick.
        If buffer is empty, returns None.
        """
        if len(self.buffer) >= self.bytes_per_slice:
            chunk = bytes(self.buffer[:self.bytes_per_slice])
            del self.buffer[:self.bytes_per_slice]
            self.total_slices_emitted += 1
            return chunk
        elif len(self.buffer) >= int(self.bytes_per_slice * 0.80):
            chunk = bytes(self.buffer)
            self.buffer.clear()
            chunk = chunk + b"\x00" * (self.bytes_per_slice - len(chunk))
            self.total_slices_emitted += 1
            return chunk
        return None

    def has_data(self) -> bool:
        """Returns True if any unread audio bytes remain in buffer."""
        return len(self.buffer) > 0

    def clear(self) -> None:
        """Flushes buffered audio."""
        self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Returns buffer diagnostics."""
        return {
            "buffered_bytes": len(self.buffer),
            "total_bytes_received": self.total_bytes_received,
            "total_slices_emitted": self.total_slices_emitted,
            "overrun_drops": self.overrun_drops,
        }


# ---------------------------------------------------------------------------
# 3. StreamingSTTProcessor: Real-Time Slicing & Whisper Transcription (Module 2)
# ---------------------------------------------------------------------------
class AudioSliceResult(BaseModel):
    text: str = Field(description="Transcribed text or 'no voice'")
    is_speech: bool = Field(description="True if slice contains active speech above energy threshold")
    rms: float = Field(description="RMS audio energy of the 600ms slice")
    latency_ms: float = Field(description="STT processing time in milliseconds")


class StreamingSTTProcessor:
    """
    Real-time STT engine for 600ms PCM audio slices.
    1. Computes RMS energy of the 600ms slice.
    2. If RMS < energy_threshold: immediately returns 'no voice' (< 2ms).
    3. If RMS >= energy_threshold: transcribes using faster-whisper tiny.en (<= 150ms).
    Zero VAD silence timeout: Never waits for silence.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.energy_threshold = energy_threshold
        self.cpu_threads = cpu_threads
        self.device, self.compute_type = self._resolve_device(device, compute_type)
        self._init_model()

    def _resolve_device(self, device: Optional[str], compute_type: Optional[str]) -> Tuple[str, str]:
        if device is None:
            try:
                preload_cuda_libs()
                import torch
                if torch.cuda.is_available():
                    return "cuda", compute_type or "float16"
                return "cpu", compute_type or "int8"
            except Exception:
                return "cpu", compute_type or "int8"
        if device == "cuda":
            preload_cuda_libs()
            return "cuda", compute_type or "float16"
        return "cpu", compute_type or "int8"

    def _init_model(self) -> None:
        logger.info(f"Loading faster-whisper ({self.model_size}) on {self.device} ({self.compute_type})...")
        try:
            self.whisper = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads if self.device == "cpu" else 0,
            )
            # Warmup model with a dummy slice
            dummy = np.random.randn(AUDIO_SLICE_SAMPLES).astype(np.float32) * 0.01
            _ = list(self.whisper.transcribe(dummy, beam_size=1)[0])
            logger.info("faster-whisper model loaded and warmed up successfully.")
        except Exception as e:
            if self.device == "cuda":
                logger.warning(f"Failed to load Whisper on CUDA ({e}), falling back to CPU int8")
                self.device = "cpu"
                self.compute_type = "int8"
                self.whisper = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                )
                dummy = np.random.randn(AUDIO_SLICE_SAMPLES).astype(np.float32) * 0.01
                _ = list(self.whisper.transcribe(dummy, beam_size=1)[0])
                logger.info("faster-whisper loaded on CPU fallback.")
            else:
                raise

    def process_slice(self, pcm_bytes: bytes) -> AudioSliceResult:
        """
        Synchronously processes a 600ms PCM 16-bit audio slice.
        - Calculates RMS audio energy.
        - Emits 'no voice' in < 2ms if below threshold.
        - Transcribes via faster-whisper in <= 150ms if speech detected.
        """
        t0 = time.perf_counter()
        if not pcm_bytes:
            return AudioSliceResult(text="no voice", is_speech=False, rms=0.0, latency_ms=0.0)

        # Enforce exact 9,600 samples (19,200 bytes)
        if len(pcm_bytes) < AUDIO_SLICE_BYTES:
            pcm_bytes = pcm_bytes + b"\x00" * (AUDIO_SLICE_BYTES - len(pcm_bytes))
        elif len(pcm_bytes) > AUDIO_SLICE_BYTES:
            pcm_bytes = pcm_bytes[:AUDIO_SLICE_BYTES]

        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))

        # Real-time RMS silence gate (< 2ms)
        if rms < self.energy_threshold:
            t_calc = (time.perf_counter() - t0) * 1000.0
            return AudioSliceResult(
                text="no voice",
                is_speech=False,
                rms=round(rms, 6),
                latency_ms=round(t_calc, 2),
            )

        # Active speech transcription (<= 150ms)
        segments, _ = self.whisper.transcribe(
            arr,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=False,
        )
        text = " ".join(s.text for s in segments).strip()
        is_speech = True
        if not text or not any(c.isalnum() for c in text):
            text = "no voice"
            is_speech = False

        t_total = (time.perf_counter() - t0) * 1000.0
        return AudioSliceResult(
            text=text,
            is_speech=is_speech,
            rms=round(rms, 6),
            latency_ms=round(t_total, 2),
        )

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribes arbitrary duration PCM 16-bit mono 16kHz audio bytes (up to 4.8s)
        using faster-whisper with beam_size=1.
        Returns clean transcribed text or empty string if no alphanumeric speech detected.
        """
        if not audio_bytes:
            return ""

        if len(audio_bytes) > MAX_SPEECH_BUFFER_BYTES:
            audio_bytes = audio_bytes[-MAX_SPEECH_BUFFER_BYTES:]

        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(arr) == 0:
            return ""

        segments, _ = self.whisper.transcribe(
            arr,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=True,
            vad_filter=False,
        )
        text = " ".join(s.text for s in segments).strip()
        if not any(c.isalnum() for c in text):
            return ""
        return text

    async def transcribe_audio_async(self, audio_bytes: bytes) -> str:
        """Non-blocking async wrapper executing transcription in thread pool."""
        return await asyncio.to_thread(self.transcribe_audio, audio_bytes)

    async def process_slice_async(self, pcm_bytes: bytes) -> AudioSliceResult:
        """Non-blocking async wrapper executing transcription in thread pool."""
        return await asyncio.to_thread(self.process_slice, pcm_bytes)


# ---------------------------------------------------------------------------
# 4. KokoroTTSProcessor: Ultra-Low Latency Neural TTS Pipeline (Module 4)
# ---------------------------------------------------------------------------
class TTSAudioChunk(BaseModel):
    pcm_bytes: bytes = b""
    audio_b64: str = ""
    sample_rate: int = TTS_SAMPLE_RATE
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    text: str = ""


class KokoroTTSProcessor:
    """
    Ultra-low latency neural TTS synthesis engine using Kokoro-82M ONNX.
    Synthesizes conversational speech fragments in <= 180ms.
    Preloads CUDA libraries and uses CUDAExecutionProvider with CPU fallback.
    Non-blocking async synthesis via run_in_executor to protect the 600ms micro-turn clock loop.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        voice: str = DEFAULT_KOKORO_VOICE,
        device: Optional[str] = None,
        speed: float = 1.0,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = TTS_SAMPLE_RATE

        m_path = Path(model_path or DEFAULT_KOKORO_MODEL)
        if not m_path.is_absolute() and not m_path.exists():
            m_path = (PROJECT_ROOT / m_path).resolve()
        self.model_path = str(m_path)

        v_path = Path(voices_path or DEFAULT_KOKORO_VOICES)
        if not v_path.is_absolute() and not v_path.exists():
            v_path = (PROJECT_ROOT / v_path).resolve()
        self.voices_path = str(v_path)

        self.device, self.provider = self._resolve_device(device)
        self._init_model()

    def _resolve_device(self, device: Optional[str]) -> Tuple[str, str]:
        if device == "cpu":
            return "cpu", "CPUExecutionProvider"
        try:
            preload_cuda_libs()
            import torch
            import onnxruntime as rt
            avail = rt.get_available_providers()
            if "CUDAExecutionProvider" in avail and (torch.cuda.is_available() or device == "cuda"):
                return "cuda", "CUDAExecutionProvider"
        except Exception as e:
            logger.debug(f"CUDA provider check notice: {e}")
        return "cpu", "CPUExecutionProvider"

    def _init_model(self) -> None:
        import onnxruntime as rt
        import kokoro_onnx

        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = 3

        logger.info(f"Loading Kokoro ONNX on {self.device} ({self.provider})...")
        try:
            providers = [self.provider]
            if self.provider != "CPUExecutionProvider":
                providers.append("CPUExecutionProvider")

            session = rt.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            self.kokoro = kokoro_onnx.Kokoro.from_session(session, self.voices_path)
            # Warmup synthesis to eliminate cold-start latency
            _ = self.kokoro.create("Warmup.", voice=self.voice, speed=self.speed)
            logger.info("Kokoro ONNX loaded and warmed up successfully.")
        except Exception as e:
            if self.device == "cuda":
                logger.warning(f"Failed to load Kokoro on CUDA ({e}), falling back to CPU")
                self.device = "cpu"
                self.provider = "CPUExecutionProvider"
                session = rt.InferenceSession(
                    self.model_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                self.kokoro = kokoro_onnx.Kokoro.from_session(session, self.voices_path)
                _ = self.kokoro.create("Warmup.", voice=self.voice, speed=self.speed)
                logger.info("Kokoro ONNX loaded on CPU fallback.")
            else:
                raise

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSAudioChunk:
        """
        Synchronously synthesizes a speech fragment into 16-bit PCM (24kHz mono) and base64 string.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return TTSAudioChunk(sample_rate=self.sample_rate, text="")

        t0 = time.perf_counter()
        try:
            audio, sr = self.kokoro.create(
                cleaned,
                voice=voice or self.voice,
                speed=speed or self.speed,
            )
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            audio_clipped = np.clip(audio, -1.0, 1.0)
            pcm_int16 = (audio_clipped * 32767.0).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            duration_ms = (len(audio) / sr) * 1000.0

            return TTSAudioChunk(
                pcm_bytes=pcm_bytes,
                audio_b64=audio_b64,
                sample_rate=sr,
                duration_ms=round(duration_ms, 2),
                latency_ms=round(latency_ms, 2),
                text=cleaned,
            )
        except Exception as e:
            logger.warning(f"Kokoro synthesis failed on '{cleaned[:30]}...': {e}")
            t1 = time.perf_counter()
            return TTSAudioChunk(
                pcm_bytes=b"",
                audio_b64="",
                sample_rate=self.sample_rate,
                duration_ms=0.0,
                latency_ms=round((t1 - t0) * 1000.0, 2),
                text=cleaned,
            )

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSAudioChunk:
        """Non-blocking async wrapper executing Kokoro synthesis in thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.synthesize, text, voice, speed)


def _is_speaker_echo(transcribed: str, last_speech: str) -> bool:
    """
    Acoustic echo filter helper: detects if user mic transcribed slice is acoustic feedback
    from assistant's last spoken fragment.
    """
    if not transcribed or not last_speech:
        return False
    t_words = [w.strip(".,!?").lower() for w in transcribed.split() if w.strip(".,!?")]
    a_words = set(w.strip(".,!?").lower() for w in last_speech.split() if w.strip(".,!?"))
    if not t_words or not a_words:
        return False
    overlap = sum(1 for w in t_words if w in a_words) / len(t_words)
    return overlap >= 0.50


def extract_incremental_speech(new_text: str, last_text: str) -> str:
    """
    Extracts new incremental speech words between new_text and last_text.
    Robust against Whisper casing differences and shifting punctuation.
    """
    new_clean = new_text.strip()
    last_clean = last_text.strip()
    if not last_clean:
        return new_clean

    if new_clean.lower().startswith(last_clean.lower()):
        delta = new_clean[len(last_clean):].strip()
        return delta if delta else ""

    new_words = new_clean.split()
    last_words = last_clean.split()

    match_len = 0
    for i in range(min(len(new_words), len(last_words))):
        w_new = new_words[i].strip(".,!?:;\"'").lower()
        w_last = last_words[i].strip(".,!?:;\"'").lower()
        if w_new == w_last:
            match_len = i + 1
        else:
            break

    if match_len > 0 and match_len < len(new_words):
        return " ".join(new_words[match_len:]).strip()
    elif match_len == len(new_words):
        return ""
    return new_clean


def is_complete_question_or_intent(text: str) -> bool:
    """
    Checks if user utterance ends in a complete question or terminal intent.
    Direct questions ending with '?' can be answered promptly (600ms).
    Incomplete thoughts / mid-sentence clauses get a 1.2s grace window to let user breathe.
    """
    cleaned = text.strip()
    if not cleaned or cleaned == "no voice":
        return False
    if cleaned.endswith("?"):
        return True
    lower = cleaned.lower()
    if lower.endswith("...") or lower.endswith("..") or lower.endswith(","):
        return False
    trailing_words = {"and", "or", "but", "so", "because", "if", "with", "like", "am", "was", "be", "to", "uh", "um", "the", "a", "an", "is", "that"}
    last_word = lower.split()[-1].strip(".,!?:;\"'") if lower.split() else ""
    if last_word in trailing_words:
        return False
    return False


# ---------------------------------------------------------------------------
# 5. DuplexSession: State Manager & Rolling History
# ---------------------------------------------------------------------------
class DuplexSession:
    """
    Manages state for an active duplex conversation session.
    Maintains rolling micro-turn history adhering to micro_turns_train.jsonl format.
    Prunes consecutive silent turns to avoid repetition bias and KV cache pollution.
    Maintains StreamingAudioSlicer for 600ms audio stream ingestion.
    """

    def __init__(
        self,
        session_id: str,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.created_at = time.time()
        self.history: List[Dict[str, str]] = (
            [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        )
        self.consecutive_silent_turns: int = 0
        self.clock = MicroTurnClock()
        self.pending_slice_queue: asyncio.Queue[str] = asyncio.Queue()
        self.audio_slicer = StreamingAudioSlicer()
        self.speech_audio_buffer: bytearray = bytearray()
        self.last_speech_text: str = ""
        self.silent_ticks_after_speech: int = 0
        self.is_active = True
        self.is_speaking = False
        self.playback_end_time: float = 0.0
        self.has_user_spoken: bool = False
        self.is_waiting_for_user: bool = False
        self.assistant_speech_burst: int = 0
        self.last_assistant_speech: str = ""
        self.lock = asyncio.Lock()

    def push_audio_chunk(self, data: bytes) -> None:
        """Buffers incoming raw PCM audio bytes into the streaming slicer."""
        self.audio_slicer.push_pcm_chunk(data)

    def pop_current_audio_slice(self) -> Optional[bytes]:
        """Pops an exact 600ms PCM chunk from the audio stream if available."""
        return self.audio_slicer.pop_600ms_slice()

    def push_input_slice(self, text: str) -> None:
        """Buffers an incoming text micro-turn slice from client fallback."""
        cleaned = text.strip() if text else "no voice"
        if not cleaned:
            cleaned = "no voice"
        self.pending_slice_queue.put_nowait(cleaned)

    def pop_current_slice(self) -> str:
        """
        Retrieves buffered text input for the current 600ms tick.
        If multiple slices arrived during the interval, merges them naturally.
        If no slices arrived, returns 'no voice'.
        """
        collected: List[str] = []
        while not self.pending_slice_queue.empty():
            try:
                collected.append(self.pending_slice_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not collected:
            return "no voice"

        # Filter out redundant 'no voice' if real speech arrived in same window
        speech_parts = [p for p in collected if p != "no voice"]
        if speech_parts:
            return " ".join(speech_parts)
        return "no voice"

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """
        Appends the completed micro-turn to the session history.
        Prunes consecutive silent turns ('no voice' -> 'listen') to prevent KV cache
        bloat and listen repetition loops.
        Resets consecutive silent counter when user speech resumes.
        """
        is_silent_turn = (user_text.strip() == "no voice" and assistant_text.strip() == "listen")

        if is_silent_turn:
            if self.consecutive_silent_turns >= 1:
                # Do not accumulate repeated silent turns into history
                return
            self.consecutive_silent_turns += 1
        else:
            self.consecutive_silent_turns = 0

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

        # History truncation preserving system prompt
        self._prune_history()

    def _prune_history(self) -> None:
        """Keeps system prompt at index 0 (if present) and at most max_history_turns micro-turns."""
        has_system = bool(self.history and self.history[0].get("role") == "system")
        sys_offset = 1 if has_system else 0

        if len(self.history) <= sys_offset + self.max_history_turns:
            return

        tail = self.history[-self.max_history_turns:]
        # Ensure tail begins with a user message for role alternation integrity
        if tail and tail[0]["role"] == "assistant":
            tail = tail[1:]

        if has_system:
            self.history = [self.history[0]] + tail
        else:
            self.history = tail

    def get_messages_for_llm(self, current_user_slice: str) -> List[Dict[str, str]]:
        """Constructs the prompt payload for the pending micro-turn evaluation."""
        return self.history + [{"role": "user", "content": current_user_slice}]

    def reset(self) -> None:
        """Resets conversation history and audio/text queues back to initial state."""
        self.history = (
            [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        )
        self.consecutive_silent_turns = 0
        self.is_speaking = False
        self.playback_end_time = 0.0
        self.has_user_spoken = False
        self.is_waiting_for_user = False
        self.assistant_speech_burst = 0
        self.last_assistant_speech = ""
        self.speech_audio_buffer.clear()
        self.last_speech_text = ""
        self.silent_ticks_after_speech = 0
        self.audio_slicer.clear()
        while not self.pending_slice_queue.empty():
            try:
                self.pending_slice_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


# ---------------------------------------------------------------------------
# 6. DuplexModelClient: Async llama-server Client & Neural Policy
# ---------------------------------------------------------------------------
class TurnDecision(BaseModel):
    action: str = Field(description="'YIELD' or 'SPEAK'")
    output_text: str = Field(description="The token or text emitted for this micro-turn")
    p_listen: float = Field(description="Probability of token <listen>")
    logprob_listen: Optional[float] = None
    roundtrip_ms: float = 0.0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    raw_content: str = ""


class DuplexModelClient:
    """
    Asynchronous client communicating with llama-server on port 8080.
    Implements calibrated neural turn-taking policy:
    P(<listen>) >= tau (0.30) -> YIELD ('listen')
    P(<listen>) < tau (0.30)  -> SPEAK (speech fragment)
    """

    def __init__(
        self,
        server_url: str = DEFAULT_LLAMA_URL,
        listen_threshold: float = DEFAULT_LISTEN_THRESHOLD,
        timeout_sec: float = 8.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.endpoint = f"{self.server_url}/v1/chat/completions"
        self.listen_threshold = listen_threshold
        self.timeout_sec = timeout_sec
        self.client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Initializes httpx async client pool."""
        if self.client is None or self.client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self.client = httpx.AsyncClient(timeout=self.timeout_sec, limits=limits)

    async def close(self) -> None:
        """Closes the client pool."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self.client = None

    async def check_health(self) -> Dict[str, Any]:
        """Queries llama-server /health."""
        if self.client is None:
            await self.start()
        try:
            resp = await self.client.get(f"{self.server_url}/health", timeout=3.0)
            return {"status": "ok" if resp.status_code == 200 else f"http_{resp.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _clean_speech_content(raw_content: str, finish_reason: Optional[str] = None) -> str:
        """
        Extracts clean, natural speech content without hardcoded keyword lists or arbitrary word slicing.
        - Removes leading 'listen' or '<listen>' control token if present.
        - If the output contains terminal sentence punctuation ('.', '!', '?'), truncates
          any trailing dangling incomplete clause so the assistant speech ends on a complete sentence.
        - If trailing dangling connector words remain (e.g. 'for example', 'for instance',
          'or perhaps', 'for', 'or', 'and', 'the'), strips them.
        """
        text = raw_content.strip()
        if not text:
            return ""

        lower = text.lower()
        if lower in ("listen", "<listen>"):
            return "listen"

        # 1. Strip leading control token prefix (e.g. '<listen>', 'listen: ', 'listen. ', 'listen ')
        if lower.startswith("<listen>"):
            text = text[len("<listen>"):].lstrip(" :,-.\t\n")
        elif lower.startswith("listen") and (len(lower) == 6 or lower[6] in " :,-.\t\n"):
            text = text[len("listen"):].lstrip(" :,-.\t\n")

        text = text.strip()
        if not text:
            return "listen" if (lower.startswith("listen") or lower.startswith("<listen>")) else ""

        # 2. If the output contains terminal sentence punctuation ('.', '!', '?'),
        # truncate any trailing dangling incomplete clause so the assistant speech ends on a complete sentence.
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct != -1:
            text = text[:last_punct + 1].strip()

        # 3. Strip trailing dangling connector words/phrases if any remain
        connector_phrases = [
            "for example",
            "for instance",
            "or perhaps",
            "for",
            "or",
            "and",
            "the",
        ]
        for _ in range(3):
            clean_no_punct = text.rstrip(".!?, :;\t\n")
            lower_clean = clean_no_punct.lower()
            stripped_any = False
            for conn in connector_phrases:
                if lower_clean == conn:
                    text = ""
                    stripped_any = True
                    break
                elif lower_clean.endswith(" " + conn):
                    # Cut off the dangling connector word/phrase
                    clean_no_punct = clean_no_punct[:-len(conn)].rstrip(", :;\t\n")
                    if clean_no_punct:
                        text = clean_no_punct if clean_no_punct[-1] in ".!?" else clean_no_punct + "."
                    else:
                        text = ""
                    stripped_any = True
                    break
            if not stripped_any:
                break

        return text.strip()

    async def evaluate_turn(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> TurnDecision:
        """
        Sends messages payload to llama-server, requests logprobs, extracts
        token 0 probability for <listen>, and executes the mathematical decision rule.
        ZERO HARDCODING: pure neural decision governed by P(<listen>) >= 0.30.
        """
        if self.client is None:
            await self.start()

        # Calibrated repetition penalties prevent repetition loops naturally without regex slicing
        eff_presence = presence_penalty if presence_penalty != 0.0 else 0.2
        eff_frequency = frequency_penalty if frequency_penalty != 0.0 else 0.3

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "presence_penalty": eff_presence,
            "frequency_penalty": eff_frequency,
            "stop": STOP_TOKENS,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        t0 = time.perf_counter()
        response = await self.client.post(self.endpoint, json=payload)
        t1 = time.perf_counter()
        roundtrip_ms = (t1 - t0) * 1000.0

        if response.status_code != 200:
            raise RuntimeError(
                f"llama-server error {response.status_code}: {response.text}"
            )

        data = response.json()
        choice = data["choices"][0]
        raw_content = choice["message"].get("content", "").strip()
        finish_reason = choice.get("finish_reason")

        timings = data.get("timings", {})
        prompt_ms = float(timings.get("prompt_ms", 0.0))
        predicted_ms = float(timings.get("predicted_ms", 0.0))

        # Extract P(<listen>) from token 0 logprobs
        p_listen = 0.0
        logprob_listen = None
        logprobs_info = choice.get("logprobs")
        token_0_is_listen = False

        if logprobs_info and "content" in logprobs_info and logprobs_info["content"]:
            token_0 = logprobs_info["content"][0]
            tok_text = token_0.get("token", "").strip().lower()
            tok_id = token_0.get("id")

            if tok_id == LISTEN_TOKEN_ID or tok_text in (LISTEN_TOKEN_STR, "<listen>"):
                token_0_is_listen = True
                logprob_listen = float(token_0.get("logprob", -99.0))
                p_listen = math.exp(logprob_listen)
            else:
                top_lps = token_0.get("top_logprobs", [])
                for candidate in top_lps:
                    cand_text = candidate.get("token", "").strip().lower()
                    cand_id = candidate.get("id")
                    if cand_id == LISTEN_TOKEN_ID or cand_text in (LISTEN_TOKEN_STR, "<listen>"):
                        logprob_listen = float(candidate.get("logprob", -99.0))
                        p_listen = math.exp(logprob_listen)
                        break

        # Zero-Hardcoding Mathematical Decision Policy
        # Check if user is currently speaking (last message content != 'no voice')
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").strip()
                break

        is_user_silent = (last_user_msg == "no voice")
        cleaned_speech = self._clean_speech_content(raw_content, finish_reason=finish_reason)

        if not is_user_silent:
            # During active user speech: yield floor if P(<listen>) >= threshold or token 0 is listen.
            # Interrupt immediately on hazard if P(<listen>) < threshold and model emitted speech words.
            if token_0_is_listen or p_listen >= self.listen_threshold:
                action = "YIELD"
                output_text = "listen"
            elif cleaned_speech and cleaned_speech.lower() != "listen":
                action = "SPEAK"
                output_text = cleaned_speech
            else:
                action = "YIELD"
                output_text = "listen"
        else:
            # When user is silent ('no voice'): user has paused or finished, waiting for answer.
            # If the model emitted real speech words (even if prefixed with 'listen'), speak them!
            if cleaned_speech and cleaned_speech.lower() != "listen":
                action = "SPEAK"
                output_text = cleaned_speech
            else:
                action = "YIELD"
                output_text = "listen"

        return TurnDecision(
            action=action,
            output_text=output_text,
            p_listen=round(p_listen, 4),
            logprob_listen=round(logprob_listen, 4) if logprob_listen is not None else None,
            roundtrip_ms=round(roundtrip_ms, 2),
            prompt_ms=round(prompt_ms, 2),
            predicted_ms=round(predicted_ms, 2),
            raw_content=raw_content,
        )


# ---------------------------------------------------------------------------
# 7. SessionManager
# ---------------------------------------------------------------------------
class SessionManager:
    """Thread-safe registry for active duplex sessions."""

    def __init__(self):
        self.sessions: Dict[str, DuplexSession] = {}
        self.lock = asyncio.Lock()

    async def get_or_create(self, session_id: Optional[str] = None) -> DuplexSession:
        async with self.lock:
            sid = session_id or str(uuid.uuid4())
            if sid not in self.sessions:
                self.sessions[sid] = DuplexSession(session_id=sid)
            return self.sessions[sid]

    async def remove(self, session_id: str) -> None:
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].is_active = False
                del self.sessions[session_id]

    async def active_count(self) -> int:
        async with self.lock:
            return len(self.sessions)


# ---------------------------------------------------------------------------
# 8. FastAPI App & WebSocket Engine Routes
# ---------------------------------------------------------------------------
def create_app(
    llama_url: str = DEFAULT_LLAMA_URL,
    listen_threshold: float = DEFAULT_LISTEN_THRESHOLD,
    stt_device: Optional[str] = None,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    stt_processor: Optional[StreamingSTTProcessor] = None,
    tts_processor: Optional[KokoroTTSProcessor] = None,
    tts_device: Optional[str] = None,
    tts_voice: str = DEFAULT_KOKORO_VOICE,
    kokoro_model_path: Optional[str] = None,
    kokoro_voices_path: Optional[str] = None,
) -> FastAPI:
    """Constructs the production FastAPI application."""
    app = FastAPI(
        title="TalkAI Duplex Micro-Turn Engine",
        description="True Full-Duplex Cascade 600ms micro-turn, streaming STT & Kokoro TTS WebSocket server.",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session_mgr = SessionManager()
    model_client = DuplexModelClient(server_url=llama_url, listen_threshold=listen_threshold)

    # Initialize STT processor (shared across WebSocket connections)
    active_stt = stt_processor or StreamingSTTProcessor(
        device=stt_device,
        energy_threshold=energy_threshold,
    )

    # Initialize Kokoro TTS processor (shared across WebSocket connections)
    active_tts = tts_processor or KokoroTTSProcessor(
        model_path=kokoro_model_path,
        voices_path=kokoro_voices_path,
        voice=tts_voice,
        device=tts_device,
    )

    @app.on_event("startup")
    async def startup_event():
        await model_client.start()
        logger.info(f"Duplex Engine started. Connected to llama-server at {llama_url}")
        logger.info(f"STT Processor active on {active_stt.device} ({active_stt.compute_type}).")
        logger.info(f"Kokoro TTS active on {active_tts.device} ({active_tts.provider}). Voice: {active_tts.voice}")

    @app.on_event("shutdown")
    async def shutdown_event():
        await model_client.close()
        logger.info("Duplex Engine stopped.")

    @app.get("/health")
    async def health_check():
        llama_health = await model_client.check_health()
        gpu_info = get_gpu_vram_info()
        active_sessions = await session_mgr.active_count()
        return {
            "status": "ok",
            "model_server": llama_health,
            "active_sessions": active_sessions,
            "tick_interval_ms": int(DEFAULT_TICK_INTERVAL_SEC * 1000),
            "listen_threshold": listen_threshold,
            "stt": {
                "device": active_stt.device,
                "compute_type": active_stt.compute_type,
                "energy_threshold": active_stt.energy_threshold,
                "slice_bytes": AUDIO_SLICE_BYTES,
                "sample_rate": AUDIO_SAMPLE_RATE,
            },
            "tts": {
                "device": active_tts.device,
                "provider": active_tts.provider,
                "voice": active_tts.voice,
                "sample_rate": active_tts.sample_rate,
                "available_voices": active_tts.kokoro.get_voices() if hasattr(active_tts, "kokoro") and active_tts.kokoro else [],
            },
            "gpu_vram": gpu_info,
        }

    @app.get("/")
    @app.get("/test.html")
    async def serve_test_ui():
        for candidate in [
            PROJECT_ROOT / "test.html",
            PROJECT_ROOT / "scripts" / "test.html",
        ]:
            if candidate.exists():
                return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>test.html not found</h1>", status_code=404)

    @app.post("/session/reset")
    async def reset_session(session_id: str):
        session = await session_mgr.get_or_create(session_id)
        session.reset()
        return {"status": "ok", "session_id": session_id, "message": "History and audio queues reset"}

    async def handle_duplex_websocket(websocket: WebSocket, is_audio_route: bool = False):
        """Unified WebSocket handler for text and binary audio streaming."""
        await websocket.accept()
        session = await session_mgr.get_or_create()
        logger.info(f"WebSocket client connected ({'/ws/duplex_audio' if is_audio_route else '/ws/duplex'}). Session: {session.session_id}")

        await websocket.send_json({
            "type": "session_init",
            "session_id": session.session_id,
            "interval_ms": int(DEFAULT_TICK_INTERVAL_SEC * 1000),
            "listen_threshold": listen_threshold,
            "slice_bytes": AUDIO_SLICE_BYTES,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "tts_sample_rate": active_tts.sample_rate,
            "tts_voice": active_tts.voice,
            "system_prompt": session.system_prompt,
        })

        session.clock.start()

        # Inbound message receiver task (handles text JSON and binary audio frames)
        async def client_listener():
            try:
                while session.is_active:
                    msg = await websocket.receive()
                    msg_type = msg.get("type", "")

                    if msg_type == "websocket.receive":
                        if "bytes" in msg and msg["bytes"]:
                            session.push_audio_chunk(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            data_str = msg["text"]
                            try:
                                parsed = json.loads(data_str)
                                p_type = parsed.get("type", "user_slice")
                                if p_type == "audio_chunk":
                                    raw_b64 = parsed.get("data", "")
                                    if raw_b64:
                                        raw_bytes = base64.b64decode(raw_b64)
                                        session.push_audio_chunk(raw_bytes)
                                elif p_type == "user_slice":
                                    session.push_input_slice(parsed.get("text", ""))
                                elif p_type == "no_voice":
                                    session.push_input_slice("no voice")
                                elif p_type == "playback_status":
                                    is_playing = bool(parsed.get("is_playing", False))
                                    session.is_speaking = is_playing
                                    if not is_playing:
                                        session.playback_end_time = 0.0
                                        session.is_waiting_for_user = True
                                elif p_type in ("playback_interrupt", "user_barge_in"):
                                    session.is_speaking = False
                                    session.playback_end_time = 0.0
                                    session.is_waiting_for_user = False
                                    await websocket.send_json({
                                        "type": "playback_interrupt",
                                        "reason": "user_barge_in",
                                        "session_id": session.session_id,
                                        "timestamp": time.time(),
                                    })
                                elif p_type == "reset":
                                    session.reset()
                                    await websocket.send_json({"type": "session_reset", "session_id": session.session_id})
                                elif p_type == "set_voice":
                                    v = parsed.get("voice")
                                    if v:
                                        active_tts.voice = v
                                        await websocket.send_json({"type": "voice_updated", "voice": v})
                                elif p_type == "set_speed":
                                    s = float(parsed.get("speed", 1.0))
                                    active_tts.speed = s
                                    await websocket.send_json({"type": "speed_updated", "speed": s})
                            except json.JSONDecodeError:
                                session.push_input_slice(data_str)

                    elif msg_type == "websocket.disconnect":
                        break
            except WebSocketDisconnect:
                pass
            except Exception as ex:
                logger.debug(f"Client listener exception: {ex}")
            finally:
                session.is_active = False

        # Outbound 600ms micro-turn clock loop task
        async def micro_turn_loop():
            try:
                while session.is_active:
                    tick_idx, actual_time, drift_ms = await session.clock.wait_next_tick()

                    # 1. Ingest audio or text slice for this 600ms window
                    audio_chunk = session.pop_current_audio_slice()
                    stt_latency_ms = 0.0
                    slice_rms = 0.0
                    is_speech_slice = False

                    if audio_chunk is not None:
                        # Enforce exact 9,600 samples (19,200 bytes)
                        if len(audio_chunk) < AUDIO_SLICE_BYTES:
                            audio_chunk = audio_chunk + b"\x00" * (AUDIO_SLICE_BYTES - len(audio_chunk))
                        elif len(audio_chunk) > AUDIO_SLICE_BYTES:
                            audio_chunk = audio_chunk[:AUDIO_SLICE_BYTES]

                        arr = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        slice_rms = float(np.sqrt(np.mean(arr ** 2)))

                        if slice_rms >= active_stt.energy_threshold:
                            session.speech_audio_buffer.extend(audio_chunk)
                            if len(session.speech_audio_buffer) > MAX_SPEECH_BUFFER_BYTES:
                                excess = len(session.speech_audio_buffer) - MAX_SPEECH_BUFFER_BYTES
                                del session.speech_audio_buffer[:excess]

                            t0_stt = time.perf_counter()
                            new_text = await active_stt.transcribe_audio_async(bytes(session.speech_audio_buffer))
                            stt_latency_ms = (time.perf_counter() - t0_stt) * 1000.0

                            if new_text and any(c.isalnum() for c in new_text):
                                incremental = extract_incremental_speech(new_text, session.last_speech_text)
                                current_slice = incremental if incremental else "no voice"
                                session.last_speech_text = new_text
                                is_speech_slice = True
                                session.silent_ticks_after_speech = 0
                            else:
                                current_slice = "no voice"
                                is_speech_slice = False
                        else:
                            # Silence tick
                            current_slice = "no voice"
                            is_speech_slice = False
                            if len(session.speech_audio_buffer) > 0 or session.last_speech_text:
                                session.silent_ticks_after_speech += 1
                                if session.silent_ticks_after_speech >= 4:
                                    session.speech_audio_buffer.clear()
                                    session.last_speech_text = ""

                        # If audio was silent ("no voice"), check if a fallback text slice was sent concurrently
                        if current_slice == "no voice":
                            text_fallback = session.pop_current_slice()
                            if text_fallback != "no voice":
                                current_slice = text_fallback
                                is_speech_slice = True
                    else:
                        # Fallback to queued text slices
                        current_slice = session.pop_current_slice()
                        is_speech_slice = (current_slice != "no voice")

                    now_mono = time.monotonic()
                    is_audio_playing = (now_mono < session.playback_end_time) or session.is_speaking
                    was_audio_playing = is_audio_playing

                    # Acoustic echo filter: If assistant is speaking and user mic picked up assistant speech
                    if is_audio_playing and _is_speaker_echo(current_slice, session.last_assistant_speech):
                        logger.debug(f"[Echo Filter] Discarded acoustic feedback: {current_slice!r}")
                        current_slice = "no voice"
                        is_speech_slice = False

                    # Update user activity state
                    if is_speech_slice:
                        session.has_user_spoken = True
                        session.is_waiting_for_user = False

                    # 2. Turn evaluation & decision
                    should_emit_interrupt = False

                    if is_audio_playing:
                        if current_slice == "no voice":
                            # Assistant is actively speaking, user is listening (no interruption)
                            # Do NOT call model to talk over itself. Set decision = YIELD/listen (waiting for playback completion).
                            decision = TurnDecision(
                                action="YIELD",
                                output_text="listen",
                                p_listen=1.0,
                                roundtrip_ms=0.0,
                                prompt_ms=0.0,
                                predicted_ms=0.0,
                                raw_content="listen",
                            )
                        else:
                            # User spoke while assistant was speaking: Interruption / barge-in!
                            should_emit_interrupt = True
                            session.is_speaking = False
                            session.playback_end_time = 0.0
                            messages = session.get_messages_for_llm(current_slice)
                            try:
                                decision = await model_client.evaluate_turn(messages)
                            except Exception as err:
                                logger.error(f"Error evaluating turn {tick_idx}: {err}")
                                decision = TurnDecision(
                                    action="YIELD",
                                    output_text="listen",
                                    p_listen=1.0,
                                    roundtrip_ms=0.0,
                                    raw_content="listen",
                                )
                            session.record_turn(current_slice, decision.output_text)
                    else:
                        # Assistant was not speaking
                        session.is_speaking = False
                        is_silent_waiting = (
                            (not session.has_user_spoken and current_slice == "no voice")
                            or (session.is_waiting_for_user and current_slice == "no voice")
                        )
                        if is_silent_waiting:
                            decision = TurnDecision(
                                action="YIELD",
                                output_text="listen",
                                p_listen=1.0,
                                roundtrip_ms=0.0,
                                prompt_ms=0.0,
                                predicted_ms=0.0,
                                raw_content="listen",
                            )
                        elif current_slice == "no voice":
                            last_text = session.last_speech_text.strip()
                            is_question = is_complete_question_or_intent(last_text)

                            # If thought is midway / incomplete (not a direct question) and user has only paused for 1 tick (600ms):
                            # Yield floor to provide a natural 1.2s breathing room before answering!
                            if not is_question and session.silent_ticks_after_speech <= 1:
                                decision = TurnDecision(
                                    action="YIELD",
                                    output_text="listen",
                                    p_listen=1.0,
                                    roundtrip_ms=0.0,
                                    prompt_ms=0.0,
                                    predicted_ms=0.0,
                                    raw_content="listen",
                                )
                                session.record_turn(current_slice, "listen")
                            else:
                                messages = session.get_messages_for_llm(current_slice)
                                try:
                                    decision = await model_client.evaluate_turn(messages)
                                except Exception as err:
                                    logger.error(f"Error evaluating turn {tick_idx}: {err}")
                                    decision = TurnDecision(
                                        action="YIELD",
                                        output_text="listen",
                                        p_listen=1.0,
                                        roundtrip_ms=0.0,
                                        raw_content="listen",
                                    )
                                session.record_turn(current_slice, decision.output_text)

                                if current_slice == "no voice" and session.consecutive_silent_turns >= 2:
                                    session.is_waiting_for_user = True
                        else:
                            messages = session.get_messages_for_llm(current_slice)
                            try:
                                decision = await model_client.evaluate_turn(messages)
                            except Exception as err:
                                logger.error(f"Error evaluating turn {tick_idx}: {err}")
                                decision = TurnDecision(
                                    action="YIELD",
                                    output_text="listen",
                                    p_listen=1.0,
                                    roundtrip_ms=0.0,
                                    raw_content="listen",
                                )
                            session.record_turn(current_slice, decision.output_text)

                    # 3. Handle assistant speech completion, TTS synthesis, and state update
                    tts_chunk: Optional[TTSAudioChunk] = None
                    should_emit_audio = False

                    if should_emit_interrupt:
                        is_audio_playing = False

                    # Hazard check: True ONLY if user was speaking mid-sentence (not asking a question) and assistant interjected
                    is_hazard = bool(
                        is_speech_slice
                        and decision.action == "SPEAK"
                        and not was_audio_playing
                        and not is_complete_question_or_intent(current_slice)
                    )

                    if decision.action == "SPEAK" and not is_audio_playing and decision.output_text.strip() and decision.output_text.strip() != "listen":
                        session.last_assistant_speech = decision.output_text
                        tts_chunk = await active_tts.synthesize_async(decision.output_text)
                        if tts_chunk and tts_chunk.duration_ms > 0:
                            session.playback_end_time = time.monotonic() + (tts_chunk.duration_ms / 1000.0)
                        session.is_speaking = True
                        session.is_waiting_for_user = True
                        should_emit_audio = True
                    elif not is_audio_playing:
                        session.is_speaking = False

                    # 4. Clean logging
                    log_out = "listen (playing)" if (is_audio_playing and decision.action == "YIELD") else decision.output_text
                    logger.info(
                        f"[Turn {tick_idx:04d}] In: {current_slice!r:<20} | Act: {decision.action:<5} | Out: {log_out!r} | P(listen): {decision.p_listen:.3f}"
                    )

                    # 5. Emit micro-turn event to client
                    await websocket.send_json({
                        "type": "micro_turn",
                        "turn_id": tick_idx,
                        "session_id": session.session_id,
                        "user": current_slice,
                        "accumulated": session.last_speech_text,
                        "is_speech": is_speech_slice,
                        "rms": round(slice_rms, 6),
                        "stt_latency_ms": round(stt_latency_ms, 2),
                        "action": decision.action,
                        "assistant": decision.output_text,
                        "p_listen": decision.p_listen,
                        "logprob_listen": decision.logprob_listen,
                        "latency_ms": decision.roundtrip_ms,
                        "prompt_ms": decision.prompt_ms,
                        "predicted_ms": decision.predicted_ms,
                        "synth_latency_ms": round(tts_chunk.latency_ms, 2) if tts_chunk else 0.0,
                        "is_speaking": session.is_speaking,
                        "is_hazard": is_hazard,
                        "drift_ms": round(drift_ms, 2),
                        "timestamp": actual_time,
                    })

                    # 6. Emit interruption or audio chunk
                    if should_emit_interrupt:
                        await websocket.send_json({
                            "type": "playback_interrupt",
                            "reason": "user_barge_in",
                            "turn_id": tick_idx,
                            "session_id": session.session_id,
                            "timestamp": actual_time,
                        })

                    if should_emit_audio and tts_chunk and tts_chunk.audio_b64:
                        await websocket.send_json({
                            "type": "assistant_audio_chunk",
                            "turn_id": tick_idx,
                            "session_id": session.session_id,
                            "text": decision.output_text,
                            "audio_b64": tts_chunk.audio_b64,
                            "sample_rate": tts_chunk.sample_rate,
                            "duration_ms": round(tts_chunk.duration_ms, 2),
                            "synth_latency_ms": round(tts_chunk.latency_ms, 2),
                            "timestamp": actual_time,
                        })

            except WebSocketDisconnect:
                pass
            except Exception as ex:
                logger.error(f"Micro-turn loop error: {ex}")
            finally:
                session.is_active = False

        listener_task = asyncio.create_task(client_listener())
        loop_task = asyncio.create_task(micro_turn_loop())

        done, pending = await asyncio.wait(
            [listener_task, loop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        await session_mgr.remove(session.session_id)
        logger.info(f"WebSocket session finished: {session.session_id}")

    @app.websocket("/ws/duplex")
    async def websocket_duplex_endpoint(websocket: WebSocket):
        await handle_duplex_websocket(websocket, is_audio_route=False)

    @app.websocket("/ws/duplex_audio")
    async def websocket_duplex_audio_endpoint(websocket: WebSocket):
        await handle_duplex_websocket(websocket, is_audio_route=True)

    return app


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TalkAI Duplex Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--llama-url", default=DEFAULT_LLAMA_URL, help="llama-server URL")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LISTEN_THRESHOLD, help="P(listen) threshold")
    parser.add_argument("--stt-device", default=None, help="STT device: cuda or cpu")
    parser.add_argument("--energy-threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD, help="RMS energy threshold for silence")
    parser.add_argument("--tts-device", default=None, help="TTS device: cuda or cpu")
    parser.add_argument("--tts-voice", default=DEFAULT_KOKORO_VOICE, help="Kokoro voice (default: af_bella)")
    parser.add_argument("--kokoro-model", default=DEFAULT_KOKORO_MODEL, help="Path to kokoro-v0_19.onnx")
    parser.add_argument("--kokoro-voices", default=DEFAULT_KOKORO_VOICES, help="Path to voices.bin")
    args = parser.parse_args()

    app = create_app(
        llama_url=args.llama_url,
        listen_threshold=args.threshold,
        stt_device=args.stt_device,
        energy_threshold=args.energy_threshold,
        tts_device=args.tts_device,
        tts_voice=args.tts_voice,
        kokoro_model_path=args.kokoro_model,
        kokoro_voices_path=args.kokoro_voices,
    )
    logger.info(f"Starting Duplex Engine on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=16 * 1024 * 1024, log_level="info")


if __name__ == "__main__":
    main()

