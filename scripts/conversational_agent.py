#!/usr/bin/env python3
"""
TalkAI - Core Audio Synthesis & Transcription Pipelines
-------------------------------------------------------
Features:
- Ultra-low latency Kokoro ONNX TTS engine with multi-voice selection and speed control.
- Sentence-level streaming chunker with smart regex punctuation boundary detection.
- Natural speech text normalization (cleaning markdown, URLs, symbols).
- Local Faster-Whisper offline transcription wrapper for speech-to-text.
- Base agent callback abstractions for streaming audio and turn lifecycle events.
"""

import os
import sys
import time
import json
import re
import io
import base64
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TalkAI.Audio")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "models" / "kokoro"
ONNX_PATH = MODELS_DIR / "kokoro-v0_19.onnx"
VOICES_PATH = MODELS_DIR / "voices.bin"


# ---------------------------------------------------------------------------
# Text Normalization for Natural Speech Delivery
# ---------------------------------------------------------------------------

def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, code blocks, URLs, and symbols so the text
    sounds natural when read aloud by Kokoro TTS.
    """
    if not text:
        return ""
    # Strip markdown code blocks
    text = re.sub(r"```[\s\S]*?```", " [code omitted] ", text)
    # Strip inline code
    text = re.sub(r"`[^`]*`", " ", text)
    # Strip markdown table formatting and delimiters
    text = re.sub(r"\|[\s\-:]+\|", " ", text)
    text = re.sub(r"\|", ", ", text)
    # Convert markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove raw URLs
    text = re.sub(r"https?://\S+", "", text)
    # Expand units and symbols for natural pronunciation
    text = re.sub(r"°C\b", " degrees Celsius", text)
    text = re.sub(r"°F\b", " degrees Fahrenheit", text)
    text = re.sub(r"%\b", " percent", text)
    text = re.sub(r"&", " and ", text)
    # Remove emojis and non-ascii decorative symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove bold, italics, headers, blockquotes, tildes
    text = re.sub(r"[*_~#>]", "", text)
    # Remove bullet symbols
    text = re.sub(r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE)
    # Replace multiple newlines or spaces with single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Streaming Sentence Chunker
# ---------------------------------------------------------------------------

class SentenceChunker:
    """
    Accumulates streaming tokens from the LLM and emits complete sentences
    so Kokoro TTS can synthesize audio concurrently while the LLM is still speaking.
    Avoids splitting on decimals (e.g. 18.1, 3.14).
    """
    def __init__(self, min_length: int = 15):
        self.min_length = min_length
        self.buffer = ""
        # Sentence boundary: period, exclamation, question mark, or newline NOT preceded or followed by digits
        self.split_pattern = re.compile(r"((?<!\d)[.!?]+(?!\d)(?:\s+|$|\n))")

    def add_token(self, token: str) -> List[str]:
        self.buffer += token
        ready_sentences = []

        parts = self.split_pattern.split(self.buffer)
        if len(parts) > 1:
            # Reconstruct sentences from split parts
            for i in range(0, len(parts) - 1, 2):
                sentence = (parts[i] + parts[i + 1]).strip()
                if sentence:
                    ready_sentences.append(sentence)
            # Remaining unsplit trailing content
            self.buffer = parts[-1]

        # If buffer is excessively long without punctuation, split at clause break
        if len(self.buffer) > 140:
            for comma in [", ", "; ", " - "]:
                if comma in self.buffer:
                    idx = self.buffer.rfind(comma)
                    if idx > 30:
                        ready_sentences.append(self.buffer[:idx].strip())
                        self.buffer = self.buffer[idx + len(comma):]
                        break

        return ready_sentences

    def flush(self) -> List[str]:
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            return [remaining]
        return []


# ---------------------------------------------------------------------------
# Kokoro ONNX TTS Engine
# ---------------------------------------------------------------------------

class KokoroTTSEngine:
    """
    High-performance wrapper for Kokoro ONNX TTS with thread-safe async synthesis.
    """
    def __init__(self, model_path: Path = ONNX_PATH, voices_path: Path = VOICES_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro ONNX model not found at: {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices.bin not found at: {voices_path}")

        logger.info(f"Loading Kokoro TTS model from {model_path}...")
        t0 = time.time()
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.available_voices = self.kokoro.get_voices()
        logger.info(f"Kokoro loaded in {(time.time() - t0)*1000:.1f}ms. Available voices: {self.available_voices}")

        # Human-friendly voice labels
        self.voice_labels = {
            "af_sarah": "Sarah (US Female - Warm & Clear)",
            "af_bella": "Bella (US Female - Expressive)",
            "af_nicole": "Nicole (US Female - Calm)",
            "af_sky": "Sky (US Female - Cheerful)",
            "am_adam": "Adam (US Male - Deep & Natural)",
            "am_michael": "Michael (US Male - Professional)",
            "bf_emma": "Emma (UK Female - Crisp)",
            "bf_isabella": "Isabella (UK Female - Soft)",
            "bm_george": "George (UK Male - Authoritative)",
            "bm_lewis": "Lewis (UK Male - Friendly)",
            "af": "Default Female (US)",
        }

    def get_voices(self) -> List[Dict[str, str]]:
        return [
            {"id": v, "label": self.voice_labels.get(v, v)}
            for v in self.available_voices
        ]

    def _sync_synthesize(self, text: str, voice: str, speed: float) -> Tuple[bytes, float, int]:
        """Synchronous Kokoro synthesis in thread worker."""
        if voice not in self.available_voices:
            voice = "af_sarah" if "af_sarah" in self.available_voices else self.available_voices[0]

        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        duration = len(samples) / sample_rate

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        return wav_bytes, duration, sample_rate

    async def synthesize(self, text: str, voice: str = "af_sarah", speed: float = 1.0) -> Tuple[bytes, str, float, int]:
        """
        Asynchronously synthesizes speech for the given text.
        Returns (wav_bytes, base64_data_uri, duration_seconds, sample_rate).
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            return b"", "", 0.0, 24000

        wav_bytes, duration, sample_rate = await asyncio.to_thread(
            self._sync_synthesize, cleaned_text, voice, speed
        )
        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        data_uri = f"data:audio/wav;base64,{b64_audio}"
        return wav_bytes, data_uri, duration, sample_rate


# ---------------------------------------------------------------------------
# Speech-to-Text Pipeline (Local Faster-Whisper)
# ---------------------------------------------------------------------------

class LocalWhisperSTT:
    """100% local, offline Speech-to-Text running on CPU/CUDA with zero cloud calls."""
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading local offline faster-whisper model ({self.model_size})...")
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info("Local faster-whisper loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize local faster-whisper: {e}")
                self._model = False
        return self._model if self._model is not False else None

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe PCM/WAV/WebM audio bytes asynchronously."""
        model = self._get_model()
        if not model:
            return ""

        def _do_transcribe():
            buf = io.BytesIO(audio_bytes)
            segments, info = model.transcribe(buf, beam_size=5, language="en")
            return " ".join([s.text for s in segments]).strip()

        return await asyncio.to_thread(_do_transcribe)


# ---------------------------------------------------------------------------
# Base Callback Interfaces
# ---------------------------------------------------------------------------

class BaseAgentCallback:
    """Base callback handler defining the complete lifecycle hooks."""
    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        pass

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]):
        pass

    async def on_tool_end(self, tool_name: str, tool_output: str, elapsed_ms: float):
        pass

    async def on_llm_start(self, model: str, message_count: int):
        pass

    async def on_llm_token(self, token: str):
        pass

    async def on_llm_end(self, full_text: str, elapsed_ms: float):
        pass

    async def on_tts_start(self, sentence: str, segment_index: int):
        pass

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        pass

    async def on_turn_finish(self, full_response: str, metrics: Dict[str, Any]):
        pass

    async def on_error(self, error_message: str):
        pass


class ConsoleCallback(BaseAgentCallback):
    """Logs colorized, structured lifecycle events to stdout/terminal."""
    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        logger.info(f"\033[1;36m[USER]\033[0m {user_input}")

    async def on_tts_start(self, sentence: str, segment_index: int):
        logger.info(f"\033[1;35m[TTS SYNTH START #{segment_index}]\033[0m \"{sentence}\"")

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        logger.info(f"\033[1;32m[TTS CHUNK #{segment_index}]\033[0m audio: {duration_sec:.2f}s (gen: {gen_time_ms:.1f}ms, rtf: {gen_time_ms/(duration_sec*1000):.2f})")
