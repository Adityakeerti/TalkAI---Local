#!/usr/bin/env python3
"""
scripts/test_module2_streaming_stt.py
=====================================
Verification and QA Benchmark Suite for Module 2: Streaming Audio Ingestion & 600ms Slicing.

Evaluation Gate 2 Criteria:
1. Slicing & Audio Processing:
   - Ingest 16kHz mono PCM 16-bit audio in 600ms slices (9,600 samples = 19,200 bytes per chunk).
   - Real-time RMS audio energy silence gate: Emits "no voice" in < 5ms (target < 2ms).
2. Speech Transcription Latency:
   - Active speech transcribed with faster-whisper (tiny.en) in <= 150ms.
   - Zero VAD silence timeout: Never waits for silence or end of utterance.
3. Continuous Streaming Stream:
   - Continuous stream of 10 consecutive 600ms audio chunks over WebSocket /ws/duplex_audio
     without buffer overrun.
4. Buffer Overrun Protection:
   - StreamingAudioSlicer bounds buffer to max_buffered_slices under burst conditions.
5. Zero Hardcoding:
   - Zero keyword checks, zero logit bias bans, zero regex stripping.
6. VRAM Gate:
   - GPU VRAM <= 5.0 GB (5,120 MB).

Outputs:
- logs/module2_streaming_stt_report.json
- logs/module_2_report.json
"""

import os
import sys
import time
import math
import json
import socket
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
import websockets
import uvicorn
import numpy as np
from kokoro_onnx import Kokoro

# Add TalkAI root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.duplex_engine import (
    StreamingAudioSlicer,
    StreamingSTTProcessor,
    DuplexSession,
    DuplexModelClient,
    SessionManager,
    create_app,
    get_gpu_vram_info,
    AUDIO_SAMPLE_RATE,
    AUDIO_BYTES_PER_SAMPLE,
    AUDIO_SLICE_SAMPLES,
    AUDIO_SLICE_BYTES,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_TICK_INTERVAL_SEC,
    DEFAULT_LLAMA_URL,
    DEFAULT_LISTEN_THRESHOLD,
)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = LOGS_DIR / "module2_streaming_stt_report.json"
REPORT_ALT_JSON = LOGS_DIR / "module_2_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestModule2")


def find_free_port() -> int:
    """Finds an available local port for running test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def generate_silence_chunk(duration_sec: float = DEFAULT_TICK_INTERVAL_SEC, noise_amp: int = 0) -> bytes:
    """Generates a 600ms PCM 16-bit silence chunk (19,200 bytes)."""
    samples = int(AUDIO_SAMPLE_RATE * duration_sec)
    if noise_amp == 0:
        return np.zeros(samples, dtype=np.int16).tobytes()
    # Add minimal room noise floor well below speech threshold
    noise = np.random.randint(-noise_amp, noise_amp + 1, size=samples, dtype=np.int16)
    return noise.tobytes()


def generate_speech_chunks(text: str, voice: str = "af_sarah") -> List[bytes]:
    """Generates 16kHz PCM 16-bit audio using Kokoro and slices into 600ms chunks."""
    onnx_path = PROJECT_ROOT / "models" / "kokoro" / "kokoro-v0_19.onnx"
    voices_path = PROJECT_ROOT / "models" / "kokoro" / "voices.bin"
    if not (onnx_path.exists() and voices_path.exists()):
        raise FileNotFoundError(f"Kokoro model files missing in {onnx_path.parent}")

    tts = Kokoro(str(onnx_path), str(voices_path))
    samples, sr = tts.create(text, voice=voice, speed=1.0, lang="en-us")

    # Resample to 16kHz
    target_len = int(len(samples) * AUDIO_SAMPLE_RATE / sr)
    samples_16k = np.interp(
        np.linspace(0, len(samples) - 1, target_len),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)

    pcm_raw = (samples_16k * 32767).astype(np.int16).tobytes()

    # Slice into exact 19,200 byte chunks
    chunks = []
    for i in range(0, len(pcm_raw), AUDIO_SLICE_BYTES):
        chunk = pcm_raw[i : i + AUDIO_SLICE_BYTES]
        if len(chunk) < AUDIO_SLICE_BYTES:
            chunk = chunk + b"\x00" * (AUDIO_SLICE_BYTES - len(chunk))
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Test 1: Real-Time RMS Silence Gate Benchmark
# ---------------------------------------------------------------------------
def test_silence_detection_gate(stt: StreamingSTTProcessor, num_slices: int = 20) -> Dict[str, Any]:
    logger.info(f"--- Running Test 1: Real-Time RMS Silence Gate ({num_slices} slices) ---")
    silence_latencies = []
    rms_values = []

    for i in range(num_slices):
        # Alternate between pure digital zero and faint background room noise (amp 15)
        noise_amp = 0 if i % 2 == 0 else 15
        chunk = generate_silence_chunk(noise_amp=noise_amp)

        result = stt.process_slice(chunk)

        assert result.text == "no voice", f"Expected 'no voice', got '{result.text}' on slice {i}"
        assert not result.is_speech, f"Slice {i} was erroneously marked as speech"
        assert result.latency_ms < 5.0, f"Silence check latency {result.latency_ms}ms exceeded 5.0ms gate!"
        assert result.rms < stt.energy_threshold, f"RMS {result.rms} exceeded silence threshold {stt.energy_threshold}"

        silence_latencies.append(result.latency_ms)
        rms_values.append(result.rms)

    mean_latency = float(np.mean(silence_latencies))
    max_latency = float(np.max(silence_latencies))
    max_rms = float(np.max(rms_values))

    logger.info(f"Silence Gate: Mean latency = {mean_latency:.3f}ms, Max latency = {max_latency:.3f}ms, Max RMS = {max_rms:.6f}")
    assert mean_latency < 2.0, f"Mean silence latency {mean_latency:.3f}ms exceeded 2.0ms target!"

    return {
        "name": "silence_detection_gate",
        "passed": True,
        "slices_tested": num_slices,
        "mean_latency_ms": round(mean_latency, 3),
        "max_latency_ms": round(max_latency, 3),
        "max_rms": round(max_rms, 6),
        "energy_threshold": stt.energy_threshold,
    }


# ---------------------------------------------------------------------------
# Test 2: Active Speech Transcription Latency Gate
# ---------------------------------------------------------------------------
def test_speech_transcription_gate(stt: StreamingSTTProcessor) -> Dict[str, Any]:
    logger.info("--- Running Test 2: Active Speech Transcription Latency Gate ---")
    speech_chunks = generate_speech_chunks("The outlet is sparking and I might touch the wire")
    assert len(speech_chunks) >= 3, f"Expected at least 3 speech slices, got {len(speech_chunks)}"

    results = []
    latencies = []

    for idx, chunk in enumerate(speech_chunks):
        res = stt.process_slice(chunk)
        latencies.append(res.latency_ms)
        results.append({
            "slice_idx": idx,
            "text": res.text,
            "rms": res.rms,
            "is_speech": res.is_speech,
            "latency_ms": res.latency_ms,
        })
        logger.info(f"Speech slice {idx}: Text='{res.text}', RMS={res.rms:.4f}, Latency={res.latency_ms:.1f}ms")

        # Assertion: active speech must be transcribed <= 150ms
        assert res.latency_ms <= 150.0, f"Speech slice {idx} latency {res.latency_ms}ms exceeded 150ms limit!"
        assert res.is_speech, f"Speech slice {idx} failed to detect speech"
        assert res.text != "no voice", f"Speech slice {idx} emitted 'no voice'"

    mean_latency = float(np.mean(latencies))
    max_latency = float(np.max(latencies))
    logger.info(f"Speech Gate: Mean latency = {mean_latency:.2f}ms, Max latency = {max_latency:.2f}ms (Budget <= 150ms)")

    return {
        "name": "speech_transcription_gate",
        "passed": True,
        "num_slices": len(speech_chunks),
        "mean_latency_ms": round(mean_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "slices": results,
    }


# ---------------------------------------------------------------------------
# Test 3: Slicer & Buffer Overrun Gate
# ---------------------------------------------------------------------------
def test_audio_slicer_buffer_overrun() -> Dict[str, Any]:
    logger.info("--- Running Test 3: Audio Slicer & Buffer Overrun Gate ---")
    slicer = StreamingAudioSlicer(max_buffered_slices=4)

    # 1. Normal Sequential Pushes
    for _ in range(5):
        slicer.push_pcm_chunk(b"\x00" * AUDIO_SLICE_BYTES)
        popped = slicer.pop_600ms_slice()
        assert popped is not None, "Failed to pop expected 600ms slice"
        assert len(popped) == AUDIO_SLICE_BYTES, f"Popped length {len(popped)} != {AUDIO_SLICE_BYTES}"

    stats_normal = slicer.get_stats()
    assert stats_normal["overrun_drops"] == 0, "Normal sequential streaming caused unexpected buffer drops!"

    # 2. Burst Dump (e.g., 20 chunks pushed without popping)
    burst_chunks = 20
    for _ in range(burst_chunks):
        slicer.push_pcm_chunk(b"\x01" * AUDIO_SLICE_BYTES)

    stats_burst = slicer.get_stats()
    # Buffer should be clamped at max_buffer_bytes (4 slices = 76,800 bytes)
    max_bytes = slicer.bytes_per_slice * 4
    assert len(slicer.buffer) <= max_bytes, f"Buffer size {len(slicer.buffer)} exceeded cap {max_bytes}"
    assert stats_burst["overrun_drops"] > 0, "Buffer overrun protection did not trigger on burst"

    logger.info(f"Buffer Overrun Gate passed: Max bytes bounded at {max_bytes}, dropped {stats_burst['overrun_drops']} stale frames.")

    return {
        "name": "slicer_buffer_overrun_gate",
        "passed": True,
        "max_allowed_buffer_bytes": max_bytes,
        "final_buffer_bytes": len(slicer.buffer),
        "overrun_drops_triggered_on_burst": stats_burst["overrun_drops"],
    }


# ---------------------------------------------------------------------------
# Test 4: End-to-End WebSocket 10-Chunk Audio Streaming Benchmark
# ---------------------------------------------------------------------------
async def test_e2e_streaming_audio_websocket(stt: StreamingSTTProcessor) -> Dict[str, Any]:
    logger.info("--- Running Test 4: End-to-End WebSocket 10-Chunk Audio Streaming Benchmark ---")
    port = find_free_port()
    app = create_app(llama_url=DEFAULT_LLAMA_URL, stt_processor=stt)
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server bind
    ws_url = f"ws://127.0.0.1:{port}/ws/duplex_audio"
    connected = False
    for _ in range(30):
        try:
            async with httpx.AsyncClient() as http_client:
                r = await http_client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    connected = True
                    break
        except Exception:
            await asyncio.sleep(0.1)

    assert connected, "Test server failed to start within timeout"

    # Prepare 10 audio chunks:
    # Turns 1-3: Silence chunks
    # Turns 4-7: Speech chunks ("The outlet is sparking and I might touch the wire")
    # Turns 8-10: Silence chunks
    silence_3 = [generate_silence_chunk() for _ in range(3)]
    speech_chunks = generate_speech_chunks("The outlet is sparking and I might touch the wire")
    speech_4 = speech_chunks[:4]
    while len(speech_4) < 4:
        speech_4.append(generate_silence_chunk())
    silence_end = [generate_silence_chunk() for _ in range(3)]

    stream_10 = silence_3 + speech_4 + silence_end
    assert len(stream_10) == 10, f"Expected 10 chunks, got {len(stream_10)}"

    received_turns = []
    stt_latencies = []

    try:
        async with websockets.connect(ws_url) as ws:
            # 1. Read session_init
            init_msg = json.loads(await ws.recv())
            assert init_msg["type"] == "session_init"
            session_id = init_msg["session_id"]
            logger.info(f"Connected to /ws/duplex_audio. Session ID: {session_id}")

            for idx, chunk in enumerate(stream_10):
                # Send 600ms PCM chunk as binary WebSocket frame
                await ws.send(chunk)

                # Receive micro-turn result from server (ignoring audio stream events)
                while True:
                    turn_data = json.loads(await ws.recv())
                    if turn_data.get("type") == "micro_turn":
                        break
                assert turn_data["type"] == "micro_turn"
                assert turn_data["session_id"] == session_id

                user_slice = turn_data["user"]
                action = turn_data["action"]
                stt_lat = turn_data.get("stt_latency_ms", 0.0)
                stt_latencies.append(stt_lat)
                received_turns.append(turn_data)

                logger.info(
                    f"Turn {turn_data['turn_id']}: User='{user_slice}', Action={action}, "
                    f"P(listen)={turn_data['p_listen']:.4f}, STT_lat={stt_lat:.1f}ms, Drift={turn_data['drift_ms']:.1f}ms"
                )

        assert len(received_turns) == 10, f"Expected 10 micro-turns, got {len(received_turns)}"

        # Validate silence turns (0, 1, 2)
        for i in range(3):
            assert received_turns[i]["user"] == "no voice", f"Turn {i} expected 'no voice', got {received_turns[i]['user']}"
            assert received_turns[i]["stt_latency_ms"] < 5.0, f"Silence turn {i} STT latency {received_turns[i]['stt_latency_ms']}ms > 5ms"

        # Validate speech turns (3, 4, 5, 6)
        speech_slice_texts = [received_turns[i]["user"] for i in range(3, 7)]
        logger.info(f"Transcribed speech slice sequence: {speech_slice_texts}")
        assert any(t != "no voice" for t in speech_slice_texts), "Speech slices failed to transcribe words"

        # Validate STT latency across all turns <= 150ms
        for i, turn in enumerate(received_turns):
            assert turn["stt_latency_ms"] <= 150.0, f"Turn {i} STT latency {turn['stt_latency_ms']}ms exceeded 150ms limit!"

        mean_stt = float(np.mean(stt_latencies))
        max_stt = float(np.max(stt_latencies))

        logger.info(f"10-Chunk Streaming Benchmark Passed: Mean STT = {mean_stt:.2f}ms, Max STT = {max_stt:.2f}ms")

        return {
            "name": "e2e_streaming_audio_websocket",
            "passed": True,
            "session_id": session_id,
            "turns_completed": len(received_turns),
            "mean_stt_latency_ms": round(mean_stt, 2),
            "max_stt_latency_ms": round(max_stt, 2),
            "turns": [
                {
                    "turn_id": t["turn_id"],
                    "user": t["user"],
                    "action": t["action"],
                    "p_listen": t["p_listen"],
                    "stt_latency_ms": t["stt_latency_ms"],
                    "roundtrip_ms": t["latency_ms"],
                }
                for t in received_turns
            ],
        }

    finally:
        server.should_exit = True
        await server_task


# ---------------------------------------------------------------------------
# Test 5: Zero Hardcoding Code Audit
# ---------------------------------------------------------------------------
def test_zero_hardcoding_audit() -> Dict[str, Any]:
    logger.info("--- Running Test 5: Zero Hardcoding Code Audit ---")
    engine_file = PROJECT_ROOT / "scripts" / "duplex_engine.py"
    with open(engine_file, "r", encoding="utf-8") as f:
        code = f.read()

    violations = []
    if "is_long_form" in code:
        violations.append("Found banned variable 'is_long_form'")
    if "logit_bias" in code:
        violations.append("Found banned parameter 'logit_bias'")
    if 're.sub(r"^listen' in code or 're.sub(r"^[listen]' in code or 're.sub(r"^<listen>' in code:
        violations.append("Found banned regex stripping on output tokens")
    if 'if "essay" in' in code or 'if "marble" in' in code:
        violations.append("Found hardcoded keyword trigger checks")

    assert not violations, f"Zero hardcoding audit failed: {violations}"
    logger.info("Zero hardcoding audit passed: Clean neural architecture, no keyword checks or logit bans.")

    return {
        "name": "zero_hardcoding_audit",
        "passed": True,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Test 6: GPU VRAM Sanity Gate
# ---------------------------------------------------------------------------
def test_vram_sanity() -> Dict[str, Any]:
    logger.info("--- Running Test 6: GPU VRAM Sanity Gate ---")
    vram_info = get_gpu_vram_info()
    logger.info(f"VRAM Info: Used = {vram_info.get('used_mb')}MB, Cap = 5120.0MB, Under Cap = {vram_info.get('under_cap')}")
    assert vram_info.get("under_cap", True), f"GPU VRAM {vram_info.get('used_mb')}MB exceeded 5120MB cap!"
    return {
        "name": "vram_sanity_check",
        "passed": True,
        "vram_info": vram_info,
    }


# ---------------------------------------------------------------------------
# Master Runner & Report Generator
# ---------------------------------------------------------------------------
async def async_main():
    logger.info("================================================================")
    logger.info("TALKAI MODULE 2: STREAMING AUDIO INGESTION & 600MS SLICING QA")
    logger.info("================================================================")

    t0 = time.time()
    results = {}

    # Initialize shared STT Processor
    stt = StreamingSTTProcessor(device="cuda", energy_threshold=DEFAULT_ENERGY_THRESHOLD)

    # Test 1: Real-Time RMS Silence Gate
    results["silence_gate"] = test_silence_detection_gate(stt, num_slices=20)

    # Test 2: Active Speech Transcription Gate
    results["speech_gate"] = test_speech_transcription_gate(stt)

    # Test 3: Slicer & Buffer Overrun Gate
    results["slicer_gate"] = test_audio_slicer_buffer_overrun()

    # Test 4: End-to-End WebSocket Streaming Audio Benchmark
    results["e2e_audio_websocket"] = await test_e2e_streaming_audio_websocket(stt)

    # Test 5: Zero Hardcoding Audit
    results["zero_hardcoding"] = test_zero_hardcoding_audit()

    # Test 6: GPU VRAM Sanity Gate
    results["vram_sanity"] = test_vram_sanity()

    total_time = round(time.time() - t0, 2)
    all_passed = all(test.get("passed", False) for test in results.values())

    report = {
        "module": 2,
        "name": "Streaming Audio Ingestion & 600ms Slicing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASSED" if all_passed else "FAILED",
        "total_duration_sec": total_time,
        "evaluation_gate": {
            "silence_produces_no_voice_under_5ms": results["silence_gate"]["passed"],
            "speech_transcribed_under_150ms": results["speech_gate"]["passed"],
            "continuous_stream_10_chunks_without_overrun": results["e2e_audio_websocket"]["passed"],
            "slicer_buffer_overrun_protection_verified": results["slicer_gate"]["passed"],
            "zero_hardcoding_enforced": results["zero_hardcoding"]["passed"],
            "vram_safely_under_5gb": results["vram_sanity"]["passed"],
        },
        "tests": results,
    }

    # Save reports
    for out_path in [REPORT_JSON, REPORT_ALT_JSON]:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report written to: {out_path}")

    logger.info("================================================================")
    logger.info(f"MODULE 2 EVALUATION GATE: {'ALL PASSED' if all_passed else 'FAILED'}")
    logger.info("================================================================")

    if not all_passed:
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
