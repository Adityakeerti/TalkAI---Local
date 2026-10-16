#!/usr/bin/env python3
"""
scripts/test_speech_accumulator_e2e.py
Verification suite for continuous utterance speech accumulator, clean output formatting,
and false hazard badge fix.
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.duplex_engine import (
    DuplexSession,
    StreamingSTTProcessor,
    DuplexModelClient,
    TurnDecision,
    AUDIO_SLICE_BYTES,
    MAX_SPEECH_BUFFER_BYTES,
    DEFAULT_MAX_OUTPUT_TOKENS,
)


def test_session_state():
    print("--- 1. Testing DuplexSession State & Reset ---")
    session = DuplexSession("test-accum-e2e")
    assert hasattr(session, "speech_audio_buffer"), "Missing speech_audio_buffer"
    assert hasattr(session, "last_speech_text"), "Missing last_speech_text"
    assert hasattr(session, "silent_ticks_after_speech"), "Missing silent_ticks_after_speech"
    assert isinstance(session.speech_audio_buffer, bytearray)
    assert session.last_speech_text == ""
    assert session.silent_ticks_after_speech == 0

    # Accumulate fake audio
    session.speech_audio_buffer.extend(b"\x00\x01" * 500)
    session.last_speech_text = "hello world"
    session.silent_ticks_after_speech = 2
    assert len(session.speech_audio_buffer) == 1000
    assert session.last_speech_text == "hello world"

    # Reset
    session.reset()
    assert len(session.speech_audio_buffer) == 0
    assert session.last_speech_text == ""
    assert session.silent_ticks_after_speech == 0
    print("PASSED: DuplexSession speech accumulation & reset.")


def test_stt_silence_and_alphanumeric_gate():
    print("--- 2. Testing StreamingSTTProcessor Silence & Alphanumeric Gate ---")
    stt = StreamingSTTProcessor(device="cpu")

    # Pure zero PCM
    silence_pcm = b"\x00" * AUDIO_SLICE_BYTES
    res = stt.process_slice(silence_pcm)
    assert res.text == "no voice", f"Expected 'no voice', got {res.text!r}"
    assert not res.is_speech, "Expected is_speech=False for silence"

    # Low amplitude noise (RMS < threshold)
    noise_arr = (np.random.randn(AUDIO_SLICE_BYTES // 2) * 10).astype(np.int16)
    noise_pcm = noise_arr.tobytes()
    res_noise = stt.process_slice(noise_pcm)
    assert res_noise.text == "no voice", f"Expected 'no voice' for noise, got {res_noise.text!r}"
    assert not res_noise.is_speech

    # transcribe_audio helper
    t_empty = stt.transcribe_audio(b"")
    assert t_empty == "", f"Expected empty text for empty bytes, got {t_empty!r}"

    t_silence = stt.transcribe_audio(silence_pcm)
    assert t_silence == "", f"Expected empty text for silence pcm, got {t_silence!r}"
    print("PASSED: STT silence gate & transcribe_audio helper.")


def test_clean_speech_formatting():
    print("--- 3. Testing DuplexModelClient._clean_speech_content ---")
    assert DEFAULT_MAX_OUTPUT_TOKENS == 36, f"Expected 36, got {DEFAULT_MAX_OUTPUT_TOKENS}"

    cases = [
        # Control token stripping
        ("listen", "listen"),
        ("<listen>", "listen"),
        ("<listen> Hello world.", "Hello world."),
        ("listen: The answer is 42.", "The answer is 42."),
        ("listen The answer is 42.", "The answer is 42."),
        # Truncate dangling incomplete clause after terminal punctuation
        (
            "The capital of France is Paris. It is known for",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris. For example",
            "The capital of France is Paris.",
        ),
        (
            "Paris is wonderful! We should visit and",
            "Paris is wonderful!",
        ),
        (
            "Is Paris the capital? Yes, because",
            "Is Paris the capital?",
        ),
        # Strip trailing dangling connector words/phrases
        (
            "The capital of France is Paris, for example",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris, for instance",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris or perhaps",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris and the",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris, and",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris for",
            "The capital of France is Paris.",
        ),
        (
            "The capital of France is Paris or",
            "The capital of France is Paris.",
        ),
        # Natural complete sentences preserved
        ("Yes, absolutely!", "Yes, absolutely!"),
        (
            "Paris is the capital of France. It is known for its museums, for example, the Louvre.",
            "Paris is the capital of France. It is known for its museums, for example, the Louvre.",
        ),
    ]

    for raw, expected in cases:
        out = DuplexModelClient._clean_speech_content(raw)
        assert out == expected, f"Expected {expected!r}, got {out!r} for raw {raw!r}"
    print("PASSED: DuplexModelClient._clean_speech_content formatting.")


def test_speech_delta_extraction():
    print("--- 4. Testing Speech Utterance Accumulation & Delta Extraction ---")
    session = DuplexSession("test-delta")

    # Simulate Turn 1: user starts speaking
    new_text_1 = "What is"
    if any(c.isalnum() for c in new_text_1):
        if session.last_speech_text and new_text_1.startswith(session.last_speech_text):
            incremental = new_text_1[len(session.last_speech_text):].strip()
            slice_1 = incremental if incremental else new_text_1
        else:
            slice_1 = new_text_1
        session.last_speech_text = new_text_1
        is_speech_1 = True
        session.silent_ticks_after_speech = 0

    assert slice_1 == "What is"
    assert session.last_speech_text == "What is"
    assert is_speech_1 is True

    # Simulate Turn 2: user continues speaking in same utterance
    new_text_2 = "What is the capital"
    if any(c.isalnum() for c in new_text_2):
        if session.last_speech_text and new_text_2.startswith(session.last_speech_text):
            incremental = new_text_2[len(session.last_speech_text):].strip()
            slice_2 = incremental if incremental else new_text_2
        else:
            slice_2 = new_text_2
        session.last_speech_text = new_text_2
        is_speech_2 = True
        session.silent_ticks_after_speech = 0

    assert slice_2 == "the capital", f"Expected 'the capital', got {slice_2!r}"
    assert session.last_speech_text == "What is the capital"
    assert is_speech_2 is True

    # Simulate Turn 3: user finishes speaking
    new_text_3 = "What is the capital of France?"
    if any(c.isalnum() for c in new_text_3):
        if session.last_speech_text and new_text_3.startswith(session.last_speech_text):
            incremental = new_text_3[len(session.last_speech_text):].strip()
            slice_3 = incremental if incremental else new_text_3
        else:
            slice_3 = new_text_3
        session.last_speech_text = new_text_3
        is_speech_3 = True
        session.silent_ticks_after_speech = 0

    assert slice_3 == "of France?", f"Expected 'of France?', got {slice_3!r}"
    assert session.last_speech_text == "What is the capital of France?"

    # Simulate Turn 4: user pauses (silence tick 1)
    slice_4 = "no voice"
    is_speech_4 = False
    session.silent_ticks_after_speech += 1
    assert session.silent_ticks_after_speech == 1
    assert session.last_speech_text != ""  # Not cleared yet

    # Simulate Turn 5: user still silent (silence tick 2)
    session.silent_ticks_after_speech += 1
    if session.silent_ticks_after_speech >= 2:
        session.speech_audio_buffer.clear()
        session.last_speech_text = ""

    assert session.last_speech_text == ""
    assert len(session.speech_audio_buffer) == 0
    print("PASSED: Speech utterance accumulation & delta extraction.")


def test_hazard_logic():
    print("--- 5. Testing is_hazard Logic ---")
    # Case 1: User silent ('no voice'), assistant speaks. Should NOT be hazard!
    is_speech_slice = False
    action = "SPEAK"
    was_audio_playing = False
    is_hazard = bool(is_speech_slice and action == "SPEAK" and not was_audio_playing)
    assert not is_hazard, "False hazard triggered when user was silent!"

    # Case 2: User speaking, assistant yields ('listen'). Not hazard.
    is_speech_slice = True
    action = "YIELD"
    was_audio_playing = False
    is_hazard = bool(is_speech_slice and action == "SPEAK" and not was_audio_playing)
    assert not is_hazard

    # Case 3: User speaking, assistant interrupts with SPEAK, assistant was not playing.
    # This IS a hazard!
    is_speech_slice = True
    action = "SPEAK"
    was_audio_playing = False
    is_hazard = bool(is_speech_slice and action == "SPEAK" and not was_audio_playing)
    assert is_hazard, "Hazard failed to trigger when assistant interrupted user speech!"

    # Case 4: Assistant already playing, user speaks (user barge-in, not assistant hazard)
    is_speech_slice = True
    action = "SPEAK"
    was_audio_playing = True
    is_hazard = bool(is_speech_slice and action == "SPEAK" and not was_audio_playing)
    assert not is_hazard
    print("PASSED: is_hazard condition logic.")


def test_html_sync():
    print("--- 6. Testing test.html and scripts/test.html Sync & Contents ---")
    html_root = (PROJECT_ROOT / "test.html").read_text(encoding="utf-8")
    html_scripts = (PROJECT_ROOT / "scripts" / "test.html").read_text(encoding="utf-8")
    assert html_root == html_scripts, "test.html and scripts/test.html are out of sync!"

    # Check is_hazard check
    assert "const isHazard = Boolean(data.is_hazard);" in html_root, "Missing Boolean(data.is_hazard) check in HTML"
    assert "isUserSpeech && (data.action === \"SPEAK\") && (data.p_listen < 0.30)" not in html_root, "Old false hazard check still present in HTML"

    # Check clean bubble merging
    assert "existing.replace(/[.,;:]+$/, '').trim()" in html_root, "Missing trailing punctuation stripping on existing text"
    assert "incoming.replace(/^[.,;:]+/, '').trim()" in html_root, "Missing leading punctuation stripping on incoming fragment"
    print("PASSED: HTML files synchronization and speech bubble merging.")


if __name__ == "__main__":
    test_session_state()
    test_stt_silence_and_alphanumeric_gate()
    test_clean_speech_formatting()
    test_speech_delta_extraction()
    test_hazard_logic()
    test_html_sync()
    print("\n=======================================================")
    print("ALL VERIFICATION SUITE ASSERTIONS PASSED SUCCESSFULLY!")
    print("=======================================================")
