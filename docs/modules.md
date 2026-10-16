
  #### Module 1: Duplex Micro-Turn Engine Core

  • Test Command:
    .venv/bin/python scripts/test_module1_engine.py

  • What It Asserts:
      • MicroTurnClock ticks at 600ms (±15ms) with zero cumulative drift.
      • Multi-session concurrency over WebSocket /ws/duplex.
      • Silent-turn pruning (prevents repetition loops and KV cache bloat).
      • Zero-hardcoding audit (confirms no keyword checks or logit bans).
  • Report: module_1_report.json
  ──────
  #### Module 2: Streaming Audio Ingestion & 600ms Slicing

  • Test Command:
    .venv/bin/python scripts/test_module2_streaming_stt.py

  • What It Asserts:
      • 600ms PCM slicing buffer overrun gate (discards stale frames, zero memory leak).
      • Faster-Whisper GPU FP16 speech gate: silence evaluation ≤ 2ms, speech transcription ≤ 150ms (actual: ≈ 18ms).
      • End-to-end 10-chunk streaming benchmark over /ws/duplex_audio.
  • Report: module_2_report.json
  ──────
  #### Module 3: Zero-Hardcoding Neural Policy & Calibrated Turn-Taking

  • Test Command:
    .venv/bin/python scripts/test_module3_neural_policy.py

  • What It Asserts:
      • Zero regex stripping, zero keyword lists, zero logit bias bans.
      • Hesitation Benchmark: Yields floor (P() ≥ 0.30 ⟹ YIELD) on hesitation prompts ("I was thinking about...").
      • Hazard Benchmark: Drops <listen> (P() < 0.30 ⟹ SPEAK) on danger slices ("outlet is sparking", "mix bleach with ammonia").
      • Direct Question Benchmark: Model outputs factual speech fragments.
  • Report: module_3_report.json
  ──────
  #### Module 4: Concurrent Kokoro TTS Synthesis & Audio Queue

  • Test Command:
    .venv/bin/python scripts/test_module4_tts_pipeline.py

  • What It Asserts:
      • Kokoro-82M ONNX 24kHz synthesis latency ≤ 180ms on speech fragments (actual: ≈ 112ms).
      • Time-To-First-Audio (TTFA) ≤ 350ms (actual: ≈ 201ms).
      • Barge-in cutoff latency ≤ 30ms (actual: ≈ 0.003ms).
      • WebSocket chunk streaming and immediate client audio cancellation on user speech.
  • Report: module_4_report.json
  ──────
  #### Module 5: Full End-to-End Cascade Validation Suite

  • Test Command:
    .venv/bin/python scripts/verify_full_duplex.py

  • What It Asserts:
      • Hesitation test case + Direct question test case + Mid-utterance hazard barge-in test case all pass with 100% assertions.
      • Mean roundtrip latency within budget SLA.
      • Comprehensive markdown report generated.
  • Report: module_5_report.json & module5_full_duplex_report.md
  ──────
  ### Current Live Service Status

  The updated daemon is active:

  • Web UI: http://localhost:8000/test.html
  • VRAM: ≈ 3, 588 MB ≤ 5, 120 MB cap on NVIDIA RTX 4050 Laptop GPU.
  • Engine: Port 8000 (scripts/duplex_engine.py) connected to llama-server on port 8080.