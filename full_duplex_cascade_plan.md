# Full-Duplex Cascade Architecture Plan: Zero-Hardcoding Real-Time Voice Agent
================================================================================

Based on **arXiv:2603.09180 (DuplexCascade)** and the system architecture detailed in [`transcript.txt`](file:///home/aditya/Desktop/STUDY/TalkAI/transcript.txt).

---

## 1. Executive Summary & Core Objective

The goal is to implement a production-grade, commercial **True Full-Duplex Cascade** voice agent that operates via continuous **600ms micro-turns**, completely eliminating half-duplex walkie-talkie behavior (no VAD silence waiting, no "press button to send" requirements).

### Absolute Requirements:
1. **Zero Hardcoding**: No keyword triggers (`if "essay" in msg`), no logit bans on `<listen>`, no regex string-stripping hacks. All conversational decisions are made natively by the fine-tuned Gemma-4-E2B weights.
2. **600ms Slicing Loop**: Audio is ingested continuously and sliced every 600ms. Slices containing speech are transcribed immediately; silence windows are tagged `<no_voice>`.
3. **Mid-Utterance Hazard Interruption**: If the user utters a hazard (e.g. *"running in front of vehicle"*, *"mix bleach with ammonia"*), the model detects the danger in that very 600ms slice, drops `<listen>`, and speaks words immediately, cutting into the user's ongoing speech without waiting for silence.
4. **Calibrated Neural Threshold**: The decision between staying silent (`<listen>`) and speaking words is governed strictly by the paper's mathematical threshold rule ($P(\text{<listen>}) \ge \tau \approx 0.30$).
5. **Ultra-Low Latency Pipeline**:
   - STT slice: $\le 150\text{ ms}$
   - LLM forward pass: $\le 120\text{ ms}$
   - Kokoro TTS fragment: $\le 180\text{ ms}$
   - Total roundtrip per turn: $\le 450\text{ ms} < 600\text{ ms}$ budget.

---

## 2. Architectural Blueprint

```
+-----------------------------------------------------------------------------------------+
|                              TRUE FULL-DUPLEX CASCADE LOOP                              |
|                                                                                         |
|   Continuous Audio Stream (Microphone)                                                  |
|        │                                                                                |
|        ▼ (every 600ms)                                                                  |
|   Streaming STT ───► Text Slice / "no voice"                                            |
|                            │                                                            |
|                            ▼                                                            |
|                    Gemma-4-E2B (GPU)                                                    |
|                   P(listen) >= 0.30?                                                    |
|                    /              \                                                     |
|                  YES               NO                                                   |
|                  /                  \                                                   |
|                 ▼                    ▼                                                  |
|           Stay Silent          Speech Fragment                                          |
|        (Yield floor /          (e.g., "Do not touch it.")                               |
|         keep listening)              │                                                  |
|                                      ▼                                                  |
|                               Kokoro TTS (ONNX)                                         |
|                                      │                                                  |
|                                      ▼                                                  |
|                           Stream Audio Chunk to Browser                                 |
|                         (Plays WHILE user continues)                                    |
+-----------------------------------------------------------------------------------------+
```

---

## 3. The 5 Modular Implementation Phases

### Module 1: Duplex Micro-Turn Engine Core (`scripts/duplex_engine.py`)
- **Objective**: Create the core async event loop that ticks precisely every 600ms per active WebSocket session.
- **Specifications**:
  - Maintain session state: rolling micro-turn history formatted exactly as in `micro_turns_train.jsonl`.
  - Connect to `llama-server` (port 8080) running `models/duplex_gemma_merged.gguf`.
  - Forward 600ms text slices (or `"no voice"`) to the model with the exact training system prompt:
    `"You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn from streaming speech recognition, or 'no voice' when no speech is detected. Output exactly 'listen' when you should yield and keep listening; otherwise output a brief, natural speech fragment that can be spoken now. Do not repeat prior fragments. Continue an unfinished answer across turns when appropriate. Yield immediately for a clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."`
- **Evaluation Gate 1**:
  - [ ] Tick frequency is stable at 600ms ($\pm 15\text{ms}$).
  - [ ] Session manages state across multiple concurrent micro-turns.
  - [ ] Reports generated in `logs/module1_engine_report.json`.

---

### Module 2: Streaming Audio Ingestion & Slicing (`test.html` & backend)
- **Objective**: Replace batch `MediaRecorder` with continuous 600ms streaming audio slices.
- **Specifications**:
  - Frontend: Web Audio API `AudioWorklet` / 16kHz PCM streaming sending raw 600ms audio chunks over WebSocket `/ws/duplex_audio`.
  - Dual-mode support: High-speed browser speech recognition streaming as an instant client fallback.
  - Backend: `faster-whisper-tiny.en` on CPU/GPU evaluates the 600ms slice in $\le 120\text{ms}$. If audio energy is below threshold, immediately emit `"no voice"`.
  - Zero VAD silence timeout: Never wait for the user to stop talking.
- **Evaluation Gate 2**:
  - [ ] Audio chunks sliced and streamed every 600ms without buffer overrun.
  - [ ] Silence produces `"no voice"` tag reliably.
  - [ ] Spoken words transcribed within 150ms.
  - [ ] Reports generated in `logs/module2_streaming_stt_report.json`.

---

### Module 3: Zero-Hardcoding Neural Policy & Calibrated Turn-Taking
- **Objective**: Remove all hardcoded keyword rules and enforce pure neural turn-taking.
- **Specifications**:
  - Remove `is_long_form = any(...)`, `logit_bias={"20923": -100}`, and regex hacks.
  - Query `/v1/chat/completions` or `/completion` with logprobs.
  - Read $P(\text{<listen>})$ (Token ID `20923`).
  - **Decision Policy**:
    $$\text{Action} = \begin{cases} \text{YIELD (listen)}, & \text{if } P(\text{<listen>}) \ge 0.30 \\ \text{SPEAK}, & \text{if } P(\text{<listen>}) < 0.30 \end{cases}$$
  - Rolling KV-cache management: Prune consecutive silent turns to prevent `listen` repetition bias as documented in `transcript.txt`.
- **Evaluation Gate 3**:
  - [ ] Zero keyword checks or logit bans exist in code.
  - [ ] Model yields `<listen>` on hesitation slices (e.g. *"I was thinking..."*).
  - [ ] Model emits immediate warning on hazard slices (e.g. *"outlet is sparking"*, *"mix bleach with ammonia"*).
  - [ ] Reports generated in `logs/module3_neural_policy_report.json`.

---

### Module 4: Concurrent Kokoro TTS Pipelining & Audio Queue
- **Objective**: Low-latency synthesis and continuous playback of assistant speech fragments.
- **Specifications**:
  - When the model outputs a speech fragment (e.g., *"Do not touch it."*), synthesize immediately with Kokoro ONNX.
  - Synthesis latency $\le 180\text{ms}$ for 3-8 word fragments.
  - Send audio chunk over WebSocket; browser queues and plays seamlessly.
  - If user utters a new input while AI is speaking, feed slice to LLM. If LLM outputs `listen` (barge-in), immediately cancel audio queue and halt playback.
- **Evaluation Gate 4**:
  - [ ] Time-To-First-Audio (TTFA) on spoken fragments $\le 350\text{ms}$.
  - [ ] Zero audio stutter across multi-fragment answers.
  - [ ] Immediate audio interruption on user barge-in within 30ms.
  - [ ] Reports generated in `logs/module4_tts_pipeline_report.json`.

---

### Module 5: Production Web UI & End-to-End Validation Suite
- **Objective**: Modern production interface and automated simulation benchmark.
- **Specifications**:
  - Frontend Layout (`test.html`):
    - Center: Prominent glowing duplex mic button (one-click start/stop session).
    - Left Drawer: Controls (voice, speed, model status, latency metrics).
    - Right Console: Live stream with user text in Electric Blue (`#38bdf8`), assistant text in White (`#ffffff`), and live token badges (`<listen>`, `<no voice>`, `<turn|>`, `[HAZARD BARGE-IN]`).
  - Automated Harness (`scripts/verify_full_duplex.py`):
    - Test Case 1: Incomplete thought / hesitation -> verified stays silent.
    - Test Case 2: Direct question -> verified answers in natural fragments.
    - Test Case 3: Mid-utterance hazard -> verified interrupts at the exact danger turn.
- **Evaluation Gate 5**:
  - [ ] All 3 test cases pass with 100% assertion success.
  - [ ] Mean turn latency $\le 450\text{ms}$ (well within the 600ms budget).
  - [ ] Full end-to-end report saved to `logs/module5_full_duplex_report.md` and `logs/module_5_report.json`.

---

## 4. Timeline & Resource Estimates

| Phase | Duration | Compute / Hardware Target |
| :--- | :--- | :--- |
| **Module 1**: Duplex Engine Core | 35 mins | FastAsyncIO / FastAPI |
| **Module 2**: Streaming STT Slicing | 30 mins | faster-whisper (CPU/GPU) + Web Audio |
| **Module 3**: Zero-Hardcoding Neural Policy | 30 mins | llama-server on GPU (RTX 4050 < 4.0 GB VRAM) |
| **Module 4**: Kokoro TTS Synthesis Queue | 25 mins | Kokoro ONNX (< 180ms synthesis) |
| **Module 5**: UI & Benchmark Verification | 25 mins | Automated test runner + test.html |
| **Total Implementation Time** | **~2.25 hours** | **Fully Autonomous & Verified** |
