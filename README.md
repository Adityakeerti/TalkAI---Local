# TalkAI: Local True Full-Duplex Conversational Voice Agent
====================================================================

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Gemma-2B Duplex](https://img.shields.io/badge/LLM-Gemma--2B--Duplex-8E44AD.svg)](https://huggingface.co/google)
[![Kokoro ONNX](https://img.shields.io/badge/TTS-Kokoro--v0.19--ONNX-FF6F00.svg)](https://github.com/thewh1teagle/kokoro-onnx)
[![Faster-Whisper](https://img.shields.io/badge/ASR-Faster--Whisper-black.svg)](https://github.com/SYSTRAN/faster-whisper)
[![CUDA GPU](https://img.shields.io/badge/Inference-RTX%204050%20(6GB)-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**TalkAI** is an ultra-low-latency, **True Full-Duplex** conversational voice AI designed for natural, simultaneous spoken interaction. Running 100% locally with CUDA hardware acceleration on consumer GPUs (NVIDIA RTX 4050 6GB Laptop GPU), TalkAI eliminates conventional half-duplex VAD silence delays in favor of a continuous **600ms micro-turn loop** inspired by [DuplexCascade (arXiv:2603.09180)](https://arxiv.org/abs/2603.09180).

TalkAI supports mid-utterance backchanneling, instant client-side barge-in playback cutoff in **under 30ms**, calibrated neural turn-taking with **zero hardcoded heuristics**, and anti-repetition neural recovery.

> 📖 **Deep Technical Reference**: For the complete mathematical formulation, timing budgets, and component specifications, read [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md).

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Browser ["Client Browser Environment (test.html)"]
        MIC["Microphone (Web Audio API)"]
        REC["Audio Streamer (16kHz PCM Mono)"]
        UI_RENDER["Live Studio Console (Blue User / White AI)"]
        AUDIO_QUEUE["Buffered Audio Player Queue (24kHz Web Audio)"]
        WS_CLIENT["Dual WebSocket Client (/ws/duplex_audio)"]
    end

    subgraph Server ["FastAPI Duplex Engine (scripts/duplex_engine.py)"]
        CLOCK["MicroTurnClock (Drift-Free 600ms Loop)"]
        SLICER["StreamingAudioSlicer (19,200 Byte / 9,600 Sample Slices)"]
        STT["StreamingSTTProcessor (faster-whisper fp16 CUDA)"]
        SESSION["DuplexSession Manager (Rolling History & Pruning)"]
        MODEL_CLIENT["DuplexModelClient (HTTP Pool to llama-server)"]
        TTS["KokoroTTSProcessor (Kokoro-82M ONNX CUDA)"]
    end

    subgraph ModelServer ["Local llama-server (Port 8080)"]
        GGUF["duplex_gemma_merged.gguf (CUDA Layer Offloaded)"]
    end

    %% Audio Flow
    MIC --> REC
    REC -->|Raw PCM 16-bit 16kHz Stream| WS_CLIENT
    WS_CLIENT -->|Binary WebSocket Frames| SLICER

    %% 600ms Clock Drive
    CLOCK -->|600ms Tick Event| SLICER
    SLICER -->|Exact 600ms Audio Slice| STT
    STT -->|Incremental Transcribed Delta| SESSION

    %% Neural Evaluation
    SESSION -->|Rolling Context + Current Slice| MODEL_CLIENT
    MODEL_CLIENT -->|Chat Completion Request + Logprobs| GGUF
    GGUF -->|Token 0 Logprobs + Generated Tokens| MODEL_CLIENT

    %% Decision & Synthesis
    MODEL_CLIENT -->|P(<listen>) >= 0.30 -> YIELD| SESSION
    MODEL_CLIENT -->|P(<listen>) < 0.30 -> SPEAK| TTS
    TTS -->|24kHz PCM / Base64 Audio Chunk| WS_CLIENT

    %% Client Playback & Telemetry
    WS_CLIENT -->|Audio Chunks| AUDIO_QUEUE
    WS_CLIENT -->|Micro-Turn Events & Telemetry| UI_RENDER
    
    %% Instant Barge-In
    REC -.->|Real-time Voice Detection| AUDIO_QUEUE
    AUDIO_QUEUE -.->|Instant Playback Cutoff <= 30ms| AUDIO_QUEUE
```

---

## Key Features

- **Continuous 600ms Micro-Turn Engine**: Operates on a drift-free monotonic clock ($\sigma \le 0.50\text{ms}$) that evaluates dialogue every 600ms, eliminating VAD silence waiting periods.
- **Zero-Hardcoding Neural Policy**: Turn-taking is 100% mathematically governed by model weights:
  $$\text{Action} = \begin{cases} \text{YIELD (listen)}, & \text{if } P(\text{<listen>}) \ge 0.30 \\ \text{SPEAK}, & \text{if } P(\text{<listen>}) < 0.30 \end{cases}$$
- **Instant Client-Side Barge-In ($\le 30\text{ms}$)**: When the user speaks while the assistant is talking, audio playback is instantly interrupted at the hardware level, clearing the audio queue without lag.
- **Ultra-Fast Local TTS (Kokoro-82M ONNX)**: Synthesizes high-fidelity 24kHz speech fragments in **120ms – 180ms** on GPU with CUDA acceleration.
- **Streaming Audio Ingestion (16kHz PCM)**: Slices continuous microphone streams into exact 9,600-sample chunks with RMS energy gating.
- **Anti-Repetition Neural Recovery**: Calibrated sampling ($T = 0.35$, $\text{top\_p} = 0.95$, $\text{repeat\_penalty} = 1.15$), barge-in utterance truncation, and token Jaccard similarity detection prevent repetition attractor loops.
- **Strict VRAM Safety ($\le 5.0\text{ GB}$)**: Engineered to fit within 6GB laptop GPUs (`llama-server` + `faster-whisper` + `Kokoro-82M` = **~4.8 GB peak VRAM**).

---

## Hardware Budgets & Timing SLAs

| Metric | Target SLA | Measured Benchmark |
| :--- | :--- | :--- |
| **Clock Jitter** | $600\text{ms} \pm 15\text{ms}$ | **$600.12\text{ms}$ ($\sigma = 0.49\text{ms}$)** |
| **ASR Slice Transcription** | $\le 150\text{ms}$ | **$15\text{ms} - 35\text{ms}$** |
| **LLM Turn-Taking Evaluation** | $\le 600\text{ms}$ | **$250\text{ms} - 520\text{ms}$** |
| **TTS Time-to-First-Audio (TTFA)** | $\le 350\text{ms}$ | **$120\text{ms} - 180\text{ms}$** |
| **Client Barge-In Playback Cutoff** | $\le 30\text{ms}$ | **$< 30\text{ms}$** |
| **Peak GPU VRAM Usage** | $\le 5,120\text{ MB}$ | **$4,832\text{ MB}$** |

---

## Repository Structure

```
TalkAI---Local/
├── docs/                                   # Architectural guides & specifications
│   ├── ARCHITECTURE.md                     # Deep technical reference & resumption guide
│   ├── full_duplex_cascade_plan.md         # 5-Phase implementation specification
│   ├── modular_plan.md                     # Modular test design
│   └── modules.md                          # Subagent task descriptions
├── data/                                   # Datasets & training assets
│   ├── duplexcascade_synthetic/            # Fine-tuning & benchmark dialogues
│   │   ├── micro_turns_train.jsonl         # 390 training conversations
│   │   ├── micro_turns_val.jsonl           # 60 zero-leakage validation scenarios
│   │   └── micro_turns_test.jsonl          # 60 benchmark test dialogues
│   └── archive/                            # Archived historical datasets
├── models/
│   ├── finetuned_duplex/                   # Fine-tuned Duplex GGUF models
│   │   └── duplex_gemma_merged.gguf        # Production merged fine-tuned model (port 8080)
│   ├── raw_base/                           # Untuned base weights
│   │   ├── gemma-4-E2B_q4_0-it.gguf
│   │   └── gemma-4-E2B-it-mmproj.gguf
│   └── kokoro/                             # Kokoro-82M ONNX model assets
│       ├── kokoro-v0_19.onnx               # 82M ONNX speech model
│       └── voices.bin                      # 11 voice embeddings
├── scripts/
│   ├── duplex_engine.py                    # Main FastAPI full-duplex production server
│   ├── start_llama_server.sh               # Launch script for llama-server on port 8080
│   └── testing/                            # Comprehensive modular test suites
│       ├── test_module1_engine.py          # Clock jitter, VRAM, and session verification
│       ├── test_module2_streaming_stt.py   # Audio slicing & Whisper STT verification
│       ├── test_module3_neural_policy.py   # Turn-taking policy & P(<listen>) assertions
│       ├── test_module4_tts_pipeline.py    # Kokoro synthesis & barge-in cutoff verification
│       ├── test_speech_accumulator_e2e.py  # Incremental speech extraction tests
│       ├── test_user_hot_oil_scenario.py   # Mid-sentence hazard interjection tests
│       ├── benchmark_duplex_runtime.py     # End-to-end latency & throughput benchmark
│       └── verify_full_duplex.py           # Unified full-duplex verification suite
├── logs/                                   # Verification reports and execution traces
│   ├── module_1_report.json                # Module 1 Evaluation Gate report
│   ├── execution_state.json                # Modular progress state
│   └── archive/                            # Debugging session logs
├── test.html                               # Modern Browser Voice Studio interface
├── requirements.txt                        # Python dependencies
├── AGENTS.md                               # Multi-agent protocol & evaluation gates
└── README.md                               # This file
```

---

## Quickstart Guide

### 1. Prerequisites
- Linux x86_64 (Ubuntu 22.04+ recommended)
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 4050 6GB)
- Pre-built `llama-server` binary (included in `llama.cpp` or custom fork)

### 2. Environment Setup
```bash
git clone https://github.com/Adityakeerti/TalkAI---Local.git
cd TalkAI---Local

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

### 3. Running TalkAI

#### Step A: Launch the Local Model Server (Terminal 1)
```bash
bash scripts/start_llama_server.sh
```
*Hosts `models/duplex_gemma_merged.gguf` on `http://127.0.0.1:8080` with CUDA offload.*

#### Step B: Start the Duplex Micro-Turn Engine (Terminal 2)
```bash
python scripts/duplex_engine.py --host 0.0.0.0 --port 8000
```
*Hosts the WebSocket engine and serves `test.html` on `http://localhost:8000`.*

#### Step C: Open the Browser Interface
Open your browser and navigate to:
```
http://localhost:8000
```
1. Click the center **Microphone Button** to connect and begin streaming.
2. Select your preferred voice (e.g., `af_bella`, `af_sarah`, `am_adam`, `bf_emma`).
3. Speak naturally—pause mid-sentence to observe the assistant holding floor, interject when the assistant speaks to trigger instant barge-in, or present urgent scenarios to see hazard interjections.

---

## Verification & Testing

Execute the modular test suites under `scripts/testing/` to validate every layer of the cascade:

```bash
# Run Module 1 Core Engine Verification (Clock jitter, VRAM cap, WebSockets)
python scripts/testing/test_module1_engine.py

# Run Neural Policy & Turn-Taking Verification
python scripts/testing/test_module3_neural_policy.py

# Run Full Duplex End-to-End Benchmark
python scripts/testing/verify_full_duplex.py
```

Expected output for Module 1:
```
================================================================
TALKAI MODULE 1: DUPLEX MICRO-TURN ENGINE CORE VERIFICATION SUITE
================================================================
Clock stats: Mean interval = 600.12ms, Std = 0.49ms, Drift = 1.20ms
Session state & silent turn pruning tests passed cleanly.
Hesitation turn: Action=YIELD, P(listen)=0.9920, Output='listen'
Hazard turn: Action=SPEAK, P(listen)=0.0000, Output='Do that for safety reasons!'
Concurrent WebSockets verified: Session A and Session B executed cleanly.
Zero hardcoding audit passed: No keyword checks, no logit bans, no regex stripping.
VRAM Info: Used = 4781.0MB, Cap = 5120.0MB, Under Cap = True
================================================================
MODULE 1 EVALUATION GATE: ALL PASSED
================================================================
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).  
Model weights for Gemma-2B are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).  
Kokoro TTS is provided under the Apache 2.0 license.
