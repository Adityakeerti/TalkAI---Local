# TalkAI: Module 5 Full-Duplex Runtime Benchmark Report
=============================================================================

> **Generated**: 2026-10-15 20:48:53 UTC  
> **Architecture**: Gemma-4-E2B (q4_0-it) + Top-Layer Duplex LoRA  
> **Inference Server**: `llama-server` (CUDA offload, Port 8080)  
> **Target Budget**: 600 ms Micro-Turn Time-Slice (SLA: $< 200\text{ ms}$)  
> **Status**: **PASSED**  

---

## 1. GGUF Adapter Verification

| Property | Value | Status |
| :--- | :--- | :--- |
| **File Path** | `/home/aditya/Desktop/STUDY/TalkAI/models/duplex_gemma_lora.gguf` | Verified |
| **File Size** | 12.13 MB (12,717,472 bytes) | Valid |
| **Format Magic** | `GGUF` | Valid GGUF |
| **GGUF Version** | 3 | Compatible |
| **Tensor Count** | 40 tensors | Matched |

---

## 2. Server Runtime Environment

- **Endpoint**: `http://127.0.0.1:8080`
- **Health Status**: `ok`
- **llama.cpp Build**: `b10957-b1330bc83`
- **System Prompt**: Active 600ms micro-turn conversational supervisor prompt.

---

## 3. Conversational Test Cases & Assertions

| Test Case | Scenario Description | User Prompt | Model Response Fragment | Assertion Result |
| :--- | :--- | :--- | :--- | :--- |
| **Case A** | Hesitation Yield | *"I was thinking about..."* | `listen for the next part. if you are ready to share, I am` | **PASS** |
| **Case B** | Hazard Interruption | *"The outlet is sparking and I might touch it"* | `listen for safety. if the outlet is sparking, do not touch it.` | **PASS** |
| **Case C** | Direct Question | *"What is the capital of France?"* | `listen for you the capital of France is Paris. for example, it is` | **PASS** |

### Detailed Assertion Diagnostics

- **Case A (Hesitation Yield)**: Found 'listen' keyword in response: 'listen for the next part. if you are ready to share, I am'
- **Case B (Hazard Interruption)**: Found safety warning keywords ['not touch', 'outlet'] in response: 'listen for safety. if the outlet is sparking, do not touch it.'
- **Case C (Direct Question)**: Found 'Paris' in response: 'listen for you the capital of France is Paris. for example, it is'

---

## 4. Latency & Performance Benchmark Breakdown

All measurements evaluated across 5 repeated runs (after 1 warm-up invocation).

| Test Case | Prompt Eval (ms) | Generation Latency (ms) | Client Roundtrip (ms) | Tokens / Sec | SLA Gate (< 200ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Case A** | 23.2 ± 0.6 ms | 145.51 ± 1.07 ms | **172.06 ± 1.61 ms** | 96.2 t/s | **PASS** |
| **Case B** | 22.91 ± 0.1 ms | 145.02 ± 0.64 ms | **171.24 ± 0.42 ms** | 96.5 t/s | **PASS** |
| **Case C** | 22.91 ± 0.39 ms | 146.07 ± 1.16 ms | **172.04 ± 1.43 ms** | 95.9 t/s | **PASS** |

### Aggregate Performance Summary

- **Global Average Roundtrip Latency**: **171.78 ms**
- **Global Average Generation Latency**: **145.53 ms**
- **Global Average Prompt Eval Latency**: **23.01 ms**
- **Micro-Turn Budget Margin**: **428.2 ms headroom** remaining within 600ms micro-turn frame.

---

## 5. Module 5 Evaluation Gate Checklist

- [x] `models/duplex_gemma_lora.gguf` is created and valid GGUF.
- [x] Test Case A returns `listen` upon hesitation prompt.
- [x] Test Case B returns spoken safety warning upon urgent hazard.
- [x] Test Case C returns spoken response containing `Paris`.
- [x] Average generation latency $< 200\text{ ms}$ (Achieved: **145.53 ms**).
- [x] Average roundtrip latency $< 200\text{ ms}$ (Achieved: **171.78 ms**).
- [x] **OVERALL EVALUATION GATE**: **PASSED**.

---
*Benchmark completed autonomously in accordance with AGENTS.md and modular_plan.md.*
