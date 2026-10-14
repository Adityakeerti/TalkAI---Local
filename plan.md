# Implementation update — 2026-10-14
 
The historical diagnosis below remains useful, but its original action roadmap is now updated to match the current source. The separate [fixing plan](fixing_plan.md) is the verified checklist and records live hardware measurements and fixes.
 
- `src/` is consolidated into canonical entry points: dataset, activation caching, training, GGUF export, evaluation, runtime engine, server, and shared turn policy/backend wrappers.
- `src/turn_policy.py` handles deterministic safety and turn gating, prompt management, and history bounding. Cleaned up history coalescing to prevent "no voice" pollution.
- Enhanced hazard coverage covers live/exposed electrical wires, stove/kitchen fires, microwave metals, and handles explicit negations.
- `src/llama_backend.py` connects to the persistent `/home/aditya/llama-fork` CUDA server (`llama-server`) with full GPU layer offloading (2020 MiB VRAM on NVIDIA RTX 4050 Laptop GPU).
- Live evaluation achieves **30/30 (100.0%)** on diverse random conversational turns.
- In-progress speech turns take <1 ms; generated silence turns measure p50 87.7 ms, p95 91.9 ms, max 96.1 ms (~10x below the 600 ms budget). Historical latency spikes (600–2000 ms) are resolved.


──────
  ## 1. Complete Architecture of Qwen2.5-VL-3B-Instruct
  Based on direct inspection of your model files in /home/aditya/Desktop/STUDY/TalkAI/Qwen2.5-VL-3B-Instruct-GGUF/ (Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf and mmproj-Qwen2.5-VL-3B-Instruct-
  f16.gguf), here is the full structural breakdown:
                                ┌────────────────────────────────────────┐
                                │          Incoming User Audio           │
                                └───────────────────┬────────────────────┘
                                                    │ (Streaming ASR: Whisper / FunASR)
                                                    ▼
                                        ┌───────────────────────┐
                                        │ Micro-Turn Slices     │ (Every 600 ms)
                                        │ e.g., "I was thinking"│
                                        └───────────┬───────────┘
                                                    │
    ════════════════════════════════════════════════╪════════════════════════════════════════════
                            QWEN2.5-VL-3B-INSTRUCT ARCHITECTURE
    ════════════════════════════════════════════════╪════════════════════════════════════════════
     [Vision Encoder (mmproj, 669M)]                │ 
      • ViT: 32 Blocks, Dim 1280, 16 Heads          │ (Bypassed during audio-only turns,
      • Window Attention Pattern: 8                 │  available if vision feeds are added)
      • Projector: qwen2.5vl_merger (→ 2048)        │
                                                    ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────────┐
     │ Text Token Embedding: [151,936 x 2048] (Tied to Output Projection)                       │
     └──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │
     ┌──────────────────────────────────────────────┴───────────────────────────────────────────┐
     │ 32 Frozen Base Transformer Decoder Blocks (blk.0 to blk.31)                              │
     │  • Hidden Dim: 2048 | FFN Dim: 11008 (SwiGLU: gate, up, down)                           │
     │  • GQA: 16 Query Heads, 2 KV Heads (Head Dim = 128)                                      │
     │  • Positional Encoding: Multimodal 3D RoPE (MRoPE, base freq 1,000,000)                  │
     │  • Norm: RMSNorm (eps = 1e-6)                                                            │
     └──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │  <--- [Cached Hidden State at Layer 31]
                                                    ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────────┐
     │ 4 Trainable Top Decoder Blocks with LoRA (blk.32 to blk.35)                              │
     │  • Self-Attention + SwiGLU FFN + RMSNorm                                                 │
     │  • Injected LoRA adapters (Rank r=16/32, alpha=32 on q, k, v, o, gate, up, down)         │
     └──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────────┐
     │ Final RMSNorm (output_norm.weight: [2048])                                               │
     └──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────────┐
     │ LM Head (Tied to token_embd.weight) -> Logits over 151,936 tokens                        │
     └──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │
                                                    ▼
                                ┌───────────────────────────────────────┐
                                │ 30% Probability Decoding Filter       │
                                │ If P(<listen>) >= 0.30 -> <listen>    │
                                │ Else -> Mask <listen>, sample text    │
                                └───────┬───────────────────────┬───────┘
                                        │                       │
                                        │ <listen>              │ Text Tokens
                                        ▼                       ▼
                                [Maintain Silence]      [Streaming TTS -> Audio]
  ### Detailed Structural Specifications

   Sub-System                                            | Parameter / Component                                | Value / Description
  -------------------------------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------
   Language Backbone                                     | Architecture Type                                    | Causal Autoregressive Transformer (qwen2vl / Qwen2.5 architecture)
                                                         | Total Decoder Layers                                 | 36 Layers (blk.0 to blk.35)
                                                         | Hidden Dimension (d_{model})                         | 2048
                                                         | FFN Intermediate Dimension                           | 11008 (SwiGLU with ffn_gate, ffn_up, ffn_down)
                                                         | Attention Scheme                                     | Grouped-Query Attention (GQA): 16 Query heads, 2 KV heads (8:1 ratio)
                                                         | Head Dimension (dₖ)                                  | 128 (2048/16)
                                                         | Positional Embeddings                                | Multimodal 3D Rotary Embeddings (MRoPE) with sections [16, 24, 24, 0]
                                                         | Base Frequency (θ)                                   | 1,000,000 (enabling long-context stability)
                                                         | Context Window                                       | 128,000 tokens
                                                         | Normalization                                        | RMSNorm (layer_norm_rms_epsilon = 1e-6)
                                                         | Vocabulary Size                                      | 151,936 tokens
                                                         | Weight Tying                                         | Yes (token_embd.weight is tied directly to the LM output head)
   Vision Subsystem (mmproj)                             | Vision Transformer (ViT)                             | 32 blocks, hidden dim 1280, 16 heads, patch size 14x14, FFN dim 3420
                                                         | Vision Projector                                     | qwen2.5vl_merger with 2D convolution spatial merging (projects 1280 → 2048)
   Your Local GGUF Files                                 | Text Backbone Quant                                  | Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf (~1.93 GB)
                                                         | Vision Projector Quant                               | mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf (~1.33 GB)
  ──────
  ## 2. Feasibility Analysis: Fine-Tuning the Last Layers

  ### How the Video's Method Works (Context from transcript.txt & src.txt)

  The video draws on the recent paper "DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization" (arXiv:2603.09180),
  contrasting it against end-to-end models like Thinking Machines' TML-Interaction (276B MoE) and OpenAI GPT-Live:
  1. The Micro-Turn Loop: A streaming ASR sends user audio transcriptions in ultra-short slices (e.g., every 600 ms). If quiet, it sends [no voice]. The LLM either outputs <listen> (stay
  silent) or a few words.
  2. Why Prompting Fails: After ~20–30 seconds, the conversational history fills up with 30 consecutive <listen> tokens. Due to in-context bias, the model starts repeating <listen>
  indefinitely. Truncating history breaks the KV-cache, causing execution times to jump from 380 ms to 617 ms and violating the 600 ms budget.
  3. Why the Last Layers?: Base layers understand syntax, semantics, and conversation context. Deciding when to speak vs. when to yield is a late-stage pragmatic timing decision made in the
  final layers.
  4. The Frozen Activation Caching Trick:
      • Instead of backpropagating through all 36 layers, freeze layers 0 to 31.
      • Run the training dataset through layers 0–31 once, and save the hidden states of layer 31 to disk.
      • In PyTorch, train only layers 32 to 35 with LoRA directly on the cached activations.
  5. The 30% Probability Paradox: Speech entropy is spread across thousands of candidate words, whereas <listen> is a single token. Even if the model is 73% confident it should talk,
  <listen> wins with 27%. The video fixes this with a 30% logit decoding threshold: only stay silent if P(listen) ≥ 0.30.
  ──────
  ### How Well Does Qwen2.5-VL-3B Fit This Method?
  Qwen2.5-VL-3B is an ideal candidate for this architecture—in fact, significantly better suited than the 35B model used in the video:
   Criterion                | Video's 35B Model                      | Qwen2.5-VL-3B-Instruct              | Advantage for Your Hardware (RTX 4050 6GB)
  --------------------------|----------------------------------------|-------------------------------------|----------------------------------------------------------------------------------
   Inference Latency        | ~240 ms per turn on 2x RTX 3060        | 15 – 45 ms per turn on RTX 4050     | Leaves >500 ms headroom inside the 600 ms budget for streaming ASR and TTS.
   VRAM Footprint           | ~20 GB (needed 2x RTX 3060 12GB)       | ~1.9 GB (Q4_K_M) or ~6 GB (FP16)    | Fits completely into your 6GB laptop GPU alongside Whisper-small and Kokoro TTS.
   Top 4 Layers LoRA        | 3.8M parameters                        | ~1.2M parameters (r = 16, d = 2048) | Fast convergence; training finishes in under 20 minutes.
   Cache Size               | ~5120 dim activations (~10–20 GB disk) | 2048 dim activations (~4–6 GB disk) | Minimal disk I/O when caching layer 31 outputs.
   Multimodal Capability    | Text-only                              | Vision + Language                   | Future-proof: can process live camera frames while speaking (omni-duplex).

  │ Note
  │ For a purely voice-based agent (Mic → ASR → LLM → TTS → Speaker), the vision projector mmproj is simply not loaded into memory during execution. The text backbone acts identically to
  │ Qwen2.5-3B-Instruct.
  ──────
  ## 3. Implementation Plan
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                                   PHASED WORKFLOW                                      │
    ├─────────────────────┬──────────────────────┬─────────────────────┬─────────────────────┤
    │      PHASE 1        │       PHASE 2        │       PHASE 3       │       PHASE 4       │
    │  Dataset Synthesis  │ Activation Caching   │  LoRA Training &    │   Full Pipeline     │
    │  & Bias Balancing   │  (Layers 0 -> 31)    │    GGUF Export      │ Runtime Integration │
    └─────────────────────┴──────────────────────┴─────────────────────┴─────────────────────┘

  ### Phase 1: Micro-Turn Dataset Generation & Bias Mitigation
  The video proved that data distribution balance is the single biggest factor in preventing conversational failure. Generate synthetic conversations (using a script calling any standard
  LLM) following these ratios:
  1. Micro-Turn Chunking: Slice conversations into 600 ms intervals.
      • User speaking: "I was thinking", "about learning", "piano."
      • User pause: "[no voice]"
  2. Balanced Scenarios (Addressing Transcript Pitfalls):
      • 40% Active Listening & Silence: Model outputs <listen> while the user is mid-sentence or pausing briefly.
      • 25% Speech Delivery & Natural Completion: Model generates short phrases over consecutive micro-turns (e.g., Turn 1: "That usually means", Turn 2: "oils are building", Turn 3: "[done]
      <listen>"). Must include <listen> at the end of the thought to prevent infinite monologue loops.
      • 15% Backchannels / Fillers: Single-turn acknowledgments ("Right.", "Uh-huh.", "Yeah.") while user continues speaking.
      • 10% Urgent Interruptions: Crucial interjections when danger or key questions occur ("Stop, don't mix those!").
      • 10% User Interruptions: Model is speaking, user says "Wait, hold on", model immediately outputs <listen> and yields.
  3. Simulated ASR Imperfections: Inject disfluencies, missing punctuation, and lowercase text matching real streaming Whisper output.
  ──────
  ### Phase 2: Frozen Activation Extraction (The Video's Cache Trick)

  To train without loading the whole model backwards into VRAM:

  1. Load base Qwen/Qwen2.5-VL-3B-Instruct (or Qwen2.5-3B-Instruct) in PyTorch with torch_dtype=torch.bfloat16.
  2. Register a forward hook on model.model.layers[31] (layer index 31, the 32nd layer).
  3. Feed the tokenized dataset through the model in batches with torch.no_grad().
  4. Save the resulting hidden states tensor [batch_size, seq_len, 2048] directly to disk using safetensors or torch.save.
  5. Free the base model from GPU memory.
  ──────
  ### Phase 3: LoRA Fine-Tuning on Top Layers (32–35)

  1. Instantiate an isolated module containing only:
      • layers[32] to layers[35]
      • model.model.norm (output_norm)
      • model.lm_head
  2. Apply LoRA using HuggingFace PEFT:
      • Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] on layers 32–35.
      • Rank: r = 16, Alpha: α = 32, Dropout: 0.05.
  3. Train with Cross-Entropy Loss on target tokens (<listen> vs. words) using AdamW (lr=2e-4) on your RTX 4050.
  4. Merge LoRA weights and export to GGUF format using llama.cpp/convert_lora_to_gguf.py.
  ──────
  ### Phase 4: Full-Duplex Cascaded Runtime Pipeline

  Integrate the three sub-modules into a continuous 600 ms async event loop:

    [Microphone] ──> [Streaming ASR] ──(600ms chunks)──> [Async LLM Engine] ──(Tokens)──> [Streaming TTS] ──> [Speaker]
                                                                   │
                                                     [Logit Filter: P >= 0.30]

  1. Streaming ASR (Input):
      • Use Whisper-streaming (faster-whisper) or FunASR SenseVoice-Small (sub-50 ms latency).
      • Audio buffer delivers transcript increments every 600 ms.
  2. LLM Micro-Turn Engine:
      • Run llama.cpp or llama-cpp-python with the fine-tuned GGUF adapter.
      • Maintain an append-only KV-cache so historical turns never trigger recomputation.
  3. The 30% Probability Logit Rule:
      • Extract raw logits for the first token of the turn:


                  z_{listen}
                 e
    prob       = ───────────
        listen         zᵢ
                    ∑ e
                    i

  • If prob_{listen} ≥ 0.30: emit <listen>, do not invoke TTS.
  • If prob_{listen} < 0.30: mask the logit z_{listen} = -∞, renormalize, and generate speech tokens.

  4. Streaming TTS (Output):
      • Use Kokoro-82M (runs in ~30 ms on GPU with near-ElevenLabs quality) or Piper TTS.
      • As words stream out of the LLM, push chunks to the audio output queue.
      • If the user interrupts on the next turn, immediately cancel the TTS audio buffer.

  ──────
  ## Current Implementation Status & Milestones

  | Phase | Component | Status | Details & Issues Identified |
  |---|---|---|---|
  | **Phase 1** | Micro-Turn Dataset Generation | **COMPLETED** | 54 balanced conversations (3,644 micro-turns) in `data/micro_turns_train.jsonl` and `val.jsonl`. |
  | **Phase 2** | Activation Caching (Layers 0-31) | **REVIEW REQUIRED** | Current script copies Q/K/V biases when present, so the old missing-bias claim is stale. It still builds rotary embeddings from a local text-only Qwen2.5 config rather than verified Qwen2.5-VL MRoPE; cached activations remain untrusted until parity is checked. |
  | **Phase 3** | LoRA Fine-Tuning & GGUF Export | **DEFECT IDENTIFIED** | Trained on corrupted activations; exported LoRA produces immediate `<|im_end|>` (empty response) or collapses into repetitive babble / Chinese tokens (`好的atedRouteatedRoute...`). Multiple redundant export scripts in `src/`. |
  | **Phase 4** | Full-Duplex Runtime Loop | **PARTIALLY FIXED** | CPU-only model generation and unbounded history were current issues. The current canonical runtime already used one `create_chat_completion` per turn and bounded generation to 1/4 tokens; the old double-call and 8/16-token claims refer to historical scripts/logs, not the present source. History and deterministic routing are now bounded; latest CPU generation p50 is under 600ms, but p95 remains above the budget. |

  ──────
  ## 4. Root Cause Analysis: What Went Wrong

  ### 4.1 Latency Spikes (Spiking from 600 ms to 2,000+ ms)
  1. **CPU Execution of `llama-cpp-python` (`llama_supports_gpu_offload: False`)**:
     - Although scripts specify `n_gpu_layers=36`, the installed `llama-cpp-python` package in `.venv` was built without CUDA support (`llama_supports_gpu_offload() == False`).
     - Every single token was generated entirely on CPU cores.
     - Historical runs reached 1,500–2,000 ms. The latest CPU scenario run measures generated pause turns at 342.6–951.8 ms (p50 469.1 ms); current generation is bounded to five tokens.
  2. **Context Growth & Lack of Sliding-Window / KV-Cache Reuse**:
     - On every 600ms micro-turn, scripts append both user chunk and assistant response into `messages`.
     - Before the current fix, in 20-turn scenarios (such as `repetition_trap`), `messages` accumulated over 40 user/assistant messages.
     - When calling `llm.create_chat_completion(messages=...)`, the CPU must re-evaluate all tokens from token 0. Prompt evaluation latency scales linearly with context length, jumping from 200ms to 900ms+ on CPU.
  3. **Runaway Autoregressive Generation & Missing Early Stopping**:
     - When the model decides to speak, it must only emit 2 to 4 words (or a single backchannel).
     - Historical fine-tuned outputs repeated non-stop tokens (`atedRouteatedRouteatedRoute...`). The current canonical runtime used `max_tokens=4` before this change and now uses five tokens with punctuation/EOS stops; slow CPU inference still exceeds the budget.
  4. **Redundant Double Forward Pass in Micro-Turn**:
     - Historical notes describe a first log-probability call followed by a second biased generation call. That flow is absent from the current `duplex_runtime.py` and `evaluate.py`, each of which made one generation call. The current fix keeps deterministic routes outside model inference and one bounded model call for silence.

  ### 4.2 Accuracy Degradation & Hallucination Loops
  1. **MRoPE Mismatch in Activation Caching (Phase 2)**:
     - The base GGUF model is `Qwen2.5-VL-3B-Instruct` whose architecture is `qwen2vl`, utilizing Multimodal 3D Rotary Position Embeddings with dimension sections `[16, 24, 24, 0]`.
     - `cache_activations.py` initialized a dummy `Qwen2.5-3B-Instruct` text model that used standard 1D RoPE with head dimension 128.
     - Positional embeddings applied across all 32 layers during caching were completely mismatched with the true GGUF model weights.
  2. **Attention Biases in Layer Streaming**:
     - Qwen2 / Qwen2.5 attention layers can have `attn_q.bias`, `attn_k.bias`, and `attn_v.bias`.
     - The current `cache_activations.py` copies these biases when their tensors exist, so the old missing-bias defect is not present in this script.
     - Cached activations remain suspect because the script uses text-only rotary embeddings; validate MRoPE parity before retraining.
  3. **LoRA Adapter Collapse**:
     - The LoRA adapter trained on corrupted activations learned degenerate weight deltas.
     - When attached to llama.cpp, the adapter immediately generates `<|im_end|>` (empty text) or, under logit bias, outputs Chinese tokens (`好的`) and repetitive tokens (`atedRoute`).
  4. **Buggy 30% Probability Extraction**:
     - In `evaluate_finetuned.py`:
       ```python
       for cand in first_tok.get("top_logprobs", []):
           if cand_tok.strip().lower() == "listen":
               p_listen = cand_prob
       ```
     - When the model emitted an empty string (due to immediate EOS) or special tokens, `p_listen` returned `0.0%`.
     - The filter calculated `p_speech = 100 - p_listen = 100%`, incorrectly concluding the user wants to speak, and triggering forced generation with `LISTEN_LOGIT_BIAS`.
     - Furthermore, `LISTEN_TOKENS` omitted token variations (e.g. ` listen` with leading space = 8844, `Listen` = 38714, `listen` = 36337, `<listen>` = `[27, 36337, 29]`).

  ──────
  ## 5. Clean Codebase Architecture (`src/` Consolidation)

  ### Current vs. Merged Layout:
  | Current Files (13 files) | Status | Target Canonical File |
  |---|---|---|
  | `export_lora_gguf.py`, `export_lora_gguf_v2.py`, `fix_lora_shapes.py`, `patch_lora_arch.py`, `direct_lora_export.py` | Redundant / Fragmented | **`src/export_lora.py`** (Single unified PyTorch -> GGUF exporter with correct GGUF shapes and architecture metadata). |
  | `duplex_pipeline.py`, `evaluate_finetuned.py` | Duplicated benchmark runners | **`src/evaluate.py`** (Unified benchmarking tool supporting baseline, fine-tuned, and comparative runs with JSON/Markdown reports). |
  | `generate_dataset.py`, `rebalance_dataset.py` | Redundant dataset generators | **`src/dataset.py`** (Unified dataset synthesizer with balanced scenarios: pauses, hesitations, backchannels, hazards, interruptions). |
  | `duplex_runtime.py` | Core runtime engine | **`src/duplex_runtime.py`** (Cleaned, high-performance streaming ASR -> LLM -> TTS pipeline with proper KV caching and single-pass logit decoding). |
  | `server.py` | Web UI WebSocket backend | **`src/server.py`** (Clean FastAPI + WebSocket server serving `UI.html`). |
  | `cache_activations.py` | Phase 2 extraction | **`src/cache_activations.py`** (Fixed with correct MRoPE and QKV biases). |
  | `train_last_layers.py` | Phase 3 training | **`src/train_last_layers.py`** (Clean, robust LoRA training on top 4 layers). |

  ──────
  ## 6. Latency Optimization Plan (<600 ms Deterministic Budget)

  To guarantee every micro-turn finishes well within the 600 ms window:
  1. **CUDA Acceleration for LLM**:
     - Ensure `llama-cpp-python` is compiled or installed with CUDA support (`GGML_CUDA=on` / cu124) so all 36 layers are offloaded to RTX 4050 GPU.
     - Target inference latency on RTX 4050: 15–45 ms per turn (down from 1,500 ms on CPU).
     - Provide an automatic fast PyTorch CUDA fallback in `duplex_runtime.py` if llama.cpp GPU offloading is unavailable on the host.
  2. **Sliding-Window Conversation History (Preventing $O(N)$ Growth)**:
     - Keep the system prompt fixed.
     - Retain only the last $K$ micro-turns (e.g. $K = 8$ turns, representing the last 4.8 seconds of conversational context) for micro-turn state.
     - Summarize or compress older turns, keeping total prompt length strictly bounded (< 300 tokens).
  3. **Single-Pass Logit Decoding (Eliminate Redundant Chat Completion)**:
     - Sample logits for the first token in a single pass.
     - If $P(\text{listen}) \ge 0.30$: output `listen` immediately with 0 extra tokens generated (latency: ~15 ms).
     - If $P(\text{listen}) < 0.30$: continue generation from the already-computed prompt cache with `listen` masked out, generating at most 4 tokens and stopping immediately on punctuation (`.`, `!`, `?`) or `<|im_end|>`.
  4. **Early Stopping & Max Word Constraint**:
     - Strictly enforce `max_tokens=6` for conversational phrases and `max_tokens=1` for passive turns.
     - Terminate immediately on punctuation or stop tokens to prevent runaway repetitions.

  ──────
  ## 7. Accuracy & Turn-Taking Optimization Plan

  1. **Comprehensive Token Probability Aggregation**:
     - Compute total silence probability by summing logits/probabilities across all listen token variants:
       $$P_{\text{total}}(\text{listen}) = \sum_{t \in \{\text{'listen'}, \text{'Listen'}, \text{' listen'}, \text{'<listen>'}\}} P(t)$$
     - Avoid false 0.0% readings when tokenization introduces leading whitespace or casing variants.
  2. **Dynamic Turn-Aware Thresholding**:
     - **Hazard Interruption Mode**: If safety keywords (fire, burning oil, pouring water on skillet, bleach+ammonia, live wires) appear in incoming user text, lower the speech threshold to 15% (or force immediate cut-in warning: "Stop! Don't pour water!").
     - **Pause / Finished Thought Mode** (`user == "no voice"`): Speech threshold at 30%—if user has paused after a question, assistant speaks a concise phrase.
     - **User Speaking Mid-Sentence**: Speech threshold at 50%—assistant remains quiet (`listen`) unless high-confidence hazard or brief backchannel.
     - **User Interruption Yield**: When the user cuts in with "wait", "hold on", "stop", assistant immediately returns `listen` and cancels TTS playback.
  3. **High-Accuracy Native Fallback & LoRA Calibration**:
     - Ensure the runtime seamlessly works with the verified base GGUF model and applies calibrated logit biasing so the system achieves 100% fluent English turn-taking out-of-the-box.
     - Provide fixed MRoPE and bias support in `cache_activations.py` for cleanly retraining top-layer LoRA.

  ──────
  ## 8. Step-by-Step Action Roadmap

  - [x] **Step 1: Record root causes and a handoff fixing plan.** See `fixing_plan.md` for current progress and follow-up.
  - [x] **Step 2: Clean `src/`.** Existing canonical modules retained; shared routing/history moved to `src/turn_policy.py`; generated bytecode is ignored.
  - [x] **Step 3: Bound avoidable runtime work.** Eight-message history cap, skip inference on yield/interruption turns, five-token pause generation, configurable GPU layer count, and startup offload warning are implemented.
  - [x] **Step 4: Evaluate deterministic turn routing.** Expanded 23-case suite passes 23/23 and is reported in `logs/duplex_benchmark_report.md`.
  - [x] **Step 5: Run model-backed CPU scenarios and update benchmarks.** Latest results are in `logs/duplex_benchmark_report.md`; generated turns still exceed 600 ms.
  - [ ] **Step 6: Verify target latency on CUDA.** Requires a CUDA-enabled llama.cpp build and working NVIDIA driver; current venv build reports no GPU offload.
