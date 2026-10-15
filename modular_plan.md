# TalkAI: Modular Fine-Tuning Specification (Gemma-4-E2B DuplexCascade)
=============================================================================

> **Target Architecture**: Google Gemma-4-E2B-it (2.3B effective, 35 decoder layers, Per-Layer Embeddings)  
> **Target Task**: Full-Duplex Micro-Turn Dialogue (600ms slices: yielding `<listen>` vs streaming speech)  
> **Hardware Constraint**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124)  
> **Base Weights**: `models/gemma-4-E2B-it/`  
> **Training Data**: `data/duplexcascade_synthetic/micro_turns_train.jsonl` (480 conversations, 4,318 micro-turns)  
> **Validation Data**: `data/duplexcascade_synthetic/micro_turns_val.jsonl` (60 conversations, 534 micro-turns)

---

## Architecture Principles & Lessons Learned
1. **No Manual Activation Caching**: Do NOT extract hidden states to disk via custom forward hooks. With a 2.3B model, PyTorch + PEFT fits directly in 6GB VRAM without risk of positional embedding (RoPE/MRoPE) corruption.
2. **Top-Layer LoRA**: Freeze layers 0–30. Train only layers 31–34 + `lm_head` + output norm. This trains only pragmatic turn-timing and speech generation (~2.5M parameters) while leaving core linguistic knowledge untouched.
3. **Discrete Action Token**: In the Gemma-4 tokenizer, `"listen"` is token ID `20923`. Train directly on `"listen"` without inventing custom multi-token XML tags.
4. **Conversational Loss Masking**: Cross-entropy loss is computed **strictly** on assistant tokens. User speech and system prompt tokens must have `label = -100`.

---

## Module 1: Tokenizer & Loss Masking Verification (Pre-Flight Check)
### 1.1 Objective
Verify that `data/duplexcascade_synthetic/micro_turns_train.jsonl` is correctly parsed, tokenized with `models/gemma-4-E2B-it`, and masked with PyTorch `CrossEntropyLoss` labels (`-100` on prompt/user tokens, active token IDs on assistant responses).

### 1.2 Implementation Requirements
- Script: `scripts/verify_dataset_tokenization.py`
- Test against: 2 sample conversations (one where assistant responds with `"listen"`, one where assistant speaks words).
- Key Checks:
  1. Tokenizer encodes `"listen"` as single token `[20923]`.
  2. For user micro-turn tokens: `labels[i] == -100`.
  3. For assistant tokens: `labels[i] == input_ids[i]`.
  4. Sequence length is bounded ($\le 512$ tokens).

### 1.3 Evaluation Gate (Pass Criteria)
- [ ] Assertion passes: `tokenizer.encode("listen", add_special_tokens=False) == [20923]`.
- [ ] Assertion passes: Number of active loss tokens equals exactly the total number of assistant tokens.
- [ ] Clean output saved to `logs/module1_tokenization_check.json`.

---

## Module 2: Model Architecture Loading, Layer Freezing & VRAM Sanity
### 2.1 Objective
Load `google/gemma-4-E2B-it` in PyTorch on CUDA, freeze layers 0 to 30, apply LoRA to layers 31–34, and execute a single dummy forward + backward pass to confirm VRAM footprint is safely under 5.0 GB.

### 2.2 Implementation Requirements
- Script: `scripts/sanity_check_model_lora.py`
- Model Loader:
  - `AutoModelForCausalLM.from_pretrained("models/gemma-4-E2B-it", torch_dtype=torch.bfloat16, device_map="cuda")`
  - Enable gradient checkpointing (`model.gradient_checkpointing_enable()`).
- Layer Freezing:
  - Iterate `model.model.layers[:31]` and set `param.requires_grad = False`.
- PEFT LoRA Config:
  - Rank $r = 16$, $\alpha = 32$, dropout $0.05$.
  - Target modules: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
  - Layers to transform: `[31, 32, 33, 34]`.
- Execution:
  - Run 1 forward pass and 1 backward pass with dummy tensor.
  - Query `torch.cuda.max_memory_allocated()`.

### 2.3 Evaluation Gate (Pass Criteria)
- [ ] Trainable parameters $\le 4.0\text{M}$ (out of $\approx 2.3\text{B}$, $<0.2\%$).
- [ ] Peak VRAM $\le 4.8\text{ GB}$ (leaving $\ge 1.2\text{ GB}$ safety buffer).
- [ ] Gradients are non-zero and contain **0 NaNs / 0 Infs**.
- [ ] Clean output saved to `logs/module2_vram_sanity.json`.

---

## Module 3: 10-Sample Micro-Overfit Test
### 3.1 Objective
Run a rapid 10-step micro-training test on just 10 conversations to confirm gradient flow, loss calculation, and optimization dynamics before committing to full training.

### 3.2 Implementation Requirements
- Script: `scripts/micro_overfit_test.py`
- Training subset: First 10 conversations from `micro_turns_train.jsonl`.
- Optimizer: `AdamW` ($lr = 2\times 10^{-4}$).
- Iterations: 10 optimization steps (batch size 1, grad accum 2).
- Monitor loss at step 1 vs step 10.

### 3.3 Evaluation Gate (Pass Criteria)
- [ ] Initial Loss at Step 1 is in range $[2.5, 4.5]$.
- [ ] Final Loss at Step 10 is $\le 1.2$ (verifying model can overfit the targets).
- [ ] No gradient explosion (grad norm stays bounded $< 2.0$).
- [ ] Clean output saved to `logs/module3_micro_overfit.json`.

---

## Module 4: Full Dataset Training & PEFT Checkpoint Validation
### 4.1 Objective
Execute complete LoRA fine-tuning on `data/duplexcascade_synthetic/micro_turns_train.jsonl` (480 conversations) for 2 epochs, validate against `micro_turns_val.jsonl`, and save PEFT adapter.

### 4.2 Implementation Requirements
- Script: `scripts/train_duplex_lora.py`
- Batch configuration: `batch_size = 1`, `gradient_accumulation_steps = 8` (effective batch size = 8).
- Epochs: 2 epochs (approx 120 optimizer steps total).
- Validation: Evaluate cross-entropy loss on `micro_turns_val.jsonl` every 30 steps.
- Output artifacts:
  - `models/duplex_gemma_lora/adapter_model.safetensors`
  - `models/duplex_gemma_lora/adapter_config.json`

### 4.3 Evaluation Gate (Pass Criteria)
- [ ] Training loss steadily drops and converges below $0.80$.
- [ ] Validation loss drops without divergence.
- [ ] Adapter files exist in `models/duplex_gemma_lora/` with size between 10MB and 40MB.
- [ ] Clean output saved to `logs/module4_training_report.json`.

---

## Module 5: GGUF Adapter Conversion & Runtime Benchmark
### 5.1 Objective
Convert the PEFT adapter into GGUF format using `llama-fork`, attach it to `llama-server`, and verify full-duplex conversational behavior with test scenarios.

### 5.2 Implementation Requirements
- Converter command:
  ```bash
  .venv/bin/python /home/aditya/llama-fork/convert_lora_to_gguf.py \
    --base models/gemma-4-E2B-it \
    --outfile models/duplex_gemma_lora.gguf \
    models/duplex_gemma_lora/
  ```
- Runtime Server:
  - Launch `llama-server` with `--model Gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf --lora models/duplex_gemma_lora.gguf`.
- Verification Script: `scripts/benchmark_duplex_runtime.py`
  - Test Case A (Hesitation): `"I was thinking about..."` $\rightarrow$ must yield `listen`.
  - Test Case B (Hazard Interruption): `"The outlet is sparking and I might touch it"` $\rightarrow$ must cut in with warning words.
  - Test Case C (Direct Question): `"What is the capital of France?"` $\rightarrow$ must speak concise answer.
  - Measure Latency: Ensure response generation finishes in $< 150\text{ ms}$ (inside the 600ms budget).

### 5.3 Evaluation Gate (Pass Criteria)
- [ ] `models/duplex_gemma_lora.gguf` is created and valid.
- [ ] Test Case A returns `listen`.
- [ ] Test Case B returns spoken safety warning.
- [ ] Test Case C returns spoken response.
- [ ] Average generation latency $< 200\text{ ms}$.
- [ ] Final benchmark report saved to `logs/module5_runtime_report.md`.
