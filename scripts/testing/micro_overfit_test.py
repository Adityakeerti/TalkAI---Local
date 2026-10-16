#!/usr/bin/env python3
"""
scripts/micro_overfit_test.py
Module 3: 10-Sample Micro-Overfit Test

Responsibilities:
1. Load Google Gemma-4-E2B-it with top-layer LoRA (layers 31..34) via setup_duplex_gemma_model().
2. Load first 10 conversations from data/duplexcascade_synthetic/micro_turns_train.jsonl via DuplexDataset.
3. Compute forward hidden states from language model, mask to only active assistant positions
   (where labels[:, 1:] != -100), and compute lm_head only on those active positions.
   This drastically cuts peak memory and prevents CUDA OOM over the 262,144 vocabulary.
4. Optimize using AdamW(lr=1e-3, batch_size=1, gradient_accumulation_steps=2).
5. Bound gradient norm strictly < 2.0 via torch.nn.utils.clip_grad_norm_.
6. Track loss trajectory from Step 1 to final step, ensuring final loss <= 1.2 (approx 10-12 steps).
7. Validate Evaluation Gate criteria:
   - Initial loss at Step 1 recorded
   - Final loss <= 1.2
   - Gradients bounded < 2.0 with 0 NaNs / 0 Infs
   - Peak VRAM <= 4.8 GB
8. Save structured reports to logs/module3_micro_overfit.json and logs/module_3_report.json.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set CUDA allocator configuration before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-E2B-it"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "duplexcascade_synthetic" / "micro_turns_train.jsonl"
LOGS_DIR = PROJECT_ROOT / "logs"

# Add project root to sys.path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sanity_check_model_lora import setup_duplex_gemma_model
from scripts.verify_dataset_tokenization import DuplexDataset, DuplexDataCollator

# Gate Thresholds & Hyperparameters
NUM_TRAIN_SAMPLES = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 20
TARGET_LOSS = 1.2
MAX_GRAD_NORM = 2.0
MAX_PEAK_VRAM_GB = 4.8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("micro_overfit_test")


def compute_masked_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes cross-entropy loss ONLY on active assistant tokens to prevent CUDA OOM
    from calculating full logits (262,144 vocab) over inactive prompt/user tokens.

    Pipeline:
    1. Pass input_ids & attention_mask through core language model to get hidden states.
    2. Align causal shift: shift_hidden = hidden_states[:, :-1, :], shift_labels = labels[:, 1:].
    3. Filter active assistant positions where shift_labels != -100.
    4. Project only the active hidden vectors through lm_head.
    5. Apply final logit softcapping (tanh) if configured.
    6. Compute cross-entropy loss against active labels in float32 for numerical stability.
    """
    core_model = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    # Core language model forward pass
    outputs = core_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    hidden_states = outputs.last_hidden_state  # Shape: (B, T, D)

    # Causal alignment: prediction at position i corresponds to target label at position i+1
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Mask to only active assistant response tokens
    active_mask = (shift_labels != -100)
    if not active_mask.any():
        raise ValueError("No active assistant tokens found in sample labels for loss computation!")

    active_hidden = shift_hidden[active_mask]      # Shape: (N_active, D)
    active_labels = shift_labels[active_mask]      # Shape: (N_active,)

    # Compute lm_head projection strictly on active positions
    active_logits = lm_head(active_hidden)         # Shape: (N_active, Vocab)

    # Apply Gemma-4 logit softcapping if configured
    if softcap is not None:
        active_logits = torch.tanh(active_logits / softcap) * softcap

    loss = F.cross_entropy(active_logits.float(), active_labels, reduction="mean")
    return loss


def run_micro_overfit_test(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    device: str = "cuda:0",
    lr: float = LEARNING_RATE,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    max_steps: int = MAX_STEPS,
    target_loss: float = TARGET_LOSS,
    max_grad_norm: float = MAX_GRAD_NORM,
    max_vram_gb: float = MAX_PEAK_VRAM_GB,
    num_samples: int = NUM_TRAIN_SAMPLES,
) -> Dict[str, Any]:
    """
    Executes the Module 3 micro-overfit loop on a 10-sample subset of micro-turn dialogues.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)

    logger.info("=================================================================")
    logger.info(" Starting Module 3: 10-Sample Micro-Overfit Test ")
    logger.info(f" Model Path: {model_path}")
    logger.info(f" Dataset Path: {dataset_path}")
    logger.info(f" Target Device: {device} | LR: {lr} | Grad Accum: {grad_accum_steps}")
    logger.info(f" Samples: {num_samples} | Target Loss: <= {target_loss} | Grad Norm Cap: < {max_grad_norm}")
    logger.info("=================================================================")

    # 1. Setup Model and Tokenizer
    model, tokenizer = setup_duplex_gemma_model(
        model_path=model_path,
        device=device,
        training=True,
    )

    # 2. Load Dataset and Slice First 10 Conversations
    full_dataset = DuplexDataset(dataset_path, tokenizer)
    selected_samples = [full_dataset[i] for i in range(min(num_samples, len(full_dataset)))]
    collator = DuplexDataCollator(tokenizer)

    logger.info(f"Loaded {len(selected_samples)} micro-turn conversations for micro-overfit test.")
    for idx, s in enumerate(selected_samples):
        logger.info(f"  Sample {idx}: ID={s.get('id', idx)}, SeqLen={len(s['input_ids'])}, ActiveTokens={s['active_loss_count']}")

    # 3. Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"Configuring AdamW optimizer for {len(trainable_params)} tensors ({trainable_count:,} parameters)...")
    optimizer = AdamW(trainable_params, lr=lr)

    # 4. Training Loop Variables
    step_records: List[Dict[str, Any]] = []
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats(target_device)

    step_1_loss: Optional[float] = None
    final_loss: Optional[float] = None
    max_observed_unclipped_norm = 0.0
    max_observed_clipped_norm = 0.0
    has_nan_or_inf_grad = False

    epoch = 0
    while step < max_steps:
        epoch += 1
        logger.info(f"--- Starting Micro-Epoch {epoch} ---")

        for sample_idx, sample in enumerate(selected_samples):
            # Collate single sample micro-batch
            batch = collator([sample])
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            labels = batch["labels"].to(target_device)

            # Compute memory-efficient active masked loss
            loss = compute_masked_loss(model, input_ids, attention_mask, labels)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            accum_loss += loss.item()

            # Optimizer step on accumulation boundary
            if (sample_idx + 1) % grad_accum_steps == 0:
                step += 1
                current_step_loss = accum_loss / grad_accum_steps

                # Record Step 1 Loss
                if step == 1:
                    step_1_loss = round(current_step_loss, 4)
                    logger.info(f"Recorded Step 1 Loss: {step_1_loss:.4f}")

                # Gradient clipping and norm monitoring
                unclipped_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm).item()

                # Calculate post-clip norm to confirm bounded < 2.0
                clipped_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in trainable_params if p.grad is not None]),
                    2,
                ).item()

                # Check for NaNs or Infs
                step_has_nan_inf = False
                for p in trainable_params:
                    if p.grad is not None:
                        if torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item():
                            step_has_nan_inf = True
                            has_nan_or_inf_grad = True
                            break

                if unclipped_norm > max_observed_unclipped_norm:
                    max_observed_unclipped_norm = unclipped_norm
                if clipped_norm > max_observed_clipped_norm:
                    max_observed_clipped_norm = clipped_norm

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                current_peak_vram = torch.cuda.max_memory_allocated(target_device) / (1024 ** 3)
                logger.info(
                    f"Epoch {epoch:2d} | Step {step:2d}/{max_steps} | "
                    f"Loss: {current_step_loss:.4f} | "
                    f"GradNorm(pre/post): {unclipped_norm:.3f}/{clipped_norm:.3f} | "
                    f"Peak VRAM: {current_peak_vram:.3f} GB"
                )

                step_records.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": round(current_step_loss, 4),
                    "unclipped_grad_norm": round(unclipped_norm, 4),
                    "clipped_grad_norm": round(clipped_norm, 4),
                    "peak_vram_gb": round(current_peak_vram, 4),
                    "has_nan_or_inf": step_has_nan_inf,
                })

                final_loss = round(current_step_loss, 4)
                accum_loss = 0.0

                # Early stopping if overfit target achieved after minimum steps
                if step >= 10 and current_step_loss <= target_loss:
                    logger.info(
                        f"Target loss condition met: step={step} >= 10, loss={current_step_loss:.4f} <= {target_loss}. "
                        "Micro-overfit test SUCCESSFUL!"
                    )
                    break

                if step >= max_steps:
                    break

        if step >= 10 and final_loss is not None and final_loss <= target_loss:
            break

    # 5. Peak VRAM measurement
    peak_vram_bytes = torch.cuda.max_memory_allocated(target_device)
    peak_vram_gb = round(peak_vram_bytes / (1024 ** 3), 4)

    # 6. Evaluation Gate Criteria
    step1_recorded_pass = (step_1_loss is not None)
    final_loss_pass = (final_loss is not None) and (final_loss <= target_loss)
    grad_norm_pass = (max_observed_clipped_norm <= max_grad_norm) and (not has_nan_or_inf_grad)
    peak_vram_pass = (peak_vram_gb <= max_vram_gb)
    all_gates_pass = (
        step1_recorded_pass
        and final_loss_pass
        and grad_norm_pass
        and peak_vram_pass
    )

    logger.info("=================================================================")
    logger.info(" Module 3 Evaluation Gate Assessment ")
    logger.info("=================================================================")
    logger.info(f"  [1] Step 1 Loss recorded: {step_1_loss} -> {'PASS' if step1_recorded_pass else 'FAIL'}")
    logger.info(f"  [2] Final Loss <= {target_loss}: {final_loss} (Step {step}) -> {'PASS' if final_loss_pass else 'FAIL'}")
    logger.info(f"  [3] Grad norm strictly bounded < {max_grad_norm}: Max post-clip = {max_observed_clipped_norm:.4f} -> {'PASS' if grad_norm_pass else 'FAIL'}")
    logger.info(f"  [4] Peak VRAM <= {max_vram_gb} GB: {peak_vram_gb:.3f} GB -> {'PASS' if peak_vram_pass else 'FAIL'}")
    logger.info(f"  Overall Status: {'PASSED' if all_gates_pass else 'FAILED'}")
    logger.info("=================================================================")

    report: Dict[str, Any] = {
        "module": 3,
        "module_name": "10-Sample Micro-Overfit Test",
        "status": "PASSED" if all_gates_pass else "FAILED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": device,
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": grad_accum_steps,
            "num_samples": num_samples,
            "max_steps": max_steps,
            "target_loss_threshold": target_loss,
            "max_grad_norm_cap": max_grad_norm,
            "max_peak_vram_threshold_gb": max_vram_gb,
        },
        "metrics": {
            "step_1_loss": step_1_loss,
            "final_loss": final_loss,
            "final_step": step,
            "total_epochs": epoch,
            "peak_vram_gb": peak_vram_gb,
            "peak_vram_bytes": peak_vram_bytes,
            "max_unclipped_grad_norm": round(max_observed_unclipped_norm, 4),
            "max_clipped_grad_norm": round(max_observed_clipped_norm, 4),
            "has_nan_or_inf_grad": has_nan_or_inf_grad,
            "step_records": step_records,
        },
        "gate_criteria": {
            "step_1_loss_recorded": step1_recorded_pass,
            "final_loss_under_threshold": final_loss_pass,
            "grad_norm_bounded_under_cap": grad_norm_pass,
            "peak_vram_under_4_8gb": peak_vram_pass,
            "all_gates_passed": all_gates_pass,
        },
    }

    # Save to both logs/module3_micro_overfit.json and logs/module_3_report.json
    out_file1 = LOGS_DIR / "module3_micro_overfit.json"
    out_file2 = LOGS_DIR / "module_3_report.json"

    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved micro-overfit report to {out_file1}")

    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved module report to {out_file2}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Module 3: 10-Sample Micro-Overfit Test")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to base model")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to jsonl dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Max optimization steps")
    parser.add_argument("--target_loss", type=float, default=TARGET_LOSS, help="Target loss threshold")
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM, help="Max grad norm")
    parser.add_argument("--max_vram_gb", type=float, default=MAX_PEAK_VRAM_GB, help="Max peak VRAM cap")
    parser.add_argument("--num_samples", type=int, default=NUM_TRAIN_SAMPLES, help="Number of samples")
    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        logger.error("CUDA is required for Module 3 micro-overfit test!")
        sys.exit(1)

    report = run_micro_overfit_test(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device=args.device,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        target_loss=args.target_loss,
        max_grad_norm=args.max_grad_norm,
        max_vram_gb=args.max_vram_gb,
        num_samples=args.num_samples,
    )

    if not report["gate_criteria"]["all_gates_passed"]:
        logger.error("Module 3 Evaluation Gate FAILED!")
        sys.exit(1)

    logger.info("Module 3 Evaluation Gate successfully PASSED!")
    sys.exit(0)


if __name__ == "__main__":
    main()
