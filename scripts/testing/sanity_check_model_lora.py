#!/usr/bin/env python3
"""
scripts/sanity_check_model_lora.py
Module 2: Model Architecture Loading, Layer Freezing & VRAM Sanity

Responsibilities:
1. Load Google Gemma-4-E2B-it on CPU in bfloat16 to avoid initial CUDA OOM.
2. Pin auxiliary 4.48 GB Per-Layer Embedding (PLE: embed_tokens_per_layer) to CPU.
3. Wrap get_per_layer_inputs to perform CPU embedding lookup and stream tensors to CUDA.
4. Move active language model layers (embed_tokens, layers 0..34, norm, rotary_emb,
   per_layer_model_projection, per_layer_projection_norm, lm_head) to cuda:0.
5. Enable gradient checkpointing and freeze layers 0..30 (and all non-LoRA parameters).
6. Attach PEFT LoRA adapter to layers 31..34 targeting attention & MLP projections.
7. Execute a dummy forward + backward pass to verify gradient propagation and measure peak VRAM.
8. Validate Evaluation Gate criteria:
   - Trainable parameters <= 4.0M
   - Peak VRAM <= 4.8 GB
   - Gradients are non-zero with 0 NaNs and 0 Infs.
9. Export reusable setup_duplex_gemma_model() for Modules 3 and 4.
10. Save structured reports to logs/module2_vram_sanity.json and logs/module_2_report.json.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Set PyTorch CUDA allocator BEFORE importing torch to ensure expandable segments are enabled
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-E2B-it"
LOGS_DIR = PROJECT_ROOT / "logs"

# Gate Thresholds
MAX_TRAINABLE_PARAMS = 4_000_000      # 4.0 Million parameters max
MAX_PEAK_VRAM_GB = 4.8                 # 4.8 GB peak VRAM cap (out of 6.0 GB)
DUMMY_SEQ_LEN = 32                     # Sanity check sequence length (32 tokens)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sanity_check_model_lora")


def wrap_per_layer_embeddings_for_cpu(model: torch.nn.Module, target_device: torch.device) -> None:
    """
    Pins the 4.48 GB embed_tokens_per_layer table to CPU and wraps
    model.model.language_model.get_per_layer_inputs so that token lookups
    occur on host RAM and the resulting small PLE tensor is moved to CUDA.
    """
    lang_model = model.model.language_model

    # Pin PLE table to CPU and freeze
    if hasattr(lang_model, "embed_tokens_per_layer") and lang_model.embed_tokens_per_layer is not None:
        lang_model.embed_tokens_per_layer.to("cpu")
        lang_model.embed_tokens_per_layer.requires_grad_(False)
        logger.info("Successfully pinned embed_tokens_per_layer (4.48 GB) to CPU.")

    orig_get_per_layer_inputs = lang_model.get_per_layer_inputs

    def custom_get_per_layer_inputs(
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CPU-offloaded PLE lookup:
        Input IDs on CUDA -> CPU lookup -> PLE tensor streamed to CUDA in bfloat16.
        """
        input_ids_cpu = input_ids.to("cpu") if input_ids is not None else None
        inputs_embeds_cpu = inputs_embeds.to("cpu") if (inputs_embeds is not None and input_ids is None) else None

        # Perform embedding lookup on CPU
        ple_cpu = orig_get_per_layer_inputs(input_ids_cpu, inputs_embeds_cpu)

        # Transfer only the small batch PLE tensor to CUDA
        return ple_cpu.to(device=target_device, dtype=torch.bfloat16)

    lang_model.get_per_layer_inputs = custom_get_per_layer_inputs
    logger.info("Wrapped language_model.get_per_layer_inputs with CPU-to-CUDA transfer.")


def move_active_layers_to_device(model: torch.nn.Module, target_device: torch.device) -> None:
    """
    Moves active language model submodules to target CUDA device while keeping
    vision/audio encoders and the large PLE table on CPU.
    """
    lang_model = model.model.language_model

    logger.info(f"Moving active language model submodules to {target_device}...")
    lang_model.embed_tokens.to(target_device)
    lang_model.layers.to(target_device)
    lang_model.norm.to(target_device)
    lang_model.rotary_emb.to(target_device)

    if hasattr(lang_model, "per_layer_model_projection") and lang_model.per_layer_model_projection is not None:
        lang_model.per_layer_model_projection.to(target_device)
    if hasattr(lang_model, "per_layer_projection_norm") and lang_model.per_layer_projection_norm is not None:
        lang_model.per_layer_projection_norm.to(target_device)

    # Move lm_head to device (tied with embed_tokens, ensures Linear layer is properly placed)
    model.lm_head.to(target_device)

    vram_after_placement = torch.cuda.memory_allocated(target_device) / (1024 ** 3)
    logger.info(f"CUDA memory allocated after active layer placement: {vram_after_placement:.3f} GB")


def setup_duplex_gemma_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str = "cuda:0",
    training: bool = True,
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """
    Reusable model and tokenizer loader for Full-Duplex Micro-Turn Dialogue.

    Features:
    - Loads Gemma-4-E2B-it on CPU in bfloat16.
    - Pins auxiliary 4.48 GB PLE embedding to CPU RAM.
    - Moves active decoder layers (0..34), embed_tokens, norm, and lm_head to CUDA.
    - Freezes layers 0..30 and all non-LoRA parameters (requires_grad = False).
    - Attaches LoRA PEFT adapter to layers 31..34:
      - r = 16, lora_alpha = 32, lora_dropout = 0.05
      - target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      - layers_to_transform = [31, 32, 33, 34]
    - Enables PyTorch gradient checkpointing with input gradients enabled.

    Returns:
        (peft_model, tokenizer)
    """
    model_path_str = str(model_path)
    target_device = torch.device(device)

    if not torch.cuda.is_available() and "cuda" in device:
        raise RuntimeError("CUDA is not available but a CUDA device was requested.")

    if "cuda" in device and torch.cuda.is_available():
        # Pre-initialize CUDA context and cuBLAS workspace before loading model weights
        torch.cuda.init()
        _warmup = torch.zeros(1, 1, dtype=torch.bfloat16, device=target_device) @ torch.zeros(1, 1, dtype=torch.bfloat16, device=target_device)
        del _warmup
        logger.info(f"Initialized CUDA context and cuBLAS workspace on {target_device}.")

    logger.info(f"Loading tokenizer from {model_path_str}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)

    logger.info(f"Loading base Gemma-4-E2B-it weights on CPU (dtype=bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        dtype=torch.bfloat16,
        device_map=None,
    )

    # 1. Pin PLE table to CPU and patch lookup
    wrap_per_layer_embeddings_for_cpu(base_model, target_device)

    # 2. Move active language model layers to CUDA
    move_active_layers_to_device(base_model, target_device)

    # 3. Enable gradient checkpointing on base model with offload to host memory
    if training:
        base_model.gradient_checkpointing_enable(offload=True)
        logger.info("Gradient checkpointing enabled on base model (offload=True).")

    # 4. Freeze all base parameters explicitly
    logger.info("Freezing base model parameters (layers 0..30, embeddings, norms)...")
    for param in base_model.parameters():
        param.requires_grad = False

    # 5. Attach PEFT LoRA adapter
    logger.info("Applying LoRA adapter to decoder layers 31..34...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=[31, 32, 33, 34],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, peft_config)

    if training:
        model.train()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            logger.info("Enabled input embeddings gradient propagation for PEFT checkpointing.")
    else:
        model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    logger.info(f"Model ready: trainable params = {trainable_params:,} / {all_params:,} ({pct:.4f}%)")

    return model, tokenizer


def execute_sanity_check(
    model: PeftModel,
    device: str = "cuda:0",
    seq_len: int = DUMMY_SEQ_LEN,
) -> Dict[str, Any]:
    """
    Executes a single dummy forward and backward pass, measuring VRAM and gradients.
    """
    target_device = torch.device(device)
    logger.info(f"Executing dummy forward + backward pass (batch_size=1, seq_len={seq_len})...")

    # Reset peak memory tracker
    torch.cuda.reset_peak_memory_stats(target_device)

    # Generate dummy batch
    dummy_input_ids = torch.randint(100, 10000, (1, seq_len), dtype=torch.long, device=target_device)
    dummy_labels = dummy_input_ids.clone()
    # Mask out prompt tokens with -100, leaving assistant response active
    dummy_labels[:, : seq_len // 2] = -100

    # Forward pass
    outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
    loss = outputs.loss
    loss_val = float(loss.item())
    logger.info(f"Forward pass completed. Dummy loss: {loss_val:.4f}")

    # Backward pass
    loss.backward()
    logger.info("Backward pass completed successfully.")

    # Measure peak VRAM
    peak_vram_bytes = torch.cuda.max_memory_allocated(target_device)
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    logger.info(f"Peak VRAM allocated: {peak_vram_gb:.3f} GB (Cap: {MAX_PEAK_VRAM_GB} GB)")

    # Inspect gradients on trainable parameters
    trainable_count = 0
    grad_non_zero_count = 0
    has_nan_or_inf = False
    param_grad_details = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            if param.grad is not None:
                is_nan = torch.isnan(param.grad).any().item()
                is_inf = torch.isinf(param.grad).any().item()
                grad_norm = float(param.grad.norm().item())
                if grad_norm > 0.0:
                    grad_non_zero_count += 1
                if is_nan or is_inf:
                    has_nan_or_inf = True
                    logger.error(f"Invalid grad in {name}: nan={is_nan}, inf={is_inf}")
                param_grad_details.append({
                    "name": name,
                    "shape": list(param.shape),
                    "grad_norm": grad_norm,
                    "is_nan": is_nan,
                    "is_inf": is_inf,
                })
            else:
                logger.warning(f"Trainable parameter {name} has no gradient!")

    logger.info(f"Checked gradients across {len(param_grad_details)} trainable LoRA tensors.")
    logger.info(f"Non-zero grad tensors: {grad_non_zero_count}/{len(param_grad_details)}")
    logger.info(f"Any NaN / Inf detected: {has_nan_or_inf}")

    return {
        "trainable_parameters": trainable_count,
        "peak_vram_gb": round(peak_vram_gb, 4),
        "peak_vram_bytes": peak_vram_bytes,
        "dummy_loss": round(loss_val, 4),
        "dummy_seq_len": seq_len,
        "trainable_tensors_count": len(param_grad_details),
        "non_zero_grad_tensors": grad_non_zero_count,
        "has_nan_or_inf": has_nan_or_inf,
        "gradients_valid": (not has_nan_or_inf) and (grad_non_zero_count > 0),
        "param_grad_details": param_grad_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Module 2: Model Architecture & LoRA VRAM Sanity Check")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to base model")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g. cuda:0)")
    parser.add_argument("--seq_len", type=int, default=DUMMY_SEQ_LEN, help="Dummy sequence length")
    args = parser.parse_args()

    logger.info("=== Starting Module 2: Gemma-4-E2B Model Architecture & LoRA VRAM Sanity Check ===")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    if "cuda" in device and not torch.cuda.is_available():
        logger.error("CUDA is required for Module 2 VRAM verification!")
        sys.exit(1)

    # 1. Setup model and tokenizer
    model, tokenizer = setup_duplex_gemma_model(
        model_path=args.model_path,
        device=device,
        training=True,
    )

    # 2. Run sanity check pass
    metrics = execute_sanity_check(model, device=device, seq_len=args.seq_len)

    # 3. Evaluate Gate Criteria
    trainable_params_pass = metrics["trainable_parameters"] <= MAX_TRAINABLE_PARAMS
    peak_vram_pass = metrics["peak_vram_gb"] <= MAX_PEAK_VRAM_GB
    gradients_pass = metrics["gradients_valid"] and (not metrics["has_nan_or_inf"])
    all_gates_pass = trainable_params_pass and peak_vram_pass and gradients_pass

    logger.info("=== Evaluation Gate Results ===")
    logger.info(f"  [1] Trainable params <= {MAX_TRAINABLE_PARAMS:,}: {metrics['trainable_parameters']:,} -> {'PASS' if trainable_params_pass else 'FAIL'}")
    logger.info(f"  [2] Peak VRAM <= {MAX_PEAK_VRAM_GB} GB: {metrics['peak_vram_gb']:.3f} GB -> {'PASS' if peak_vram_pass else 'FAIL'}")
    logger.info(f"  [3] Gradients valid (non-zero, 0 NaN, 0 Inf): {metrics['gradients_valid']} -> {'PASS' if gradients_pass else 'FAIL'}")
    logger.info(f"  Overall Status: {'PASSED' if all_gates_pass else 'FAILED'}")

    report: Dict[str, Any] = {
        "module": 2,
        "module_name": "Model Architecture Loading, Layer Freezing & VRAM Sanity",
        "status": "PASSED" if all_gates_pass else "FAILED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(DEFAULT_MODEL_PATH),
        "device": device,
        "metrics": {
            "trainable_parameters": metrics["trainable_parameters"],
            "max_trainable_param_threshold": MAX_TRAINABLE_PARAMS,
            "peak_vram_gb": metrics["peak_vram_gb"],
            "max_peak_vram_threshold_gb": MAX_PEAK_VRAM_GB,
            "dummy_loss": metrics["dummy_loss"],
            "dummy_seq_len": metrics["dummy_seq_len"],
            "trainable_tensors_count": metrics["trainable_tensors_count"],
            "non_zero_grad_tensors": metrics["non_zero_grad_tensors"],
            "has_nan_or_inf": metrics["has_nan_or_inf"],
        },
        "gate_criteria": {
            "trainable_params_under_4m": trainable_params_pass,
            "peak_vram_under_4_8gb": peak_vram_pass,
            "gradients_valid_nonzero_no_nan_inf": gradients_pass,
            "all_gates_passed": all_gates_pass,
        },
    }

    # Save to both logs/module2_vram_sanity.json and logs/module_2_report.json
    out_file1 = LOGS_DIR / "module2_vram_sanity.json"
    out_file2 = LOGS_DIR / "module_2_report.json"

    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved sanity report to {out_file1}")

    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved module report to {out_file2}")

    if not all_gates_pass:
        logger.error("Module 2 Evaluation Gate FAILED!")
        sys.exit(1)

    logger.info("Module 2 Evaluation Gate successfully PASSED!")
    sys.exit(0)


if __name__ == "__main__":
    main()
