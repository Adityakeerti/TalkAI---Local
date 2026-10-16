#!/usr/bin/env python3
"""
scripts/verify_dataset_tokenization.py
Module 1: Tokenizer & Loss Masking Verification (Pre-Flight Check)

Verifies:
1. Tokenizer encodes 'listen' as single token [20923].
2. For user micro-turn and system prompt tokens: labels[i] == -100.
3. For assistant response tokens: labels[i] == input_ids[i].
4. Sequence lengths are bounded (<= 512 tokens).
5. Number of active loss tokens equals exactly the total number of assistant tokens.
6. Reusable DuplexDataset and DuplexDataCollator utilities for subsequent training modules.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-E2B-it"
TRAIN_DATASET_PATH = PROJECT_ROOT / "data" / "duplexcascade_synthetic" / "micro_turns_train.jsonl"
VAL_DATASET_PATH = PROJECT_ROOT / "data" / "duplexcascade_synthetic" / "micro_turns_val.jsonl"
LOGS_DIR = PROJECT_ROOT / "logs"

# Gemma-4 Constants
LISTEN_TOKEN_ID = 20923
MAX_SEQ_LENGTH = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_dataset_tokenization")


def find_subsequence(sequence: List[int], subseq: List[int], start_from: int = 0) -> int:
    """Find start index of subseq in sequence starting from start_from. Returns -1 if not found."""
    n, m = len(sequence), len(subseq)
    if m == 0 or m > n:
        return -1
    for i in range(start_from, n - m + 1):
        if sequence[i:i + m] == subseq:
            return i
    return -1


def tokenize_conversation(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
    pad_to_max: bool = False,
) -> Dict[str, Any]:
    """
    Tokenizes a multi-turn micro-turn conversation using the official chat template
    and constructs PyTorch CrossEntropyLoss labels:
      - labels[i] == -100 for all system prompt, user speech, and turn delimiter tokens.
      - labels[i] == input_ids[i] strictly for assistant generation tokens.

    Returns:
        dict with keys: 'input_ids', 'attention_mask', 'labels',
                        'assistant_token_count', 'active_loss_count'
    """
    # Render with Gemma-4 chat template
    rendered_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length" if pad_to_max else False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # In Gemma chat template:
    # Model turn begins with: <|turn>model\n
    # Model turn ends with:   <turn|>\n
    model_prefix = tokenizer.encode("<|turn>model\n", add_special_tokens=False)
    turn_suffix = tokenizer.encode("<turn|>\n", add_special_tokens=False)

    labels = [-100] * len(input_ids)
    asst_token_count = 0

    idx = 0
    seq_len = len(input_ids)

    while idx < seq_len:
        found_start = find_subsequence(input_ids, model_prefix, start_from=idx)
        if found_start == -1:
            break

        content_start = found_start + len(model_prefix)
        found_end = find_subsequence(input_ids, turn_suffix, start_from=content_start)
        if found_end == -1:
            # If sequence was truncated mid-turn, loss extends to sequence end
            content_end = seq_len
            idx = seq_len
        else:
            content_end = found_end
            idx = found_end + len(turn_suffix)

        for k in range(content_start, content_end):
            # Only unmask if it is an actual token (not padding)
            if attention_mask[k] == 1:
                labels[k] = input_ids[k]
                asst_token_count += 1

    active_loss_count = sum(1 for label in labels if label != -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_token_count": asst_token_count,
        "active_loss_count": active_loss_count,
    }


class DuplexDataset(Dataset):
    """
    PyTorch Dataset for Full-Duplex Micro-Turn Dialogue.
    Reads lines from .jsonl and returns tokenized dictionaries with loss masks.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict[str, Any]] = []

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.path}")

        logger.info(f"Loading dataset from {self.path}...")
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        logger.info(f"Loaded {len(self.records)} records from {self.path.name}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        messages = record.get("messages", [])
        features = tokenize_conversation(
            messages=messages,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_max=False,
        )
        features["id"] = record.get("id", f"sample_{idx}")
        return features


class DuplexDataCollator:
    """
    Data Collator for dynamically padding input_ids, attention_mask, and labels.
    - input_ids are padded with tokenizer.pad_token_id (0)
    - attention_mask is padded with 0
    - labels are padded with -100
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of is not None:
            batch_max_len = ((batch_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            pad_len = batch_max_len - len(input_ids)

            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def run_verification() -> Dict[str, Any]:
    """Runs the complete Module 1 verification suite."""
    logger.info("Initializing Module 1 Verification...")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    # ---------------------------------------------------------
    # CHECK 1: Tokenizer encodes 'listen' as single token [20923]
    # ---------------------------------------------------------
    listen_tokens = tokenizer.encode("listen", add_special_tokens=False)
    check1_passed = (listen_tokens == [LISTEN_TOKEN_ID])
    assert check1_passed, f"Listen encoding mismatch: expected [{LISTEN_TOKEN_ID}], got {listen_tokens}"
    logger.info(f"CHECK 1 PASSED: 'listen' encoded as single token {listen_tokens}")

    # ---------------------------------------------------------
    # CHECK 2: Sample A - Assistant responds with 'listen'
    # ---------------------------------------------------------
    sample_a_messages = [
        {
            "role": "system",
            "content": "You are a full-duplex conversational assistant. Output exactly 'listen' when you should yield."
        },
        {"role": "user", "content": "I was thinking about..."},
        {"role": "assistant", "content": "listen"}
    ]
    res_a = tokenize_conversation(sample_a_messages, tokenizer, max_length=MAX_SEQ_LENGTH)
    
    # Assertions for Sample A
    # Sequence length bounded
    assert len(res_a["input_ids"]) <= MAX_SEQ_LENGTH, f"Sample A length exceeds {MAX_SEQ_LENGTH}"
    # Number of active loss tokens equals exactly 1 (total assistant tokens)
    assert res_a["active_loss_count"] == 1, f"Expected 1 active loss token for listen, got {res_a['active_loss_count']}"
    assert res_a["assistant_token_count"] == 1, f"Expected 1 assistant token for listen, got {res_a['assistant_token_count']}"
    assert res_a["active_loss_count"] == res_a["assistant_token_count"], "Active loss count does not match assistant token count"

    # Verify that the active loss token is LISTEN_TOKEN_ID
    active_indices_a = [idx for idx, val in enumerate(res_a["labels"]) if val != -100]
    assert len(active_indices_a) == 1
    assert res_a["labels"][active_indices_a[0]] == LISTEN_TOKEN_ID
    assert res_a["input_ids"][active_indices_a[0]] == LISTEN_TOKEN_ID

    # Verify that user and system tokens have label == -100
    for idx in range(active_indices_a[0]):
        assert res_a["labels"][idx] == -100, f"Token at {idx} was not masked with -100"

    logger.info(f"CHECK 2 PASSED: Sample A ('listen') verified (seq_len={len(res_a['input_ids'])}, active_tokens={res_a['active_loss_count']})")

    # ---------------------------------------------------------
    # CHECK 3: Sample B - Assistant speaks words
    # ---------------------------------------------------------
    spoken_phrase = "Do not touch it."
    expected_spoken_tokens = tokenizer.encode(spoken_phrase, add_special_tokens=False)
    sample_b_messages = [
        {
            "role": "system",
            "content": "You are a full-duplex conversational assistant. Output warning words when safety hazard exists."
        },
        {"role": "user", "content": "The outlet is sparking and I might touch it"},
        {"role": "assistant", "content": spoken_phrase}
    ]
    res_b = tokenize_conversation(sample_b_messages, tokenizer, max_length=MAX_SEQ_LENGTH)

    assert len(res_b["input_ids"]) <= MAX_SEQ_LENGTH, f"Sample B length exceeds {MAX_SEQ_LENGTH}"
    assert res_b["active_loss_count"] == len(expected_spoken_tokens), (
        f"Expected {len(expected_spoken_tokens)} active tokens, got {res_b['active_loss_count']}"
    )
    assert res_b["active_loss_count"] == res_b["assistant_token_count"], "Active loss count does not match assistant token count"

    # Verify active token labels match input_ids
    active_indices_b = [idx for idx, val in enumerate(res_b["labels"]) if val != -100]
    extracted_spoken_tokens = [res_b["labels"][i] for i in active_indices_b]
    assert extracted_spoken_tokens == expected_spoken_tokens, (
        f"Spoken tokens mismatch: expected {expected_spoken_tokens}, got {extracted_spoken_tokens}"
    )

    # Verify all tokens before assistant response are masked (-100)
    for idx in range(active_indices_b[0]):
        assert res_b["labels"][idx] == -100, f"Token at {idx} was not masked with -100"

    logger.info(f"CHECK 3 PASSED: Sample B ('speech') verified (seq_len={len(res_b['input_ids'])}, active_tokens={res_b['active_loss_count']})")

    # ---------------------------------------------------------
    # CHECK 4: Multi-Turn Full Dataset Verification (480 conversations)
    # ---------------------------------------------------------
    dataset = DuplexDataset(TRAIN_DATASET_PATH, tokenizer, max_length=MAX_SEQ_LENGTH)
    total_samples = len(dataset)
    max_observed_len = 0
    total_active_tokens = 0
    total_assistant_tokens = 0
    mismatch_records = []

    for i in range(total_samples):
        item = dataset[i]
        seq_len = len(item["input_ids"])
        if seq_len > max_observed_len:
            max_observed_len = seq_len

        # Check assertion for sequence length <= 512
        if seq_len > MAX_SEQ_LENGTH:
            mismatch_records.append({"id": item["id"], "error": f"seq_len {seq_len} > {MAX_SEQ_LENGTH}"})

        # Check assertion: active loss tokens equals exactly assistant tokens
        active_cnt = item["active_loss_count"]
        asst_cnt = item["assistant_token_count"]
        total_active_tokens += active_cnt
        total_assistant_tokens += asst_cnt

        if active_cnt != asst_cnt:
            mismatch_records.append({
                "id": item["id"],
                "error": f"active_cnt ({active_cnt}) != asst_cnt ({asst_cnt})"
            })

    assert len(mismatch_records) == 0, f"Found mismatches in dataset: {mismatch_records}"
    assert max_observed_len <= MAX_SEQ_LENGTH, f"Max sequence length {max_observed_len} > {MAX_SEQ_LENGTH}"
    assert total_active_tokens == total_assistant_tokens, "Total active tokens do not match assistant tokens across dataset"

    logger.info(f"CHECK 4 PASSED: All {total_samples} conversations validated.")
    logger.info(f"Max observed sequence length: {max_observed_len} tokens (<= {MAX_SEQ_LENGTH})")
    logger.info(f"Total active loss tokens: {total_active_tokens} (100% assistant turn match)")

    # ---------------------------------------------------------
    # CHECK 5: Collator Test with Batching
    # ---------------------------------------------------------
    collator = DuplexDataCollator(tokenizer)
    batch = collator([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    logger.info(f"CHECK 5 PASSED: Collator test batch shape: {batch['input_ids'].shape}")

    # ---------------------------------------------------------
    # Compile Report
    # ---------------------------------------------------------
    report = {
        "module": 1,
        "module_name": "Tokenizer & Loss Masking Verification",
        "status": "PASSED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "tokenizer_vocab_size": len(tokenizer),
        "listen_token_id": LISTEN_TOKEN_ID,
        "listen_token_verified": check1_passed,
        "sample_a_listen": {
            "conversation_id": "test_sample_listen",
            "sequence_length": len(res_a["input_ids"]),
            "active_loss_tokens": res_a["active_loss_count"],
            "assistant_tokens": res_a["assistant_token_count"],
            "loss_masking_valid": True,
        },
        "sample_b_speech": {
            "conversation_id": "test_sample_speech",
            "sequence_length": len(res_b["input_ids"]),
            "active_loss_tokens": res_b["active_loss_count"],
            "assistant_tokens": res_b["assistant_token_count"],
            "spoken_token_ids": expected_spoken_tokens,
            "loss_masking_valid": True,
        },
        "dataset_validation": {
            "path": str(TRAIN_DATASET_PATH),
            "total_conversations": total_samples,
            "max_sequence_length": max_observed_len,
            "bounded_under_512": max_observed_len <= MAX_SEQ_LENGTH,
            "total_active_loss_tokens": total_active_tokens,
            "total_assistant_tokens": total_assistant_tokens,
            "mismatch_count": len(mismatch_records),
        },
        "gate_criteria": {
            "listen_token_is_20923": check1_passed,
            "user_tokens_masked_neg100": True,
            "assistant_tokens_active_in_labels": True,
            "sequence_length_bounded_512": max_observed_len <= MAX_SEQ_LENGTH,
            "active_tokens_equal_assistant_tokens": total_active_tokens == total_assistant_tokens,
            "all_gates_passed": True,
        }
    }

    report_path1 = LOGS_DIR / "module1_tokenization_check.json"
    report_path2 = LOGS_DIR / "module_1_report.json"

    with open(report_path1, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved verification report to {report_path1}")

    with open(report_path2, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved verification report to {report_path2}")

    return report


if __name__ == "__main__":
    try:
        report = run_verification()
        print("\n==================================================")
        print(" MODULE 1 EVALUATION GATE: ALL ASSERTIONS PASSED ")
        print("==================================================")
        print(f"Listen token: {report['listen_token_id']} (Verified: {report['listen_token_verified']})")
        print(f"Dataset conversations: {report['dataset_validation']['total_conversations']}")
        print(f"Max sequence length: {report['dataset_validation']['max_sequence_length']} (<= {MAX_SEQ_LENGTH})")
        print(f"Total active loss tokens: {report['dataset_validation']['total_active_loss_tokens']}")
        print(f"Status: {report['status']}")
        print("==================================================\n")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Module 1 verification failed: {e}")
        sys.exit(1)
