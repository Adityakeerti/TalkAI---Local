# Dataset Generation & Stabilization Suite

This directory contains the scripts and prompts used to rewrite and stabilize the full-duplex micro-turn dataset (`data/duplexcascade_synthetic/`).

## Contents

- `generate_train_part1.py` & `build_part1_data.py`: Generates Part 1 of the rewritten training dialogues (130 conversations).
- `generate_train_part2.py`: Generates Part 2 of the rewritten training dialogues (130 conversations).
- `generate_train_part3.py`: Generates Part 3 of the rewritten training dialogues (130 conversations).
- `generate_val.py`: Generates unique validation dialogues with 0% train leakage (60 conversations).
- `generate_test.py`: Generates unique test dialogues with 0% train/val leakage (60 conversations).

## Subdirectories

- `prompts/`: Scenario outlines and prompt files used during generation (`train_prompts.txt`, `val_prompts.txt`).
- `intermediate/`: Intermediate batch slices (`train_part1.jsonl`, `train_part2.jsonl`, `train_part3.jsonl`), which are merged into `data/duplexcascade_synthetic/micro_turns_train.jsonl`.
