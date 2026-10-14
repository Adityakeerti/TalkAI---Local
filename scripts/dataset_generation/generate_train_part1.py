#!/usr/bin/env python3
"""
Generate rewritten natural, distinct conversations for Part 1 of the training dataset.
Replaces the first 130 duplicate conversations (indices 0 to 129 among duplicates)
in micro_turns_train.jsonl with unique, diverse, high-quality full-duplex conversations.
"""

import json
import os
import sys

# Ensure local directory is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_part1_data import CONVERSATIONS

SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. Output exactly 'listen' "
    "when you should yield and keep listening; otherwise output a brief, natural speech fragment "
    "that can be spoken now. Do not repeat prior fragments. Continue an unfinished answer across turns "
    "when appropriate. Yield immediately for a clear interruption or urgent safety risk. "
    "Do not treat every pause as the end of the user's thought."
)

TRAIN_PATH = os.path.join(SCRIPT_DIR, "micro_turns_train.jsonl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "train_part1.jsonl")

def main():
    # 1. Load train dataset and identify duplicates
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]

    seen = set()
    duplicates = []
    for i, rec in enumerate(train_records):
        key = json.dumps(rec["micro_turns"])
        if key in seen:
            duplicates.append((i, rec))
        else:
            seen.add(key)

    print(f"Total train records: {len(train_records)}")
    print(f"Total duplicates found: {len(duplicates)}")

    part1_duplicates = duplicates[:130]
    print(f"Processing Part 1 duplicates: {len(part1_duplicates)} conversations")

    # 2. Build rewritten records
    rewritten_records = []
    for idx, (orig_idx, orig_rec) in enumerate(part1_duplicates):
        cid = orig_rec["id"]
        scenario = orig_rec["scenario"]
        topic = orig_rec["topic"]

        if cid not in CONVERSATIONS:
            raise KeyError(f"Conversation ID {cid} missing from CONVERSATIONS dictionary!")

        micro_turns = CONVERSATIONS[cid]

        # Build messages list matching the duplex starter format
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in micro_turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        record = {
            "id": cid,
            "scenario": scenario,
            "topic": topic,
            "micro_turns": micro_turns,
            "messages": messages,
        }
        rewritten_records.append(record)

    # 3. Write output to train_part1.jsonl
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in rewritten_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully saved {len(rewritten_records)} rewritten conversations to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
