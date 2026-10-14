DuplexCascade-style synthetic micro-turn dataset (Stabilized & Deduplicated)

Files:
- micro_turns_train.jsonl (480 conversations)
- micro_turns_val.jsonl (60 conversations)
- micro_turns_test.jsonl (60 conversations)

Schema:
Matches standard micro-turn JSONL:
- id: unique conversation identifier
- scenario: conversation scenario category (e.g. urgent_chemical_safety, hesitation_wait, backchannel_story, etc.)
- topic: specific topic variant
- micro_turns: list of {"user": str, "assistant": str} for each 600 ms window
- messages: flattened OpenAI-style chat messages format

Action semantics:
- assistant="listen": Yield and remain silent this micro-turn.
- assistant="<words>": Concise speech fragment (3-7 words) spoken by TTS during this micro-turn.
- user="no voice": User silence / no speech detected by streaming ASR in this 600 ms window.

Stabilization & Audit Summary:
- 100% Unique Dialogue Sequences: All 600 conversations are completely unique (all 503 duplicate/template instances were rewritten).
- 0% Split Leakage: Completely disjoint across train, val, and test splits (no dialogue overlap or leaked scenarios across splits).
- Train Split: 480 conversations, 4,318 micro-turns | Listen: 2,319 (53.7%) | Speak: 1,999 (46.3%) | Scenarios: 21
- Val Split: 60 conversations, 534 micro-turns | Listen: 294 (55.1%) | Speak: 240 (44.9%) | Scenarios: 20
- Test Split: 60 conversations, 562 micro-turns | Listen: 305 (54.3%) | Speak: 257 (45.7%) | Scenarios: 20
- End Turn Constraint: 100% of conversations terminate with {"user": "no voice", "assistant": "listen"}.

Generation Scripts:
Generation, prompts, and intermediate generation parts are archived in scripts/dataset_generation/.
Original raw starter dataset is preserved in data/archive/duplexcascade_synthetic_dataset.zip.
