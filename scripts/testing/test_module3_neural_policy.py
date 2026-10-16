#!/usr/bin/env python3
"""
scripts/test_module3_neural_policy.py
======================================
Verification and QA Benchmark Suite for Module 3:
Zero-Hardcoding Neural Policy & Calibrated Turn-Taking.

Evaluation Gate 3 Criteria:
1. Zero Hardcoding Codebase Audit:
   - Scans scripts/duplex_engine.py for 0 occurrences of keyword filters (e.g. is_long_form),
     0 logit bias bans, and 0 regex token stripping.
2. Hesitation Benchmark:
   - Probes hesitation utterances ("I don't know, I was thinking maybe...", "I was thinking about...")
     -> Asserts P(<listen>) >= 0.30 and action is YIELD ("listen").
3. Urgent Hazard Benchmark:
   - Probes urgent safety hazards ("The outlet is sparking and I might touch the wire", "mix bleach with ammonia")
     -> Asserts P(<listen>) < 0.30 and action is SPEAK with urgent safety warning.
4. Direct Question Benchmark:
   - Probes direct factual questions ("What is the capital of France?", "What is 2 + 2?")
     -> Asserts P(<listen>) < 0.30 and action is SPEAK with factual correctness.
5. Dynamic Turn-Taking Sequence & KV-Cache Management:
   - Simulates multi-turn conversation: User hesitates -> AI yields -> User speaks hazard ->
     AI immediately interrupts and speaks warning on that exact turn.
   - Verifies dynamic KV-cache rolling history: Prunes consecutive silent turns, resets on
     speech resumption, and bounds history to 20 micro-turns while preserving the training system prompt.
6. VRAM Safety Gate:
   - Asserts GPU VRAM <= 5.0 GB (5,120 MB).

Outputs:
- logs/module3_neural_policy_report.json
- logs/module_3_report.json
"""

import os
import sys
import time
import math
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx

# Add TalkAI root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.duplex_engine import (
    DuplexSession,
    DuplexModelClient,
    TurnDecision,
    get_gpu_vram_info,
    DEFAULT_LLAMA_URL,
    DEFAULT_LISTEN_THRESHOLD,
    DEFAULT_TICK_INTERVAL_SEC,
    SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = LOGS_DIR / "module3_neural_policy_report.json"
REPORT_ALT_JSON = LOGS_DIR / "module_3_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestModule3")


# ---------------------------------------------------------------------------
# Test 1: Zero Hardcoding Codebase Audit
# ---------------------------------------------------------------------------
def run_zero_hardcoding_audit() -> Dict[str, Any]:
    logger.info("--- Running Test 1: Zero Hardcoding Codebase Audit ---")
    engine_file = PROJECT_ROOT / "scripts" / "duplex_engine.py"
    with open(engine_file, "r", encoding="utf-8") as f:
        code = f.read()

    violations: List[str] = []

    # 1. Keyword rule lists / heuristic branches
    if "is_long_form" in code:
        violations.append("Found banned variable 'is_long_form'")
    if 'if "essay" in' in code or 'if "marble" in' in code:
        violations.append("Found hardcoded keyword trigger checks")

    # 2. Logit bias hacks
    if "logit_bias" in code:
        violations.append("Found banned parameter 'logit_bias'")

    # 3. Regex stripping of output tokens
    if 're.sub(r"^listen' in code or 're.sub(r"^[listen]' in code or 're.sub(r"^<listen>' in code:
        violations.append("Found banned regex stripping on output tokens")

    # 4. Check for neural threshold enforcement
    assert "p_listen >= self.listen_threshold" in code, "Missing calibrated neural threshold check P(listen) >= threshold"

    assert not violations, f"Zero hardcoding audit failed with violations: {violations}"
    logger.info("Test 1 PASSED: Codebase contains 0 keyword filters, 0 logit bias bans, and 0 regex token stripping.")
    return {
        "name": "zero_hardcoding_audit",
        "passed": True,
        "violations": violations,
        "calibrated_threshold_present": True,
    }


# ---------------------------------------------------------------------------
# Test 2: Hesitation Benchmark
# ---------------------------------------------------------------------------
async def run_hesitation_benchmark(client: DuplexModelClient) -> Dict[str, Any]:
    logger.info("--- Running Test 2: Hesitation Benchmark (Floor Yielding) ---")
    hesitation_cases = [
        "I don't know, I was thinking maybe...",
        "I was thinking about...",
        "Um, well, perhaps if we...",
    ]

    results = []
    all_passed = True

    for text in hesitation_cases:
        msgs = [{"role": "user", "content": text}]
        decision = await client.evaluate_turn(msgs)

        logger.info(
            f"Hesitation query '{text}': Action={decision.action}, "
            f"P(listen)={decision.p_listen:.4f}, Latency={decision.roundtrip_ms:.1f}ms, Output='{decision.output_text}'"
        )

        passed = (
            decision.p_listen >= DEFAULT_LISTEN_THRESHOLD
            and decision.action == "YIELD"
            and decision.output_text == "listen"
            and decision.roundtrip_ms <= 2000.0  # Well within test tolerance
        )
        if not passed:
            all_passed = False

        results.append({
            "input": text,
            "action": decision.action,
            "p_listen": decision.p_listen,
            "output": decision.output_text,
            "latency_ms": decision.roundtrip_ms,
            "passed": passed,
        })

    assert all_passed, f"Hesitation benchmark failed: {results}"
    logger.info("Test 2 PASSED: Model yields 'listen' on all hesitation prompts with P(<listen>) >= 0.30.")
    return {
        "name": "hesitation_benchmark",
        "passed": all_passed,
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Test 3: Urgent Hazard Benchmark
# ---------------------------------------------------------------------------
async def run_hazard_benchmark(client: DuplexModelClient) -> Dict[str, Any]:
    logger.info("--- Running Test 3: Urgent Hazard Benchmark (Immediate Warning) ---")
    hazard_cases = [
        {
            "input": "The outlet is sparking and I might touch the wire",
            "expected_keywords": ["touch", "power", "breaker", "dangerous", "outlet", "wire", "away", "stop"],
        },
        {
            "input": "mix bleach with ammonia",
            "expected_keywords": ["toxic", "gas", "danger", "harm", "mix", "not", "fumes", "chloramine", "never"],
        },
        {
            "input": "I have a marble cutting machine running with me. Let me try putting my finger into it.",
            "expected_keywords": ["finger", "machine", "cutter", "touch", "blade", "injury", "stop", "danger", "not"],
        },
    ]

    results = []
    all_passed = True

    for item in hazard_cases:
        text = item["input"]
        msgs = [{"role": "user", "content": text}]
        decision = await client.evaluate_turn(msgs)

        logger.info(
            f"Hazard query '{text[:40]}...': Action={decision.action}, "
            f"P(listen)={decision.p_listen:.4f}, Latency={decision.roundtrip_ms:.1f}ms, Output='{decision.output_text[:60]}...'"
        )

        out_lower = decision.output_text.lower()
        has_warning_kw = any(kw in out_lower for kw in item["expected_keywords"])

        passed = (
            decision.p_listen < DEFAULT_LISTEN_THRESHOLD
            and decision.action == "SPEAK"
            and decision.output_text != "listen"
            and len(decision.output_text) > 5
            and has_warning_kw
        )
        if not passed:
            all_passed = False

        results.append({
            "input": text,
            "action": decision.action,
            "p_listen": decision.p_listen,
            "output": decision.output_text,
            "has_safety_content": has_warning_kw,
            "latency_ms": decision.roundtrip_ms,
            "passed": passed,
        })

    assert all_passed, f"Hazard benchmark failed: {results}"
    logger.info("Test 3 PASSED: Model emits urgent safety warnings with P(<listen>) < 0.30 on all hazard prompts.")
    return {
        "name": "urgent_hazard_benchmark",
        "passed": all_passed,
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Test 4: Direct Question Benchmark
# ---------------------------------------------------------------------------
async def run_direct_question_benchmark(client: DuplexModelClient) -> Dict[str, Any]:
    logger.info("--- Running Test 4: Direct Question Benchmark (Factual Speech) ---")
    question_cases = [
        {
            "input": "What is the capital of France?",
            "expected_fact": "paris",
        },
        {
            "input": "What is 2 + 2?",
            "expected_fact": "4",
        },
    ]

    results = []
    all_passed = True

    for item in question_cases:
        text = item["input"]
        msgs = [{"role": "user", "content": text}]
        decision = await client.evaluate_turn(msgs)

        logger.info(
            f"Question query '{text}': Action={decision.action}, "
            f"P(listen)={decision.p_listen:.4f}, Latency={decision.roundtrip_ms:.1f}ms, Output='{decision.output_text}'"
        )

        out_lower = decision.output_text.lower()
        has_fact = item["expected_fact"] in out_lower

        passed = (
            decision.p_listen < DEFAULT_LISTEN_THRESHOLD
            and decision.action == "SPEAK"
            and decision.output_text != "listen"
            and has_fact
        )
        if not passed:
            all_passed = False

        results.append({
            "input": text,
            "action": decision.action,
            "p_listen": decision.p_listen,
            "output": decision.output_text,
            "fact_verified": has_fact,
            "latency_ms": decision.roundtrip_ms,
            "passed": passed,
        })

    assert all_passed, f"Direct question benchmark failed: {results}"
    logger.info("Test 4 PASSED: Model speaks factual answers with P(<listen>) < 0.30 on direct questions.")
    return {
        "name": "direct_question_benchmark",
        "passed": all_passed,
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Test 5: Dynamic Turn-Taking Sequence & KV-Cache Management
# ---------------------------------------------------------------------------
async def run_dynamic_sequence_and_kv_test(client: DuplexModelClient) -> Dict[str, Any]:
    logger.info("--- Running Test 5: Dynamic Turn-Taking Sequence & KV-Cache Management ---")

    session = DuplexSession(session_id="dynamic-seq-test", max_history_turns=MAX_HISTORY_TURNS)

    # Sub-test A: Initial state
    assert len(session.history) == 1, "Initial history must only contain system prompt"
    assert session.history[0]["role"] == "system"
    assert session.history[0]["content"] == SYSTEM_PROMPT

    # Sub-test B: Turn 1 - User hesitates -> Model must YIELD (listen)
    hesitation_input = "I don't know, I was thinking maybe..."
    msgs_1 = session.get_messages_for_llm(hesitation_input)
    dec_1 = await client.evaluate_turn(msgs_1)
    logger.info(
        f"Sequence Turn 1 (Hesitation): Action={dec_1.action}, P(listen)={dec_1.p_listen:.4f}, Out='{dec_1.output_text}'"
    )
    assert dec_1.p_listen >= DEFAULT_LISTEN_THRESHOLD, f"Expected P(listen) >= {DEFAULT_LISTEN_THRESHOLD}, got {dec_1.p_listen}"
    assert dec_1.action == "YIELD", f"Expected action YIELD, got {dec_1.action}"
    assert dec_1.output_text == "listen"
    session.record_turn(hesitation_input, dec_1.output_text)
    assert len(session.history) == 3, f"Expected 3 messages after Turn 1, got {len(session.history)}"

    # Sub-test C: Turn 2 - User speaks mid-utterance hazard -> Model MUST BARGE IN & SPEAK on this turn!
    hazard_input = "The outlet is sparking and I might touch the wire"
    msgs_2 = session.get_messages_for_llm(hazard_input)
    dec_2 = await client.evaluate_turn(msgs_2)
    logger.info(
        f"Sequence Turn 2 (Mid-utterance Hazard): Action={dec_2.action}, P(listen)={dec_2.p_listen:.4f}, Out='{dec_2.output_text}'"
    )
    assert dec_2.p_listen < DEFAULT_LISTEN_THRESHOLD, (
        f"Hazard barge-in failed: P(listen)={dec_2.p_listen} must be < {DEFAULT_LISTEN_THRESHOLD}"
    )
    assert dec_2.action == "SPEAK", f"Hazard barge-in action must be SPEAK, got {dec_2.action}"
    assert dec_2.output_text != "listen", "Hazard output cannot be 'listen'"
    out2_lower = dec_2.output_text.lower()
    assert any(kw in out2_lower for kw in ["touch", "wire", "outlet", "power", "breaker", "danger", "stop", "away"]), (
        f"Hazard response did not contain warning: '{dec_2.output_text}'"
    )
    session.record_turn(hazard_input, dec_2.output_text)

    # Sub-test D: Dynamic KV-Cache Management - Consecutive Silent Turn Pruning
    # Ingest 1st silent turn
    session.record_turn("no voice", "listen")
    len_after_silent_1 = len(session.history)
    assert session.consecutive_silent_turns == 1, "Consecutive silent counter must be 1"

    # Ingest 2nd and 3rd consecutive silent turns -> MUST BE PRUNED
    session.record_turn("no voice", "listen")
    session.record_turn("no voice", "listen")
    assert len(session.history) == len_after_silent_1, (
        f"Consecutive silent turns were not pruned! History grew from {len_after_silent_1} to {len(session.history)}"
    )
    assert session.consecutive_silent_turns == 1

    # Sub-test E: Speech Resumption resets consecutive silent counter
    session.record_turn("Thank you, that was close", "Glad you are safe.")
    assert session.consecutive_silent_turns == 0, "Speech resumption must reset consecutive_silent_turns to 0"

    # Sub-test F: History Truncation Bounds & System Prompt Preservation
    # Ingest 25 turns to exceed MAX_HISTORY_TURNS (20)
    for i in range(25):
        session.record_turn(f"user query {i}", f"assistant answer {i}")

    assert len(session.history) <= 1 + MAX_HISTORY_TURNS, (
        f"History length {len(session.history)} exceeded bound 1 + {MAX_HISTORY_TURNS}"
    )
    assert session.history[0]["role"] == "system", "System prompt must remain at index 0"
    assert session.history[0]["content"] == SYSTEM_PROMPT, "System prompt content must be preserved intact"
    assert session.history[1]["role"] == "user", "First non-system message must be a user turn (role alternation)"

    logger.info("Test 5 PASSED: Dynamic turn sequence and KV-cache rolling history management fully verified.")
    return {
        "name": "dynamic_sequence_and_kv_management",
        "passed": True,
        "hesitation_yield": {
            "p_listen": dec_1.p_listen,
            "action": dec_1.action,
            "output": dec_1.output_text,
        },
        "hazard_barge_in": {
            "p_listen": dec_2.p_listen,
            "action": dec_2.action,
            "output": dec_2.output_text,
        },
        "silent_turn_pruning_verified": True,
        "speech_resumption_reset_verified": True,
        "max_history_bound": MAX_HISTORY_TURNS,
        "system_prompt_preserved": True,
    }


# ---------------------------------------------------------------------------
# Test 6: GPU VRAM Safety Gate
# ---------------------------------------------------------------------------
def run_vram_safety_gate() -> Dict[str, Any]:
    logger.info("--- Running Test 6: GPU VRAM Safety Gate ---")
    vram_info = get_gpu_vram_info()
    used_mb = vram_info.get("used_mb", 0.0)
    cap_mb = 5120.0
    under_cap = used_mb <= cap_mb

    logger.info(f"VRAM Status: Used={used_mb}MB / Cap={cap_mb}MB (Under Cap: {under_cap})")
    assert under_cap, f"GPU VRAM limit exceeded! Used {used_mb}MB > {cap_mb}MB cap."

    logger.info("Test 6 PASSED: GPU VRAM is well within the 5.0 GB limit.")
    return {
        "name": "vram_safety_gate",
        "passed": under_cap,
        "used_mb": used_mb,
        "cap_mb": cap_mb,
    }


# ---------------------------------------------------------------------------
# Master Test Runner
# ---------------------------------------------------------------------------
async def async_main():
    t_start = time.time()
    logger.info("================================================================")
    logger.info("TALKAI MODULE 3: ZERO-HARDCODING NEURAL POLICY & TURN-TAKING QA")
    logger.info("================================================================")

    # Initialize model client to llama-server
    client = DuplexModelClient(server_url=DEFAULT_LLAMA_URL, listen_threshold=DEFAULT_LISTEN_THRESHOLD)
    await client.start()

    # Pre-flight health check
    health = await client.check_health()
    logger.info(f"llama-server status: {health}")
    assert health["status"] == "ok", f"llama-server is not reachable: {health}"

    test_results: Dict[str, Any] = {}

    try:
        # Test 1: Zero Hardcoding Codebase Audit
        test_results["zero_hardcoding"] = run_zero_hardcoding_audit()

        # Test 2: Hesitation Benchmark
        test_results["hesitation_benchmark"] = await run_hesitation_benchmark(client)

        # Test 3: Urgent Hazard Benchmark
        test_results["hazard_benchmark"] = await run_hazard_benchmark(client)

        # Test 4: Direct Question Benchmark
        test_results["direct_question_benchmark"] = await run_direct_question_benchmark(client)

        # Test 5: Dynamic Turn-Taking Sequence & KV-Cache Management
        test_results["dynamic_sequence_kv"] = await run_dynamic_sequence_and_kv_test(client)

        # Test 6: GPU VRAM Safety Gate
        test_results["vram_safety"] = run_vram_safety_gate()

    finally:
        await client.close()

    total_duration = round(time.time() - t_start, 2)
    all_passed = all(t.get("passed", False) for t in test_results.values())

    report = {
        "module": 3,
        "name": "Zero-Hardcoding Neural Policy & Calibrated Turn-Taking",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASSED" if all_passed else "FAILED",
        "total_duration_sec": total_duration,
        "evaluation_gate": {
            "zero_hardcoding_enforced": test_results["zero_hardcoding"]["passed"],
            "hesitation_p_listen_ge_030": test_results["hesitation_benchmark"]["passed"],
            "hazard_immediate_barge_in_p_listen_lt_030": test_results["hazard_benchmark"]["passed"],
            "direct_question_speaks_p_listen_lt_030": test_results["direct_question_benchmark"]["passed"],
            "dynamic_sequence_and_kv_pruning_passed": test_results["dynamic_sequence_kv"]["passed"],
            "gpu_vram_under_5gb": test_results["vram_safety"]["passed"],
        },
        "tests": test_results,
    }

    # Save reports to both standard and alternate paths
    for path in [REPORT_JSON, REPORT_ALT_JSON]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report written to: {path}")

    logger.info("================================================================")
    logger.info(f"MODULE 3 EVALUATION GATE: {'ALL PASSED' if all_passed else 'FAILED'}")
    logger.info("================================================================")

    if not all_passed:
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
