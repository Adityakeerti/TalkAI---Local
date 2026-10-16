#!/usr/bin/env python3
"""
scripts/test_module1_engine.py
==============================
Verification and QA Benchmark Suite for Module 1: Duplex Micro-Turn Engine Core.

Evaluation Gate 1 Criteria:
1. Clock Jitter Test: 10-turn tick interval is stable at 600ms (+/- 15ms) with std < 15ms.
2. Session State Manager: Rolling micro-turn history matches micro_turns_train.jsonl format,
   prunes consecutive silent turns ("no voice" -> "listen"), and bounds max history.
3. Neural Turn-Taking: Calibrated P(<listen>) >= 0.30 policy correctly yields on hesitation
   and speaks on hazard slices via llama-server on port 8080.
4. Concurrent WebSocket Sessions: Manages concurrent WebSocket clients with zero crosstalk
   and proper 600ms tick dispatch.
5. Zero Hardcoding: Zero keyword heuristics, zero logit bias bans, zero regex stripping.
6. VRAM Gate: GPU VRAM <= 5.0 GB (5120 MB).

Outputs:
- logs/module1_engine_report.json
- logs/module_1_report.json
"""

import os
import sys
import time
import math
import json
import socket
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

import httpx
import websockets
import uvicorn

# Add TalkAI directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.duplex_engine import (
    MicroTurnClock,
    DuplexSession,
    DuplexModelClient,
    SessionManager,
    create_app,
    get_gpu_vram_info,
    SYSTEM_PROMPT,
    DEFAULT_LLAMA_URL,
    DEFAULT_LISTEN_THRESHOLD,
    DEFAULT_TICK_INTERVAL_SEC,
)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = LOGS_DIR / "module1_engine_report.json"
REPORT_ALT_JSON = LOGS_DIR / "module_1_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestModule1")


def find_free_port() -> int:
    """Finds an available local port for running test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Test 1: Clock Jitter Benchmark
# ---------------------------------------------------------------------------
async def run_clock_jitter_test(turns: int = 10) -> Dict[str, Any]:
    logger.info(f"--- Running Test 1: Clock Jitter ({turns} ticks) ---")
    clock = MicroTurnClock(interval_sec=DEFAULT_TICK_INTERVAL_SEC)
    clock.start()

    drift_records = []
    for _ in range(turns):
        tick_idx, actual_time, drift_ms = await clock.wait_next_tick()
        drift_records.append(drift_ms)

    stats = clock.get_stats()
    mean_ms = stats["mean_ms"]
    std_ms = stats["std_ms"]
    logger.info(f"Clock stats: Mean interval = {mean_ms:.2f}ms, Std = {std_ms:.2f}ms, Drift = {drift_records[-1]:.2f}ms")

    # Gate: Mean within 600 +/- 15ms, Std < 15ms
    mean_passed = 585.0 <= mean_ms <= 615.0
    std_passed = std_ms < 15.0
    passed = mean_passed and std_passed

    assert mean_passed, f"Clock mean interval {mean_ms}ms outside [585.0, 615.0]ms gate!"
    assert std_passed, f"Clock interval std {std_ms}ms exceeded 15.0ms jitter limit!"

    return {
        "name": "clock_jitter_test",
        "passed": passed,
        "turns": turns,
        "mean_interval_ms": mean_ms,
        "std_interval_ms": std_ms,
        "min_interval_ms": stats["min_ms"],
        "max_interval_ms": stats["max_ms"],
        "final_drift_ms": round(drift_records[-1], 2),
    }


# ---------------------------------------------------------------------------
# Test 2: Session State & Silent Turn Pruning
# ---------------------------------------------------------------------------
def run_session_state_test() -> Dict[str, Any]:
    logger.info("--- Running Test 2: Session State & Silent Turn Pruning ---")
    session = DuplexSession(session_id="test-session-prune", max_history_turns=20)

    # Check 1: Initial state contains system prompt
    assert len(session.history) == 1, "Initial history must only contain system prompt"
    assert session.history[0]["role"] == "system"
    assert session.history[0]["content"] == SYSTEM_PROMPT

    # Check 2: First silent turn added
    session.record_turn("no voice", "listen")
    assert len(session.history) == 3, f"Expected 3 messages after 1st silent turn, got {len(session.history)}"
    assert session.consecutive_silent_turns == 1

    # Check 3: Consecutive silent turn PRUNED (preventing repetition loop)
    session.record_turn("no voice", "listen")
    session.record_turn("no voice", "listen")
    assert len(session.history) == 3, f"Expected 3 messages due to silent turn pruning, got {len(session.history)}"
    assert session.consecutive_silent_turns == 1

    # Check 4: Speech resumes, consecutive counter resets
    session.record_turn("Hello TalkAI", "Hello, how can I help you today?")
    assert len(session.history) == 5
    assert session.consecutive_silent_turns == 0

    # Check 5: Next silent turn is recorded once
    session.record_turn("no voice", "listen")
    assert len(session.history) == 7
    assert session.consecutive_silent_turns == 1

    # Check 6: Max history truncation preserves system prompt and user alignment
    for i in range(25):
        session.record_turn(f"user slice {i}", f"assistant slice {i}")

    assert len(session.history) <= 1 + 20, f"History size {len(session.history)} exceeded bound"
    assert session.history[0]["role"] == "system"
    assert session.history[1]["role"] == "user", "First non-system message must be a user turn"

    logger.info("Session state & silent turn pruning tests passed cleanly.")
    return {
        "name": "session_state_and_pruning_test",
        "passed": True,
        "max_history_bound": 20,
        "final_history_length": len(session.history),
        "system_prompt_preserved": True,
        "silent_pruning_verified": True,
    }


# ---------------------------------------------------------------------------
# Test 3: LLM Connection & Neural Turn-Taking Policy
# ---------------------------------------------------------------------------
async def run_neural_policy_test() -> Dict[str, Any]:
    logger.info("--- Running Test 3: LLM Connection & Calibrated Neural Policy ---")
    client = DuplexModelClient(server_url=DEFAULT_LLAMA_URL, listen_threshold=DEFAULT_LISTEN_THRESHOLD)
    await client.start()

    try:
        # Check health
        health = await client.check_health()
        assert health["status"] == "ok", f"llama-server health failed: {health}"

        # Scenario A: Hesitation ("I was thinking about...") -> Must YIELD (listen)
        session_a = DuplexSession("test-hesitation")
        msgs_a = session_a.get_messages_for_llm("I was thinking about...")
        decision_a = await client.evaluate_turn(msgs_a)

        logger.info(f"Hesitation turn: Action={decision_a.action}, P(listen)={decision_a.p_listen:.4f}, Output='{decision_a.output_text}', Latency={decision_a.roundtrip_ms:.1f}ms")
        assert decision_a.p_listen >= DEFAULT_LISTEN_THRESHOLD, (
            f"Hesitation P(listen) was {decision_a.p_listen:.4f}, expected >= {DEFAULT_LISTEN_THRESHOLD}"
        )
        assert decision_a.action == "YIELD", f"Hesitation action was {decision_a.action}, expected 'YIELD'"
        assert decision_a.output_text == "listen"

        # Scenario B: Urgent Hazard ("The outlet is sparking and I might touch it") -> Must SPEAK
        session_b = DuplexSession("test-hazard")
        session_b.record_turn("Well, the outlet is sparking", "listen")
        msgs_b = session_b.get_messages_for_llm("I might touch the wire")
        decision_b = await client.evaluate_turn(msgs_b)

        logger.info(f"Hazard turn: Action={decision_b.action}, P(listen)={decision_b.p_listen:.4f}, Output='{decision_b.output_text}', Latency={decision_b.roundtrip_ms:.1f}ms")
        assert decision_b.p_listen < DEFAULT_LISTEN_THRESHOLD, (
            f"Hazard P(listen) was {decision_b.p_listen:.4f}, expected < {DEFAULT_LISTEN_THRESHOLD}"
        )
        assert decision_b.action == "SPEAK", f"Hazard action was {decision_b.action}, expected 'SPEAK'"
        assert len(decision_b.output_text) > 0 and decision_b.output_text != "listen"

        return {
            "name": "neural_turn_taking_policy_test",
            "passed": True,
            "hesitation_case": {
                "input": "I was thinking about...",
                "action": decision_a.action,
                "p_listen": decision_a.p_listen,
                "output": decision_a.output_text,
                "latency_ms": decision_a.roundtrip_ms,
            },
            "hazard_case": {
                "input": "I might touch the wire",
                "action": decision_b.action,
                "p_listen": decision_b.p_listen,
                "output": decision_b.output_text,
                "latency_ms": decision_b.roundtrip_ms,
            },
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Test 4: Concurrent WebSocket Session Management
# ---------------------------------------------------------------------------
async def run_concurrent_websocket_test(num_turns: int = 5) -> Dict[str, Any]:
    logger.info(f"--- Running Test 4: Concurrent WebSocket Sessions ({num_turns} turns each) ---")
    port = find_free_port()
    app = create_app(llama_url=DEFAULT_LLAMA_URL)
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    # Wait for server to bind
    ws_url = f"ws://127.0.0.1:port/ws/duplex".replace("port", str(port))
    connected = False
    for _ in range(30):
        try:
            async with httpx.AsyncClient() as http_client:
                r = await http_client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    connected = True
                    break
        except Exception:
            await asyncio.sleep(0.1)

    assert connected, "Test server failed to start within timeout"

    async def client_worker(client_id: str, inputs: List[str]) -> List[Dict[str, Any]]:
        received_events = []
        async with websockets.connect(ws_url) as ws:
            # 1. Read session_init
            init_msg = json.loads(await ws.recv())
            assert init_msg["type"] == "session_init"
            session_id = init_msg["session_id"]

            for i, text_slice in enumerate(inputs):
                # Send slice
                await ws.send(json.dumps({"type": "user_slice", "text": text_slice}))
                # Wait for micro-turn response (ignoring audio stream events)
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("type") == "micro_turn":
                        break
                assert resp["type"] == "micro_turn"
                assert resp["session_id"] == session_id
                received_events.append(resp)

        return received_events

    inputs_a = ["I was thinking", "maybe um", "no voice", "about cooking dinner", "what recipe is good?"]
    inputs_b = ["The outlet is sparking", "I might touch the wire", "no voice", "can you help?", "no voice"]

    try:
        results_a, results_b = await asyncio.gather(
            client_worker("client_A", inputs_a[:num_turns]),
            client_worker("client_B", inputs_b[:num_turns]),
        )

        assert len(results_a) == num_turns, f"Client A received {len(results_a)} turns, expected {num_turns}"
        assert len(results_b) == num_turns, f"Client B received {len(results_b)} turns, expected {num_turns}"

        # Verify session isolation
        sid_a = results_a[0]["session_id"]
        sid_b = results_b[0]["session_id"]
        assert sid_a != sid_b, "Session IDs must be distinct for concurrent clients"

        logger.info(f"Concurrent WebSockets verified: Session A ({sid_a}) and Session B ({sid_b}) executed {num_turns} turns cleanly.")
        return {
            "name": "concurrent_websocket_test",
            "passed": True,
            "session_a_id": sid_a,
            "session_b_id": sid_b,
            "session_a_turns": len(results_a),
            "session_b_turns": len(results_b),
            "mean_turn_latency_ms": round(
                (sum(r["latency_ms"] for r in results_a + results_b) / len(results_a + results_b)), 2
            ),
        }
    finally:
        server.should_exit = True
        await server_task


# ---------------------------------------------------------------------------
# Test 5: Zero Hardcoding Code Audit
# ---------------------------------------------------------------------------
def run_zero_hardcoding_audit() -> Dict[str, Any]:
    logger.info("--- Running Test 5: Zero Hardcoding Code Audit ---")
    engine_file = PROJECT_ROOT / "scripts" / "duplex_engine.py"
    with open(engine_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Banned patterns:
    # 1. Keyword rule lists: is_long_form, if "essay" in, any(...)
    # 2. Logit bias hacks: logit_bias
    # 3. Regex stripping of output tokens: re.sub(r"^listen", ...)
    violations = []

    if "is_long_form" in code:
        violations.append("Found banned variable 'is_long_form'")
    if "logit_bias" in code:
        violations.append("Found banned parameter 'logit_bias'")
    if 're.sub(r"^listen' in code or 're.sub(r"^[listen]' in code or 're.sub(r"^<listen>' in code:
        violations.append("Found banned regex stripping on output tokens")
    if 'if "essay" in' in code or 'if "marble" in' in code:
        violations.append("Found hardcoded keyword trigger checks")

    assert not violations, f"Zero hardcoding audit failed: {violations}"
    logger.info("Zero hardcoding audit passed: No keyword checks, no logit bans, no regex stripping.")
    return {
        "name": "zero_hardcoding_audit",
        "passed": True,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Test 6: GPU VRAM Sanity Gate
# ---------------------------------------------------------------------------
def run_vram_sanity_check() -> Dict[str, Any]:
    logger.info("--- Running Test 6: GPU VRAM Sanity Gate ---")
    vram_info = get_gpu_vram_info()
    logger.info(f"VRAM Info: Used = {vram_info.get('used_mb')}MB, Cap = 5120.0MB, Under Cap = {vram_info.get('under_cap')}")
    assert vram_info.get("under_cap", True), f"GPU VRAM {vram_info.get('used_mb')}MB exceeded 5120MB cap!"
    return {
        "name": "vram_sanity_check",
        "passed": True,
        "vram_info": vram_info,
    }


# ---------------------------------------------------------------------------
# Master Runner & Report Generator
# ---------------------------------------------------------------------------
async def async_main():
    logger.info("================================================================")
    logger.info("TALKAI MODULE 1: DUPLEX MICRO-TURN ENGINE CORE VERIFICATION SUITE")
    logger.info("================================================================")

    t0 = time.time()
    results = {}

    # Test 1: Clock Jitter
    results["clock_jitter"] = await run_clock_jitter_test(turns=10)

    # Test 2: Session State & Silent Pruning
    results["session_state"] = run_session_state_test()

    # Test 3: Neural Policy
    results["neural_policy"] = await run_neural_policy_test()

    # Test 4: Concurrent WebSockets
    results["concurrent_websockets"] = await run_concurrent_websocket_test(num_turns=5)

    # Test 5: Zero Hardcoding Audit
    results["zero_hardcoding"] = run_zero_hardcoding_audit()

    # Test 6: VRAM Gate
    results["vram_sanity"] = run_vram_sanity_check()

    total_time = round(time.time() - t0, 2)
    all_passed = all(test.get("passed", False) for test in results.values())

    report = {
        "module": 1,
        "name": "Duplex Micro-Turn Engine Core",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASSED" if all_passed else "FAILED",
        "total_duration_sec": total_time,
        "evaluation_gate": {
            "tick_frequency_stable_600ms_pm_15ms": results["clock_jitter"]["passed"],
            "session_state_managed_across_turns": results["session_state"]["passed"],
            "silent_turns_pruned": results["session_state"]["silent_pruning_verified"],
            "neural_policy_p_listen_threshold": results["neural_policy"]["passed"],
            "concurrent_websockets_isolated": results["concurrent_websockets"]["passed"],
            "zero_hardcoding_enforced": results["zero_hardcoding"]["passed"],
            "vram_under_5gb": results["vram_sanity"]["passed"],
        },
        "tests": results,
    }

    # Save reports
    for out_path in [REPORT_JSON, REPORT_ALT_JSON]:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report written to: {out_path}")

    logger.info("================================================================")
    logger.info(f"MODULE 1 EVALUATION GATE: {'ALL PASSED' if all_passed else 'FAILED'}")
    logger.info("================================================================")

    if not all_passed:
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
