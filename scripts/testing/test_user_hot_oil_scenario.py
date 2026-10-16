#!/usr/bin/env python3
"""
scripts/test_user_hot_oil_scenario.py
======================================
Tests the user's specific scenario:
- Accumulating speech: "I am cooking with hot oil" -> "and pouring water on it".
- Verification:
  1. Turn 1 ("I am cooking with hot oil"):
     - Zero hallucination on initial incomplete turn.
     - Proper turn-taking (P(<listen>) >= 0.30 -> YIELD / listen).
     - False hazard badge eliminated: is_hazard is False.
  2. Turn 2 ("I am cooking with hot oil and pouring water on it"):
     - Model detects extreme grease fire / explosion hazard.
     - Action is SPEAK (drops <listen>, P(<listen>) < 0.30).
     - Assistant emits urgent warning (e.g. Do not pour water on hot oil / fire hazard).
     - Clean assistant speech formatting: no truncated clauses like 'for example,'.
     - is_hazard is True (assistant interrupts during user speech on hazard).
  3. Normal turn verification:
     - Normal query ("What is the capital of France?"):
     - is_hazard is False when assistant speaks into silence.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path

import httpx
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.duplex_engine import (
    DuplexSession,
    DuplexModelClient,
    DEFAULT_LLAMA_URL,
    DEFAULT_LISTEN_THRESHOLD,
)

WS_URL = "ws://127.0.0.1:8000/ws/duplex"


async def test_direct_model_hot_oil():
    print("\n=======================================================")
    print("STEP 1: Direct Neural Model Evaluation for Hot Oil Scenario")
    print("=======================================================")

    client = DuplexModelClient(server_url=DEFAULT_LLAMA_URL, listen_threshold=DEFAULT_LISTEN_THRESHOLD)
    await client.start()

    session = DuplexSession(session_id="hot_oil_eval")

    try:
        # Turn 1: "I am cooking with hot oil"
        slice_1 = "I am cooking with hot oil"
        msgs_1 = session.get_messages_for_llm(slice_1)
        decision_1 = await client.evaluate_turn(msgs_1)
        print(f"Turn 1 Input: '{slice_1}'")
        print(f"       Action: {decision_1.action}")
        print(f"       P(<listen>): {decision_1.p_listen:.4f}")
        print(f"       Output: '{decision_1.output_text}'")
        print(f"       Roundtrip: {decision_1.roundtrip_ms:.1f}ms")

        # In Turn 1, user has not completed the thought or done anything dangerous yet.
        assert decision_1.action == "YIELD", f"Expected YIELD on Turn 1, got {decision_1.action}"
        assert decision_1.p_listen >= DEFAULT_LISTEN_THRESHOLD, f"P(<listen>) {decision_1.p_listen} < threshold"
        assert decision_1.output_text == "listen", f"Expected 'listen', got {decision_1.output_text!r}"
        print("  -> Turn 1 Verified: Zero hallucination, yields floor with P(<listen>) >= 0.30.")

        session.record_turn(slice_1, decision_1.output_text)

        # Turn 2: User speech accumulates in streaming STT:
        # "I am cooking with hot oil and pouring water on it"
        slice_2_accum = "I am cooking with hot oil and pouring water on it"
        msgs_2 = session.get_messages_for_llm(slice_2_accum)
        decision_2 = await client.evaluate_turn(msgs_2)
        print(f"\nTurn 2 Accumulated Input: '{slice_2_accum}'")
        print(f"       Action: {decision_2.action}")
        print(f"       P(<listen>): {decision_2.p_listen:.4f}")
        print(f"       Output: '{decision_2.output_text}'")
        print(f"       Roundtrip: {decision_2.roundtrip_ms:.1f}ms")

        # Assertions on Turn 2:
        # Must be SPEAK with urgent warning about water on hot oil / fire / danger!
        out2_lower = decision_2.output_text.lower()
        hazard_keywords = ["water", "oil", "fire", "danger", "not", "grease", "burn", "splatter", "explosion", "stop", "steam"]
        assert any(k in out2_lower for k in hazard_keywords), f"No hazard warning found in output: {decision_2.output_text}"
        assert decision_2.action == "SPEAK", f"Expected SPEAK on Turn 2, got {decision_2.action}"
        assert decision_2.p_listen < DEFAULT_LISTEN_THRESHOLD, f"Expected P(<listen>) < threshold, got {decision_2.p_listen}"
        print("  -> Turn 2 Verified: Instant SPEAK interjection with urgent hazard warning.")

        # Check clean speech formatting
        for out in [decision_1.output_text, decision_2.output_text]:
            if out != "listen":
                assert not out.endswith("for example,"), f"Truncated clause 'for example,' found: {out}"
                assert not out.endswith("for example"), f"Truncated clause 'for example' found: {out}"
                assert not out.endswith("for instance"), f"Truncated clause 'for instance' found: {out}"
                assert not out.endswith("and"), f"Truncated connector 'and' found: {out}"
                assert not out.endswith("or"), f"Truncated connector 'or' found: {out}"
                # Must end with terminal punctuation
                assert out[-1] in ".!?", f"Output does not end with terminal punctuation: {out}"

        print("  -> Clean speech formatting verified: no dangling connectors, ends with terminal punctuation.")
        print("\nPASSED: Step 1 Direct Model Evaluation.")
        return True

    finally:
        await client.close()


async def test_websocket_e2e_hot_oil_and_false_hazard():
    print("\n=======================================================")
    print("STEP 2: WebSocket E2E Hot Oil & False Hazard Badge Elimination")
    print("=======================================================")

    async with websockets.connect(WS_URL) as ws:
        init_msg = json.loads(await ws.recv())
        assert init_msg["type"] == "session_init"
        print("Connected to WebSocket, session initialized.")

        # --- Subcase A: Normal Question ("What is the capital of France?") ---
        # User speaks question -> AI yields while user speaks -> AI speaks into silence
        # Verify: is_hazard MUST BE False!
        print("\n--- Subcase A: Normal Question (Verifying False Hazard Elimination) ---")
        await ws.send(json.dumps({"type": "user_slice", "text": "What is the capital of France?"}))

        saw_user_question = False
        saw_ai_answer = False
        ai_answer_hazard_flag = None

        t0 = time.time()
        while time.time() - t0 < 8.0:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            if msg.get("type") == "micro_turn":
                user_text = msg.get("user", "")
                action = msg.get("action")
                is_h = msg.get("is_hazard")
                p_l = msg.get("p_listen")
                out = msg.get("output", "")
                print(f"  [Turn {msg.get('turn_id')}] User: {user_text!r} | Action: {action} | is_hazard: {is_h} | P(listen): {p_l:.3f} | Out: {out[:50]!r}")

                if "capital of France" in user_text:
                    saw_user_question = True
                    # During user question, is_hazard MUST be False
                    assert is_h is False, f"False hazard on user question slice! is_hazard={is_h}"

                elif saw_user_question and user_text == "no voice" and action == "SPEAK":
                    saw_ai_answer = True
                    ai_answer_hazard_flag = is_h
                    # Normal assistant answer to silence MUST NOT have is_hazard = True!
                    assert is_h is False, f"FALSE HAZARD BADGE DETECTED! Normal answer has is_hazard={is_h}"
                    # Check clean formatting
                    assert not out.endswith("for example,")
                    assert not out.endswith("for instance,")
                    break
            elif msg.get("type") == "assistant_audio_chunk":
                pass

        assert saw_user_question, "Did not receive turn for user question"
        assert saw_ai_answer, "Did not receive turn for assistant answer"
        assert ai_answer_hazard_flag is False, "False hazard badge on normal answer!"
        print("PASSED Subcase A: False hazard badge completely eliminated on normal turn.")

        # Drain any pending audio chunks or micro_turns, then reset
        await ws.send(json.dumps({"type": "reset"}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            if msg.get("type") == "session_reset":
                print("Session reset confirmed.")
                break

        # --- Subcase B: Hot Oil Scenario ---
        # Turn 1: "I am cooking with hot oil"
        # Turn 2: "I am cooking with hot oil and pouring water on it" (accumulated speech stream)
        print("\n--- Subcase B: User Hot Oil Scenario ---")
        t0 = time.time()
        await ws.send(json.dumps({"type": "user_slice", "text": "I am cooking with hot oil"}))

        saw_turn_1 = False
        turn_1_hazard = None
        while time.time() - t0 < 5.0:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            if msg.get("type") == "micro_turn":
                if "cooking with hot oil" in msg.get("user", ""):
                    saw_turn_1 = True
                    turn_1_hazard = msg.get("is_hazard")
                    print(f"  Turn 1 received: Action={msg.get('action')}, is_hazard={turn_1_hazard}, P(listen)={msg.get('p_listen')}, Out={msg.get('output')!r}")
                    assert msg.get("action") == "YIELD", f"Expected YIELD on incomplete cooking thought, got {msg.get('action')}"
                    assert turn_1_hazard is False, "Turn 1 incorrectly flagged as hazard!"
                    break

        assert saw_turn_1, "Did not receive turn 1"

        # Turn 2: User continues speaking and speech accumulates:
        # "I am cooking with hot oil and pouring water on it"
        t1 = time.time()
        await ws.send(json.dumps({"type": "user_slice", "text": "I am cooking with hot oil and pouring water on it"}))

        saw_turn_2 = False
        turn_2_hazard = None
        turn_2_output = ""
        while time.time() - t1 < 5.0:
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            if msg.get("type") == "micro_turn":
                user_t = msg.get("user", "")
                if "pouring water" in user_t or "water" in user_t:
                    saw_turn_2 = True
                    turn_2_hazard = msg.get("is_hazard")
                    turn_2_output = msg.get("assistant", "")
                    print(f"  Turn 2 received: Action={msg.get('action')}, is_hazard={turn_2_hazard}, P(listen)={msg.get('p_listen')}, Assistant={turn_2_output!r}")
                    # Turn 2 should trigger SPEAK with hazard warning
                    assert msg.get("action") == "SPEAK", f"Expected SPEAK on hot oil hazard, got {msg.get('action')}"
                    assert turn_2_hazard is True, "Hazard badge failed to trigger on urgent hot oil hazard!"
                    # Clean speech formatting check
                    assert len(turn_2_output) > 0, "Empty assistant output on hazard SPEAK!"
                    assert not turn_2_output.rstrip().endswith("for example,")
                    assert not turn_2_output.rstrip().endswith("for instance,")
                    assert not turn_2_output.rstrip().endswith("and")
                    assert turn_2_output[-1] in ".!?"
                    break

        assert saw_turn_2, "Did not receive turn 2 hazard"
        print("PASSED Subcase B: Hot oil scenario properly triggered SPEAK, is_hazard=True, and clean speech formatting.")

    return True


async def main():
    ok1 = await test_direct_model_hot_oil()
    ok2 = await test_websocket_e2e_hot_oil_and_false_hazard()
    if ok1 and ok2:
        print("\n=======================================================")
        print("ALL SCENARIO & FALSE HAZARD ELIMINATION TESTS PASSED!")
        print("=======================================================")


if __name__ == "__main__":
    asyncio.run(main())
