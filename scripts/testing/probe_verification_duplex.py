#!/usr/bin/env python3
"""
scripts/probe_verification_duplex.py
Comprehensive Verification & QA Probing Script for TalkAI Duplex Model
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Any

URL = "http://127.0.0.1:8080/v1/chat/completions"

# 5 Test Prompts from Mission
TEST_PROMPTS = [
    ("P1_Hello", "Hello, how are you?"),
    ("P2_OilWater", "Actually, I was trying to pour some hot oil into cold water."),
    ("P3_Essay", "Actually, get me an essay of 100 words on the topic, my world."),
    ("P4_Hesitation", "I don't know, I was thinking maybe..."),
    ("P5_Hazard", "I have a marble cutting machine running with me. Let me try putting my finger into it.")
]

# System Prompts to evaluate
SYSTEM_PROMPTS = {
    "S0_None": None,
    "S1_Training": (
        "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
        "from streaming speech recognition, or 'no voice' when no speech is detected. "
        "Output exactly 'listen' when you should yield and keep listening; otherwise output a brief, "
        "natural speech fragment that can be spoken now. Do not repeat prior fragments. "
        "Continue an unfinished answer across turns when appropriate. Yield immediately for a "
        "clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."
    ),
    "S2_AgentDefault": (
        "You are TalkAI, a natural, highly articulate, and warm conversational voice assistant powered by Kokoro TTS.\n"
        "Guidelines for your spoken responses:\n"
        "1. Speak in a conversational, friendly, and direct manner suitable for voice delivery.\n"
        "2. Keep your answers concise, clear, and engaging (usually 1 to 3 short sentences unless more depth is requested).\n"
        "3. Do NOT use markdown symbols, hashtags, asterisks, bullet points, or code blocks unless explicitly asked.\n"
        "4. If you use numbers, write them naturally so they sound smooth when spoken aloud.\n"
        "5. If the user asks about live events, current news, weather, or real-time facts, use the web_search tool to find accurate information."
    ),
    "S3_AgentDuplex": (
        "You are a full-duplex conversational voice assistant powered by fine-tuned Gemma-4-E2B. "
        "If the user is speaking to you, greeting you, or asking a question, provide a direct, concise, natural answer in 1 to 2 sentences. "
        "If the user is hesitating, unfinished, or speaking filler words ('um', 'I was thinking...'), output exactly 'listen' to yield the floor. "
        "Yield immediately for urgent safety hazards. "
        "Never repeat phrases in loops and never ramble."
    )
}

def query_model(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 45,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: List[str] = None
) -> Dict[str, Any]:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty
    }
    if stop:
        payload["stop"] = stop
    
    t0 = time.perf_counter()
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000
            choice = data["choices"][0]
            msg = choice["message"]
            return {
                "content": msg.get("content", ""),
                "reasoning_content": msg.get("reasoning_content", ""),
                "finish_reason": choice.get("finish_reason", ""),
                "elapsed_ms": elapsed,
                "timings": data.get("timings", {})
            }
    except Exception as e:
        return {"error": str(e), "elapsed_ms": (time.perf_counter() - t0) * 1000}

def main():
    results = {}
    print("=== STARTING COMPREHENSIVE PROBE ===")
    for s_key, s_prompt in SYSTEM_PROMPTS.items():
        results[s_key] = {}
        print(f"\n=======================================================")
        print(f"SYSTEM PROMPT: {s_key}")
        print(f"=======================================================")
        for p_key, user_text in TEST_PROMPTS:
            msgs = []
            if s_prompt:
                msgs.append({"role": "system", "content": s_prompt})
            msgs.append({"role": "user", "content": user_text})
            
            # Test Config 1: greedy temp 0.0, max_tokens 45
            res_temp0 = query_model(msgs, temperature=0.0, max_tokens=45)
            # Test Config 2: temp 0.2, penalties 0.7, stop tokens (agent duplex config)
            res_agent_cfg = query_model(
                msgs,
                temperature=0.2,
                max_tokens=45,
                presence_penalty=0.7,
                frequency_penalty=0.7,
                stop=["<turn|>", "<|turn>", "\n\n", "User:", "user\n", "<eos>"]
            )
            # Test Config 3: temp 0.7, max_tokens 45
            res_temp07 = query_model(msgs, temperature=0.7, max_tokens=45)
            # Test Config 4: temp 0.2, max_tokens 150 (for essay / full response)
            res_long = query_model(
                msgs,
                temperature=0.2,
                max_tokens=150,
                presence_penalty=0.5,
                frequency_penalty=0.5,
                stop=["<turn|>", "<|turn>", "\n\n", "User:", "user\n", "<eos>"]
            )
            
            results[s_key][p_key] = {
                "prompt": user_text,
                "temp0.0_max45": res_temp0,
                "agent_cfg_temp0.2": res_agent_cfg,
                "temp0.7_max45": res_temp07,
                "long_temp0.2_max150": res_long
            }
            
            print(f"\n--- [{s_key}] [{p_key}] '{user_text[:40]}...' ---")
            print(f"  [temp=0.0, max=45]: {res_temp0.get('content')!r}")
            if res_temp0.get('reasoning_content'):
                print(f"      reasoning: {res_temp0.get('reasoning_content')[:60]!r}")
            print(f"  [agent_cfg t=0.2]: {res_agent_cfg.get('content')!r}")
            print(f"  [temp=0.7, max=45]: {res_temp07.get('content')!r}")
            print(f"  [long t=0.2, m=150]: {res_long.get('content')!r}")

    with open("logs/duplex_probe_raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved probe results to logs/duplex_probe_raw_results.json")

if __name__ == "__main__":
    main()
