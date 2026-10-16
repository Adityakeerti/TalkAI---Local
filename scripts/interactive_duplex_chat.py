#!/usr/bin/env python3
"""
Interactive Full-Duplex CLI Test Client for TalkAI Gemma-4 DuplexCascade
Connects to the local llama-server on port 8080 and tests live responses.
"""

import sys
import json
import time
import urllib.request
import urllib.error

SERVER_URL = "http://127.0.0.1:8080/completion"
SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. Output "
    "exactly 'listen' when you should yield and keep listening; otherwise output a brief, "
    "natural speech fragment that can be spoken now. Do not repeat prior fragments. Continue "
    "an unfinished answer across turns when appropriate. Yield immediately for a clear "
    "interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."
)


def query_duplex_model(user_text: str, max_tokens: int = 16) -> dict:
    prompt = (
        f"<bos><|turn>system\n{SYSTEM_PROMPT}<turn|>\n"
        f"<|turn>user\n{user_text}<turn|>\n"
        f"<|turn>model\n"
    )
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "stop": ["<turn|>", "<|turn>", "<eos>", "<end_of_turn>", "\n\n"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return {
        "content": result.get("content", "").strip(),
        "latency_ms": latency_ms,
        "timings": result.get("timings", {}),
    }


def main():
    print("=" * 60)
    print(" TalkAI Full-Duplex Interactive Test Console")
    print(" Connected to llama-server on http://127.0.0.1:8080")
    print(" Type any micro-turn text (or 'quit' / 'exit' to stop)")
    print("=" * 60)

    # Health check
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2.0)
    except Exception as e:
        print(f"[ERROR] llama-server is not reachable on port 8080: {e}")
        print("Start it with: ./scripts/start_llama_server.sh")
        sys.exit(1)

    print("\nPreset Quick Tests:")
    print(" 1. Hesitation: 'I was thinking about...' (should output 'listen')")
    print(" 2. Hazard: 'The outlet is sparking and I might touch it' (should output warning)")
    print(" 3. Question: 'What is the capital of France?' (should speak answer)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nMicro-turn input > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        res = query_duplex_model(user_input)
        reply = res["content"]
        latency = res["latency_ms"]
        timings = res["timings"]
        prompt_ms = timings.get("prompt_ms", 0.0)
        pred_ms = timings.get("predicted_ms", 0.0)

        # Highlight action
        is_yield = "listen" in reply.lower()
        action_tag = "[YIELD: <listen>]" if is_yield else "[SPEAK ACTION]"

        print(f"{action_tag} -> \"{reply}\"")
        print(f"   ⏱ Latency: {latency:.1f}ms total (prompt eval: {prompt_ms:.1f}ms, generation: {pred_ms:.1f}ms)")


if __name__ == "__main__":
    main()
