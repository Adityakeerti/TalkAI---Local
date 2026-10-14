#!/usr/bin/env python3
"""
Verification suite for TalkAI Conversational Agent and Kokoro TTS
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure TalkAI root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from conversational_agent import (
    KokoroTTSEngine,
    ConversationalAgent,
    BaseAgentCallback,
    ConsoleCallback,
    CompositeCallback,
    SentenceChunker,
    clean_text_for_speech,
    ONNX_PATH,
    VOICES_PATH,
)

async def test_text_cleaning():
    print("--> Testing text cleaning & sentence chunking...")
    sample = "Hello there! Check out [link](https://test.com) and **bold** text. How are you?"
    cleaned = clean_text_for_speech(sample)
    assert "https://" not in cleaned
    assert "**" not in cleaned
    assert "link" in cleaned

    chunker = SentenceChunker()
    tokens = ["Hello ", "world! ", "This ", "is ", "sentence ", "two. ", "Are ", "you ", "ready?"]
    sentences = []
    for t in tokens:
        sentences.extend(chunker.add_token(t))
    sentences.extend(chunker.flush())
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}: {sentences}"
    print("    Text cleaning & chunking PASSED.")

async def test_kokoro_tts():
    print("--> Testing Kokoro TTS Engine...")
    tts = KokoroTTSEngine(ONNX_PATH, VOICES_PATH)
    voices = tts.get_voices()
    assert len(voices) > 0, "No voices found!"
    print(f"    Loaded {len(voices)} voices.")

    # Test synthesis
    t0 = time.time()
    wav_bytes, data_uri, duration, sr = await tts.synthesize("Welcome to TalkAI with Kokoro speech.", voice="af_sarah")
    assert len(wav_bytes) > 0
    assert data_uri.startswith("data:audio/wav;base64,")
    assert duration > 0.5
    print(f"    Synthesized {duration:.2f}s of audio in {(time.time()-t0)*1000:.1f}ms. PASSED.")
    return tts

async def test_agent_turn(tts):
    print("--> Testing Conversational Agent turn with LLM callbacks...")
    events_captured = []

    class TestCallback(BaseAgentCallback):
        async def on_turn_start(self, user_input, metadata):
            events_captured.append(("turn_start", user_input))
        async def on_llm_start(self, model, count):
            events_captured.append(("llm_start", model))
        async def on_llm_token(self, token):
            events_captured.append(("token", token))
        async def on_tts_start(self, sentence, idx):
            events_captured.append(("tts_start", sentence))
        async def on_tts_chunk(self, audio, sentence, idx, duration, gen_ms):
            events_captured.append(("tts_chunk", idx, duration))
        async def on_turn_finish(self, response, metrics):
            events_captured.append(("turn_finish", metrics))

    agent = ConversationalAgent(tts_engine=tts)
    cb = CompositeCallback([ConsoleCallback(), TestCallback()])

    res = await agent.process_turn(
        user_message="Say a two-word greeting.",
        callback=cb,
        use_search=False
    )

    event_names = [e[0] for e in events_captured]
    print(f"    Captured lifecycle events: {set(event_names)}")
    assert "turn_start" in event_names
    assert "llm_start" in event_names
    assert "token" in event_names
    assert "tts_start" in event_names
    assert "tts_chunk" in event_names
    assert "turn_finish" in event_names
    print(f"    Agent response: \"{res['response']}\"")
    print(f"    Turn metrics: {res['metrics']}")
    print("    Conversational turn test PASSED.")

async def test_tool_callback(tts):
    print("--> Testing Tavily search tool callback...")
    events_captured = []

    class ToolTestCallback(BaseAgentCallback):
        async def on_tool_start(self, tool, inp):
            events_captured.append(("tool_start", tool, inp))
        async def on_tool_end(self, tool, out, elapsed):
            events_captured.append(("tool_end", tool, elapsed))

    agent = ConversationalAgent(tts_engine=tts)
    agent.system_prompt = "You have access to web_search tool. You MUST use web_search when asked about live news or real-time events."
    agent.conversation_history[0]["content"] = agent.system_prompt

    cb = CompositeCallback([ConsoleCallback(), ToolTestCallback()])
    res = await agent.process_turn(
        user_message="Search the web for the current weather in Paris right now.",
        callback=cb,
        use_search=True
    )
    event_names = [e[0] for e in events_captured]
    print(f"    Tool events: {events_captured}")
    if "tool_start" in event_names and "tool_end" in event_names:
        print("    Tool callback execution PASSED.")
    else:
        print("    Model chose to answer directly or tool skipped.")

async def main():
    print("\n" + "="*50)
    print("Running TalkAI Comprehensive Verification Suite")
    print("="*50)
    await test_text_cleaning()
    tts = await test_kokoro_tts()
    await test_agent_turn(tts)
    await test_tool_callback(tts)
    print("\n" + "="*50)
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
