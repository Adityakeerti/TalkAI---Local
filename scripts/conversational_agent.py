#!/usr/bin/env python3
"""
TalkAI - Conversational Voice Agent with Kokoro TTS and .env LLM Callbacks
--------------------------------------------------------------------------
Features:
- Ultra-low latency voice-to-voice / text-to-voice conversational agent.
- Kokoro ONNX TTS integration with multiple realistic voices and speed controls.
- Full LLM and tool lifecycle callbacks (on_llm_start, on_token, on_tool_start,
  on_tool_end, on_tts_start, on_tts_chunk, on_turn_finish).
- Streaming LLM generation (Gemma-4-E2B / GPT-OSS models) with sentence-level
  concurrent TTS audio synthesis.
- Tavily real-time web search tool integration with automatic tool calling.
- Integrated FastAPI and WebSocket server serving interactive browser test.html.
- Audio transcription via Groq Whisper (whisper-large-v3-turbo) for full cross-browser
  microphone support.
- Standalone CLI mode with system speaker playback via sounddevice.
"""

import os
import sys
import time
import json
import re
import io
import base64
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

# Third-party dependencies
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from groq import Groq
from openai import AsyncOpenAI
from tavily import TavilyClient

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TalkAI")

# Directory paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "models" / "kokoro"
ONNX_PATH = MODELS_DIR / "kokoro-v0_19.onnx"
VOICES_PATH = MODELS_DIR / "voices.bin"

# ---------------------------------------------------------------------------
# Text Normalization for Natural Speech Delivery
# ---------------------------------------------------------------------------

def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, code blocks, URLs, and symbols so the text
    sounds natural when read aloud by Kokoro TTS.
    Also strips discrete full-duplex action tokens ('listen') and action prefixes.
    """
    if not text:
        return ""

    # Full-Duplex action token handling:
    # If the text is strictly a yield action token ('listen')
    t_raw = text.strip()
    if t_raw.lower() in ("listen", "listen.", "listen!", "<listen>", "[listen]"):
        return ""

    # Strip leading duplex action tokens and filler prefixes ONLY when acting as standalone prefix
    # Preserve legitimate verb phrases like 'Listen to...', 'Listen carefully...'
    text = re.sub(r"^listen[\s,\.!\:\-]+(?!(?:to|carefully|closely|in)\b)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:for\s+you|for\s+safety)[\s,\.!\:\-]+", "", text, flags=re.IGNORECASE).strip()

    # Capitalize the leading character if stripping exposed a lowercase start
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if not text or text.lower() in ("listen", "listen.", "listen!", "<listen>"):
        return ""

    # Strip markdown code blocks
    text = re.sub(r"```[\s\S]*?```", " [code omitted] ", text)
    # Strip inline code
    text = re.sub(r"`[^`]*`", " ", text)
    # Strip markdown table formatting and delimiters
    text = re.sub(r"\|[\s\-:]+\|", " ", text)
    text = re.sub(r"\|", ", ", text)
    # Convert markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove raw URLs
    text = re.sub(r"https?://\S+", "", text)
    # Expand units and symbols for natural pronunciation
    text = re.sub(r"°C\b", " degrees Celsius", text)
    text = re.sub(r"°F\b", " degrees Fahrenheit", text)
    text = re.sub(r"%\b", " percent", text)
    text = re.sub(r"&", " and ", text)
    # Remove emojis and non-ascii decorative symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove bold, italics, headers, blockquotes, tildes
    text = re.sub(r"[*_~#>]", "", text)
    # Remove bullet symbols
    text = re.sub(r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE)
    # Replace multiple newlines or spaces with single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentenceChunker:
    """
    Two-stage adaptive streaming sentence chunker for ultra-low TTFA (<800ms):
    - Stage 1 (First Chunk): Emits on first clause break (3-6 words / 18-35 chars) or punctuation
      so Kokoro TTS starts synthesizing immediately while LLM continues generating.
    - Stage 2 (Subsequent Chunks): Emits on full terminal punctuation (.!?) to preserve
      natural prosody and intonation with zero stutter or buffer starvation.
    """
    def __init__(self, first_min_words: int = 3, first_max_words: int = 6):
        self.first_min_words = first_min_words
        self.first_max_words = first_max_words
        self.buffer = ""
        self.is_first_chunk = True
        self.terminal_re = re.compile(r"((?<!\d)[.!?]+(?!\d)(?:\s+|$|\n))")
        self.clause_re = re.compile(r"([,;:\u2014]|\s-\s)")

    def add_token(self, token: str) -> List[str]:
        self.buffer += token
        ready = []

        if self.is_first_chunk:
            words = self.buffer.strip().split()
            word_count = len(words)

            # 1. Immediate terminal punctuation (.!?)
            m_term = self.terminal_re.search(self.buffer)
            if m_term and word_count >= 1:
                end_pos = m_term.end()
                chunk = self.buffer[:end_pos].strip()
                if chunk:
                    ready.append(chunk)
                    self.buffer = self.buffer[end_pos:]
                    self.is_first_chunk = False
                    return ready

            # 2. Early clause break (,:;-) in Goldilocks zone (>= 3 words or >= 18 chars)
            m_clause = self.clause_re.search(self.buffer)
            if m_clause:
                end_pos = m_clause.end()
                sub = self.buffer[:end_pos].strip()
                sub_words = sub.split()
                if len(sub_words) >= self.first_min_words or len(sub) >= 18:
                    ready.append(sub)
                    self.buffer = self.buffer[end_pos:]
                    self.is_first_chunk = False
                    return ready

            # 3. Soft word cap: if no punctuation seen after 5-6 words (>= 26 chars)
            if word_count >= self.first_max_words and len(self.buffer.strip()) >= 26:
                space_indices = [i for i, ch in enumerate(self.buffer) if ch.isspace()]
                if len(space_indices) >= (self.first_min_words + 1):
                    split_idx = space_indices[self.first_min_words]
                    chunk = self.buffer[:split_idx].strip()
                    if chunk:
                        ready.append(chunk)
                        self.buffer = self.buffer[split_idx:]
                        self.is_first_chunk = False
                        return ready

            return ready

        # Stage 2: Standard full sentence chunking for natural Kokoro prosody
        parts = self.terminal_re.split(self.buffer)
        if len(parts) > 1:
            for i in range(0, len(parts) - 1, 2):
                sentence = (parts[i] + parts[i + 1]).strip()
                if sentence:
                    ready.append(sentence)
            self.buffer = parts[-1]

        # Safety fallback for run-on sentences (> 120 chars)
        if len(self.buffer) > 120:
            for sep in [", ", "; ", " - ", " \u2014 "]:
                if sep in self.buffer:
                    idx = self.buffer.rfind(sep)
                    if idx > 30:
                        ready.append(self.buffer[:idx].strip())
                        self.buffer = self.buffer[idx + len(sep):]
                        break

        return ready

    def flush(self) -> List[str]:
        rem = self.buffer.strip()
        self.buffer = ""
        self.is_first_chunk = True
        return [rem] if rem else []


# ---------------------------------------------------------------------------
# Kokoro TTS Engine
# ---------------------------------------------------------------------------

class KokoroTTSEngine:
    """
    High-performance wrapper for Kokoro ONNX TTS with thread-safe async synthesis.
    """
    def __init__(self, model_path: Path = ONNX_PATH, voices_path: Path = VOICES_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro ONNX model not found at: {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices.bin not found at: {voices_path}")

        logger.info(f"Loading Kokoro TTS model from {model_path}...")
        t0 = time.time()
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.available_voices = self.kokoro.get_voices()
        logger.info(f"Kokoro loaded in {(time.time() - t0)*1000:.1f}ms. Available voices: {self.available_voices}")

        # Human-friendly voice labels
        self.voice_labels = {
            "af_sarah": "Sarah (US Female - Warm & Clear)",
            "af_bella": "Bella (US Female - Expressive)",
            "af_nicole": "Nicole (US Female - Calm)",
            "af_sky": "Sky (US Female - Cheerful)",
            "am_adam": "Adam (US Male - Deep & Natural)",
            "am_michael": "Michael (US Male - Professional)",
            "bf_emma": "Emma (UK Female - Crisp)",
            "bf_isabella": "Isabella (UK Female - Soft)",
            "bm_george": "George (UK Male - Authoritative)",
            "bm_lewis": "Lewis (UK Male - Friendly)",
            "af": "Default Female (US)",
        }

    def get_voices(self) -> List[Dict[str, str]]:
        return [
            {"id": v, "label": self.voice_labels.get(v, v)}
            for v in self.available_voices
        ]

    def _sync_synthesize(self, text: str, voice: str, speed: float) -> tuple[bytes, float, int]:
        """Synchronous Kokoro synthesis in thread worker."""
        if voice not in self.available_voices:
            voice = "af_sarah" if "af_sarah" in self.available_voices else self.available_voices[0]

        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        duration = len(samples) / sample_rate

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        return wav_bytes, duration, sample_rate

    async def synthesize(self, text: str, voice: str = "af_sarah", speed: float = 1.0) -> tuple[bytes, str, float, int]:
        """
        Asynchronously synthesizes speech for the given text.
        Returns (wav_bytes, base64_data_uri, duration_seconds, sample_rate).
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            return b"", "", 0.0, 24000

        wav_bytes, duration, sample_rate = await asyncio.to_thread(
            self._sync_synthesize, cleaned_text, voice, speed
        )
        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        data_uri = f"data:audio/wav;base64,{b64_audio}"
        return wav_bytes, data_uri, duration, sample_rate


# ---------------------------------------------------------------------------
# LLM & Agent Callbacks System
# ---------------------------------------------------------------------------

class BaseAgentCallback:
    """Base callback handler defining the complete lifecycle hooks."""
    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        pass

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]):
        pass

    async def on_tool_end(self, tool_name: str, tool_output: str, elapsed_ms: float):
        pass

    async def on_llm_start(self, model: str, message_count: int):
        pass

    async def on_llm_token(self, token: str):
        pass

    async def on_llm_end(self, full_text: str, elapsed_ms: float):
        pass

    async def on_tts_start(self, sentence: str, segment_index: int):
        pass

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        pass

    async def on_turn_finish(self, full_response: str, metrics: Dict[str, Any]):
        pass

    async def on_error(self, error_message: str):
        pass


class ConsoleCallback(BaseAgentCallback):
    """Logs colorized, structured lifecycle events to stdout/terminal."""
    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        logger.info(f"\033[1;36m[USER]\033[0m {user_input}")

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]):
        logger.info(f"\033[1;33m[TOOL CALL: {tool_name}]\033[0m input: {tool_input}")

    async def on_tool_end(self, tool_name: str, tool_output: str, elapsed_ms: float):
        preview = tool_output[:120].replace('\n', ' ') + "..." if len(tool_output) > 120 else tool_output
        logger.info(f"\033[1;33m[TOOL DONE: {tool_name}]\033[0m in {elapsed_ms:.1f}ms: {preview}")

    async def on_llm_start(self, model: str, message_count: int):
        logger.info(f"\033[1;34m[LLM START]\033[0m model: {model} (context: {message_count} messages)")

    async def on_llm_end(self, full_text: str, elapsed_ms: float):
        logger.info(f"\033[1;34m[LLM FINISH]\033[0m in {elapsed_ms:.1f}ms ({len(full_text.split())} words)")

    async def on_tts_start(self, sentence: str, segment_index: int):
        logger.info(f"\033[1;35m[TTS SYNTH START #{segment_index}]\033[0m \"{sentence}\"")

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        logger.info(f"\033[1;32m[TTS CHUNK #{segment_index}]\033[0m audio: {duration_sec:.2f}s (gen: {gen_time_ms:.1f}ms, rtf: {gen_time_ms/(duration_sec*1000):.2f})")

    async def on_turn_finish(self, full_response: str, metrics: Dict[str, Any]):
        ttft = metrics.get('ttft_ms', 0)
        ttfa = metrics.get('ttfa_ms', 0)
        total = metrics.get('total_ms', 0)
        logger.info(f"\033[1;32m[TURN COMPLETE]\033[0m TTFT: {ttft:.1f}ms | TTFA: {ttfa:.1f}ms | Total: {total:.1f}ms")

    async def on_error(self, error_message: str):
        logger.error(f"\033[1;31m[ERROR]\033[0m {error_message}")


class WebSocketCallback(BaseAgentCallback):
    """Dispatches JSON telemetry and audio chunks directly to browser WebSocket."""
    def __init__(self, websocket: WebSocket):
        self.ws = websocket

    async def _send(self, payload: Dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception as e:
            logger.debug(f"WebSocket send failed: {e}")

    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        await self._send({"type": "turn_start", "input": user_input, "metadata": metadata})

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]):
        await self._send({"type": "tool_start", "tool": tool_name, "input": tool_input})

    async def on_tool_end(self, tool_name: str, tool_output: str, elapsed_ms: float):
        await self._send({"type": "tool_end", "tool": tool_name, "output": tool_output, "elapsed_ms": elapsed_ms})

    async def on_llm_start(self, model: str, message_count: int):
        await self._send({"type": "llm_start", "model": model, "message_count": message_count})

    async def on_llm_token(self, token: str):
        await self._send({"type": "token", "token": token})

    async def on_llm_end(self, full_text: str, elapsed_ms: float):
        await self._send({"type": "llm_end", "full_text": full_text, "elapsed_ms": elapsed_ms})

    async def on_tts_start(self, sentence: str, segment_index: int):
        await self._send({"type": "tts_start", "sentence": sentence, "segment_index": segment_index})

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        await self._send({
            "type": "tts_chunk",
            "audio": audio_data_uri,
            "sentence": sentence,
            "segment_index": segment_index,
            "duration": duration_sec,
            "gen_time_ms": gen_time_ms
        })

    async def on_turn_finish(self, full_response: str, metrics: Dict[str, Any]):
        await self._send({"type": "turn_finish", "full_response": full_response, "metrics": metrics})

    async def on_error(self, error_message: str):
        await self._send({"type": "error", "error": error_message})


class CompositeCallback(BaseAgentCallback):
    """Multiplexes callbacks to multiple handlers."""
    def __init__(self, handlers: List[BaseAgentCallback]):
        self.handlers = handlers

    async def on_turn_start(self, user_input: str, metadata: Dict[str, Any]):
        for h in self.handlers:
            await h.on_turn_start(user_input, metadata)

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]):
        for h in self.handlers:
            await h.on_tool_start(tool_name, tool_input)

    async def on_tool_end(self, tool_name: str, tool_output: str, elapsed_ms: float):
        for h in self.handlers:
            await h.on_tool_end(tool_name, tool_output, elapsed_ms)

    async def on_llm_start(self, model: str, message_count: int):
        for h in self.handlers:
            await h.on_llm_start(model, message_count)

    async def on_llm_token(self, token: str):
        for h in self.handlers:
            await h.on_llm_token(token)

    async def on_llm_end(self, full_text: str, elapsed_ms: float):
        for h in self.handlers:
            await h.on_llm_end(full_text, elapsed_ms)

    async def on_tts_start(self, sentence: str, segment_index: int):
        for h in self.handlers:
            await h.on_tts_start(sentence, segment_index)

    async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                           duration_sec: float, gen_time_ms: float):
        for h in self.handlers:
            await h.on_tts_chunk(audio_data_uri, sentence, segment_index, duration_sec, gen_time_ms)

    async def on_turn_finish(self, full_response: str, metrics: Dict[str, Any]):
        for h in self.handlers:
            await h.on_turn_finish(full_response, metrics)

    async def on_error(self, error_message: str):
        for h in self.handlers:
            await h.on_error(error_message)


# ---------------------------------------------------------------------------
# Tavily Search Tool Callback
# ---------------------------------------------------------------------------

class TavilySearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    async def search(self, query: str, max_results: int = 2) -> str:
        if not self.client:
            return "Tavily search is not configured (missing TAVILY_API_KEY in .env)."
        try:
            t0 = time.time()
            res = await asyncio.to_thread(self.client.search, query=query, max_results=max_results)
            snippets = []
            for item in res.get("results", []):
                title = item.get("title", "")
                content = item.get("content", "")
                snippets.append(f"[{title}]: {content}")
            result_str = "\n\n".join(snippets)
            return result_str or "No relevant search results found."
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# Conversational Agent
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are TalkAI, a natural, articulate, warm, and highly expressive conversational voice assistant like ChatGPT Voice, powered by Kokoro TTS.
Guidelines for your spoken responses:
1. Speak in a natural, friendly, and direct conversational tone suitable for live voice delivery.
2. Keep everyday answers concise, engaging, and clear (usually 2 to 4 sentences) so conversations flow effortlessly.
3. When the user asks for an explanation, essay, story, or detailed thoughts, provide a rich, coherent, and complete response without arbitrary length limits.
4. Do NOT use markdown symbols, hashtags, asterisks, or bullet points in spoken responses because they will be read aloud by Kokoro TTS.
5. If you use numbers or abbreviations, write them naturally as words so they sound smooth when spoken.
6. Never start responses with 'listen', filler words, or robotic acknowledgments. Answer directly and warmly.
7. If the user asks about live events, current news, weather, or real-time facts, use the web_search tool to find accurate information.
"""

AVAILABLE_MODELS = [
    {"id": "local/gemma-4-e2b", "name": "Fine-Tuned Gemma-4-E2B Duplex (llama-fork GPU)", "provider": "local_llama"},
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Groq)", "provider": "groq"},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B (Groq)", "provider": "groq"},
]

class LocalWhisperSTT:
    """100% local, offline Speech-to-Text running on CPU/CUDA with zero cloud calls."""
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading local offline faster-whisper model ({self.model_size})...")
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info("Local faster-whisper loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize local faster-whisper: {e}")
                self._model = False
        return self._model if self._model is not False else None

    def transcribe(self, audio_bytes: bytes) -> str:
        model = self._get_model()
        if not model:
            return ""
        segments, _ = model.transcribe(io.BytesIO(audio_bytes))
        return " ".join(seg.text for seg in segments).strip()


class ConversationalAgent:
    """
    Full-duplex capable conversational agent orchestrating LLMs, tools, and Kokoro TTS.
    """
    def __init__(
        self,
        tts_engine: KokoroTTSEngine,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        default_model: str = "local/gemma-4-e2b",
        default_voice: str = "af_sarah",
        default_speed: float = 1.0,
    ):
        self.tts = tts_engine
        self.system_prompt = system_prompt
        self.default_model = default_model
        self.default_voice = default_voice
        self.default_speed = default_speed

        # API Clients from .env
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None
        self.tavily_tool = TavilySearchTool()

        # Multi-turn conversation history
        self.conversation_history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Cancellation / interruption signal
        self._current_task_cancel = False

    def clear_history(self, custom_system_prompt: Optional[str] = None):
        """Resets conversation history."""
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        logger.info("Conversation history cleared.")

    def interrupt(self):
        """Signals interruption to immediately halt in-flight turn."""
        self._current_task_cancel = True
        logger.info("Agent interrupted by user.")


    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribes user voice recording. Prioritizes local offline faster-whisper
        (zero cloud cost/quota), with fallback to Groq Whisper if needed.
        """
        # Try 100% local offline faster-whisper first
        try:
            if not hasattr(self, "_local_stt"):
                self._local_stt = LocalWhisperSTT()
            local_text = await asyncio.to_thread(self._local_stt.transcribe, audio_bytes)
            if local_text:
                logger.info(f"Local offline Whisper transcribed: '{local_text}'")
                return local_text
        except Exception as e:
            logger.debug(f"Local whisper fallback: {e}")

        # Fallback to Groq Whisper if local is unavailable
        if self.groq_client:
            try:
                transcription = await asyncio.to_thread(
                    self.groq_client.audio.transcriptions.create,
                    file=(filename, audio_bytes),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
                return str(transcription).strip()
            except Exception as e:
                logger.error(f"Cloud whisper failed: {e}")
                raise

        raise RuntimeError("No STT engine available for transcription.")

    async def process_turn(
        self,
        user_message: str,
        callback: BaseAgentCallback,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        model: Optional[str] = None,
        use_search: bool = True,
    ) -> Dict[str, Any]:
        """
        Orchestrates an entire conversational turn with LLM streaming & concurrent TTS.
        """
        self._current_task_cancel = False
        voice = voice or self.default_voice
        speed = speed or self.default_speed
        model = model or self.default_model

        start_time = time.time()
        ttft_recorded = False
        ttfa_recorded = False
        ttft_ms = 0.0
        ttfa_ms = 0.0

        await callback.on_turn_start(user_message, {
            "voice": voice,
            "speed": speed,
            "model": model,
            "use_search": use_search
        })

        # Append user turn
        self.conversation_history.append({"role": "user", "content": user_message})

        # Keep context window bounded (last 16 messages + system)
        if len(self.conversation_history) > 17:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-16:]

        is_local = model.startswith("local") or "gemma" in model.lower()

        if not is_local and not self.groq_client:
            err = "GROQ_API_KEY is missing in .env. Please set GROQ_API_KEY."
            await callback.on_error(err)
            return {"error": err}

        # Check for tool calling capability
        tools = None
        if use_search and not is_local and self.tavily_tool.client:
            tools = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the live web for current events, breaking news, weather, or real-time facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
                        },
                        "required": ["query"]
                    }
                }
            }]

        # Step 1: Query LLM (with tool check)
        await callback.on_llm_start(model, len(self.conversation_history))
        t_llm_start = time.time()

        try:
            # First check if tool is triggered
            tool_detected = False
            first_response = None
            if tools and not is_local:
                check_res = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=model,
                    messages=self.conversation_history,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=600
                )
                choice = check_res.choices[0]
                if choice.message.tool_calls:
                    tool_detected = True
                    for tool_call in choice.message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except Exception:
                            args = {"query": tool_call.function.arguments}

                        if func_name == "web_search":
                            query = args.get("query", user_message)
                            t_tool_start = time.time()
                            await callback.on_tool_start("web_search", {"query": query})
                            search_result = await self.tavily_tool.search(query)
                            elapsed_tool = (time.time() - t_tool_start) * 1000
                            await callback.on_tool_end("web_search", search_result, elapsed_tool)

                            # Append tool response
                            self.conversation_history.append(choice.message)
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": "web_search",
                                "content": search_result
                            })
                else:
                    first_response = choice.message.content

            # Step 2: Stream final answer to SentenceChunker and Kokoro TTS
            chunker = SentenceChunker()
            full_response_text = ""
            segment_index = 0

            # Queue for TTS audio tasks
            tts_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

            # Background worker that processes TTS chunks sequentially for smooth ordering
            async def tts_worker():
                nonlocal segment_index, ttfa_recorded, ttfa_ms
                while True:
                    item = await tts_queue.get()
                    if item is None:  # Sentinel
                        tts_queue.task_done()
                        break
                    idx, sentence = item
                    if self._current_task_cancel:
                        tts_queue.task_done()
                        break

                    t_synth_start = time.time()
                    await callback.on_tts_start(sentence, idx)
                    try:
                        wav_bytes, data_uri, duration, sr = await self.tts.synthesize(sentence, voice=voice, speed=speed)
                        gen_ms = (time.time() - t_synth_start) * 1000
                        if not ttfa_recorded:
                            ttfa_recorded = True
                            ttfa_ms = (time.time() - start_time) * 1000

                        if not self._current_task_cancel and data_uri:
                            await callback.on_tts_chunk(data_uri, sentence, idx, duration, gen_ms)
                    except Exception as e:
                        logger.error(f"TTS chunk synthesis error: {e}")
                        await callback.on_error(f"TTS error: {str(e)}")
                    finally:
                        tts_queue.task_done()

            worker_task = asyncio.create_task(tts_worker())

            if is_local:
                # Stream from local llama-server (Gemma-4-E2B with ChatGPT Voice experience)
                clean_history = []
                # Use configured system prompt or rich ChatGPT Voice prompt
                sys_content = self.system_prompt.strip() if self.system_prompt else DEFAULT_SYSTEM_PROMPT.strip()
                clean_history.append({"role": "system", "content": sys_content})
                for m in self.conversation_history[1:]:
                    if isinstance(m, dict) and "role" in m and "content" in m:
                        clean_history.append({"role": m["role"], "content": str(m["content"])})

                local_client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
                stream = await local_client.chat.completions.create(
                    model="local/gemma-4-e2b",
                    messages=clean_history,
                    stream=True,
                    max_tokens=300,
                    temperature=0.65,
                    top_p=0.9,
                    frequency_penalty=0.05,
                    presence_penalty=0.05,
                    stop=["<turn|>", "<|turn>", "<eos>"],
                    logit_bias={
                        "20923": -100,  # 'listen'
                        "10763": -100,  # ' listen'
                        "27752": -100   # 'Listen'
                    },
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "repeat_penalty": 1.15
                    }
                )

                async for chunk in stream:
                    if self._current_task_cancel:
                        break
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if not ttft_recorded:
                            ttft_recorded = True
                            ttft_ms = (time.time() - start_time) * 1000

                        full_response_text += delta
                        await callback.on_llm_token(delta)

                        sentences = chunker.add_token(delta)
                        for s in sentences:
                            cleaned_s = clean_text_for_speech(s)
                            if cleaned_s:
                                segment_index += 1
                                await tts_queue.put((segment_index, cleaned_s))

            elif first_response and not tool_detected:
                full_response_text = first_response
                # Simulate token stream
                words = first_response.split(" ")
                for i, w in enumerate(words):
                    if self._current_task_cancel:
                        break
                    tok = w + (" " if i < len(words) - 1 else "")
                    await callback.on_llm_token(tok)
                    sentences = chunker.add_token(tok)
                    for s in sentences:
                        cleaned_s = clean_text_for_speech(s)
                        if cleaned_s:
                            segment_index += 1
                            await tts_queue.put((segment_index, cleaned_s))
            else:
                # Stream from Groq
                stream = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=model,
                    messages=self.conversation_history,
                    stream=True,
                    max_tokens=250,
                    temperature=0.7,
                )

                for chunk in stream:
                    if self._current_task_cancel:
                        break
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if not ttft_recorded:
                            ttft_recorded = True
                            ttft_ms = (time.time() - start_time) * 1000

                        full_response_text += delta
                        await callback.on_llm_token(delta)

                        sentences = chunker.add_token(delta)
                        for s in sentences:
                            cleaned_s = clean_text_for_speech(s)
                            if cleaned_s:
                                segment_index += 1
                                await tts_queue.put((segment_index, cleaned_s))

            # Flush remaining buffer in chunker
            remaining_sentences = chunker.flush()
            for s in remaining_sentences:
                cleaned_s = clean_text_for_speech(s)
                if cleaned_s:
                    segment_index += 1
                    await tts_queue.put((segment_index, cleaned_s))

            # Wait for all TTS chunks to finish synthesizing
            await tts_queue.put(None)  # Sentinel to stop worker
            await worker_task

            elapsed_llm = (time.time() - t_llm_start) * 1000
            await callback.on_llm_end(full_response_text, elapsed_llm)

            # Record final assistant turn in history
            self.conversation_history.append({"role": "assistant", "content": full_response_text})

            total_ms = (time.time() - start_time) * 1000
            is_yield = full_response_text.strip().lower() in ("listen", "listen.", "<listen>", "[listen]")
            action = "YIELD (listen)" if is_yield else "SPEAK"
            metrics = {
                "ttft_ms": ttft_ms,
                "ttfa_ms": ttfa_ms,
                "total_ms": total_ms,
                "segments": segment_index,
                "model": model,
                "model_detail": "Fine-Tuned Gemma-4-E2B (Merged GGUF via llama-fork GPU)" if is_local else model,
                "backend": "llama-fork GPU (:8080)" if is_local else "Groq Cloud API",
                "action": action,
                "voice": voice,
            }

            await callback.on_turn_finish(full_response_text, metrics)
            return {
                "response": full_response_text,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Error in conversational agent turn: {e}", exc_info=True)
            await callback.on_error(str(e))
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI & WebSocket Web Server
# ---------------------------------------------------------------------------

def create_app(agent: ConversationalAgent) -> FastAPI:
    app = FastAPI(title="TalkAI Conversational Agent", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Path to test.html (can be in root or scripts)
    html_file = ROOT_DIR / "test.html"
    if not html_file.exists():
        html_file = SCRIPT_DIR / "test.html"

    @app.get("/")
    @app.get("/test.html")
    async def get_test_page():
        if html_file.exists():
            return FileResponse(str(html_file), media_type="text/html")
        return HTMLResponse("<h1>test.html not found</h1>", status_code=404)

    @app.get("/api/voices")
    async def get_voices():
        return {"voices": agent.tts.get_voices()}

    @app.get("/api/models")
    async def get_models():
        return {"models": AVAILABLE_MODELS}

    @app.get("/api/status")
    async def get_status():
        return {
            "status": "online",
            "kokoro_voices_loaded": len(agent.tts.available_voices),
            "groq_configured": bool(agent.groq_client),
            "tavily_configured": bool(agent.tavily_tool.client),
            "default_model": agent.default_model,
            "default_voice": agent.default_voice
        }

    @app.post("/api/test_tts")
    async def test_tts(payload: Dict[str, Any]):
        voice = payload.get("voice", "af_sarah")
        speed = float(payload.get("speed", 1.0))
        text = payload.get("text", "Hello there! Kokoro TTS is active and ready.")
        wav_bytes, data_uri, duration, sr = await agent.tts.synthesize(text, voice=voice, speed=speed)
        return {
            "audio": data_uri,
            "duration": duration,
            "sample_rate": sr,
            "text": text
        }

    @app.post("/api/transcribe")
    async def transcribe_audio(file: UploadFile = File(...)):
        try:
            content = await file.read()
            filename = file.filename or "recording.webm"
            if "." not in filename:
                filename = f"{filename}.webm"
            logger.info(f"Received audio upload: {len(content)} bytes, filename: {filename}")
            text = await agent.transcribe_audio(content, filename=filename)
            logger.info(f"Whisper transcription result: '{text}'")
            return {"text": text}
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/chat")
    async def websocket_chat_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_callback = WebSocketCallback(websocket)
        composite_cb = CompositeCallback([ConsoleCallback(), ws_callback])

        logger.info("Client connected to conversational WebSocket.")

        active_turn_task: Optional[asyncio.Task] = None

        try:
            while True:
                data_text = await websocket.receive_text()
                try:
                    payload = json.loads(data_text)
                except Exception:
                    payload = {"type": "message", "text": data_text}

                msg_type = payload.get("type", "message")

                if msg_type == "interrupt":
                    agent.interrupt()
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    await websocket.send_text(json.dumps({"type": "interrupted"}))
                    continue

                if msg_type == "clear_history":
                    agent.interrupt()
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    custom_prompt = payload.get("system_prompt")
                    agent.clear_history(custom_prompt)
                    await websocket.send_text(json.dumps({"type": "history_cleared"}))
                    continue

                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                if msg_type == "message":
                    user_text = payload.get("text", "").strip()
                    if not user_text:
                        continue

                    # Cancel any prior active turn
                    if active_turn_task and not active_turn_task.done():
                        agent.interrupt()
                        active_turn_task.cancel()

                    voice = payload.get("voice", agent.default_voice)
                    speed = float(payload.get("speed", agent.default_speed))
                    model = payload.get("model", agent.default_model)
                    use_search = bool(payload.get("use_search", True))

                    # Spawn turn as concurrent background task so WebSocket loop remains responsive
                    active_turn_task = asyncio.create_task(
                        agent.process_turn(
                            user_message=user_text,
                            callback=composite_cb,
                            voice=voice,
                            speed=speed,
                            model=model,
                            use_search=use_search
                        )
                    )

        except WebSocketDisconnect:
            logger.info("Client disconnected from WebSocket.")
            if active_turn_task and not active_turn_task.done():
                active_turn_task.cancel()
        except Exception as e:
            logger.error(f"WebSocket session error: {e}")
            if active_turn_task and not active_turn_task.done():
                active_turn_task.cancel()

    return app


# ---------------------------------------------------------------------------
# Interactive Terminal CLI Mode
# ---------------------------------------------------------------------------

async def run_cli_loop(agent: ConversationalAgent, voice: str, speed: float, model: str):
    import sounddevice as sd

    console_cb = ConsoleCallback()
    print("\n" + "=" * 65)
    print("TalkAI Conversational Agent (CLI Mode)")
    print(f"Model: {model} | Voice: {voice} | Speed: {speed}x")
    print("Type your message and press Enter. (Type 'quit' or 'exit' to exit)")
    print("=" * 65 + "\n")

    class SpeakerAudioCallback(BaseAgentCallback):
        async def on_tts_chunk(self, audio_data_uri: str, sentence: str, segment_index: int,
                               duration_sec: float, gen_time_ms: float):
            # Decode wav data uri
            b64_part = audio_data_uri.split(",", 1)[-1]
            raw_wav = base64.b64decode(b64_part)
            data, sr = sf.read(io.BytesIO(raw_wav))
            sd.play(data, sr)
            sd.wait()

    composite_cb = CompositeCallback([console_cb, SpeakerAudioCallback()])

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            await agent.process_turn(
                user_message=user_input,
                callback=composite_cb,
                voice=voice,
                speed=speed,
                model=model,
                use_search=True
            )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TalkAI Conversational Agent with Kokoro TTS")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for web server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")
    parser.add_argument("--voice", default="af_sarah", help="Default Kokoro voice ID (e.g. af_sarah, am_adam, bf_emma)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--model", default="local/gemma-4-e2b", help="Default LLM model (default: local/gemma-4-e2b)")
    parser.add_argument("--web", action="store_true", help="Run web server (default)")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal CLI with speaker output instead of server")
    args = parser.parse_args()

    # Initialize Kokoro Engine
    try:
        tts_engine = KokoroTTSEngine(ONNX_PATH, VOICES_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro TTS engine: {e}")
        sys.exit(1)

    agent = ConversationalAgent(
        tts_engine=tts_engine,
        default_model=args.model,
        default_voice=args.voice,
        default_speed=args.speed
    )

    if args.cli:
        asyncio.run(run_cli_loop(agent, args.voice, args.speed, args.model))
    else:
        app = create_app(agent)
        logger.info(f"Starting TalkAI Web Server at http://{args.host}:{args.port}/")
        logger.info(f"Open http://localhost:{args.port}/ to view the test interface.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
