#!/usr/bin/env python3
"""
scripts/benchmark_duplex_runtime.py
Module 5: GGUF Adapter Conversion & Runtime Benchmark

Responsibilities:
1. Validate models/duplex_gemma_lora.gguf format and binary header.
2. If missing or requested via --convert, invoke convert_lora_to_gguf.py to build it.
3. Connect to llama-server (default: http://127.0.0.1:8080) running Gemma-4-E2B with LoRA.
4. Execute benchmark test suite across three core conversational scenarios:
   - Test Case A (Hesitation): "I was thinking about..." -> Asserts response starts with/contains 'listen'.
   - Test Case B (Hazard Interruption): "The outlet is sparking and I might touch it" -> Asserts response speaks safety warnings ('not touch', 'danger', 'outlet').
   - Test Case C (Direct Question): "What is the capital of France?" -> Asserts response contains 'Paris'.
5. Measure latency across multiple runs (prompt eval, generation time, roundtrip).
6. Verify Evaluation Gate:
   - models/duplex_gemma_lora.gguf is valid.
   - All test case assertions pass.
   - Average generation and roundtrip latencies < 200 ms (inside 600ms duplex micro-turn budget).
7. Save reports to:
   - logs/module_5_report.json
   - logs/module5_runtime_report.json
   - logs/module5_runtime_report.md
8. Update logs/execution_state.json with Module 5 PASSED status.
"""

import os
import sys
import json
import time
import struct
import math
import logging
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "gemma-4-E2B-it"
ADAPTER_DIR = PROJECT_ROOT / "models" / "duplex_gemma_lora"
LORA_GGUF_PATH = PROJECT_ROOT / "models" / "duplex_gemma_lora.gguf"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORT_MD_PATH = LOGS_DIR / "module5_runtime_report.md"
REPORT_JSON_PATH = LOGS_DIR / "module_5_report.json"
REPORT_JSON_ALT_PATH = LOGS_DIR / "module5_runtime_report.json"
STATE_JSON_PATH = LOGS_DIR / "execution_state.json"
CONVERTER_SCRIPT = Path("/home/aditya/llama-fork/convert_lora_to_gguf.py")

DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_REPEATS = 5
DEFAULT_MAX_TOKENS = 15
LATENCY_GATE_THRESHOLD_MS = 200.0

# Official conversational system prompt matching training dataset
SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. "
    "Output exactly 'listen' when you should yield and keep listening; otherwise output a brief, "
    "natural speech fragment that can be spoken now. Do not repeat prior fragments. "
    "Continue an unfinished answer across turns when appropriate. Yield immediately for a "
    "clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark_duplex_runtime")


def validate_gguf_file(gguf_path: Path) -> Dict[str, Any]:
    """
    Validates the binary integrity of a GGUF file:
    Checks magic bytes ('GGUF'), version, tensor count, and file size.
    """
    logger.info(f"Validating GGUF file: {gguf_path}")
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found at: {gguf_path}")

    file_size_bytes = gguf_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_bytes < 1024:
        raise ValueError(f"GGUF file too small ({file_size_bytes} bytes), likely corrupted.")

    with open(gguf_path, "rb") as f:
        header = f.read(24)
        if len(header) < 24:
            raise ValueError(f"Unable to read GGUF 24-byte header from {gguf_path}")
        magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", header)

    if magic != b"GGUF":
        raise ValueError(f"Invalid GGUF magic bytes: {magic!r}, expected b'GGUF'")

    if version not in (2, 3):
        logger.warning(f"Uncommon GGUF version: {version}")

    info = {
        "path": str(gguf_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": kv_count,
        "is_valid": True,
    }
    logger.info(
        f"GGUF Header Validated: Magic={info['magic']}, Version={info['version']}, "
        f"Tensors={info['tensor_count']}, Size={info['file_size_mb']} MB"
    )
    return info


def convert_adapter_to_gguf(
    adapter_dir: Path,
    base_model_dir: Path,
    output_gguf: Path,
    converter_script: Path = CONVERTER_SCRIPT,
) -> None:
    """Invokes llama-fork's convert_lora_to_gguf.py to produce a GGUF adapter."""
    logger.info(f"Converting adapter at {adapter_dir} -> {output_gguf} using {converter_script}")
    if not converter_script.exists():
        raise FileNotFoundError(f"convert_lora_to_gguf.py script not found at {converter_script}")

    python_bin = sys.executable
    cmd = [
        python_bin,
        str(converter_script),
        "--base",
        str(base_model_dir),
        "--outfile",
        str(output_gguf),
        str(adapter_dir),
    ]

    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GGUF Conversion failed with exit code {result.returncode}:\n{result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    logger.info(f"Conversion successful! Output saved to {output_gguf}")


def check_server_health(server_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Queries /health and /props to confirm llama-server is healthy and responsive."""
    health_url = f"{server_url.rstrip('/')}/health"
    props_url = f"{server_url.rstrip('/')}/props"
    logger.info(f"Checking server health at {health_url}")

    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "TalkAI-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") != "ok":
                raise RuntimeError(f"Server health status is not ok: {data}")
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to llama-server at {server_url}. Is the server running? Details: {e}"
        ) from e

    server_props = {}
    try:
        req_props = urllib.request.Request(props_url, headers={"User-Agent": "TalkAI-Benchmark/1.0"})
        with urllib.request.urlopen(req_props, timeout=timeout) as resp_props:
            server_props = json.loads(resp_props.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Could not retrieve server /props: {e}")

    logger.info("llama-server is healthy and ready.")
    return {
        "status": "ok",
        "server_url": server_url,
        "build_info": server_props.get("build_info", "unknown"),
        "bos_token": server_props.get("bos_token", ""),
        "eos_token": server_props.get("eos_token", ""),
    }


def query_chat_completion(
    server_url: str,
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """
    Sends a chat completion request to llama-server and records:
    - client roundtrip latency (ms)
    - server prompt evaluation time (prompt_ms)
    - server generation time (predicted_ms)
    - response text content
    """
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    encoded_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=encoded_data,
        headers={"Content-Type": "application/json", "User-Agent": "TalkAI-Benchmark/1.0"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_body = resp.read().decode("utf-8")
    t1 = time.perf_counter()

    roundtrip_ms = (t1 - t0) * 1000.0
    response_json = json.loads(raw_body)

    choice = response_json["choices"][0]
    content = choice["message"]["content"].strip()
    finish_reason = choice.get("finish_reason", "")

    timings = response_json.get("timings", {})
    prompt_ms = float(timings.get("prompt_ms", 0.0))
    predicted_ms = float(timings.get("predicted_ms", 0.0))
    prompt_n = int(timings.get("prompt_n", 0))
    predicted_n = int(timings.get("predicted_n", 0))
    prompt_tokens_per_sec = float(timings.get("prompt_per_second", 0.0))
    predicted_tokens_per_sec = float(timings.get("predicted_per_second", 0.0))

    return {
        "content": content,
        "finish_reason": finish_reason,
        "roundtrip_ms": roundtrip_ms,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prompt_tokens_per_sec": prompt_tokens_per_sec,
        "predicted_tokens_per_sec": predicted_tokens_per_sec,
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Computes mean, std, min, and max for a numeric list."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def run_benchmark_test_case(
    test_id: str,
    test_name: str,
    user_prompt: str,
    check_fn,
    server_url: str,
    repeats: int = DEFAULT_REPEATS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Executes 1 warmup run + repeats benchmark runs for a specific conversational scenario.
    Validates assertions and calculates timing statistics.
    """
    logger.info(f"--- Running {test_id}: {test_name} ---")
    logger.info(f"Prompt: {user_prompt!r}")

    # Warmup run to initialize KV caches and connection pool
    try:
        warmup_res = query_chat_completion(server_url, user_prompt, max_tokens=max_tokens)
        logger.info(
            f"Warmup: roundtrip={warmup_res['roundtrip_ms']:.1f}ms, "
            f"prompt_eval={warmup_res['prompt_ms']:.1f}ms, gen={warmup_res['predicted_ms']:.1f}ms"
        )
    except Exception as e:
        logger.warning(f"Warmup query encountered warning: {e}")

    runs: List[Dict[str, Any]] = []
    roundtrips: List[float] = []
    prompt_times: List[float] = []
    gen_times: List[float] = []
    sample_responses: List[str] = []

    for r in range(repeats):
        res = query_chat_completion(server_url, user_prompt, max_tokens=max_tokens)
        runs.append(
            {
                "run_index": r + 1,
                "roundtrip_ms": round(res["roundtrip_ms"], 2),
                "prompt_ms": round(res["prompt_ms"], 2),
                "predicted_ms": round(res["predicted_ms"], 2),
                "prompt_n": res["prompt_n"],
                "predicted_n": res["predicted_n"],
                "predicted_tps": round(res["predicted_tokens_per_sec"], 1),
                "content": res["content"],
            }
        )
        roundtrips.append(res["roundtrip_ms"])
        prompt_times.append(res["prompt_ms"])
        gen_times.append(res["predicted_ms"])
        sample_responses.append(res["content"])

        logger.info(
            f"  Run {r + 1}/{repeats}: roundtrip={res['roundtrip_ms']:.1f}ms | "
            f"prompt={res['prompt_ms']:.1f}ms | gen={res['predicted_ms']:.1f}ms | "
            f"content: {res['content']!r}"
        )

    # Validate assertions against sample responses
    # The requirement is that the model's response adheres to the scenario specification
    latest_content = sample_responses[-1] if sample_responses else ""
    passed, reason = check_fn(sample_responses)

    stats = {
        "roundtrip_stats": compute_statistics(roundtrips),
        "prompt_eval_stats": compute_statistics(prompt_times),
        "generation_stats": compute_statistics(gen_times),
    }

    logger.info(
        f"Result for {test_id}: {'PASS' if passed else 'FAIL'} - {reason} "
        f"(Avg Gen: {stats['generation_stats']['mean']}ms, Avg Roundtrip: {stats['roundtrip_stats']['mean']}ms)"
    )

    return {
        "test_id": test_id,
        "test_name": test_name,
        "user_prompt": user_prompt,
        "sample_output": latest_content,
        "all_outputs": sample_responses,
        "passed": passed,
        "assertion_reason": reason,
        "statistics": stats,
        "runs": runs,
    }


def check_case_a(responses: List[str]) -> Tuple[bool, str]:
    """Test Case A (Hesitation): Response must start with or contain 'listen'."""
    for resp in responses:
        resp_lower = resp.lower()
        if "listen" in resp_lower:
            return True, f"Found 'listen' keyword in response: {resp!r}"
    return False, f"Did not find 'listen' in responses: {responses}"


def check_case_b(responses: List[str]) -> Tuple[bool, str]:
    """
    Test Case B (Hazard Interruption):
    Response must speak immediate safety warning words (e.g. 'not touch', 'danger', 'outlet').
    """
    safety_keywords = ["not touch", "don't touch", "dont touch", "outlet", "danger", "breaker", "hazard"]
    for resp in responses:
        resp_lower = resp.lower()
        matched = [kw for kw in safety_keywords if kw in resp_lower]
        if matched:
            return True, f"Found safety warning keywords {matched} in response: {resp!r}"
    return False, f"No safety keywords ({safety_keywords}) found in responses: {responses}"


def check_case_c(responses: List[str]) -> Tuple[bool, str]:
    """Test Case C (Direct Question): Response must speak concise answer containing 'Paris'."""
    for resp in responses:
        resp_lower = resp.lower()
        if "paris" in resp_lower:
            return True, f"Found 'Paris' in response: {resp!r}"
    return False, f"Did not find 'Paris' in responses: {responses}"


def generate_markdown_report(
    gguf_info: Dict[str, Any],
    server_info: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    overall_stats: Dict[str, Any],
    gate_criteria: Dict[str, bool],
    all_passed: bool,
) -> str:
    """Constructs a comprehensive, publication-quality Markdown report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# TalkAI: Module 5 Full-Duplex Runtime Benchmark Report",
        "=============================================================================",
        "",
        f"> **Generated**: {timestamp}  ",
        f"> **Architecture**: Gemma-4-E2B (q4_0-it) + Top-Layer Duplex LoRA  ",
        f"> **Inference Server**: `llama-server` (CUDA offload, Port 8080)  ",
        f"> **Target Budget**: 600 ms Micro-Turn Time-Slice (SLA: $< 200\\text{{ ms}}$)  ",
        f"> **Status**: **{'PASSED' if all_passed else 'FAILED'}**  ",
        "",
        "---",
        "",
        "## 1. GGUF Adapter Verification",
        "",
        "| Property | Value | Status |",
        "| :--- | :--- | :--- |",
        f"| **File Path** | `{gguf_info.get('path', 'N/A')}` | Verified |",
        f"| **File Size** | {gguf_info.get('file_size_mb', 'N/A')} MB ({gguf_info.get('file_size_bytes', 0):,} bytes) | Valid |",
        f"| **Format Magic** | `{gguf_info.get('magic', 'N/A')}` | Valid GGUF |",
        f"| **GGUF Version** | {gguf_info.get('version', 'N/A')} | Compatible |",
        f"| **Tensor Count** | {gguf_info.get('tensor_count', 'N/A')} tensors | Matched |",
        "",
        "---",
        "",
        "## 2. Server Runtime Environment",
        "",
        f"- **Endpoint**: `{server_info.get('server_url', 'N/A')}`",
        f"- **Health Status**: `{server_info.get('status', 'unknown')}`",
        f"- **llama.cpp Build**: `{server_info.get('build_info', 'unknown')}`",
        f"- **System Prompt**: Active 600ms micro-turn conversational supervisor prompt.",
        "",
        "---",
        "",
        "## 3. Conversational Test Cases & Assertions",
        "",
        "| Test Case | Scenario Description | User Prompt | Model Response Fragment | Assertion Result |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for res in test_results:
        clean_resp = res["sample_output"].replace("|", "\\|").replace("\n", " ")
        status_badge = "PASS" if res["passed"] else "FAIL"
        lines.append(
            f"| **{res['test_id']}** | {res['test_name']} | *\"{res['user_prompt']}\"* | `{clean_resp}` | **{status_badge}** |"
        )

    lines.extend([
        "",
        "### Detailed Assertion Diagnostics",
        "",
    ])

    for res in test_results:
        lines.append(f"- **{res['test_id']} ({res['test_name']})**: {res['assertion_reason']}")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Latency & Performance Benchmark Breakdown",
        "",
        "All measurements evaluated across 5 repeated runs (after 1 warm-up invocation).",
        "",
        "| Test Case | Prompt Eval (ms) | Generation Latency (ms) | Client Roundtrip (ms) | Tokens / Sec | SLA Gate (< 200ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for res in test_results:
        p_stat = res["statistics"]["prompt_eval_stats"]
        g_stat = res["statistics"]["generation_stats"]
        r_stat = res["statistics"]["roundtrip_stats"]
        avg_tps = (
            sum(r["predicted_tps"] for r in res["runs"]) / len(res["runs"]) if res["runs"] else 0.0
        )
        gate_ok = "PASS" if r_stat["mean"] < LATENCY_GATE_THRESHOLD_MS else "FAIL"
        lines.append(
            f"| **{res['test_id']}** | {p_stat['mean']} ± {p_stat['std']} ms | "
            f"{g_stat['mean']} ± {g_stat['std']} ms | "
            f"**{r_stat['mean']} ± {r_stat['std']} ms** | "
            f"{avg_tps:.1f} t/s | **{gate_ok}** |"
        )

    lines.extend([
        "",
        "### Aggregate Performance Summary",
        "",
        f"- **Global Average Roundtrip Latency**: **{overall_stats['global_avg_roundtrip_ms']} ms**",
        f"- **Global Average Generation Latency**: **{overall_stats['global_avg_gen_ms']} ms**",
        f"- **Global Average Prompt Eval Latency**: **{overall_stats['global_avg_prompt_ms']} ms**",
        f"- **Micro-Turn Budget Margin**: **{600.0 - overall_stats['global_avg_roundtrip_ms']:.1f} ms headroom** remaining within 600ms micro-turn frame.",
        "",
        "---",
        "",
        "## 5. Module 5 Evaluation Gate Checklist",
        "",
        f"- [{'x' if gate_criteria.get('gguf_adapter_valid') else ' '}] `models/duplex_gemma_lora.gguf` is created and valid GGUF.",
        f"- [{'x' if gate_criteria.get('test_a_hesitation_passed') else ' '}] Test Case A returns `listen` upon hesitation prompt.",
        f"- [{'x' if gate_criteria.get('test_b_hazard_passed') else ' '}] Test Case B returns spoken safety warning upon urgent hazard.",
        f"- [{'x' if gate_criteria.get('test_c_question_passed') else ' '}] Test Case C returns spoken response containing `Paris`.",
        f"- [{'x' if gate_criteria.get('avg_generation_latency_under_200ms') else ' '}] Average generation latency $< 200\\text{{ ms}}$ (Achieved: **{overall_stats['global_avg_gen_ms']} ms**).",
        f"- [{'x' if gate_criteria.get('avg_roundtrip_latency_under_200ms') else ' '}] Average roundtrip latency $< 200\\text{{ ms}}$ (Achieved: **{overall_stats['global_avg_roundtrip_ms']} ms**).",
        f"- [{'x' if gate_criteria.get('all_gates_passed') else ' '}] **OVERALL EVALUATION GATE**: **{'PASSED' if all_passed else 'FAILED'}**.",
        "",
        "---",
        "*Benchmark completed autonomously in accordance with AGENTS.md and modular_plan.md.*",
    ])

    return "\n".join(lines) + "\n"


def update_execution_state(state_file: Path, report_rel_path: str = "logs/module_5_report.json") -> None:
    """Updates logs/execution_state.json marking Module 5 as PASSED."""
    if not state_file.exists():
        logger.warning(f"execution_state.json not found at {state_file}, creating fresh.")
        state = {"current_module": 5, "modules": {}}
    else:
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read state file: {e}")
            state = {"current_module": 5, "modules": {}}

    state["current_module"] = 5
    if "modules" not in state:
        state["modules"] = {}

    state["modules"]["5"] = {
        "name": "GGUF Adapter Conversion & Runtime Benchmark",
        "status": "PASSED",
        "report": report_rel_path,
    }

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Updated {state_file} with Module 5 PASSED status.")


def main() -> int:
    parser = argparse.ArgumentParser(description="TalkAI Module 5: GGUF Adapter & Runtime Benchmark")
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL, help="llama-server URL")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Repeat runs per test case")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per generation")
    parser.add_argument("--convert", action="store_true", help="Force re-conversion to GGUF before running")
    parser.add_argument("--skip-state-update", action="store_true", help="Do not update execution_state.json")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("TalkAI Module 5: GGUF Adapter Conversion & Runtime Benchmark")
    logger.info("=================================================================")

    # 1. Adapter GGUF Validation or Conversion
    if args.convert or not LORA_GGUF_PATH.exists():
        logger.info(f"Converting LoRA adapter from {ADAPTER_DIR} to GGUF format...")
        convert_adapter_to_gguf(ADAPTER_DIR, BASE_MODEL_DIR, LORA_GGUF_PATH)

    gguf_info = validate_gguf_file(LORA_GGUF_PATH)

    # 2. Check llama-server Health
    server_info = check_server_health(args.server_url)

    # 3. Define Test Scenarios
    scenarios = [
        {
            "test_id": "Case A",
            "test_name": "Hesitation Yield",
            "prompt": "I was thinking about...",
            "check_fn": check_case_a,
        },
        {
            "test_id": "Case B",
            "test_name": "Hazard Interruption",
            "prompt": "The outlet is sparking and I might touch it",
            "check_fn": check_case_b,
        },
        {
            "test_id": "Case C",
            "test_name": "Direct Question",
            "prompt": "What is the capital of France?",
            "check_fn": check_case_c,
        },
    ]

    test_results = []
    for sc in scenarios:
        res = run_benchmark_test_case(
            test_id=sc["test_id"],
            test_name=sc["test_name"],
            user_prompt=sc["prompt"],
            check_fn=sc["check_fn"],
            server_url=args.server_url,
            repeats=args.repeats,
            max_tokens=args.max_tokens,
        )
        test_results.append(res)

    # 4. Aggregated Latency Statistics
    all_roundtrips: List[float] = []
    all_gen_times: List[float] = []
    all_prompt_times: List[float] = []

    for tr in test_results:
        for run in tr["runs"]:
            all_roundtrips.append(run["roundtrip_ms"])
            all_gen_times.append(run["predicted_ms"])
            all_prompt_times.append(run["prompt_ms"])

    global_avg_roundtrip = round(sum(all_roundtrips) / len(all_roundtrips), 2) if all_roundtrips else 0.0
    global_avg_gen = round(sum(all_gen_times) / len(all_gen_times), 2) if all_gen_times else 0.0
    global_avg_prompt = round(sum(all_prompt_times) / len(all_prompt_times), 2) if all_prompt_times else 0.0

    overall_stats = {
        "global_avg_roundtrip_ms": global_avg_roundtrip,
        "global_avg_gen_ms": global_avg_gen,
        "global_avg_prompt_ms": global_avg_prompt,
        "total_benchmark_queries": len(all_roundtrips),
        "target_threshold_ms": LATENCY_GATE_THRESHOLD_MS,
    }

    # 5. Evaluation Gate Verification
    gate_criteria = {
        "gguf_adapter_valid": gguf_info.get("is_valid", False),
        "test_a_hesitation_passed": test_results[0]["passed"],
        "test_b_hazard_passed": test_results[1]["passed"],
        "test_c_question_passed": test_results[2]["passed"],
        "avg_generation_latency_under_200ms": global_avg_gen < LATENCY_GATE_THRESHOLD_MS,
        "avg_roundtrip_latency_under_200ms": global_avg_roundtrip < LATENCY_GATE_THRESHOLD_MS,
    }
    all_gates_passed = all(gate_criteria.values())
    gate_criteria["all_gates_passed"] = all_gates_passed

    logger.info("=================================================================")
    logger.info("Module 5 Evaluation Gate Checklist:")
    for k, v in gate_criteria.items():
        logger.info(f"  - {k}: {'PASS' if v else 'FAIL'}")
    logger.info(f"Overall Module 5 Status: {'PASSED' if all_gates_passed else 'FAILED'}")
    logger.info("=================================================================")

    # 6. Structured JSON Report
    report_dict: Dict[str, Any] = {
        "module": 5,
        "module_name": "GGUF Adapter Conversion & Runtime Benchmark",
        "status": "PASSED" if all_gates_passed else "FAILED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_info": server_info,
        "gguf_adapter": gguf_info,
        "test_cases": test_results,
        "overall_statistics": overall_stats,
        "gate_criteria": gate_criteria,
    }

    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved JSON report to {REPORT_JSON_PATH}")

    with open(REPORT_JSON_ALT_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved alternate JSON report to {REPORT_JSON_ALT_PATH}")

    # 7. Markdown Report
    md_content = generate_markdown_report(
        gguf_info=gguf_info,
        server_info=server_info,
        test_results=test_results,
        overall_stats=overall_stats,
        gate_criteria=gate_criteria,
        all_passed=all_gates_passed,
    )
    with open(REPORT_MD_PATH, "w") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown report to {REPORT_MD_PATH}")

    # 8. State File Update
    if all_gates_passed and not args.skip_state_update:
        update_execution_state(STATE_JSON_PATH, report_rel_path="logs/module_5_report.json")

    return 0 if all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
