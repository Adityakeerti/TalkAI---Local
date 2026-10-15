# AGENTS.md: Autonomous Multi-Agent Orchestration Protocol
=============================================================

This document defines the autonomous multi-agent operating procedure for executing the 5-phase fine-tuning pipeline specified in `modular_plan.md`.

---

## 1. Multi-Agent Team Structure

Each module in `modular_plan.md` MUST be executed by three specialized subagents coordinated by the **Lead Orchestrator**:

```
                       ┌──────────────────────┐
                       │  Lead Orchestrator   │
                       └──────────┬───────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Coder Subagent  │───>│ Reviewer Subagent│───>│ Tester Subagent  │
│ (Implementation) │    │  (Code Auditor)  │    │  (Verification)  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

| Agent Role | Subagent Type | Key Responsibility | Tool Access |
| :--- | :--- | :--- | :--- |
| **Lead Orchestrator** | Root / Parent Agent | Manages roadmap, selects active module, passes tasks to subagents, tracks `logs/execution_state.json`. | Read/Write, Subagent management |
| **Coder** | `TypeName: "self"`, `Role: "Implementation Engineer"` | Writes scripts, PyTorch modules, dataset loaders, and training loops. | Read/Write, Run command |
| **Reviewer** | `TypeName: "self"`, `Role: "Code & Math Auditor"` | Audits code for silent traps: RoPE mismatches, gradient leaks, tokenization bugs, VRAM safety. | Read-only |
| **Tester** | `TypeName: "self"`, `Role: "QA & Verification Engineer"` | Executes tests, measures VRAM/loss metrics, checks assertions, and writes `logs/module_X_report.json`. | Run command, Read-only |

---

## 2. Autonomous Execution Cycle per Module

For every module ($k \in [1, 2, 3, 4, 5]$):

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THE 5-STEP MODULAR CYCLE                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. PICK MODULE   : Read requirements & evaluation gate from plan.md.   │
│ 2. IMPLEMENT     : Spawn Coder to write code and unit harness.         │
│ 3. CODE REVIEW   : Spawn Reviewer to audit code against safety rules.  │
│ 4. EXECUTE & TEST: Spawn Tester to run script and record metrics.      │
│ 5. GATE CHECK    :                                                     │
│    • IF ALL GATES PASS -> Save report, commit state, advance to k+1.   │
│    • IF ANY GATE FAILS -> Spawn Debugger, diagnose, fix, re-test.      │
└────────────────────────────────────────────────────────────────────────┘
```

### Step A: Module Selection & State Check
1. The Orchestrator inspects `logs/execution_state.json` to identify the current module.
2. Reads the exact requirements, files, and **Evaluation Gate** criteria from `modular_plan.md`.

### Step B: Implementation (Coder Subagent)
1. Orchestrator invokes `Coder` subagent with prompt:
   ```markdown
   "Implement Module {K}: {Module_Name} as specified in modular_plan.md.
    Write the code to {script_path}. Write clear assertions and log results in JSON."
   ```
2. Coder writes the script using `.venv/bin/python`.

### Step C: Independent Code Review (Reviewer Subagent)
1. Orchestrator invokes `Reviewer` subagent to audit the newly written code:
   - Check 1: Does it avoid manual activation caching to disk?
   - Check 2: Are user tokens masked with `label = -100`?
   - Check 3: Are frozen layers explicitly set to `requires_grad = False`?
   - Check 4: Is gradient checkpointing enabled to prevent CUDA OOM?
2. If Reviewer flags any issues, Orchestrator instructs Coder to address them before execution.

### Step D: Execution & Testing (Tester Subagent)
1. Orchestrator invokes `Tester` subagent:
   ```markdown
   "Execute {script_path} using .venv/bin/python.
    Validate all assertions specified in Module {K} Evaluation Gate.
    Record memory, loss, and latency metrics in logs/module_{K}_report.json."
   ```
2. Tester executes the script in bash and captures standard output and metrics.

### Step E: Evaluation Gate & Debug Loop
* **If PASSED**:
  - Tester confirms all checkboxes for Module $K$ pass.
  - Orchestrator updates `logs/execution_state.json` (`status: "PASSED"`).
  - Proceed immediately to Module $K+1$.
* **If FAILED**:
  - Orchestrator captures the traceback and metrics.
  - Spawns Coder to fix the root cause.
  - Re-triggers Tester until the gate passes. Never proceed with a failing gate.

---

## 3. Environment & Hardware Ground Rules

All subagents MUST adhere to these environment invariants:
- **Python**: `/home/aditya/Desktop/STUDY/TalkAI/.venv/bin/python`
- **PyTorch / CUDA**: Device `cuda:0` (NVIDIA RTX 4050 6GB Laptop GPU).
- **VRAM Cap**: Never exceed **5.0 GB allocated VRAM** at any step.
- **Model Checkpoint**: Base safetensors at `/home/aditya/Desktop/STUDY/TalkAI/models/gemma-4-E2B-it`.
- **Inference Server**: `/home/aditya/llama-fork/build/bin/llama-server` on port 8080.
- **Logs Directory**: All metrics and gate reports must be saved in `logs/`.

---

## 4. One-Shot Master Prompt for the Next AGY Instance

To run this entire automated pipeline in a fresh AGY CLI instance, feed this single instruction:

```text
Follow AGENTS.md and modular_plan.md to autonomously execute the 5-phase Gemma-4-E2B fine-tuning pipeline.
For each module (1 through 5):
1. Spawn Coder to write the module implementation.
2. Spawn Reviewer to audit code for bugs and RoPE/loss-masking safety.
3. Spawn Tester to execute the script and verify evaluation gate assertions.
4. If a gate fails, debug and fix before advancing.
5. Save results to logs/module_X_report.json.
Do not stop until Module 5 benchmark is completed and reported.
```
