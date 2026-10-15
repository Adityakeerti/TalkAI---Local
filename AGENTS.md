# AGENTS.md: Autonomous Multi-Agent Full-Duplex Implementation Protocol
=======================================================================

This document defines the autonomous multi-agent operating procedure for executing the 5-phase **True Full-Duplex Cascade** pipeline specified in `full_duplex_cascade_plan.md`.

---

## 1. Multi-Agent Team Structure

Every module in `full_duplex_cascade_plan.md` MUST be executed by three specialized subagents coordinated by the **Lead Orchestrator**:

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
| **Lead Orchestrator** | Root / Parent Agent | Manages plan progress, selects active module, passes full context to subagents, tracks `logs/execution_state.json`. | Read/Write, Subagent management |
| **Coder** | `TypeName: "self"`, `Role: "Implementation Engineer"` | Writes backend scripts, WebSocket handlers, audio slicers, and frontend UI. | Read/Write, Run command |
| **Reviewer** | `TypeName: "self"`, `Role: "Code & Math Auditor"` | Audits code for: **Zero Hardcoding** (no keyword lists, no regex bans, no logit bias hacks), VRAM safety ($\le 5.0\text{ GB}$), and 600ms timing budgets. | Read-only |
| **Tester** | `TypeName: "self"`, `Role: "QA & Verification Engineer"` | Executes benchmarks, validates all Evaluation Gate assertions, and writes `logs/module_{K}_report.json`. | Run command, Read-only |

---

## 2. Invariants & Context for Every Subagent

To prevent subagents from hallucinating or going off track midway, every prompt dispatched to a subagent MUST inject this exact operational context:

1. **Architecture Goal**: Continuous 600ms micro-turn loop based on **arXiv:2603.09180 (DuplexCascade)**.
2. **Zero Hardcoding Invariant**:
   - **Never** add keyword lists (e.g. `is_long_form = any(...)`).
   - **Never** add artificial token bans (e.g. `logit_bias={"20923": -100}`).
   - **Never** use regex stripping on output tokens (`re.sub(r"^listen", ...)`).
   - Decisions are strictly governed by model weights and the calibrated threshold:
     $$\text{Action} = \begin{cases} \text{YIELD (listen)}, & \text{if } P(\text{<listen>}) \ge 0.30 \\ \text{SPEAK}, & \text{if } P(\text{<listen>}) < 0.30 \end{cases}$$
3. **Environment Paths**:
   - Python Interpreter: `/home/aditya/Desktop/STUDY/TalkAI/.venv/bin/python`
   - Active GPU: `cuda:0` (NVIDIA RTX 4050 6GB Laptop GPU, strict VRAM cap: 5.0 GB)
   - Model Server: `llama-server` on port 8080 running `models/duplex_gemma_merged.gguf`
   - Web Server: FastAPI / Uvicorn running on port 8000
   - Logs Directory: `logs/`

---

## 3. Autonomous Execution Cycle per Module

For every module ($k \in [1, 2, 3, 4, 5]$) defined in `full_duplex_cascade_plan.md`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THE 5-STEP MODULAR CYCLE                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. CONTEXT BRIEF : Orchestrator reads module spec from plan.md.        │
│ 2. IMPLEMENT     : Coder writes code and unit test harness.            │
│ 3. AUDIT         : Reviewer audits code for zero-hardcoding & safety.  │
│ 4. VERIFY        : Tester executes script and asserts evaluation gate. │
│ 5. GATE CHECK    :                                                     │
│    • IF PASSED  -> Update logs/execution_state.json, advance to k+1.   │
│    • IF FAILED  -> Coder fixes root cause, re-tests until gate passes. │
└────────────────────────────────────────────────────────────────────────┘
```

### Step A: Module Selection & Context Dispatch
The Orchestrator reads `full_duplex_cascade_plan.md` for Module $K$ requirements and Evaluation Gate criteria.

### Step B: Implementation (Coder Subagent)
Orchestrator invokes `Coder` subagent with full context:
```markdown
"Implement Module {K}: {Module_Name} as specified in full_duplex_cascade_plan.md.
 Target files: {target_files}.
 Rules:
 - Zero hardcoded keywords or regex stripping.
 - Python environment: /home/aditya/Desktop/STUDY/TalkAI/.venv/bin/python.
 - Maintain VRAM <= 5.0 GB and 600ms latency budget.
 Write unit assertions and confirm clean syntax."
```

### Step C: Independent Code Review (Reviewer Subagent)
Orchestrator invokes `Reviewer` subagent:
```markdown
"Perform an independent code audit on {target_files} for Module {K}:
 - Check 1: Is there ZERO hardcoding (no keyword checks, no logit bans, no regex stripping)?
 - Check 2: Does turn-taking follow P(<listen>) >= 0.30 neural policy?
 - Check 3: Are audio slices kept strictly at 600ms?
 - Check 4: Is VRAM safely under 5.0 GB?
 Reply with APPROVE or REQUEST_CHANGES."
```
If Reviewer flags any issues, Orchestrator instructs Coder to resolve them before execution.

### Step D: Execution & Testing (Tester Subagent)
Orchestrator invokes `Tester` subagent:
```markdown
"Execute test suite for Module {K} using .venv/bin/python.
 Validate all assertions specified in Module {K} Evaluation Gate.
 Record latency, memory, and assertion pass rates in logs/module_{K}_report.json."
```

### Step E: Evaluation Gate Check
- **PASSED**: Update `logs/execution_state.json` (`module_K: "PASSED"`), advance to Module $K+1$.
- **FAILED**: Capture traceback, instruct Coder to fix the specific failure, re-run Tester. Never proceed with a failing gate.

---

## 4. The 5 Modules & Evaluation Gates

1. **Module 1: Duplex Micro-Turn Engine Core (`scripts/duplex_engine.py`)**
   - *Gate*: 600ms clock tick ($\pm 15\text{ms}$), async WebSocket `/ws/duplex` session management.
2. **Module 2: Streaming Audio Ingestion & 600ms Slicing (`test.html` & backend)**
   - *Gate*: Web Audio 600ms PCM/Web Speech stream; silence tagged `"no voice"`; speech transcribed $\le 150\text{ms}$; zero VAD silence wait.
3. **Module 3: Zero-Hardcoding Neural Policy & Calibrated Turn-Taking**
   - *Gate*: Zero keyword heuristics in codebase; $P(\text{<listen>}) \ge 0.30$ policy; instant warning on hazard slices.
4. **Module 4: Concurrent Kokoro TTS Synthesis & Audio Queue**
   - *Gate*: TTFA $\le 350\text{ms}$ on speech fragments; instant client barge-in playback cutoff $\le 30\text{ms}$.
5. **Module 5: Production Web UI & End-to-End Validation Suite**
   - *Gate*: Center mic, left drawer, right live console (blue user, white AI); hesitation test, question-answer test, and mid-utterance hazard test all pass; report saved to `logs/module5_full_duplex_report.md`.

---

## 5. One-Shot Master Prompt for the Next AGY Instance

To autonomously execute this entire plan in another fresh AGY CLI instance, feed this single prompt:

```text
Follow AGENTS.md and full_duplex_cascade_plan.md to autonomously implement and verify the True Full-Duplex Cascade system.
Ensure zero hardcoding anywhere in the pipeline (no keyword triggers, no logit bans, no regex text-stripping).
For each module (1 through 5):
1. Spawn Coder to implement the module specifications.
2. Spawn Reviewer to audit code for zero-hardcoding and 600ms timing safety.
3. Spawn Tester to execute tests and verify all evaluation gate assertions.
4. If any gate fails, debug and fix before advancing.
5. Save results to logs/module_X_report.json.
Do not stop until Module 5 end-to-end benchmark is fully completed and reported.
```
