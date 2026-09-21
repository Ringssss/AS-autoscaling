# Hybrid Agentic State Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:executing-plans to execute this plan task-by-task.

**Goal:** Quantify how Qwen3.5 hybrid attention state changes AgentShift cache accounting and migration choices.

**Hypothesis:** Token-only KV accounting misplaces hybrid-agent state because full-attention KV scales with context while linear-attention state scales with live branch snapshots.

**Validation scope:** See `2026-09-12-hybrid-state-discovery-design.md`; L0 and L3 are runtime smoke checks, L1 is deterministic state/trace accounting, and training L2 is skipped.

**Architecture:** Add a standalone experiment script under `experiments/hybrid_state_20260912/` that parses model config and existing NPZ traces, emits machine-readable summaries, and is validated by deterministic unit tests. Then launch one local SGLang instance on an idle H100 for a cache-hit smoke test. No production AgentShift control-plane behavior is changed in the first batch.

---

## Shared Scaffold

### Existing Infra

- AgentShift repo: `/data-aisoft/zhujianian/cloud/AS-autoscaling-upstream`
- Current SGLang source: `/data-aisoft/zhujianian/cloud/h3i/sglang`
- Model config: `/data-aisoft/zhujianian/Uneed/huggingface_download/Qwen3.5-35B-A3B/config.json`
- Anonymous Qwen traces: `/home/zhujianian/agentic-trace-study-20260908/results/qwen_*_requests.npz`
- Python: `/home/zhujianian/miniconda3/envs/hid/bin/python`

### Needs Setup

- `experiments/hybrid_state_20260912/analyze_hybrid_state.py`
- `tests/test_hybrid_state_analysis.py`
- Result JSON/Markdown directory: `experiments/hybrid_state_20260912/results/`

## Subtask 1: Deterministic State Accounting

**Hypothesis:** Exact Qwen3.5 config implies a full-KV versus linear-state crossover near a few thousand tokens.

**Implementation:** Parse `config.json`, compute full-attention KV bytes/token and Qwen linear conv/temporal bytes/state directly from architecture fields, and validate the formulas against SGLang's `KimiLinearStateShape` and `mamba_cache_per_req`.

**Unit tests:** Verify state formulas, MiB conversion, crossover, and trace summary invariants with synthetic arrays.

**Validation:** `pytest -q tests/test_hybrid_state_analysis.py`, then run the script on all four local Qwen traces.

**Expected conclusion:** Obtain exact bytes/token, bytes/state, crossover, and trace percentiles with reproducible output.

## Subtask 2: Chain/Branch Cost Replay

**Hypothesis:** Real linked-chain prefixes show where hybrid state migration changes decisions compared with AgentShift's homogeneous MHA cost model.

**Implementation:** For each usable parent link, compare MHA-only migration bytes with hybrid full-KV plus one or more SSM snapshots. Report gap-overlap sensitivity at 25/50/100 GiB/s and identify requests whose full state cannot fit common tool gaps.

**Unit tests:** Validate cost calculation and bandwidth boundaries on synthetic links.

**Validation:** Inspect emitted JSON for finite metrics and exact request counts matching each NPZ.

**Expected conclusion:** Identify split-specific crossover and overlap feasibility distributions without claiming measured transfer performance.

## Subtask 3: Runtime Smoke

**Hypothesis:** Current SGLang recognizes Qwen3.5 and allocates separate full-attention and Mamba pools, while AgentShift's pinned data plane cannot express both.

**Implementation:** Start one SGLang server on an idle H100 with a small context, send a repeated-prefix request, record pool telemetry and prefix-hit metrics, then probe API compatibility assumptions without modifying production code.

**Unit tests:** No new deterministic unit test; runtime facts are recorded in JSON/Markdown with server logs.

**Validation:** Successful health check, exact repeated-prefix cached-token count, and a clean server shutdown.

**Expected conclusion:** Establish whether hybrid runtime support is present and whether migration is only a control-plane mismatch or also missing data-plane state.
