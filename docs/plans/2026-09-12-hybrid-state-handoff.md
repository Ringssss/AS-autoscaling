# Hybrid State Handoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:executing-plans to execute this plan task-by-task.

**Goal:** Demonstrate real two-engine Qwen3.5 full-state handoff and exact continuation equivalence.

**Hypothesis:** Copying both full-attention KV and one valid Mamba snapshot, then atomically installing them as a `MambaRadixCache` prefix, is sufficient for exact greedy continuation on a compatibility-matched destination.

**Validation scope:** See `2026-09-12-heterogeneous-state-mobility-design.md`. L0/L1/L3 are enabled; training L2 is skipped.

**Architecture:** Patch a clean SGLang worktree with scheduler-side export/import methods, tokenizer control messages, and admin HTTP endpoints. Staging uses a restricted local checkpoint directory. An AgentShift experiment script drives source generation, export, destination import, three continuation controls, and artifact collection.

---

## Shared Scaffold

- Clean SGLang worktree: `/data-aisoft/zhujianian/cloud/sglang-hybrid-state`
- AgentShift repo: `/data-aisoft/zhujianian/cloud/AS-autoscaling-upstream`
- Model: `/data-aisoft/zhujianian/Uneed/huggingface_download/Qwen3.5-35B-A3B`
- Python: `/home/zhujianian/miniconda3/envs/sdmllm/bin/python`

## Subtask 1: Scheduler State Bundle

Implement `agentshift_export_hybrid_state` and `agentshift_import_hybrid_state` on Scheduler. Export matches a completed prefix, copies full KV rows and the branch Mamba slot to CPU, and saves a manifest. Import validates compatibility, reserves destination resources, copies tensors, and inserts into `MambaRadixCache`.

Tests: static unit tests for path validation are not possible without SGLang test setup; rely first on source review and runtime errors.

Validation: source export reports exact token and byte counts matching `20,480 * tokens + 64,389,120`.

## Subtask 2: Control Messages And Endpoints

Add request/output dataclasses, tokenizer communicators, scheduler dispatch, and `/agentshift/export_hybrid_state` plus `/agentshift/import_hybrid_state`.

Tests: successful HTTP round trips on both engines.

Validation: malformed checkpoint IDs and mismatched token IDs are rejected.

## Subtask 3: Two-Engine Equivalence

Start source on GPU1 and destination on GPU3 with identical arguments. Generate a deterministic token prefix on source, export it, import on destination, then issue identical continuation requests to source, imported destination, and a cold destination.

Validation: source and imported destination output IDs match exactly; imported request reports the full prefix cached; cold destination reports zero cached tokens.

## Subtask 4: Record And Generalize

Persist timings, descriptor fields, cache-hit telemetry, output hashes, and logs. Only after correctness passes, replace host staging with direct grouped GPU transfer and add the materialization planner.

## Execution Status

- Subtask 1: complete; see `experiments/hybrid_state_mobility/RESULTS.md`.
- Subtask 2: complete; export/import endpoints passed positive and negative HTTP round trips.
- Subtask 3: complete; three final-code repetitions passed exact output and cache-hit checks.
- Subtask 4: recording complete; planner, token-only replay, direct NCCL transfer, and durable ownership commit are implemented. Non-blocking concurrent serving and crash recovery remain the next batch.
