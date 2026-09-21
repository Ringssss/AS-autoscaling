# FluidKV local serving-pipeline stall Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task.

**Goal:** Determine whether local (PP=1) hybrid copy causes repeatable short serving stalls concealed by mean decode cost, and only if supported test a bounded mitigation.

**Hypothesis:** Source packing/synchronization, NCCL transport, destination scatter/installation or control handling can disturb token cadence even when average decode overhead is small. GPU gaps are not automatically schedulable slack.

**Validation scope:** User explicitly rejected PP and approved local trace-first investigation with conditional small experiments. L0 clocks/backends/observer overhead/stream integrity and L3 actual valid snapshot transfer+consumption enabled. Training gradient/overfit layers inapplicable. CPU deterministic parser/interval tests. No PP launch or stage-aware protocol, no absolute zero-stall guarantee.

**Architecture:** Keep current PP1/TP1 SGLang runtime unchanged initially. External new code experiments/local_pipeline; inspect saved Torch traces and runtime code. Existing agentic_profile generation stores token emissions for CPU runs, but old state runs only first/end; hence paired live per-token measurements are needed. Same two dedicated H100 GPU0/1, prior graph extra_buffer/FP32 recurrent128kKV64snapshot backend. Preserve existing dirty work and every earlier result.

## Subtask1 offline diagnosis

Analyze prior state-profile and state-main artifacts for CPU control RPC envelopes, CUDA Graph launch cadence, sync calls and packet/installation kernels. Distinguish host correlation from causal mechanism. Determine what timestamps are missing. Review source pack synchronous wait and destination install on status path; verify thread/scheduler execution context. Output offline-review.md/json with line refs and explicit limitations.

## Subtask2 exact streaming observation

Implement benchmark_local_copy.py using existing safe coordinator and observed cleanup clients. Record every cumulative stream emission timestamp/count, TTFT/end, token-ID digest/length, CPU control RPC path/role/boundary, migration phase responses, source and target ready times. Use monotonic+epoch anchors. Per-token gap valid only when adjacent counts differ1; keep coalesced chunks explicit. Profilers off for performance; no host global process scans in client event loop.

## Subtask3 bounded paired comparison

48 initial runs: foreground source/destination × concurrency2/8 × prefix8192/24576 × no-copy/copy ×3 paired repetitions, randomized order seed1515. One verified retained checkpoint per run (not a burst of multiple independent copies), disjoint1024token foregrounds with256 outputs. Warm all shapes and transfer paths before observations; source checkpoint construction outside measurement and equal across controls. Trigger copy after observed first token plus fixed100ms delay, ensure foreground still active. No-copy schedules same marker/delay but no transfer. Validate target cache and source-cached continuation after copy outside measured window; failed/stale controls preserved and stopped before unsafe flush. TwoGPU32engineCPU+4controller budget constant across groups. No-copy is interference attribution, not equal-work speed comparison.

Report per-run/client observed decode gap p50/p95/p99/max, stalls above20/50ms, clustered short windows around control phases, total foreground finish time, migration completion, actual overlap, bytes and cache/quality checks. Three repeats are descriptive paired evidence, not token-level significance. Explicitly check client chunking/buffering and multi-request simultaneous stalls. A candidate for mitigation requires repeatable added tail in paired runs beyond ordinary variation (screen p99 >10% and >2ms, plus phase/trace evidence); threshold is a screening rule not a zero-stall theorem.

## Subtask4 conditional attribution/mitigation

If meaningful copy tails exist, capture a short representative joint CPU/CUDA trace. Compare source-only packaging/control-only or an appropriate bounded scheduling ablation to isolate the cause without equating unrelated kernels to transfer. Choose smallest intervention from evidence: move blocking staging out of foreground scheduler path, bound chunk submission, or change installation admission. Any transport change must retain prepared-both-end safety, complete snapshot semantics, capacity guards and cleanup. Do not change runtime between paired variants without matching baselines/provenance. If no meaningful copy tail is demonstrated, stop and report no justification for local bubble-filling controller. Do not implement a new optimizer merely to obtain a win.

## Completion

Independent audit of phase clocks, emission counts/lengths, paired workloads, resource budget, real copy consumption and attribution. Report limitations and whether pipeline-aware migration adds value beyond existing async copy and replay placement. Run appropriate/fullCPU checks, archive exact source, release only dedicated drained engines. No added PP or indefinite experiments.

## Completion and user steering

Completed48 profiler-off local PP1 conditions (240requests,61200valid token intervals),24verified copies and4separate profiler diagnostic conditions. All raw-audit checks pass. Repeatable source24k client tails exist, but paired totalforeground completion median+0.201%; profiles do not show large emptyGPU bubbles or long scheduler-main-thread stream synchronization. No immediate asynchronous-staging rewrite or broad bubble scheduler is justified by current attribution; runtime remained unchanged.318 CPUtests pass; dedicated engines released. Details experiments/local_pipeline/results/20260915/DECISION.md.

User concurrently requested FreeToken desktop assessment. Read-only pinnedclone afd99cb3b7c118dce4a9835359bcbc33b70c3cb8 and linkedpaper reviewed. NoFreeToken installation/GPU/contribution. Existingsemanticanchors/CPU-GPU expertpolicy/elasticpool features overlap broadideas. Concreteunmeasuredcandidate is selectivelegalagent-statepreservation across actualKV/recurrent/windowpoolresizes (expert-onlyresize maypreservestate). Recommendboundedfuturevalidation ratherthannewlargeimplementation; noFreeToken speedup orpublicationclaim.
