# Agentic serving disaggregation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task.

**Goal:** Measure whether independently schedulable agent-serving phases contend and whether separating them improves end-to-end service at equal resources; decide whether to pivot or retain FluidKV.

**Hypothesis:** Real CPU tool/context work or hybrid-state preparation can interfere with inference; resource separation may remove enough interference to cover handoff and lost pooling costs. Null and negative outcomes are equally valid.

**Validation scope:** User approved design on 2026-09-15. L0 backend/resource/profiler-overhead validation and L3 small end-to-end inference/tool workflow are enabled. Training gradients and overfitting do not apply. Deterministic accounting/affinity/identity logic receives CPU tests. Final performance runs disable heavyweight profilers, use matched work and resources, two load levels and three paired repetitions. No extra approval or commit of unrelated dirty work.

**Architecture:** New external observation/benchmark code under experiments/agentic_profile; preserve prior serving/runtime implementations. Reuse Qwen3.5-35B-A3B graph/extra_buffer backend and existing owned-engine lifecycle. Record one monotonic application timeline, process CPU/RSS/context-switch counters, GPU telemetry, sampled engine Torch CPU/CUDA traces and Nsight Systems when supported. Anonymous trace replays and constructed actual-tool workflows remain explicitly distinct.

## Shared scaffold

Existing: scripts/agent_tool_operations.py has real AST/JSON/PDF operations; prior tool-gap benchmarks do not feed actual tool text back and are not sufficient. SGLang /start_profile supports CPU/GPU activities. nsys2024.4.1 and ncu available. Start from saved dedicated engine identities, preserve runtime configs, discover CPU topology and affinity. Do not use extra GPUs in a treatment without charging/control comparison.

New files: experiments/agentic_profile/{cpu_workloads.py,telemetry.py,run_profile_campaign.py,report_profile_campaign.py}; tests/test_agentic_profile.py; outputs under results/20260915. Modules may be split further as needed. No existing serving code imports profiling toolkit.

## Subtask 1 — CPU/tool/context primitives and topology

Implement fixed corpus-backed AST/PDF or JSON tools, actual deterministic text outputs, context assembly/tokenization and worker process isolation. Record monotonic start/finish, CPU time, bytes, output hash and errors. Affinity uses physical cores with documented NUMA; shared and partitioned modes have equal union of cores and GPU budget. Tool process workers and shared mode are separate processes already, so the test isolates contention rather than introducing a blocking event-loop bug. Include tool-only/context-only measurements and full available-CPU characterization. Validate hashes/lengths, no fake tool sleeps as primary evidence. Parent/controller pinned outside experiment core pool in every variant and separately charged.

## Subtask 2 — Engine baseline and timeline

Launch dedicated engine from prior identities. Warm exact graph/backend workload. Verify original input/output and tokenization boundaries in a two-turn real-tool fixture. Sample profiler on/off and capture PyTorch CPU/CUDA plus nsys cross-process/cuda/NVTX timeline if supported. Record unsupported OS sampling honestly. Lightweight process/GPU observers run identically in all performance variants. Profile time windows need known epoch/monotonic alignment; never sum overlapping stage durations into end-to-end time. Single-agent wait/dependency is not a contention diagnosis.

## Subtask 3 — Matched CPU separation campaign

Screen real tool and context work at modest/busy concurrency with inference-only and operation-only controls. Freeze work counts, prompt/output lengths, same physical CPU union, model/GPU, client and policy seeds before treatment outcomes. Compare shared pool vs partitioned CPU pools (local isolation, not remote-node claim), with non-interfering precomputed-output control for attribution. All actual tool results feed subsequent context, carry checksums. Measure task E2E, TTFT/ITL, stage queues and service, tool latency, CPU-seconds/RSS, GPU occupancy, success denominators. Two load levels,3 paired reps; record all runs. No significance claims from three repeats.

## Subtask 4 — State preparation boundary

Measure foreground inference with and without matched checkpoint-copy/replay work using current safe transfer coordinator; prerequisite reservations and cleanup confirmed. Separate source/destination impacts and compare sequential/co-running scheduling under equal two-GPU resources. Any isolated-GPU scenario counts both GPUs and foreground opportunity cost; never compare1 vs2 GPUs as free isolation. Use existing validated materialization path and fixed workloads; small baseline copy vs reconstruct cost evidence reused only with scope. Diagnose only demonstrated kernel hotspots with NCU, not entire application replay. At least3 paired repeats per retained load level. Avoid uncertain transfer cleanup or OOM retries; preserve failed controls.

## Subtask 5 — Review, report, decision

Independent measurement review checks equal budgets, logical dependencies, actual consumption, tokenization, profiler overhead, invalid runs, CPU quota versus raw node capacity and source state correctness. Generate machine-readable metrics, paired deltas, CPU/GPU timeline and bottleneck figures. Validate remaining full CPU suite and source diffs; archive sources and logs. Shut down only owned, drained experiment engines.

Decision: pursue separation only if co-running interference exists and equal-budget separation repeatedly improves application outcomes after handoff costs. Continue focused FluidKV only if reusable state recovery/retention dominates demonstrated loss. Otherwise recommend against enlarging either design. No inference of NSDI/SIGCOMM novelty from profiling alone. Multi-node and arbitrary production workloads require separate evidence.

## Exposure audit correction before conclusions

The first three low-concurrency burst runs expose only0.33%–1.24% of tool execution to other request lifetimes and zero to other decode lifetimes. Preserve all24 burst runs as synchronization/workload controls. To actually test approved cross-agent contention, add fixed staggered arrivals0.25s with otherwise unchanged16agents/tools/prompts/outputs: shared/frontend-isolated×concurrency2/8×3reps on8physical cores (12runs). The already planned64core characterization uses the same staggered arrivals and two policies (12runs), permitting interpretation under wider CPU availability. This change is driven by exposure validity, not favorable policy results. No fabricated extra tool passes. Total CPU48 bounded runs; state84 bounded micro-controls follow. Main source remains frozen.

## Real CPU-heavy coverage

The initial tool parses26files149764B, while a real SGLang SRT corpus has995files14.79MB. CPU-only3repeat characterization on reserved nonexperimentCPU80 measured3.21s wall/3.17s CPU per single AST analysis. After the current frozen48 CPU runs finish, preserve their exact source snapshot and add an optional frozen tool-manifest input to a new harness revision. Heavy control: same16agents/2048firsttokens/128outputs/6000char result cap/arrival.25; whole-agent concurrency8/16; shared8cores vs4engine+4tool and2engine+6tool, three paired repeats,18runs. The2core reservation is chosen before heavy outcomes from measured ~1CPU engine consumption, not from selecting a winning result. This tests a real larger corpus once, not artificial passes. Tool/preparation output and workload identities audited as before. All configurations keep8pool+4controller physical cores and1GPU. No further CPU variants after this coverage test; finish planned state matrix and conclude.

## Completion

Completed66 CPU profiler-off conditions (1056 constructed two-turn agents),84 GPU-state conditions (300 foreground requests,144copies,144coldreplays), all execution/cleanup successful. Independent audits verify workload/resource identities with heavyCPU numerical exceptions retained; all288 state consumption/cached-output pairs pass. Short Torch/Nsight, real-heavy-workflow and dualengine state profiles saved separately, excluded from performance comparisons. Source/workload frozen per revision; initial observer/affinity bugs repaired before main and preserved as diagnostics. All dedicated engines released. Main conclusion in experiments/agentic_profile/results/20260915/RESULTS.md: no demonstrated net gain from tested staticCPU splits; replay/decode placement conflict is real, copy can overlap with lowforeground impact. Retain focused FluidKV mechanisms, defer broadMILP/disaggregation/system claims. No additional performance experiments pending.
