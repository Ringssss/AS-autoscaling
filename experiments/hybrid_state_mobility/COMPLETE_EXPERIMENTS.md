# Complete Mobility Experiment Results

Date: 2026-09-13. This run completes the missing concurrency, sensitivity,
elasticity, and branch-validity experiments using the Kimi-timing /
LooGLE-content / ToolBench-boundary replay.

## Non-Blocking Transfer

The direct path now has start/status/cleanup semantics:

```text
PREPARING -> COPYING -> COPY_DONE -> INSTALLING -> COMPLETE -> CLEANED
```

The source prefix is pinned during transfer. Destination full-KV token slots and
Mamba snapshot slots are reserved privately. The transfer worker uses a side
CUDA stream and grouped NCCL P2P operations. Public radix visibility occurs only
when `direct_transfer_status` installs the complete bundle.

The scheduler idle checker now recognizes active mobility reservations instead
of treating pinned source state or private destination state as a cache leak.

## Concurrent Interference

Configuration:

- source and destination each served one foreground request;
- foreground prompt: 2,048 tokens;
- foreground output: 96 tokens;
- three repetitions;
- one concurrent 55-58 MiB hybrid bundle migration.

| Engine | E2E overhead | Client-time overhead |
|---|---:|---:|
| source | 1.00298x | 1.00268x |
| destination | 0.99828x | 0.99843x |

All concurrent migrations committed ownership and all next turns had complete
parent-prefix hits. Scheduler-side concurrent transfer time was 134-137ms,
versus about 54ms without foreground load. The measured foreground effect was
within run-to-run noise in this three-sample experiment.

Artifact: `results/kimi_loogle_toolbench/async_interference.json`.

## 32-Agent Trace Window

The real two-engine window was expanded from four to 32 agents.

| Metric | Value |
|---|---:|
| Successful agents | 32/32 |
| Owner commits | 32/32 |
| Destination full parent hits | 32/32 |
| Migration wall p50 / p90 / max | 62.8ms / 66.5ms / 163.7ms |
| Scheduler transfer p50 / p90 | 54.7ms / 57.2ms |
| Total moved bytes | 5,637,918,720 |
| Median parent prefix | 5,696 tokens |
| Minimum gap margin after migration | 146.4ms |

Artifact: `results/kimi_loogle_toolbench/trace32/trace_direct_window.json`.

## Global Sensitivity

The full sweep covered:

- bandwidth: 1-32 GiB/s;
- tool-gap scale: 0.05x-2x;
- recurrent snapshot capacity: 4-48 slots/engine;
- full-KV capacity: 65,536-327,680 token slots/engine;
- scale-in time: 30-120s;
- drain deadline: 120-300s.

Most important results at the 180s deadline:

| Tool-gap scale | Sticky residue | Token-only residue | Heterogeneous residue |
|---:|---:|---:|---:|
| 0.05x | 66 | 64 | 31 |
| 0.10x | 66 | 46 | 24 |
| 0.25x | 66 | 21 | 16 |
| 1.00x | 66 | 18 | 14 |

Snapshot capacity is a hard gate:

| Snapshot slots/engine | Sticky | Token-only | Heterogeneous |
|---:|---:|---:|---:|
| 16 | 58 | 62 | 62 |
| 24 | 66 | 32 | 32 |
| 32 | 66 | 18 | 14 |

At low snapshot capacity, all policies are constrained by the same physical
recurrent-state limit. Heterogeneous-aware planning differs once enough
snapshot slots exist to permit selective migration.

Artifact: `results/kimi_loogle_toolbench/global_mobility_sweep.json`.

A separate simulation calibrated to the measured async path used 50ms fixed
overhead and 4.37 GiB/s. It produced 285 heterogeneous migrations and reduced
retiring-engine state to 14, versus 19 for token-only and 66 for sticky.

Artifact:
`results/kimi_loogle_toolbench/global_mobility_simulation_measured_async.json`.

## Three-Engine Scale-Out / Scale-In

Engines:

- source: H100 PCIe;
- destination: H100 PCIe;
- scale-out/scale-in engine: A100-SXM4.

Two three-turn replay agents executed:

```text
source turn 1
  -> migrate parent to scale-out engine
  -> scale-out turn 2 full parent hit
  -> migrate advanced branch to destination
  -> destination turn 3 full parent hit
```

All checks passed for both agents:

- source first-turn zero cache hit;
- scale-out second-turn full parent hit;
- destination third-turn full parent hit;
- both owner-epoch commits succeeded;
- final owner was destination;
- both tool gaps covered migration;
- bundle sizes matched the hybrid formula.

Measured walls:

| Agent | Source -> scale-out | Scale-out -> destination |
|---|---:|---:|
| lifecycle-00 | 216.1ms | 343.1ms |
| lifecycle-01 | 63.4ms | 238.7ms |

The slower scale-in path includes heterogeneous H100/A100 transfer and custom
NCCL group behavior. It remained within the replay-derived tool gaps.

Artifact: `results/kimi_loogle_toolbench/lifecycle/three_engine_lifecycle.json`.

## Branch Fanout

A migrated 4,096-token parent state was concurrently extended by two distinct
256-token child branches.

Results:

- parent source request: zero cached tokens;
- migration bundle: 148,275,200 bytes;
- owner commit: successful;
- child 1 cached parent tokens: 4,096/4,096;
- child 2 cached parent tokens: 4,096/4,096;
- both child generations completed.

This validates Mamba copy-on-write branching from one migrated parent snapshot.

Artifact: `results/kimi_loogle_toolbench/branch_fanout.json`.

## Remaining Boundary

The complete experiment now covers correctness, non-blocking concurrent
migration, policy sensitivity, real scale-out/scale-in, and branch fanout. The
remaining productionization gaps are:

- destination-crash recovery during async transfer;
- explicit cancellation when a tool returns before transfer completes;
- larger multi-process workload generation with continuous arrivals;
- longer SLO/GPU-second campaigns.
