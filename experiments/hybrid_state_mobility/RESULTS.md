# Heterogeneous State Mobility Milestone

Date: 2026-09-12. This experiment upgrades the repo from hybrid-state
characterization to a working two-engine state handoff.

## Result

**Effective for same-host, TP=1, idle-scheduler correctness.** A completed
Qwen3.5 branch's full-attention KV and Mamba conv/temporal snapshot can be
exported from one SGLang engine, imported by another compatibility-matched
engine, atomically installed in `MambaRadixCache`, and resumed with exact greedy
output equivalence.

This is not yet a performance result: staging uses a local checkpoint file and
both schedulers must be idle. It closes the largest correctness gap and gives
the first end-to-end implementation of the `StateDescriptor` idea.

## Implementation

Clean SGLang worktree:

- `/data-aisoft/zhujianian/cloud/sglang-hybrid-state`
- branch: `hybrid-state-mobility`
- base: `009eee85a`

The generated patch is
`results/runtime2/sglang-hybrid-state.patch` (SHA-256
`60dd8c31a9c41a88727e81200887c1dd4840a268a2951336ae2ab45d90b8d9e1`).

It adds:

- `HybridStateTransferManager`, which exports/imports one branch-valid bundle.
- A descriptor containing architecture, dtype, TP/layout, token hash, weight
  metadata, KV shapes, and Mamba state shapes.
- Full-attention K/V row transfer plus Mamba temporal and conv snapshot
  transfer.
- Destination reservation in both token slots and Mamba snapshot slots.
- Public installation only through `MambaRadixCache.insert`.
- Scheduler control messages and admin HTTP endpoints:
  `/agentshift/export_hybrid_state` and `/agentshift/import_hybrid_state`.
- A direct NCCL path between two engines:
  `/agentshift/hybrid_state/direct_group/init` and
  `/agentshift/hybrid_state/direct_transfer`.
- Idle-scheduler and checkpoint/path/hash/branch checks.

Current SGLang upstream already has Mamba state support in its PD-disaggregation
transfer path. The delta here is not "Mamba bytes can move"; it is mobility of a
completed branch into a destination radix cache, with a state descriptor,
dual-resource reservation, and atomic public visibility.

## Correctness Evidence

Configuration:

- Model: Qwen3.5-35B-A3B, BF16, TP=1.
- Source: physical GPU1, H100 PCIe, port 31080.
- Destination: physical GPU3, H100 PCIe, port 31081.
- Context: 4,096 tokens; full-KV capacity: 4,096 tokens; Mamba slots: 8.
- Full-attention backend: FA3; linear-attention backend: Triton.
- Prefix: 2,048 tokens; continuation suffix: 16 tokens.

Three final-code repetitions all passed every check:

| Check | Result |
|---|---:|
| Source continuation matches imported destination | 3/3 |
| Source continuation matches cold destination | 3/3 |
| Source full prefix hit | 3/3 |
| Imported destination full prefix hit | 3/3 |
| Cold destination zero prefix hit | 3/3 |
| State bytes equal `2048*20480 + 64389120` | 3/3 |

The exported state is 106,332,160 bytes (about 101.4 MiB). Median scheduler-side
export time was 274.7 ms and median import time was 134.0 ms through the local
checkpoint. These are correctness-path timings, not transfer-performance
results.

All engines produced:

```text
[220, 151, 109, 151, 108, 151, 108, 151, 108,
 151, 108, 151, 108, 151, 108, 151]
```

Artifacts:

- `results/runtime2/verified-repeat-{1,2,3}.json`
- `results/runtime2/source-verified.log`
- `results/runtime2/destination-verified.log`

## Mamba Ablation

The import endpoint can deliberately install full KV while leaving the imported
Mamba snapshot zeroed. It still reports a full 2,048-token prefix hit, but the
next 16 tokens diverge:

```text
source/imported/cold:
[220, 151, 109, 151, 108, 151, 108, 151, 108,
 151, 108, 151, 108, 151, 108, 151]

KV-only / zeroed Mamba:
[220, 16, 15, 15, 15, 15, 15, 15, 15,
 15, 15, 15, 15, 15, 15, 15]
```

This is the key control: a prefix-cache hit alone does not prove that hybrid
execution state is valid. The recurrent snapshot is semantically necessary.

Artifact: `results/runtime2/verified-mamba-ablation.json`.

## Negative Controls

Final-code negative controls all returned HTTP 400:

- incorrect checkpoint SHA-256;
- token IDs from a different branch;
- checkpoint ID that attempts path escape.

Artifact: `results/runtime2/verified-negative-controls.json`.

## Direct Two-Engine Transfer

The direct path uses a persistent two-rank NCCL group and sends four grouped
tensors: all full-attention K layers, all V layers, the Mamba temporal state,
and the Mamba conv state. The destination reserves both resource families before
receiving and installs the complete bundle into `MambaRadixCache`.

Every direct-transfer check passed:

| Prefix | Bundle bytes | Wall time | Effective throughput |
|---:|---:|---:|---:|
| 512 | 74,874,880 | 26.23 ms | 2.66 GiB/s |
| 1,024 | 85,360,640 | 20.10 ms | 3.95 GiB/s |
| 2,048 | 106,332,160 | 22.16 ms | 4.47 GiB/s |
| 3,072 | 127,303,680 | 21.05 ms | 5.63 GiB/s |

A five-run reuse experiment over one persistent group transferred the 2,048-token
bundle with a 22.66 ms median wall time and 4.37 GiB/s median effective rate.
All runs had exact source/destination output, full 2,048-token destination hits,
and byte counts matching the heterogeneous formula.

This throughput is below the earlier raw H100 peer-copy microbenchmark because
the measured path includes scheduler control, NCCL launch/fixed overhead, staging,
destination writes, radix insertion, and barriers. It is nevertheless fast enough
to hide a 2,048-token Qwen3.5 bundle inside a roughly 25-30 ms tool gap in this
small configuration.

Artifacts:

- `results/runtime2/direct-grouped-{512,1024,2048,3072}.json`
- `results/runtime2/direct-reuse-2048.json`
- `results/runtime2/source-batched-2.log`
- `results/runtime2/destination-batched-2.log`

## Durable Coordinator

`HybridMigrationCoordinator` connects the direct bundle path to the durable
AgentShift ownership store. It initializes the transfer group, drives source and
destination transfer concurrently, verifies token and byte accounting, moves the
migration to `DEST_READY`, and only then commits ownership.

The live-engine result:

- source epoch: 7;
- destination epoch after commit: 8;
- migration state: `COMMITTED`;
- transferred bundle: 106,332,160 bytes;
- destination prefix hit: 2,048/2,048;
- source and destination continuation outputs matched exactly.

Failure tests also verify that a transfer exception or source/destination byte
count disagreement preserves the source owner and marks the migration `ABORTED`.
Artifact: `results/runtime2/hybrid_coordinator_result.json`.

## Control-Plane Upgrade

AgentShift now has:

- `StateDescriptor`, with independent token-slot and snapshot-slot footprints.
- `BranchStatePlanner`, which chooses among:
  - full bundle copy;
  - copy an ancestor snapshot and replay the suffix;
  - reconstruct from tokens.
- Placement support for heterogeneous `state_bytes` and destination snapshot
  capacity, while retaining the legacy MHA path.

Tests cover descriptor invariants, replay/reconstruct selection, dual-resource
capacity rejection, and placement use of the heterogeneous footprint.

## Token-Only Policy Replay

The real trace replay compares a Qwen3-8B token-only estimate
(147,456 B/token) with Qwen3.5's actual `20480 * tokens + 64389120` bundle at
20.28 GiB/s. Decision mismatch means the token-only policy and heterogeneous
cost model disagree on whether a direct move fits the gap.

| Gap (ms) | A mismatch | Coder mismatch | Thinking mismatch | Dominant error |
|---:|---:|---:|---:|---|
| 1 | 14.155% | 5.595% | 2.250% | optimistic |
| 2 | 17.753% | 8.768% | 5.917% | optimistic |
| 5 | 33.908% | 20.325% | 11.833% | conservative |
| 10 | 35.151% | 45.450% | 38.167% | conservative |
| 20 | 25.690% | 54.387% | 52.333% | conservative |
| 30 | 17.432% | 43.138% | 51.250% | conservative |
| 50 | 7.671% | 25.830% | 33.583% | conservative |
| 100 | 0.972% | 0.241% | 12.833% | conservative |

Short gaps make token-only scheduling optimistic because the fixed Mamba
snapshot dominates. Moderate gaps make it conservative because Qwen3.5's hybrid
bundle is much smaller than Qwen3-8B MHA KV for the same long prefix. This is
the first quantitative evidence that a token-only mobility policy makes both
kinds of errors on real traces.

Artifacts: `results/policy/policy_replay.{json,md}`.

## Boundaries And Next Step

A second milestone now consumes the provided Kimi/LooGLE/ToolBench workload:
see [TRACE_REPLAY.md](TRACE_REPLAY.md). It includes a 442-event replay
manifest, an event-driven global mobility simulation, and a four-agent real
two-engine trace window.

The complete 2026-09-13 experiment batch adds non-blocking concurrent transfer,
a 32-agent trace window, global sensitivity sweeps, real three-engine
scale-out/scale-in, and branch fanout:
[COMPLETE_EXPERIMENTS.md](COMPLETE_EXPERIMENTS.md).

The paper-story batch adds a one-hour, 512-agent/1,754-event replay plus four
600-dpi figures quantifying state gravity, existing-policy limits, AgentShift
load balance, and per-request migration directions:
[PAPER_STORY_RESULTS.md](PAPER_STORY_RESULTS.md).

The current prototype requires:

- same host and shared checkpoint root;
- identical model, dtype, TP=1, page size 1;
- idle source and destination schedulers;
- a prompt-boundary branch (this SGLang configuration does not expose the
  generated output token as a Mamba-radix boundary);
- no active request, concurrent transfer, crash rollback, or ownership commit.

Host-file staging, direct GPU transfer, and durable ownership commit are now
covered by separate correctness experiments. The remaining system work is to
make direct transfer non-blocking under concurrent serving, add source release
and crash recovery, and run prefix/gap plus scale-in/scale-out sweeps against
always-copy, always-replay, tiered-KV, and token-only baselines.
