# AgentShift Async Handoff Evaluation

## Frozen System

AgentShift now implements the three frozen components:

1. **Durable Agent Continuation**: SQLite WAL continuation, owner epoch, step
   claims, tool mailbox, managed effects, and post-commit recovery.
2. **Completed-Prefix Mobility**: source pin, destination reservation,
   rank-to-rank KV copy, full-token-key installation, and delayed source release.
3. **Gap-Aware Semantic Handoff**: persistent bounded transfer workers,
   independent CUDA streams, async status polling, ownership CAS, and tool-gap
   overlap.

The main matrix uses Qwen3-8B on one 8xH100 80GB NVLink host. TP=1 experiments
use two single-GPU engines and TP=2 experiments use two two-GPU engines. A
generality experiment uses Qwen3-32B with two TP=4 engines across all eight
GPUs. The mechanism supports completed MHA prefixes only; it does not migrate
active decode state.

## RQ1: Locality After Placement Change

The original TP=1 next-turn sweep remains the cleanest locality result:

| Prefix | Reroute + re-prefill | AgentShift | Speedup | Target hit |
|---:|---:|---:|---:|---:|
| 1K | 57.1 ms | 34.6 ms | 1.65x | Full |
| 4K | 130.7 ms | 34.4 ms | 3.80x | Full |
| 16K | 528.2 ms | 47.8 ms | 11.05x | Full |
| 32K | 1295.7 ms | 68.5 ms | 18.93x | Full |

Raw data: `e2e-1784487320239905011.json`.

TP=2 shows the same trend, despite faster tensor-parallel re-prefill:

| Prefix | Sticky | Reroute | AgentShift | Speedup vs reroute | Migration |
|---:|---:|---:|---:|---:|---:|
| 4K | 71.9 ms | 133.9 ms | 73.7 ms | 1.82x | 34.0 ms |
| 16K | 70.1 ms | 354.8 ms | 70.2 ms | 5.06x | 27.1 ms |
| 32K | 75.4 ms | 704.3 ms | 83.2 ms | 8.46x | 38.5 ms |

Raw data: `e2e-1784537113268610412.json`.

The larger Qwen3-32B TP=4 configuration also preserves sticky-like locality:

| Prefix | Sticky | Reroute | AgentShift | Speedup vs reroute | Migration |
|---:|---:|---:|---:|---:|---:|
| 4K | 113.0 ms | 248.6 ms | 128.1 ms | 1.94x | 57.0 ms |
| 16K | 112.4 ms | 742.4 ms | 116.8 ms | 6.36x | 43.1 ms |
| 32K | 118.7 ms | 1637.3 ms | 120.5 ms | 13.58x | 57.1 ms |

At 32K, the destination hits all 32,772 completed-prefix tokens and AgentShift
adds only 1.8 ms over sticky while changing placement. A separate deterministic
run produced identical 32-token outputs for sticky, full re-prefill, and the
migrated prefix. Raw data: `e2e-1784539658778872653.json` and
`correctness-state-1784539567544681245.db`.

## RQ2: Gap Overlap and Baselines

The following table uses a 100 ms blocked window and a 50 ms TTL. `TTL` keeps
the prefix on the source before expiry and evicts it after expiry. `On-return`
uses the same KV transfer mechanism as AgentShift but starts after the tool
returns.

| Prefix | Sticky | Reroute | TTL | On-return | AgentShift |
|---:|---:|---:|---:|---:|---:|
| 4K | 43.2 ms | 135.7 ms | 137.7 ms | 84.2 ms | 44.3 ms |
| 16K | 46.2 ms | 515.4 ms | 525.4 ms | 101.9 ms | 48.2 ms |
| 32K | 51.9 ms | 1193.4 ms | 1219.2 ms | 127.2 ms | 57.1 ms |

AgentShift is 1.90-2.23x faster than on-return migration and 3.06-20.9x
faster than stateless rerouting. It stays close to sticky latency while changing
the owner. TTL matches sticky before expiry, but after expiry it re-prefills on
the original engine and still cannot change placement.

At a 25 ms gap, transfer is only partly hidden. For 32K, post-tool latency is
103.0 ms for AgentShift, 134.4 ms for on-return, and 1194.3 ms for reroute.

Raw data: `blocked-window-1784536919459548545.json`.

For Qwen3-32B at TP=4, a 32K prefix requires about 32 GiB of physical KV traffic
across four rank pairs. The proactive/on-return comparison remains favorable:

| Blocked window | Reroute | On-return | AgentShift | Hidden ratio |
|---:|---:|---:|---:|---:|
| 0 ms | 1640.5 ms | 235.3 ms | 202.6 ms | 6.0% |
| 50 ms | 1636.4 ms | 213.1 ms | 157.1 ms | 57.7% |
| 100 ms | 1631.0 ms | 229.1 ms | 118.7 ms | 100% |
| 250 ms | 1631.6 ms | 214.1 ms | 124.1 ms | 100% |

At a 100 ms gap, proactive handoff is 1.93x faster than waiting until tool
return to start the identical transfer. Raw data:
`blocked-window-1784539937523322519.json`.

## RQ3: Correlated Return Bursts

With 16 agents, 128 output tokens, and a small 32-token tool observation:

| Prefix | Sticky | Reroute | AgentShift | vs reroute | Re-prefilled history |
|---:|---:|---:|---:|---:|---:|
| 8K | 2.57 s | 4.23 s | 2.60 s | 1.63x | 0 vs 65,568 |
| 16K | 2.64 s | 6.45 s | 2.66 s | 2.42x | 0 vs 131,104 |

Here sticky is already close to optimal because one batch of 16 decodes
efficiently and the new observation is small. AgentShift preserves that latency
while relocating half the future computation.

With a 1K-token tool observation, correlated prefill work makes placement more
important:

| Prefix | Sticky | Reroute | AgentShift | vs sticky | vs reroute |
|---:|---:|---:|---:|---:|---:|
| 2K | 2.94 s | 3.11 s | 2.76 s | 1.06x | 1.13x |
| 8K | 3.06 s | 4.49 s | 2.82 s | 1.09x | 1.60x |

Raw data: `burst-matrix-1784536247087227712.json` and
`burst-matrix-1784536382649192042.json`.

The earlier 2.00x result versus sticky used a 45K-token pool and included cache
pressure. It remains valid for that configuration, but the larger 270K-token
pool results above separate load placement from forced cache eviction.

## RQ4: Foreground Interference

The async path removes NCCL copy from the scheduler control loop. The table
reports a single foreground streaming decode and the mean maximum token gap over
three runs:

| Migrated prefix | No migration | Sync transfer | Async transfer |
|---:|---:|---:|---:|
| 4K | 20.1 ms | 25.4 ms | 24.9 ms |
| 16K | 20.5 ms | 34.5 ms | 26.9 ms |
| 32K | 20.0 ms | 43.1 ms | 30.0 ms |

For 32K, async transfer reduces excess stall over baseline by 56.9%. With four
foreground streams at 32K, async throughput is 192.4 tok/s versus 195.5 tok/s
without migration, a 1.6% reduction. Async copy is not free and can still
compete for memory bandwidth; bounded concurrency remains necessary.

Raw data: `interference-1784534986077328311.json`.

## RQ5: Real Tool Windows

The coding-tool replay executes real subprocesses in this repository. Values
below are three-run means for a 32K prefix:

| Tool | Duration | Reroute | On-return | AgentShift | Fully hidden |
|---|---:|---:|---:|---:|---:|
| `git status --short` | 2.8 ms | 1202.4 ms | 125.9 ms | 113.9 ms | 0% |
| state-store tests | 390.5 ms | 1194.9 ms | 130.0 ms | 53.5 ms | 100% |
| control-plane tests | 452.2 ms | 1198.4 ms | 126.5 ms | 54.1 ms | 100% |

Long coding tools completely hide migration. Very short tools do not, which is
why the scheduler must estimate remaining gap instead of moving every blocked
agent.

Raw data: `real-tools-1784536691098166342.json`.

## RQ6: Correctness and Recovery

- AgentShift control-plane tests: 14 passed.
- SGLang prefix lifecycle and TP aggregation tests: 9 passed.
- TP=1, TP=2, and TP=4 deterministic 32-token outputs match sticky and full
  re-prefill.
- Async destination state becomes visible only after every TP rank reports
  `COMPLETE` and the full token key is installed.
- TP control operations aggregate over the CPU process group, so a rank-zero
  success cannot hide a lagging or failed rank. This was exercised at TP=4,
  where the unfenced version could report `COMPLETE` while another rank was
  still at `COPY_DONE`.
- In-flight destination reservations are included in SGLang's strict pool
  invariant, preventing false leak reports without disabling the checker.
- Terminal transfer records are explicitly cleaned on both engines. Failed
  destination reservations are returned to the allocator, while successfully
  installed rows remain owned by RadixCache; subsequent status polls report an
  unknown migration instead of retaining stale GPU index tensors indefinitely.
- Pre-commit failure leaves the source owner unchanged.
- Post-commit/pre-first-token failure rebinds the source shadow and advances
  durable ownership from destination `epoch+1` to source `epoch+2`.
- ACK/source release is retryable.
- A restarted coordinator can recover a `COMMITTED` migration from the same WAL.
- A real external SQLite counter is incremented once under duplicate managed
  effect execution.

The cleanup-enabled live recovery validation preserved a full 4100-token source
hit, rejected the failed destination epoch, and recovered ownership from
`engine-b@2` to `engine-a@3`. Raw state:
`recovery-state-1784538068899360049.db`.

## Trace Scope

- Kimi 1-day trace: 245,555 requests, context p50/p90/p95 of
  8.1K/38.7K/60.4K. It has no session or tool labels.
- FlowPrefill: 43,058 requests and 19,957 parent-linked subsequent turns. Its
  inter-turn delta is only a blocked-window proxy, not a labeled tool duration.

Raw analysis: `trace-analysis-1784487419038306012.json`.

## Remaining Work

- Controlled CPU offload/reload baseline is not implemented yet.
- Placement currently uses a conservative queue/gap heuristic; the full policy
  sweep and multi-migration bandwidth scheduler remain incomplete.
- Recovery is validated by protocol-level failure injection and a restarted
  controller object, not by killing a live SGLang process mid-commit.
- Cross-node transfer and transport contention remain untested.
- Local Kimi-Linear-48B is a hybrid linear/recurrent model. Migrating only its MHA
  KV would be semantically incomplete, so it is intentionally excluded rather
  than reported as a larger-model result.
