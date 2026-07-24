# AgentShift PoC Results

## Setup

- 8x H100 80GB NVLink host
- Qwen3-8B, BF16 KV, 45K-token KV capacity per engine
- Main performance runs: two TP=1 SGLang engines
- TP validation: two TP=2 SGLang engines
- Single-turn comparison uses one generated token after a 1K-token tool result
- Burst comparison uses 16 agents, 2K warm prefix, 1K tool result, 128 output tokens
- Three repeats per reported point

Raw result: `e2e-1784487320239905011.json`

## Next-Turn Latency

| Historical prefix | Sticky | Reroute + re-prefill | AgentShift | Speedup vs reroute | Migration |
|---:|---:|---:|---:|---:|---:|
| 1K | 32.9 ms | 57.1 ms | 34.6 ms | 1.65x | 16.0 ms |
| 4K | 33.0 ms | 130.7 ms | 34.4 ms | 3.80x | 11.7 ms |
| 16K | 47.6 ms | 528.2 ms | 47.8 ms | 11.05x | 22.7 ms |
| 32K | 66.9 ms | 1295.7 ms | 68.5 ms | 18.93x | 41.0 ms |

AgentShift preserved complete target cache hits at every length. A 250 ms blocked
window hides every measured migration, including the 4.50 GiB 32K prefix.

The 1K migration includes fixed per-layer and control overhead. Sustained transfer
reaches roughly 100 GiB/s for the 16K and 32K cases.

## Correlated Return Burst

| Policy | Makespan | Mean p95 | Mean cached tokens |
|---|---:|---:|---:|
| Sticky source engine | 5.69 s | 5.69 s | 1667 |
| Stateless half reroute | 3.24 s | 3.24 s | 1026 |
| AgentShift half handoff | 2.84 s | 2.84 s | 2052 |

AgentShift is 2.00x faster than sticky and 1.14x faster than stateless rerouting.
Sticky loses some retained prefixes under KV pressure; rerouting balances compute but
re-prefills half the sessions. AgentShift both balances ownership and retains locality.

## Correctness

- Control-plane tests: 10 passed.
- SGLang prefix lifecycle tests: 3 passed.
- A 4K deterministic 32-token continuation is identical for sticky, full re-prefill,
  and migrated KV paths.
- The destination reports a full 4100-token hit after TP=1 and TP=2 migration.
- Destination failure injection leaves `engine-a@epoch` authoritative and records
  ABORTED.
- Old epoch and duplicate step claims are rejected before inference.
- Managed tool results and managed effect submission are idempotent.

## Trace Calibration

Raw analysis: `trace-analysis-1784487419038306012.json`

- Kimi: 245,555 requests; context p50/p90/p95 = 8.1K/38.7K/60.4K; 87.5% fit in 32K.
- FlowPrefill: 43,058 requests and 19,957 subsequent parent-linked turns; cumulative
  prefix p50/p95 = 1.1K/8.8K.
- FlowPrefill inter-turn deltas exceed the measured migration estimate for all sampled
  subsequent turns. This is only a blocked-window proxy, not a labeled tool duration.

## Residual Risk

The transport currently blocks the scheduler control path and uses per-layer staging;
foreground interference and concurrent migration scheduling remain future work. Source
release in the non-streaming benchmark happens after destination completion, which is
more conservative than the intended first-token ACK. Cross-node results, other model
architectures, TP=4/70B, crash-stop server replay, and real managed external tools have
not yet been evaluated.
