# Paper-Story Experiment Results

Date: 2026-09-13.

This batch follows the paper narrative:

```text
prefix-heavy agentic workloads
-> state gravity and global hot/cold imbalance
-> existing policies either preserve state or balance compute
-> AgentShift moves branch-valid heterogeneous state and authority
-> one-hour replay shows balanced load and explicit migration directions
```

## Trace And Replay

### Real trace evidence

Input:

- Kimi Finish log extract: 245,555 requests over 24 hours;
- representative window: hour 18;
- hour-18 requests: 6,700;
- prompt p50 / p90 / p99: 7,208 / 39,610 / 103,674 tokens;
- cached-token p50 / p90 / p99: 5,760 / 6,784 / 58,562 tokens.

Prefix-cache weight:

| Metric | Value |
|---|---:|
| Requests reusing at least 50% of input | 70.87% |
| Cached/prompt ratio p50 | 86.15% |
| Cached/prompt ratio p90 | 99.47% |
| Cached tokens / input-token mass | 47.18% |

Thus, this is not a lightly cached workload: most requests carry a large warm
prefix, and a large fraction of total input-token work is already stateful.

Global hot/cold severity is measured per minute using logged cache-token
arrivals on the eight original DP ranks:

| Metric | p50 | p90 | max |
|---|---:|---:|---:|
| hot-engine / mean-engine cache load | 1.555x | 1.955x | 2.331x |
| hot-engine / cold-engine cache load | 2.388x | 3.600x | - |
| top-2-of-8 engines' cache share | 36.06% | 42.28% | - |

The important observation is that the load is not uniformly warm: at any minute,
a small subset of engines carries most of the cache-token traffic while other
engines remain cold. This is the state-gravity condition that defeats naive
compute elasticity.

Artifact:
`results/paper_figures/paper_figure_metrics.json`.

## One-Hour Policy Replay

The replay uses:

- 512 agents;
- 1,754 turn events;
- 9,838,401 stored tokens;
- event span 3,649 seconds;
- eight logical engines;
- scale-in from eight to four engines at 1,800 seconds;
- 300-second warm-state retention;
- 50ms transfer fixed overhead;
- 4.37 GiB/s effective bandwidth;
- full-KV plus recurrent-snapshot capacity accounting.

Construction remains trace-shaped: Kimi hour 18 provides timing, lengths,
cached-token shape and original DP placement; LooGLE provides real long-context
token content; ToolBench-comb provides nested tool boundaries and tool names.
It is not claimed to recover the original redacted Kimi token IDs.

### Existing methods

| Policy | What it does | Post-scale hot/mean p50 / p90 | Retiring states at deadline | Prefix hits | Recomputed tokens | Transfer |
|---|---|---:|---:|---:|---:|---:|
| Sticky | Keeps state and execution on the original engine | 1.467x / 1.857x | 11 | 6.858M | 0 | 0 |
| Stateless reroute | Balances compute but reconstructs context | 1.048x / 1.111x | 0 | 0 | 6.246M | 0 |
| Token-only migration | Models state as `tokens * MHA bytes/token` | 1.061x / 1.125x | 0 | 6.858M | 0 | 169.21 GiB |

Sticky preserves cache but leaves state gravity intact. Stateless reroute
removes state gravity but throws away the warm prefix. Token-only migration can
balance this particular hour, but its cost model is not execution-state valid:
earlier short-gap sweeps showed up to 54.39% decision mismatch, and KV-only
migration produces a full prefix hit while producing incorrect continuation.

### AgentShift

AgentShift's heterogeneous-aware policy produces:

| Metric | Value |
|---|---:|
| Post-scale hot/mean p50 / p90 | 1.059x / 1.143x |
| Retiring states at deadline | 0 |
| Prefix hits preserved | 6.858M tokens |
| Recomputed tokens | 0 |
| Total migrations | 1,152 |
| Drain-urgent migrations | 119 |
| Load-balance migrations | 1,033 |
| Hybrid state transferred | 185.46 GiB |

Compared with sticky, AgentShift reduces median post-scale hot/mean imbalance
from 1.467x to 1.059x. Compared with stateless reroute, it avoids 6.246M tokens
of reconstruction. Compared with token-only migration, it uses the same balance
opportunity but represents the actual full-KV plus recurrent-snapshot bundle and
enforces both token-slot and snapshot-slot capacity.

The aggregate one-hour numbers alone do not expose all token-only errors: with
long tool gaps and ample capacity, token-only can make successful moves. The
correctness boundary is in the earlier Mamba ablation and short-gap sensitivity
sweeps, not in this aggregate hour.

## Migration Directions

AgentShift made 549 migration decisions after the 1,800-second scale-in event.
The largest aggregated directions were:

| Source | Destination | Requests |
|---|---|---:|
| E03 | E02 | 100 |
| E03 | E01 | 56 |
| E03 | E00 | 54 |
| E02 | E03 | 52 |
| E06 | E03 | 47 |
| E02 | E01 | 45 |
| E04 | E03 | 44 |
| E05 | E03 | 42 |
| E01 | E03 | 39 |
| E07 | E03 | 31 |

The policy uses both load-balancing moves before scale-in and drain-urgent moves
from retiring engines E04-E07 after scale-in.

## Figures

All figures are 600-dpi PNG plus vector PDF and use the requested white
background, black axes, large fonts, and restrained blue/orange palette.

1. `results/paper_figures/fig1_workload_state_gravity.{png,pdf}`
   - prefix-cache CDF;
   - context-length demand;
   - per-engine cache-token heat map;
   - per-minute hot/mean and hot/cold imbalance.

2. `results/paper_figures/fig2_policy_comparison.{png,pdf}`
   - sticky versus stateless reroute versus token-only versus AgentShift;
   - post-scale imbalance, retiring-state residue, prefix hits, and bytes moved.

3. `results/paper_figures/fig3_one_hour_load_balance.{png,pdf}`
   - rolling hot/mean load for all policies over the full one-hour replay;
   - marks the 8-to-4 GPU scale-in event.

4. `results/paper_figures/fig4_migration_directions.{png,pdf}`
   - AgentShift engine-load heat map;
   - every per-request migration as a source-to-destination arrow over time;
   - aggregated source-destination request matrix.

## Boundary

The one-hour policy comparison is a trace-shaped simulation, not a claim of
physical cross-GPU execution for all 512 agents. Real-engine evidence already
covers exact-output handoff, concurrent non-blocking transfer, 32-agent windows,
three-engine scale-out/scale-in, and branch fanout. The remaining paper-scale
step is a longer continuous multi-engine serving deployment with measured SLO
goodput and GPU-seconds.
