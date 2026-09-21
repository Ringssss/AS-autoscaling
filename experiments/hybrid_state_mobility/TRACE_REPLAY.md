# Kimi/LooGLE/ToolBench Mobility Replay

Date: 2026-09-12. Artifacts are under
`experiments/hybrid_state_mobility/results/kimi_loogle_toolbench/`.

## Input Audit

The Kimi dataset contains 245,555 Finish records over 24 hours. Its token IDs
and image URLs are redacted, and the visible records do not carry conversation
IDs. The fields that remain reliable are request received/finished timestamps,
prompt length, completion length, cached-token count, E2E latency, TTFT, DP
rank, bootstrap host, and RID.

Consequently, this workload is explicitly **trace-shaped replay**, not recovery
of original conversations:

- Kimi supplies timing, prompt/output/cache length shape, and source placement;
- LooGLE `longdep_qa` supplies deterministic real long-context token content;
- ToolBench-comb supplies nested tool-boundary prompts, tool names, and branch
  nesting;
- Tool durations are sampled deterministically from the repository's prior
+measured formal tool-gap campaign.

No claim is made that reconstructed token IDs are the original Kimi IDs.

The extracted Kimi distributions are:

| Field | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| prompt tokens | 8,090 | 38,738 | 102,749 | 125,446 |
| cached tokens | 5,760 | 5,888 | 50,240 | 114,816 |
| completion tokens | 350 | 1,665 | 7,233 | 16,000 |

Artifact: `data/kimi_finish.npz`.

## Replay Construction

Selected source window: Kimi minute 206, the highest-arrival minute with
prompts capped at 32K.

Generator settings:

- 128 agents;
- 442 turn events;
- 8 logical engines;
- Qwen3.5 tokenizer;
- LooGLE long-dependency QA contexts;
- ToolBench-comb repeat 0 rows;
- seed 20260407.

For each agent, the same LooGLE token base is prepended to a nested
ToolBench-comb branch family. A deeper turn is normalized to the prior turn's
token prefix followed by its post-LCP suffix; this preserves exact token-prefix
nesting even when BPE boundary tokens differ slightly from text nesting.

Summary:

| Metric | Value |
|---|---:|
| Agents | 128 |
| Events | 442 |
| Prompt p50 / p90 / p99 / max | 5,824 / 6,324 / 17,643 / 19,643 |
| Median parent-prefix ratio | 97.07% |
| Tool duration p50 / p90 / max | 1.057s / 1.977s / 3.088s |
| Replay event span | 0-257.1s |
| Stored token arrays | 442 |
| Stored tokens | 2,516,053 |

Artifacts:

- `replay_manifest.jsonl`
- `replay_tokens.npz`
- `replay_summary.json`

Deterministic invariants passed:

- every event has a token array;
- each deeper turn exactly extends the prior turn's token prefix;
- next-turn arrival is at least the previous tool end.

## Global Mobility Simulation

The new `GlobalMobilityPolicy` is an event-driven controller for agent/tool
boundaries and elasticity events. It uses:

- branch-valid `StateDescriptor`;
- destination full-KV token capacity;
- destination recurrent snapshot capacity;
- direct-copy/replay/reconstruct timing;
- source versus destination load;
- retiring-engine drain urgency.

Simulation settings:

- 8 engines;
- 262,144 token slots and 32 recurrent snapshot slots per engine;
- scale-in at 60s from 8 to 4 engines;
- 180s drain deadline;
- measured direct-transfer model: 4.37 GiB/s plus 20ms fixed overhead.

| Policy | Migrations | Drain migrations | Retiring states at 180s | Load p50/p90/max |
|---|---:|---:|---:|---:|
| Sticky | 0 | 0 | 66 | 14 / 18 / 22 |
| Token-only | 246 | 46 | 18 | 12 / 25 / 28 |
| Heterogeneous-aware | 285 | 50 | 14 | 13 / 26 / 32 |

The main result is drain effectiveness, not peak load: heterogeneous-aware
mobility leaves 14 recurrent states on retiring engines at the deadline, versus
18 for token-only and 66 for sticky. It also performs 285 branch-valid
materializations while respecting both token and snapshot capacity.

Artifact: `global_mobility_simulation.json`.

## Real Two-Engine Trace Window

Four selected replay agents ran on two real Qwen3.5 engines. Each agent:

1. prefilled its first LooGLE+ToolBench prompt on source;
2. entered the ToolBench tool wait;
3. migrated the complete hybrid bundle through the durable direct coordinator;
4. continued its nested next turn on destination.

| Agent | Tool | Tool gap | Parent prefix | Migration wall | Checks |
|---|---|---:|---:|---:|---:|
| trace-window-00 | `pe_demo` | 0.973s | 5,824 | 511ms* | 5/5 |
| trace-window-01 | `amazon_data_scraper_v2` | 1.388s | 5,824 | 33.8ms | 5/5 |
| trace-window-02 | `reddit_meme` | 0.358s | 5,559 | 33.3ms | 5/5 |
| trace-window-03 | `lm_api` | 1.075s | 5,696 | 33.6ms | 5/5 |

*The first wall time includes persistent-group warm-up. Its scheduler-side
transfer time was 34.3ms; the following scheduler-side transfers were around
26.8-27.1ms.

All four checks passed for every agent:

- destination owner committed;
- first source request had zero cached tokens;
- next destination request hit the complete parent prefix;
- transferred bundle size matched `prefix * 20480 + 64389120`;
- tool gap covered source generation plus materialization.

Artifacts:

- `trace_direct_window.json`
- `source-trace.log`
- `destination-trace.log`

## Boundary

The global simulation is a placement/lifecycle replay, not a full serving
throughput experiment. The real trace window is synchronous and uses idle
schedulers. The next step remains non-blocking transfer under concurrent
foreground serving, followed by longer multi-engine scale-in/out runs with SLO
and GPU-seconds accounting.
