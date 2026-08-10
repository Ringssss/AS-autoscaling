# AgentShift NSDI Figure Plan

This plan freezes a 17-figure narrative. A figure occupies a slot only when it
answers one reviewer question. Autellix/Agentix is not an experimental baseline:
it is an orthogonal program scheduler and appears only in Related Work.

## Global visual rules

- Experimental plots: Times New Roman or a metrically compatible serif fallback,
  8 pt labels, 7.5 pt ticks, 1.0 pt axes, and 1.5 pt data lines. Export vector
  PDF and 600 dpi PNG.
- Palette: AgentShift `#0072B2`, Sticky `#4D4D4D`, Reroute `#D55E00`,
  On-return `#E69F00`, residency baselines `#009E73`, and unsupported/TODO
  regions `#BDBDBD`. Use both color and marker/hatch encodings.
- Layout: no chart title, no decorative background, no gradient, white canvas,
  light horizontal grid only when values need alignment, and legends inside
  unused plot space.
- Error bars: show min--max for three repeats and 95% confidence intervals when
  at least five independent repetitions are available. State the convention in
  each caption.
- Schematics: square-cornered engine and store containers, left-to-right flow,
  solid arrows for data, dashed arrows for control, and red crossed arrows for
  forbidden execution.

## Frozen figure sequence

| Fig. | Type | Evidence status | What it shows and how to draw it | Claim supported |
|---:|---|---|---|---|
| 1 | Trace + baseline experiment | Ready | Two panels, with no AgentShift result. (a) FlowPrefill turn index versus cumulative prefix p50/p90. (b) Measured post-tool time to first token at 4K/16K/32K for staying on the source with KV reuse and rerouting to another engine with historical re-prefill. | Agent state grows across turns. As it grows, KV locality increasingly constrains placement: staying preserves reuse, while rerouting reconstructs a growing history. |
| 2 | Workload experiment | TODO for full version | Four-state lifecycle breakdown for real coding and function-calling agents: `LLM_RUNNING`, `TOOL_BLOCKED`, `READY`, `FINISHED`. Left panel is stacked time share; right panel is blocked-agent KV GiB over time with GPU compute utilization overlaid. Collect timestamps from a labeled agent runtime; do not derive this figure from Kimi or FlowPrefill. | Agents spend measurable intervals unable to compute while retaining placement-coupled KV. |
| 3 | Trace experiment | Partly ready | Two panels. (a) Turn index versus cumulative prefix p50/p90/p99 from parent-linked FlowPrefill sessions. (b) Prefix length versus measured stateless reroute cost for Qwen3-8B and 32B, including 4K--130K. Use a log y-axis only if 4K values become unreadable. | Relocation value increases as agents age and accumulated history grows. |
| 4 | Trace experiment | Ready as proxy | Left: CDF of child arrivals per 10/50/100/500 ms window. Right: one 5 s replay timeline with tool-return proxies colored by source engine and queue depth below it. Explicitly label FlowPrefill deltas as return proxies, not tool durations. | Warm-agent returns can cluster and amplify a source-local queue. |
| 5 | Architecture schematic | Ready to draw | Four horizontal layers: agent runtime; continuation/mailbox/effect store; mobility controller; source and destination SGLang engines. Solid blue path moves completed-prefix KV; dashed black path performs readiness and ownership CAS; green path delivers the tool result to the new owner. | AgentShift coordinates logical continuation, acceleration state, and execution authority. |
| 6 | Protocol sequence diagram | Ready to draw | Swimlanes for runtime, controller/store, source, destination, and tool. Show `TOOL_SUBMIT`, source pin, destination reserve, async copy, all-rank `DEST_READY`, owner CAS `epoch e -> e+1`, mailbox rebind, tool completion, next-turn claim, first-token ACK, and source release. Shade the tool gap and mark the CAS as the commit point. | State preparation precedes authority transfer; source state is released only after useful destination execution. |
| 7 | State-machine schematic | Ready to draw | States: `PREPARING`, `SOURCE_PINNED`, `DEST_RESERVED`, `COPYING`, `DEST_READY`, `COMMITTED`, `FIRST_TOKEN`, `SOURCE_RELEASED`, plus `ABORTED`. Color all pre-CAS states source-owned and post-CAS states destination-owned. Add recovery arrows for controller restart, destination failure, and lost ACK. | Every fault point has a deterministic owner and cleanup/recovery action. |
| 8 | Prefix-mobility schematic | Ready to draw | Show TP rank pairs and per-layer KV shards. Source ranks pin blocks, destination ranks reserve blocks, NCCL arrows copy matching shards, and a barrier feeds one atomic RadixCache install. Include exact full-token-key validation and a crossed-out partially installed node. | A destination prefix is invisible until every layer and TP rank is complete. |
| 9 | Core performance experiment | Ready | Three panels. (a) Post-tool latency versus prefix length for Sticky, Reroute, On-return, AgentShift. (b) 130K Qwen3-8B TP=1 bars. (c) 130K Qwen3-32B TP=4 bars. Annotate destination full-hit rate and KV GiB moved, not only speedup. | AgentShift changes placement with Sticky-like latency and the benefit grows at long context. |
| 10 | Gap-overlap experiment | Partly ready | Left: post-tool latency versus tool gap (0--1000 ms) for On-return and AgentShift, with the Sticky floor. Overlay the measured `max(0, Tcopy - Tgap)` envelope. Right: heatmap of real labeled tool gaps that fully cover migration by prefix size and backend. Keep the heatmap grey/TODO until labeled traces or cross-node data exist. | Tool stalls hide completed-prefix mobility when the remaining gap exceeds transfer time. |
| 11 | End-to-end/policy experiment | Ready for current controlled replay | Left: three-turn coding replay makespan and relocated-owner fraction. Right: in-gap completion coverage for FIFO, shortest-KV, earliest-return, admissible-first, and oracle. Add p95 exposed delay as dots so coverage is not presented as the only objective. | Mobility persists across turns, and admission/order determine whether migrations finish in-gap. |
| 12 | Hotspot experiment | Ready | Left: Pareto scatter with post-tool latency on x and source-owner relief on y; filled markers mean full prefix hit and thick outlines mean owner fencing. Right: source and destination queue depth over time for Sticky, Reroute, On-return, and AgentShift in the eight-agent 32K burst. | AgentShift reaches the low-latency, high-relief region unavailable to locality-only or stateless policies. |
| 13 | Mechanism-baseline experiment | Ready, mechanism-equivalent | Three panels under the same fixed relocation/pressure scenarios: TTL cache-hit and makespan versus gap; TokenCake-Source HBM relief versus owner relief; direct GPU handoff versus shared-tier prefetch+CAS latency/bytes. Label every literature-inspired implementation `mechanism-equivalent`, never `official`. | Retention/offload manages residency, and shared prefetch manages availability; neither alone is semantic execution handoff. |
| 14 | Warm scale-out experiment | Ready | Stacked readiness timeline from target admission: model-ready, state-ready, authority-ready, first useful token. Plot Sticky, semantic reroute, On-return, and AgentShift; model-ready begins at zero because this is a warm pool. Add relocated-owner fraction above each bar. | Model-loaded capacity is not useful to warm agents until state and authority are ready. |
| 15 | Semantic scale-in experiment | Ready | Source-owned agent count over time after `DRAINING` for Sticky, semantic reroute, On-return, and AgentShift. Add a compact inset for re-prefilled tokens and wasted GPU-seconds. Sticky remains above zero and must be marked `does not drain`. | Completed-turn handoff drains a stateful engine without waiting for agent termination or rebuilding history. |
| 16 | Interference/concurrency experiment | Partly ready | Matrix of foreground load (25/50/75/90%) by concurrent migrations (1/2/4/8). Each cell reports TTFT p99 overhead; a lower strip reports migration p95 and completed-in-gap fraction. Use the existing one-migration/eight-stream point as ready and grey all unmeasured cells. | Asynchronous transfer needs admission control to bound foreground tail latency. |
| 17 | Fault/recovery experiment | Ready within current model | Left: fault-point matrix with duplicate steps, lost effects, leaked blocks, and recovery result. Right: recovery latency bars for source shadow versus cold reconstruction, with 48.1 ms and 523.2 ms anchors. A no-fencing inset shows both executors accepting the same step. | Epoch fencing preserves single-owner progress, and source shadow reduces recovery cost without becoming authoritative state. |

## Placement in the paper

- Pages 1--4: Figures 1--4 establish the problem before the system design.
- Design: Figures 5--8 map one-to-one to overview, protocol, data plane, and
  recovery semantics.
- Evaluation: Figures 9--17 follow the evaluation questions in causal order:
  locality, overlap, multi-turn policy, hotspots, alternatives, elasticity,
  interference, and faults.
- Control-plane throughput remains a table because a separate plot would add a
  figure without advancing the main execution-mobility argument.
