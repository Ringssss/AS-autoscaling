# Figure Asset Map

The full drawing specification is in `FIGURE_PLAN.md`. The LaTeX draft reserves
17 slots; Figure 1 is measured and the remaining slots use placeholder boxes.
Replacing a placeholder must not change its caption-level claim.

| Figure | Draft asset | Status | Evidence anchor | LaTeX label |
|---:|---|---|---|---|
| 1 | Growing agent state + rerouting cost | Final measured PDF and 600 dpi PNG | Raw FlowPrefill trace, E4 | `fig:motivation` |
| 2 | Suspended-warm lifecycle | Placeholder, experiment TODO | New labeled agent-runtime trace required | `fig:lifecycle` |
| 3 | Agent aging and reconstruction cost | Placeholder, data partly ready | E1--E5, E19 | `fig:aging` |
| 4 | Correlated return proxies | Placeholder, data ready as proxy | E3, E6 | `fig:returns` |
| 5 | System architecture | Placeholder; prior editable asset exists as `figures/fig_architecture.drawio` | Design and implementation | `fig:architecture` |
| 6 | Semantic handoff sequence | Placeholder, ready to draw | Protocol implementation and E10 | `fig:protocol` |
| 7 | Handoff/recovery state machine | Placeholder, ready to draw | Durable states and E10 | `fig:state-machine` |
| 8 | TP completed-prefix mobility | Placeholder, ready to draw | SGLang patch, E11, E18, E19 | `fig:prefix-mobility` |
| 9 | Core context/model/TP performance | Placeholder, experiment ready | E4, E5, E11, E19 | `fig:core-performance` |
| 10 | Gap overlap and opportunity coverage | Placeholder, left ready/right TODO | E4, E5, E8; labeled gap trace still needed | `fig:gap` |
| 11 | Multi-turn replay and migration policy | Placeholder, controlled replay ready | E16, E17 | `fig:replay-policy` |
| 12 | Hotspot Pareto and queue relief | Placeholder, experiment ready | E6 | `fig:hotspot` |
| 13 | TTL/offload/shared-tier alternatives | Placeholder, mechanism-equivalent data ready | E6, E7 and shared-tier result | `fig:alternatives` |
| 14 | Warm scale-out readiness | Placeholder, experiment ready | E12, E13 | `fig:scaleout` |
| 15 | Semantic scale-in drain | Placeholder, experiment ready | E12, E13 | `fig:scalein` |
| 16 | Foreground interference/concurrency | Placeholder, one point ready/matrix TODO | E9; broader matrix required | `fig:interference` |
| 17 | Fault semantics and shadow recovery | Placeholder, experiment ready within current model | E10 | `fig:fault-recovery` |

Existing generated figures remain useful as data checks, but they do not yet
match the frozen 17-figure narrative. Every final experimental asset must use
the Times-compatible style specified in `FIGURE_PLAN.md`.
