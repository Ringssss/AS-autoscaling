# Source Map

Only local AgentShift artifacts support claims about AgentShift. External papers
support background, related-work boundaries, and writing structure.

| Source ID | Local source | Use |
|---|---|---|
| S1 | `docs/implementation_plan.md` | Frozen scope, protocol invariants, implementation boundary |
| S2 | `docs/evaluation_report_20260721.md` | Consolidated, manually checked result summary |
| S3 | `results/blocked-window-1784566204404099436.json` | Five-run 32K locality headline |
| S4 | `results/blocked-window-1784565792789187724.json` | Prefix/gap/baseline matrix |
| S5 | `results/hotspot-1784566696908767902.json` | Five-run correlated-return hotspot |
| S6 | `results/ttl-pressure-1784562956744594834.json` | Calibrated TTL under capacity pressure |
| S7 | `results/real-tools-all-1784548799599095961.json` | Real local coding-tool durations |
| S8 | `results/interference-1784566871298243631.json` | Five-run foreground interference headline |
| S9 | `results/interference-1784564046361063009.json` | Prefix/concurrency interference matrix |
| S10 | `results/fault-matrix-1784829560507240459.json` | Eight-case real-engine fault matrix with real DEST_READY boundary injection |
| S11 | `results/e2e-1784537113268610412.json` | Qwen3-8B TP=2 result |
| S12 | `results/e2e-1784539658778872653.json` | Qwen3-32B TP=4 result |
| S13 | `results/agent-workloads-1784819110111467791.json` | Kimi and FlowPrefill characterization |
| S14 | `results/control-plane-1784819214062797714.json` | SQLite control-plane scaling |
| S15 | `results/elasticity-*.json` | Warm scale-out and semantic scale-in; select only after validation |
| S16 | `patches/agentshift-sglang.patch` | SGLang data-plane implementation relative to the pinned upstream commit |
| S17 | `agentshift/` and `tests/` | Durable runtime, controller, and unit tests |
| S18 | `results/coding-agent-replay-1784829317183895026.json` | Eight-agent, three-turn controlled coding replay |
| S19 | `results/migration-policy-1784829520064490654.json` | Heterogeneous-prefix migration-policy matrix |
| S20 | `paper_rewriting_output/figures/fig_replay_policy.pdf` | Arial replay and scheduling figure generated from S18--S19 |
| S21 | `results/long-context-128k/RESULTS.md` and raw JSON files in that directory | 130,004-token Qwen3-8B TP=1 and Qwen3-32B TP=4 relocation results |

## External Sources

| Source ID | Paper | Role |
|---|---|---|
| R1 | FastServe, arXiv:2305.05920 | Exemplar: measured bottleneck to scheduling mechanism |
| R2 | Autellix/Agentix, arXiv 2025 / NSDI 2026 | Orthogonal program scheduling work and agent-workload framing; not an experimental baseline |
| R3 | Symphony, NSDI 2026 | Closest disaggregated KV prefetch design |
| R4 | BlitzScale, OSDI 2025 | Stateful elasticity composition and live provisioning exemplar |
| R5 | Llumnix, OSDI 2024 | Active-request migration boundary |
| R6 | Continuum, arXiv 2025/2026 | Agent KV retention and TTL policy |
| R7 | TokenCake, arXiv 2025 | Tool-gap KV offload and predictive reload |
| R8 | ServerlessLLM, OSDI 2024 | Cold model readiness and autoscaling background |
