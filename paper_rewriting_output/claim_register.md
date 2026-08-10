# Claim Register

| Claim | Evidence | Strength | Allowed wording | Avoid |
|---|---|---|---|---|
| Suspended warm state is important | E1--E3 | Moderate | Real traces show long contexts and parent-linked turns; FlowPrefill suggests clustered return opportunities | Real agents spend a measured majority of time blocked |
| AgentShift preserves locality after relocation | E4, E5, E11 | Strong | AgentShift installs a full destination prefix and matches Sticky-like next-turn latency on identical engines | AgentShift is always faster than Sticky |
| Gap overlap is a distinct contribution | E4, E5, E8 | Strong | Proactive handoff is 2.41x faster than same-transport on-return at 32K/500 ms | Every tool gap hides migration |
| AgentShift relieves correlated-return hotspots | E6 | Strong for tested case | It relocates 50% of owners with full hits and reduces the tested burst makespan | It eliminates all hotspots |
| AgentShift is better than retention/offload when placement must move | E6, E7 | Strong under constrained objective | Source-side policies match latency only by retaining source placement; under tested pressure/hotspots AgentShift is faster | AgentShift officially beats TokenCake or Continuum |
| AgentShift supports stateful elasticity | E12, E13 | Strong for warm pool | AgentShift turns already model-ready capacity into warm-agent capacity and drains an engine semantically | AgentShift implements cold autoscaling or BlitzScale |
| Async transfer has bounded interference | E9 | Moderate | In the tested eight-stream case throughput is unchanged within noise and excess arrival TTFT is 10.6 ms | Migration has zero foreground impact |
| Ownership protocol prevents double advancement | E10, E20 | Strong within model | Under injected duplicate delivery, router-only prefetch executes at both real engines while fencing rejects the stale source before GPU admission | Epoch fencing is the only possible single-winner mechanism; arbitrary third-party effects are exactly-once |
| Source shadow improves recovery | E10 | Strong | It reduces the tested recovery point by about 10.9x; cold reconstruction remains correct | Source shadow is required for correctness |
| Controller scales adequately for prototype | E14 | Moderate | Control throughput exceeds the current serialized KV data plane; concurrent SQLite writes have high tail latency | SQLite is cluster-scale or highly available |
| Multi-turn replay preserves mobility benefits | E16 | Moderate | In the controlled eight-agent coding replay, AgentShift moves half the owners with full hits and the lowest makespan | AgentShift has been validated on complete SWE-agent or BFCL tasks |
| Gap-aware ordering improves completion coverage | E17 | Moderate | Admissible-first ordering completes 83.3% of the tested handoffs in-gap versus 33.3--50.0% for simple orders | The policy minimizes every latency metric or is globally optimal |
| Cross-node applicability | Design only | Unsupported experimentally | The protocol is transport-independent in structure and cross-node evaluation is future work | AgentShift has RDMA results |
| 130K mobility | E19 | Strong for tested configurations | AgentShift moves and fully hits a 130,004-token prefix on Qwen3-8B TP=1 and Qwen3-32B TP=4; at a 250 ms gap it is 132.30x and 75.54x faster than reroute | Native Qwen3 context is 128K without YaRN; every longer prefix is always beneficial |
