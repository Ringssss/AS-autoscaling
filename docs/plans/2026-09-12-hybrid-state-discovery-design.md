# Hybrid Agentic State Discovery Design

## Goal

Identify whether Qwen3.5-35B-A3B's hybrid full-attention plus linear-attention state changes the migration, cache accounting, and autoscaling assumptions of AgentShift.

## Hypotheses

1. Hybrid models have two independently bounded cache resources: token-indexed full-attention KV and fixed-size linear-attention recurrent states. A token-only scheduler can make invalid placement or eviction decisions.
2. For agentic chains, the relative migration/recompute cost crosses over with prefix length and branch count. Long chains favor migrating both state families, while many branches can amplify linear-state cost because full-attention KV is shared by prefix but each branch requires an SSM state.
3. Existing AgentShift results for homogeneous MHA are not directly transferable because its SGLang data plane assumes homogeneous MHA KV layout.

## Validation Scope

- L0 engineering: verify local SGLang imports, inspect exact Qwen3.5 memory-pool dimensions, launch a small smoke server if model loading fits, and record startup/cache-pool facts.
- L1 process/state metrics: derive state sizes from source and config, validate tree accounting invariants against SGLang code, and replay real anonymous agentic traces with exact per-state-family costs.
- L2 overfitting: not applicable; this is inference/serving analysis, not model training.
- L3 end-to-end: issue cold and repeated-prefix requests to a launched server, record prefix hit and latency. Cross-engine migration is expected to fail or be unsupported for hybrid state and that outcome is itself a compatibility finding, not a performance conclusion.

## Approaches

1. Source-first structural audit plus trace-driven cost replay. Lowest risk and directly testable.
2. One-engine Qwen3.5 serving smoke to validate pool sizing and cache-hit behavior. Moderate cost, needed before claiming runtime relevance.
3. Two-engine migration attempt only after one-engine smoke passes. Highest implementation cost because AgentShift is pinned to older homogeneous MHA SGLang.

The chosen order is 1, then 2, then only the compatibility probe in 3.

## Controls and Metrics

- Full-attention KV bytes per token, linear state bytes per branch snapshot, crossover context length, unique branch count, tool-gap overlap budget, and cache-pool split.
- Trace metrics are separated by split and by chain length; no synthetic fanout is injected.
- Simulated transfer times are reported only as bandwidth-normalized sensitivity, not as measured migration performance.
