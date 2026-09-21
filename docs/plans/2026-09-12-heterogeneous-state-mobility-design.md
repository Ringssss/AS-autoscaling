# Heterogeneous State Mobility Design

## Goal

Upgrade AgentShift from homogeneous-MHA prefix migration to safe mobility of a hybrid LLM agent's complete execution state.

## Approved Thesis

Elastic compute requires mobile execution state, but modern hybrid LLM state is heterogeneous, branch-sensitive, and layout-dependent. A scheduler cannot safely move an agent from a token count alone.

## Core Abstraction

`StateDescriptor` describes the components required to make one branch executable at a destination:

- component type: token-indexed full-attention KV, recurrent linear-attention state, or control metadata;
- validity scope: model, dtype, TP/layout, prefix, position, and branch epoch;
- footprint: destination token slots and recurrent snapshot slots;
- materialization choices: direct copy, grouped-layer copy, replay, or reconstruction;
- install dependencies: when the bundle is complete and may become publicly visible.

## Main Mechanism

Gap-aware heterogeneous state materialization is the primary mechanism. During a tool wait, the system reserves destination resources and materializes a valid state bundle privately. The destination does not expose the prefix or claim execution authority until the full bundle is ready. `PREPARE`, `COMMIT`, `PROMOTE`, and `RELEASE` retain their AgentShift meanings but operate on a bundle rather than KV rows alone.

Branch-aware state planning is a policy under this mechanism. For each movable branch, it selects among copying an existing recurrent snapshot, replaying a suffix, or reconstructing state, subject to branch validity and the remaining tool gap.

## First Implementation Milestone

Build a correctness-first, same-host/two-engine Qwen3.5 prototype:

1. Export a completed branch's full-attention K/V rows and Mamba conv/temporal snapshot from SGLang's `MambaRadixCache`.
2. Save a `StateDescriptor` manifest plus tensors to an engine-local checkpoint.
3. Import into a second, compatibility-matched engine by reserving token and Mamba slots, copying both state families, and atomically inserting into `MambaRadixCache`.
4. Send identical continuation token IDs to source and destination and compare output IDs.

This is not yet the performance path. Host-file staging is deliberately simple so correctness and install semantics can be validated before adding direct GPU/RDMA transfer.

## Validation Scope

- L0: two real Qwen3.5 servers initialize with hybrid pools and the export/import endpoints.
- L1: descriptor shape, dtype, compatibility, and branch-validity checks.
- L3: cold source, imported destination, cold destination, and source continuation all produce identical greedy token IDs; imported destination reports an exact prefix hit.
- Training L2 is not applicable.

## Non-Goals For This Milestone

- No generic recurrent-cache replacement.
- No cross-model or cross-TP conversion.
- No claim of optimized cross-node transfer.
- No ownership store changes until state bundles are demonstrably correct.
