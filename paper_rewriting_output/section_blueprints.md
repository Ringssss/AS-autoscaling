# Section Blueprints

## Abstract

- Define the locality--placement conflict for suspended agents.
- State the completed-turn insight.
- Name the three mechanisms and their benefit.
- Report 130K latency, burst, elasticity, and fault evidence with scope qualifiers.

## 1 Introduction

- Open with long-lived agents alternating between LLM turns and interrupts.
- Define suspended-but-warm and show why request-oriented serving misses it.
- Explain why routing, retention/offload/prefetch, and active migration each leave a gap.
- Present AgentShift as execution mobility, not KV copy.
- Summarize evidence and list three contributions.

## 2 Background and Motivation

- Explain cross-turn KV locality and its placement consequence.
- Use Kimi/FlowPrefill characterization with explicit schema limitations and reserve the true lifecycle panel as TODO until a labeled runtime trace exists.
- Formalize the constrained objective: destination execution, full hit, hidden preparation, one owner.
- State design principles: durable progress is authoritative; KV is reconstructable acceleration state.

## 3 Overview

- Define the completed-turn boundary and supported engine compatibility.
- Walk through tool start, pin/reserve/copy, destination readiness, ownership CAS, mailbox rebind, ACK, release.
- State safety and visibility invariants.

## 4 Continuation and Ownership

- Motivate why moving KV does not transfer authority.
- Define continuation, epoch lease, step claim, mailbox, and effect lifecycle.
- Specify pre-commit abort and post-commit recovery.

## 5 Completed-Prefix Mobility

- Motivate avoiding historical prefill after placement changes.
- Describe source pin, destination reservation, rank-pair transfer, all-rank completion, exact-key installation, cleanup.
- Explain immutable prefix and identical TP/model assumptions.

## 6 Gap-Aware Semantic Handoff

- Define eligibility and simple cost/admission policy.
- Explain proactive versus on-return timing.
- Describe async worker, bounded transfer concurrency, readiness/commit ordering, and source shadow release.
- Present warm elasticity as a consequence.

## 7 Implementation

- Identify SGLang commit, MHA/RadixCache scope, NCCL groups, CUDA streams, SQLite WAL, HTTP control path.
- Report the concurrent-ACK bug and per-engine serialization fix as implementation detail, not a contribution.

## 8 Evaluation

- Setup and baseline capability table.
- Do not include Autellix/Agentix as an experimental baseline; keep it in Related Work as orthogonal program scheduling.
- RQ1 locality after relocation.
- RQ2 gap overlap, multi-turn coding replay, and heterogeneous migration ordering.
- RQ3 correlated-return hotspots and TTL/offload boundaries.
- RQ4 warm scale-out and semantic scale-in.
- RQ5 foreground interference.
- RQ6 fault semantics and recovery.
- RQ7 model/TP and control-plane scale.

## 9 Discussion and Limitations

- Single node and no cross-node RDMA.
- Identical model/revision/TP and MHA-only completed turns.
- Warm pool, not cold model autoscaling.
- Central SQLite fault model and high concurrent-write p99.

## 10 Related Work and 11 Conclusion

- Group related work by routing/scheduling, KV residency/prefetch, migration/autoscaling, and durable runtimes.
- Close with execution mobility as the new abstraction and cross-node transport as the next validation step.
