# SOTA Gap Map

| Design point | State action | Can change next-turn engine? | Preserves full prefix? | Transfers durable owner? | Remaining gap |
|---|---|---:|---:|---:|---|
| Sticky | Retain on source | No | Yes | No | Source hotspot remains |
| Program-aware call scheduling (Related Work only) | Prioritize and place arrived calls | Not a mobility action | Not evaluated here | No cross-turn handoff | Orthogonal objective; no performance baseline |
| Continuum-style TTL | Retain, then evict | No | Until expiry | No | No relocation action |
| TokenCake-style | Offload and reload | Usually no | Yes after reload | No | Memory mobility is not execution mobility |
| Symphony-style | Prefetch from shared tier | Yes | Yes | No | Availability does not fence stale execution |
| Llumnix | Migrate active request | Yes | Yes | Request scoped | No active request exists during the target gap |
| On-return handoff | Direct transfer after tool return | Yes | Yes | Yes | Copy remains on the critical path |
| AgentShift | Proactive direct transfer plus CAS | Yes | Yes | Yes | Limited to compatible engines and completed turns |

## Defensible Novelty

AgentShift is the first evaluated design in this testbed that simultaneously
relocates future execution, preserves a complete prefix hit, prepares state
during the blocked interval, and durably fences the old owner. This is a claim
about the evaluated capability set, not an assertion that no unpublished or
concurrent system can compose similar mechanisms.
