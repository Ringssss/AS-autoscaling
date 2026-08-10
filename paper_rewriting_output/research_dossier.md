# Research Dossier

## Target Venue

AgentShift targets USENIX NSDI. The paper must therefore establish a systems
abstraction, a protocol with explicit safety properties, an implementation in a
real serving engine, and experiments that separate effectiveness, causality,
interference, failure behavior, and scale. The current single-node prototype can
support the core execution-mobility claim. It cannot yet support claims about
cross-node RDMA performance or independent node-failure domains.

## Deep-Read Papers

### FastServe (arXiv:2305.05920)

FastServe starts with one measured bottleneck: head-of-line blocking dominates
latency under skewed LLM requests. It then derives token-granularity preemption,
shows why preemption creates a KV-memory problem, and introduces proactive
offload as the enabling mechanism. Its reusable writing move is to make each
component the consequence of the previous component. AgentShift should follow
the same causal discipline: placement stickiness motivates state mobility;
state mobility requires ownership; exposed mobility motivates gap overlap.

### Autellix / Agentix (arXiv 2025 / NSDI 2026)

Autellix, later published as Agentix, first defines agentic programs as dynamic DAGs of LLM calls and
interrupts. It then measures program-level waiting and shows that call-level
schedulers lack program context. PLAS/ATLAS and program-aware routing follow
from that observation. It is an orthogonal scheduling system, not an
apple-to-apple AgentShift baseline. AgentShift uses its workload and rhetorical
organization as an exemplar, and cites the system in Related Work only.

### Symphony (NSDI 2026)

Symphony quantifies state-compute coupling, session stickiness, and load
imbalance before proposing disaggregated KV storage. Advisory requests move KV
off the critical path, while priority and cooperative memory management handle
unreliable hints. This is the closest performance design point. AgentShift must
not claim that shared storage is invalid. It should show that state availability
is complementary to, but does not define, single-owner continuation handoff.

### BlitzScale (OSDI 2025)

BlitzScale separates model autoscaling into data-plane readiness and live
execution. It uses the compute network to multicast parameters and layer-level
cooperative execution to avoid stop-the-world loading. The reusable abstraction
is readiness decomposition. AgentShift similarly distinguishes model-ready,
state-ready, and authority-ready capacity. The systems are complementary:
BlitzScale creates model capacity; AgentShift makes existing warm agents able to
use that capacity without reconstructing history.

### Llumnix (OSDI 2024)

Llumnix uses live migration to reschedule active LLM requests across instances.
It overlaps append-only KV transfer with continued decoding and unifies load
balancing, defragmentation, priorities, and autoscaling. AgentShift deliberately
targets another semantic point: no active request exists, the completed prefix
is immutable, and a pending external interrupt separates two LLM turns. This
boundary enables transfer of continuation and future authority, not only a
running request's memory.

### Continuum (2025/2026)

Continuum identifies end-of-turn eviction as harmful for short tool calls. It
assigns a TTL that balances recomputation/reload and queueing benefits against
GPU residency cost. Its action space is retain or evict on the source. AgentShift
adds relocate. A fair comparison therefore needs both latency and whether the
source owner and future computation moved.

### TokenCake (2025)

TokenCake uses function-call stalls to offload inactive KV and reload it before
the next turn. It validates the same opportunity window but optimizes memory-tier
residency. AgentShift directly transfers a completed prefix to another engine and
commits semantic ownership. The local TokenCake-style implementation is a
mechanism-equivalent baseline, not an official artifact reproduction.

### ServerlessLLM (OSDI 2024)

ServerlessLLM accelerates cold model startup through multi-tier parameter
caching. It establishes model readiness, not warm-session readiness. It belongs
in the elasticity discussion and is not a baseline for completed-prefix
handoff between already loaded identical engines.

## Research Synthesis

Prior systems expose three separate axes: program-aware placement, KV residency,
and active-request migration. None defines a commit protocol for relocating a
suspended agent's immutable prefix, pending continuation, and right to execute
the next turn. AgentShift's novelty is the composition of those states at a
semantic boundary, not a new tensor-copy primitive.
