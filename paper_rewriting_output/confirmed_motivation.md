# Confirmed Motivation

The manuscript uses **agent execution mobility** as its controlling motivation.
Long-lived agents repeatedly enter a suspended-but-warm state between LLM turns:
the completed prefix is immutable and valuable, the agent cannot execute while
waiting for an external interrupt, and future computation remains pinned by KV
locality. Existing routing, retention, offload, prefetch, and active-request
migration mechanisms do not jointly relocate this state and its execution
authority.

AgentShift treats the completed-turn boundary as a safe handoff point. It moves
the completed prefix during the blocked interval and atomically commits which
engine may advance the next step. Warm-pool scale-out and semantic scale-in are
presented as consequences of this abstraction, not as a separate cold-start
autoscaling contribution. Cross-node RDMA is explicitly outside the current
evaluation scope.
