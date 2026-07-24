# Exemplar Learning Dossier

| Example | Structural feature learned | Transfer to AgentShift | Constraint |
|---|---|---|---|
| FastServe | Lead with a measured bottleneck, then make each mechanism answer a new consequence | Start with locality versus placement; ownership follows mobility; overlap follows exposed copy | Do not reuse FastServe's request-preemption framing |
| Agentix | Define the workload abstraction before discussing the scheduler | Define `suspended-but-warm` and the completed-turn boundary before implementation | Keep Agentix as routing, not a strawman |
| Symphony | Quantify state coupling and treat prediction uncertainty as a design input | Pair gap CDF/proxy with migration time; state the short-gap failure mode | Do not claim local shared-memory numbers represent official Symphony |
| BlitzScale | Decompose readiness and evaluate time to useful capacity | Separate model-ready, state-ready, and authority-ready events | Warm-pool results are not cold autoscaling |
| Llumnix | Use an OS analogy only after defining exact migration state | Contrast active-request and completed-turn migration with a timeline | Do not imply active migration is inferior outside AgentShift's boundary |
| Continuum | Tie retention policy to measured tool and queue costs | Calibrate TTL and report both cache hit and owner relocation | Mechanism-equivalent implementation must be labeled |

## Style Moves

1. Put the paragraph claim in its first sentence.
2. Use one measured number only when it changes the argument.
3. Name the exact missing capability before naming AgentShift.
4. State mechanism boundaries in the same paragraph as a comparison.
5. Use captions as claims: setting first, takeaway second.
6. Keep implementation nouns concrete: prefix pin, destination reservation,
   rank-pair transfer, ownership CAS, first-token ACK.
