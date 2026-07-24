# Final Adversarial Self-Review

## Contribution

**Pass.** The paper identifies a state not represented by active-request migration: an agent with no active request, an immutable completed prefix, a pending external result, and live future authority. The contribution is the semantic handoff of continuation, acceleration state, and ownership at this boundary. The Introduction and capability table make the abstraction separable from a faster tensor copy.

## Writing Clarity

**Pass.** Every design section begins with the problem its component solves. Terminology is stable: `suspended-but-warm`, `completed-prefix mobility`, and `semantic handoff` retain one meaning. The state machine gives a concrete commit boundary, and implementation scope names the exact SGLang commit and compatibility assumptions.

## Experimental Strength

**Pass within the tested single-node setting.** The strongest controlled comparison is On-return because it shares transfer, installation, and ownership code; AgentShift is 2.41x faster at 32K/500 ms. Reroute, hotspot, elasticity, interference, TP/model, TTL, multi-turn coding replay, migration ordering, and fault results test distinct claims. The short 8.9 ms tool and the deferred policy tail are reported as failure modes rather than hidden.

## Evaluation Completeness

**Needs new infrastructure-dependent experiments before an NSDI submission.** The paper now includes an eight-agent three-turn replay and a serialized migration-policy sweep, but it still lacks a complete SWE-agent/BFCL replay, concurrent transfer groups, independent-process crash-stop injection, and cross-node network results. Mechanism-equivalent Agentix/Continuum/TokenCake/Symphony comparisons are labeled and must not be described as official reproductions. These gaps are explicit in Section 9 and do not invalidate the prototype claims.

## Method Soundness

**Pass for homogeneous warm pools.** Epoch CAS and step claims define one valid executor; full-rank completion defines prefix visibility; pre/post-commit recovery is deterministic. The method deliberately excludes active decode, heterogeneous TP, MLA/Mamba, cold model loading, arbitrary exactly-once effects, and multi-controller availability.

## Claim-Evidence Audit

| Claim | Evidence | Status |
|---|---|---|
| Relocation preserves locality | E4, E5, E11 | Supported |
| Gap overlap is distinct from mobility | E4, E5, E8 | Supported |
| Correlated-return load can be relieved | E6, E7 | Supported for tested bursts |
| Warm scale-out and semantic drain work | E12, E13 | Supported for model-ready targets |
| Async copy has bounded interference | E9 | Supported for one migration/eight streams |
| Fencing prevents double advancement | E10 | Supported within evaluated fault model |
| Source shadow accelerates recovery | E10 | Supported; performance, not correctness |
| Multi-turn coding replay preserves full-hit mobility | E16 | Supported for controlled subprocess tools |
| Admissible-first ordering improves in-gap coverage | E17 | Supported for one serialized trace |
| Official SOTA systems are outperformed | None | Not claimed |
| Cross-node performance is established | None | Not claimed |
