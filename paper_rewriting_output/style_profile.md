# Style Profile

The paper uses concise, information-dense systems prose. Each paragraph carries
one claim and opens with that claim. Sentences prefer direct subject-verb-object
structure. Technical nouns are defined once and then used consistently.

## Terminology

- Use **agent execution mobility** for the paper-level abstraction.
- Use **suspended-but-warm** for a completed LLM turn blocked on an interrupt
  while its prefix remains useful.
- Use **completed-prefix mobility** for KV transfer and installation.
- Use **semantic handoff** for destination readiness plus ownership CAS.
- Use **mechanism-equivalent** for locally reimplemented literature baselines.
- Use **warm-pool elasticity**, not autoscaling, for already loaded targets.

## Claim Discipline

- Say `matches the sticky latency floor`, not `beats Sticky`, when differences
  are within run noise.
- Say `among evaluated relocation designs`, not `outperforms all SOTA systems`.
- Qualify failure claims with `within the evaluated fault model`.
- Qualify managed effects as `at-most-once submission`, not arbitrary exactly-once.
- Call FlowPrefill inter-turn deltas a proxy, never a measured tool duration.

## Visual Style

All experimental plots use Arial, 7--9 pt text, colorblind-safe colors, vector
PDF plus 300 dpi PNG, light or no grid lines, no chart title, and captions that
state the experimental message. Architecture diagrams use square-cornered
containers, restrained blue/green/orange accents, left-to-right data flow, and
no decorative gradients.
