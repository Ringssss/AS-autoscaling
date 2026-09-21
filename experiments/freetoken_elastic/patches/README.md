# Research changes to pinned FreeToken

Base revision: afd99cb3b7c118dce4a9835359bcbc33b70c3cb8.

`overlap-length-fix.patch` is the shared output-budget correctness repair, not a FluidKV performance contribution. Apply it to a separate research checkout.

`checkpoint_preservation.py` is the exact tested core helper. Copy it to `python/freetoken/kvcache/checkpoint_preservation.py` in that checkout. It imports no experiment/test modules and is deliberately restricted to an idle, same-engine TP1 MHA/linear-state pool with page size1.

The opt-in external `../bootstrap/` wrapper attaches it for sequential experiments. No public FreeToken API or upstream contribution is implied. Keep the source/target schema, host budget, staging limit and cache publication checks; do not restore old page indices. Nonaligned semantic/final snapshots may be legal but are not selected by this baseline's aligned-boundary policy.
