# Qwen3.5 Hybrid State Discovery

Date: 2026-09-12. Artifacts are in `experiments/hybrid_state_20260912/results/`.

## Findings

### 1. Hybrid state is two independently constrained resources

Qwen3.5-35B-A3B has 40 decoder layers: 10 full-attention layers and 30
linear-attention layers. Its state does not fit the homogeneous-MHA model used
by the current AgentShift patch.

| State family | Exact size | Scaling |
|---|---:|---|
| Full-attention KV | 20,480 B/token | Linear in context tokens |
| One linear-attention snapshot | 64,389,120 B = 61.40625 MiB | Fixed per branch/completed request |
| Hybrid equals full KV alone | 3,144 tokens | Crossover |
| Hybrid equals Qwen3-8B MHA KV | about 507 tokens | Crossover |

The linear snapshot is 60 MiB of FP32 temporal state plus 1.40625 MiB of BF16
conv state. Below about 507 tokens, migrating the complete Qwen3.5 state can be
larger than migrating the same prefix for Qwen3-8B; above that, it is smaller.
At 32K, the complete hybrid state is about 701 MiB versus 4.5 GiB for
Qwen3-8B.

### 2. Real agentic links fall on both sides of the crossover

The trace replay uses verified parent links and `parent_prefix_tokens`; it does
not inject synthetic fanout.

| Split | Requests | Usable links | Prefix p50/p90/p99 | Hybrid-vs-Qwen3-8B bytes p50 | Ratio-distribution p90 |
|---|---:|---:|---:|---:|---:|
| A | 43,058 | 19,957 | 1,168/6,544/14,631 | 0.51x | 5.60x |
| Coder | 43,011 | 16,605 | 3,488/11,024/13,632 | 0.26x | 1.33x |
| Thinking | 10,812 | 1,200 | 5,032/16,965/28,744 | 0.22x | 1.50x |

The ratio-distribution p90 means at least 10% of usable links are short enough
that the fixed SSM state dominates. A scheduler that maps context length to a
single bytes-per-token coefficient can therefore both underestimate and
overestimate migration cost. The aggregate link linear overhead is 127.8% for
A, 67.5% for Coder, and 48.6% for Thinking.

The drain replay treats full KV as deduplicated by maximum context length and
counts one linear snapshot per completed request as a conservative upper bound.
It is not an exact resident-cache measurement because the public NPZ does not
carry the full token-hash tree.

### 3. Current SGLang supports hybrid prefix reuse

A real-weight server ran on one H100 PCIe:

- TP=1, context 4,096, `max_total_tokens=4,096`, 8 Mamba slots.
- Weights: 65.49 GB.
- Mamba pool: 0.53 GiB SSM plus 0.01 GiB conv (the implementation allocates
  `size + 1`, so 8 visible slots allocate 9 padded slots).
- Full KV pool: 0.04 GiB K plus 0.04 GiB V.
- Backends: FA3 full attention, Triton linear attention.

An exact-prefix sweep reported:

| Prefix | First cached tokens | Repeat cached tokens |
|---:|---:|---:|
| 512 | 0 | 512 |
| 1,024 | 0 | 1,024 |
| 2,048 | 0 | 2,048 |
| 3,072 | 0 | 3,072 |

For the 2,048 case, the second request prefilled only one new suffix token.
This verifies that `MambaRadixCache` can restore both state families for an
exact completed prefix on one engine.

### 4. Per-agent SSM layout can reduce copy efficiency

The SGLang pool is layer-first. A per-agent snapshot is therefore strided
across 30 linear layers. On two H100 PCIe GPUs with peer access enabled:

| Copy | Bytes | Median |
|---|---:|---:|
| One temporal slot, one bulk strided copy | 60 MiB | 4.212 ms, 13.91 GiB/s |
| One temporal slot, 30 layer copies | 60 MiB | 3.325 ms, 17.62 GiB/s |
| Full 8-slot temporal pool | 540 MiB | 26.000 ms, 20.28 GiB/s |
| Full 2,048-token Qwen3.5 KV | 40 MiB | 0.974 ms, 20.05 GiB/s |

The current bulk per-agent form reaches only 69% of the full-pool bandwidth.
Layer-grouped copy is better in this microbenchmark but still below the full
pool. Adding conv state, one complete state-family copy is about 4.6 ms under
this layout. This is a data-plane layout issue, not just a control-plane
scheduling issue. A heterogeneous-H100/A100 path without peer access showed
roughly 4 GiB/s, so topology and peer access remain first-class placement
inputs.

### 5. AgentShift's current data plane cannot carry this state

The current patch's `AgentPrefixManager` explicitly checks for `k_buffer` and
`v_buffer` and rejects non-MHA pools. Qwen3.5 uses `HybridLinearKVPool` plus a
`MambaPool`; even if full KV rows were copied, linear temporal and conv states
would be absent. The patch also no longer applies to the current SGLang tree.
Thus hybrid support is not a small cost-formula change: it needs a state
descriptor, destination reservation for both resource families, transfer, and
atomic installation into `MambaRadixCache`.

## Research Direction

The strongest upgrade is a **heterogeneous global state plane**: a catalog and
migration substrate where state is represented by capabilities rather than a
token count. A state object contains full-attention token ranges, SSM snapshot
IDs, model/TP/dtype/layout identity, causal epoch, and size in both token slots
and snapshot slots. Placement becomes multi-resource and topology-aware, with
explicit choices among keep, migrate, tier, reconstruct, and drop.

Two natural deep extensions are:

1. **Branch-aware SSM cache management.** Full KV is prefix-shared, but a
   linear state is valid only at a specific sequence position and branch. This
   motivates checkpoint selection, copy-on-write snapshots, and reconstruction
   by replay when fanout or staleness makes migration expensive.
2. **Gap-aware state materialization with a correctness ledger.** Tool calls
   create a deadline known only probabilistically. Transfer should be layered
   by readiness and consumable privately, while public cache visibility,
   ownership, and tool-effect commit remain fenced. This generalizes the
   progressive-handoff idea from layer groups to heterogeneous state families.

## Boundaries

The trace replay is cost accounting, not measured serving throughput. The
runtime smoke is single-engine at 4K context and concurrency two. The copy test
uses synthetic tensors with SGLang's exact shapes, not an integrated NCCL
migration path. No claim is made about two-engine Qwen3.5 migration performance
until those transfers and output equivalence are measured.
