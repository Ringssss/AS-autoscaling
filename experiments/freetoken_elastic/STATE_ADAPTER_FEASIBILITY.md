# Exact hybrid checkpoint preservation across pool rebuild

Code review only, 2026-09-15. No FreeToken changes, GPU experiments, or numerical validation performed for this note. Source: `/tmp/freetoken-review-20260915`, commit `afd99cb3b7c118dce4a9835359bcbc33b70c3cb8`.

## Minimal supported scope

TP=1, Qwen3.5 text-only, MHAKVCache plus LinearStatePool and HybridRadixCache, idle scheduler, one exact committed checkpoint initially. Reject other pool types rather than silently omitting compressed/index/window states. Pool resizing already preserves model geometry; prohibit model, dtype, page-size, positional convention, and layer-map changes. A checkpoint boundary must be an existing live snapshot node, not an arbitrary token truncation.

The baseline really destroys reusable state. `scheduler/scheduler.py:160` calls the engine rebuild and then `CacheManager.rebuild` for any supplied KV/mamba/window resize; this resets the prefix tree and free lists (`scheduler/cache.py:551`). Even a mamba-only resize discards the KV tree although its tensor is not reallocated. Expert-only resizing skips the tree reset. Runtime callers distinguish touched pools by non-None arguments (`scheduler/scheduler.py:673`), not by changed values; an expert-only request must omit KV/mamba/window parameters. The experiment uses actual geometry changes.

## Concrete external helper interface

An external experiment adapter can define `export_checkpoint(scheduler, exact_tokens, host_budget_bytes)` and `restore_checkpoint(scheduler, saved)` and wrap the existing `scheduler.rebuild_cache(**target_geometry)`. No new distributed protocol is required for this idle-only baseline.

Export:

1. Gate new admission and wait for every in-flight operation, including final request cleanup. The production scheduler's rebuild machinery is the safest entry seam. Synchronize the engine stream.
2. Inspect `cm.prefix_cache._snapshot_nodes()`. Concatenate node `_key` values along root-to-node path for exact token identity, and obtain `_collect_kv(node)`. Alternatively `match_prefix(tokens)` returns a valid checkpoint but may truncate to an ancestor and mutate tree splits/LRU; record the returned boundary exactly. Check node references are idle and its `mamba_value` is non-null.
3. Hold `inc_lock(node)` while staging if any interleaving could occur. For each physical KV page in `indices[::page_size] // page_size`, copy all K/V storage-layer slices from `engine.kv_cache._kv_buffer` to bounded host storage. The shape is `[2, storage_layers, pages, page_size, local_kv_heads, head_dim]` (`kvcache/mha_pool.py:49`). Verify indices are full contiguous pages internally, not merely a list of page heads.
4. Copy `linear_state_pool.conv_states[:, slot]`, `recurrent_states[:, slot]`, and EVERY entry of `slot_states[name][:, slot]`. Save shapes, dtypes, layer maps, model identity, token identity and boundary. `LinearStatePool.copy_from` (`linear_state_pool.py:188`) explicitly includes these sibling states.
5. Complete D2H before freeing old pools. Release the export lock in a finally block. Retain no old device views, handles, or slot references across rebuild.

Restore after the existing `Scheduler.rebuild_cache` returns:

1. Validate saved schema, shapes/dtypes and target capacity before allocating. Leave sufficient page capacity for continuation and at least the configured non-evictable recurrent working set. A fit in the raw pool does not imply sustainable serving.
2. Allocate fresh pages with `cm._allocate(boundary // page_size)` and convert to fresh token indices with `cm._page_to_token(page_heads)`. Allocate one new donated snapshot with `pool.alloc(1)[0]`. Never reuse the previous indices.
3. Copy host KV into the corresponding fresh pages and all saved state tensors into the new slot, then synchronize completion.
4. Publish with `cm.prefix_cache.insert(tokens, fresh_token_indices, new_slot)`. For the first isolated checkpoint into an empty tree, expect `(0, False)`. A returned `mamba_exist=True` leaves ownership with the caller, which must free its duplicate slot. With multiple checkpoints, free only the allocated duplicated prefix pages according to returned `prefix_len`; do not leave them allocated or free canonical tree pages. Build multi-checkpoint publication with explicit ownership tracking before extending this baseline.
5. Call `cm.check_integrity`, plus strengthen idle slot conservation to equality: free slots + donated snapshot count == usable slots. Query the restored prefix and confirm its exact boundary and new indices. Only then admit requests.
6. On prepublication failure, free allocated pages and slot. On failure after partial publication, discard the newly built tree and reclaim all state/page bookkeeping, returning an empty but consistent cache. Do not pretend the original cache survived a destructive rebuild.

Page tables do not need persistent request rows restored: there are no live requests. Normal admission obtains the restored HybridCacheHandle and page indices and COW-restores snapshot to a new live slot (`scheduler/prefill.py:126`, `scheduler/scheduler.py:623`). Copying old page tables would be wrong.

## Graph capture and footprint

`engine/engine.py:832` prevalidates, synchronizes, destroys graphs, reallocates pools, refreshes page tables, and recaptures the prior graph batch-size set. `Scheduler.rebuild_cache` resets the cache manager AFTER graph recapture. Therefore restore AFTER the entire scheduler rebuild: restoring earlier is erased by manager reset and may be overwritten by capture. Graphs reference final tensor allocations; writing values into freshly reserved slots does not itself require another recapture. Baseline must confirm this with actual continuation outputs.

For the provided Qwen3.5-35B-A3B config, TP=1/BF16 KV and conv/FP32 recurrent: full KV is 20,480 bytes/token; recurrent+conv snapshot is 64,389,120 bytes (61.40625 MiB), before any extra declared slot_states. One 8,192-token checkpoint is 221.40625 MiB. Use runtime tensor accounting as authoritative. Whole-checkpoint advanced indexing can transiently allocate another GPU-sized payload; a page/chunk staging loop avoids accidentally violating the claimed device budget. Host pinned buffers also need explicit accounting and lifetime.

## Existing support and tests

No host checkpoint storage/restore or cross-pool preservation API found in `kvcache/` or CacheManager. Its tier terminology describes device-side window/compressed/index tiers; CPU tensors in CacheManager are page-table staging, not KV offload. Expert host banks are separate and are retained by runtime rebuild. This is a pinned-tree finding, not a claim about all versions or other projects.

Useful existing tests: `tests/scheduler/test_hybrid_cache_manager.py` constructs real CPU LinearStatePool, CacheManager and donated checkpoints; `tests/scheduler/test_cache_rebuild.py` covers idle gating and bookkeeping; `tests/kvcache/test_kv_cache_rebuild.py` covers pool identity and allocation. CPU fake state tests establish remapping only, not model correctness. Model snapshot tests such as `tests/models/test_glm5_next_kda_snapshot.py` and `tests/models/qwen4_exp/test_ple.py` demonstrate boundary-state versus independent-prefix and continuation numerical comparisons, but are not Qwen3.5 end-to-end evidence.

Mandatory actual-engine comparisons: source cached continuation versus backup/rebuild/restore continuation, and independently versus cold recomputation. A cached/cold numerical discrepancy must remain distinct from preservation correctness. Count rebuild pause, graph capture, D2H/H2D, first resumed TTFT, full task time, failures and actual cache hit boundary. Using next-user input to choose a preserved snapshot would give the mechanism test oracle knowledge; label that scope and use only already-observed state for any policy claim.

## Implication for three components

The minimal baseline needs legal checkpoint export, capacity-bounded restore, and safe fresh-index publication. Three research contributions do not follow automatically. Only measured competition between expert residency, retained checkpoints and restoration timing could motivate a joint materialization/budget scheduler. Single-engine idle rebuild does not need distributed owner epochs or multi-stage commit.
