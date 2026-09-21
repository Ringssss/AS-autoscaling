"""Independent CPU inspection checks using the runtime's actual radix keys."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_hybrid_transfer_capacity import runtime  # Shared CPU-only AST loader.


@pytest.fixture
def inspected_tree(runtime):
    path = Path(__file__).resolve().parents[2]/"sglang-hybrid-state/python/sglang/srt/mem_cache/radix_cache.py"
    parsed = ast.parse(path.read_text())
    names = {"RadixKey", "get_child_key", "_key_match_page_size1", "_check_extra_key"}
    definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions], type_ignores=[])
    env = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    Key = env["RadixKey"]
    runtime.RadixKey = lambda tokens, extra: Key(tokens, extra)

    def node(tokens, mamba=True, extra=None):
        return SimpleNamespace(key=Key(tokens, extra), value=torch.arange(len(tokens)),
            mamba_value=torch.tensor([0]) if mamba else None, children={},
            last_access_time=19, full_lock_ref=0, mamba_lock_ref=0)

    root = node([], False)
    first = node([1, 2])
    tombstone = node([3, 4, 5], False)
    leaf = node([6, 7, 8, 9])
    namespaced = node([1, 2, 3, 4, 5, 6], True, "other-cache-salt")
    root.children[1] = first
    root.children[("other-cache-salt", 1)] = namespaced
    first.children[3] = tombstone
    tombstone.children[6] = leaf
    cache = SimpleNamespace(root_node=root, disable=False,
        get_child_key_fn=env["get_child_key"], key_match_fn=env["_key_match_page_size1"],
        full_evictable_size=lambda: 9, mamba_evictable_size=lambda: 2)
    cache.match_prefix = lambda *_args: pytest.fail("Inspection must not call mutating match_prefix")
    cache.inc_lock_ref = cache.dec_lock_ref = cache.match_prefix
    cache.full_lru_list = cache.mamba_lru_list = SimpleNamespace(reset_node_and_parents_mru=cache.match_prefix)
    allocator = SimpleNamespace(available_size=lambda: 10)
    pool = SimpleNamespace(mamba_pool=SimpleNamespace(available_size=lambda: 3))
    manager = runtime.HybridStateTransferManager.__new__(runtime.HybridStateTransferManager)
    manager.scheduler = SimpleNamespace(model_config=SimpleNamespace(context_len=32768, vocab_size=32000),
                                        server_args=SimpleNamespace(context_length=32768))
    manager._validate_runtime = lambda: (cache, allocator, pool)
    return manager, cache, [root, first, tombstone, leaf, namespaced]


def fingerprint(nodes):
    return [(tuple(node.key), node.key.extra_key, tuple(node.children), node.last_access_time,
             node.full_lock_ref, node.mamba_lock_ref,
             None if node.value is None else tuple(node.value.tolist()),
             None if node.mamba_value is None else tuple(node.mamba_value.tolist())) for node in nodes]


def test_actual_radix_key_offsets_cross_tombstone_without_mutation(inspected_tree):
    manager, _, nodes = inspected_tree
    before = fingerprint(nodes)
    result = manager.inspect_prefix(token_ids=list(range(1, 10)))
    assert result["snapshot_boundaries"] == [2, 9]
    assert result["matched_tokens"] == 9
    assert fingerprint(nodes) == before


@pytest.mark.parametrize("tokens", [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 99], [1, 2, 3, 4, 5, 6]])
def test_partial_edge_and_tombstone_never_create_snapshot(inspected_tree, tokens):
    manager, _, nodes = inspected_tree
    before = fingerprint(nodes)
    result = manager.inspect_prefix(token_ids=tokens)
    assert result["snapshot_boundaries"] == [2]
    assert result["matched_tokens"] == 2
    assert fingerprint(nodes) == before


@pytest.mark.parametrize("bad_value", [None, torch.arange(2)])
def test_missing_or_short_kv_edge_blocks_later_recurrent_snapshot(inspected_tree, bad_value):
    manager, _, nodes = inspected_tree
    nodes[2].value = bad_value
    assert manager.inspect_prefix(token_ids=list(range(1, 10)))["snapshot_boundaries"] == [2]


def test_disabled_and_other_cache_namespace_are_not_reported(inspected_tree):
    manager, cache, nodes = inspected_tree
    del nodes[0].children[1]
    assert manager.inspect_prefix(token_ids=list(range(1, 7)))["snapshot_boundaries"] == []
    cache.disable = True
    nodes[0].children[1] = nodes[1]
    assert manager.inspect_prefix(token_ids=list(range(1, 10)))["matched_tokens"] == 0
