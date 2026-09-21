"""CPU checks of actual runtime admission/rollback methods, without SGLang boot.

The class is compiled from the sibling runtime source. Only runtime dependencies
are replaced; admission, job ownership and cleanup code execute unchanged.
"""

import ast
import dataclasses
import os
from pathlib import Path
import re
import sys
import threading
import time
import types
from array import array

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    source = Path(os.environ.get("FLUIDKV_HYBRID_RUNTIME_SOURCE", str(
        Path(__file__).resolve().parents[2] / "sglang-hybrid-state/python/sglang/srt/mem_cache/hybrid_state_transfer.py")))
    if not source.exists():
        pytest.skip("Requires sibling sglang-hybrid-state runtime checkout")
    parsed = ast.parse(source.read_text(), filename=str(source))
    selected = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                and node.name in ("DirectTransferJob", "HybridStateTransferManager")]
    module = types.ModuleType("_fluidkv_runtime_capacity_test")
    module.__dict__.update({
        "torch": torch, "dataclass": dataclasses.dataclass,
        "threading": threading, "time": time, "array": array,
        "_CHECKPOINT_ID_RE": re.compile(r"^[A-Za-z0-9_.-]{1,128}$"),
        "MatchPrefixParams": types.SimpleNamespace,
        "RadixKey": lambda tokens, _extra: tokens,
        "EvictParams": lambda num_tokens, mamba_num=0: types.SimpleNamespace(
            num_tokens=num_tokens, mamba_num=mamba_num),
    })
    monkeypatch.setitem(sys.modules, module.__name__, module)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(code, str(source), "exec"), module.__dict__)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    return module


class Allocator:
    device = "cpu"

    def __init__(self, free_indices):
        self.free_indices = list(free_indices)
        self.freed = []
        self.allocations = []

    def available_size(self):
        return len(self.free_indices)

    def alloc(self, count):
        assert count <= self.available_size()
        indices = torch.tensor(self.free_indices[:count], dtype=torch.int64)
        del self.free_indices[:count]
        self.allocations.append(indices.tolist())
        return indices

    def free(self, indices):
        values = indices.tolist()
        self.freed.append(values)
        self.free_indices.extend(values)


def manager_fixture(runtime, *, free_kv=0, evictable_kv=0,
                    free_mamba=1, evictable_mamba=0):
    prefix_node = types.SimpleNamespace(lock_count=0, mamba_value=torch.tensor([0]))
    indices = torch.tensor([0, 1], dtype=torch.int64)
    allocator = Allocator(range(2 + evictable_kv, 2 + evictable_kv + free_kv))
    mamba = Allocator(range(1 + evictable_mamba, 1 + evictable_mamba + free_mamba))
    mamba.mamba_cache = types.SimpleNamespace(
        temporal=torch.zeros((1, 4, 2)), conv=[torch.zeros((1, 4, 2))])
    events = []
    unlocked_kv = list(range(2, 2 + evictable_kv))
    unlocked_mamba = list(range(1, 1 + evictable_mamba))

    def pin(node):
        assert node is prefix_node
        node.lock_count += 1
        events.append("pin")

    def unpin(node):
        assert node is prefix_node and node.lock_count == 1
        node.lock_count -= 1
        events.append("unpin")

    def evict(params):
        # If admission evicts before pinning, the matched prefix would itself be
        # an eligible LRU victim. Assert protection for both state dimensions.
        assert prefix_node.lock_count == 1, "matched destination prefix was not protected"
        events.append(("evict", params.num_tokens, params.mamba_num))
        reclaimed_kv = min(params.num_tokens, len(unlocked_kv))
        allocator.free_indices.extend(unlocked_kv[:reclaimed_kv])
        del unlocked_kv[:reclaimed_kv]
        reclaimed_mamba = min(params.mamba_num, len(unlocked_mamba))
        mamba.free_indices.extend(unlocked_mamba[:reclaimed_mamba])
        del unlocked_mamba[:reclaimed_mamba]
        # Admission must use actual free capacity, not trust eviction counters.
        return types.SimpleNamespace(num_tokens_evicted=999, mamba_num_evicted=999)

    cache = types.SimpleNamespace(
        match_prefix=lambda _params: types.SimpleNamespace(
            device_indices=indices, last_device_node=prefix_node),
        inc_lock_ref=pin, dec_lock_ref=unpin, evict=evict,
    )
    pool = types.SimpleNamespace(
        full_kv_pool=types.SimpleNamespace(k_buffer=[torch.zeros((8, 1))],
                                         v_buffer=[torch.zeros((8, 1))]),
        mamba_pool=mamba,
    )
    manager = runtime.HybridStateTransferManager.__new__(runtime.HybridStateTransferManager)
    manager.direct_jobs = {}
    manager._direct_job_lock = threading.Lock()
    manager._validate_runtime = lambda: (cache, allocator, pool)
    manager._direct_group = lambda _name, _port: None
    return manager, allocator, mamba, prefix_node, events


def prepare(manager, tokens=6):
    return manager.start_direct_transfer(transfer_id="capacity-test", role="destination",
        token_ids=list(range(tokens)), group_name="cpu-test", port=32000, prepare_only=True)


def inspection_fixture(runtime):
    def node(key, snapshot, children=None):
        return types.SimpleNamespace(key=list(key), value=torch.arange(len(key)),
            mamba_value=torch.tensor([1]) if snapshot else None, children=children or {},
            last_access_time=17, lock_ref=0)
    leaf = node([4,5], True)
    middle = node([2,3], False, {4:leaf})
    first = node([0,1], True, {2:middle})
    root = node([], False, {0:first})
    def common(a,b):
        count=0
        for left,right in zip(a,b):
            if left != right:break
            count+=1
        return count
    cache=types.SimpleNamespace(root_node=root,disable=False,
        get_child_key_fn=lambda key:key[0],key_match_fn=common,
        full_evictable_size=lambda:6,mamba_evictable_size=lambda:2)
    manager=runtime.HybridStateTransferManager.__new__(runtime.HybridStateTransferManager)
    manager.scheduler=types.SimpleNamespace(model_config=types.SimpleNamespace(context_len=32768,vocab_size=100000))
    manager._validate_runtime=lambda:(cache,Allocator(range(10)),types.SimpleNamespace(mamba_pool=Allocator(range(3))))
    return manager,[root,first,middle,leaf]


def test_prefix_inspection_reports_only_existing_hybrid_nodes_without_mutation(runtime):
    manager,nodes=inspection_fixture(runtime)
    before=[(tuple(n.key),tuple(n.children),n.last_access_time,n.lock_ref) for n in nodes]
    result=manager.inspect_prefix(token_ids=list(range(6)))
    assert result['snapshot_boundaries']==[2,6]
    assert result['matched_tokens']==6
    assert result['available_token_slots']==10
    assert result['evictable_token_slots']==6
    assert result['available_snapshot_slots']==3
    assert result['evictable_snapshot_slots']==2
    assert before==[(tuple(n.key),tuple(n.children),n.last_access_time,n.lock_ref) for n in nodes]


def test_inspection_partial_node_match_does_not_create_a_snapshot(runtime):
    manager,nodes=inspection_fixture(runtime)
    for tokens in (list(range(5)),[0,1,2,9],[0,1,2,3]):
        result=manager.inspect_prefix(token_ids=tokens)
        assert result['matched_tokens']==2
        assert result['snapshot_boundaries']==[2]
    assert list(nodes[2].children)==[4]
    assert nodes[2].mamba_value is None
    assert manager.inspect_prefix(token_ids=[])['matched_tokens']==0


@pytest.mark.parametrize('tokens',[[True],[-1],[100000],[0]*32769])
def test_inspection_rejects_invalid_or_oversized_query(runtime,tokens):
    manager,_=inspection_fixture(runtime)
    with pytest.raises(ValueError):
        manager.inspect_prefix(token_ids=tokens)


def test_reclaims_unlocked_full_cache_and_keeps_matching_prefix_pinned(runtime):
    manager, allocator, mamba, node, events = manager_fixture(runtime, evictable_kv=4)
    result = prepare(manager)
    assert result["state"] == "PREPARED"
    assert result["new_token_count"] == 4
    assert events == ["pin", ("evict", 4, 0)]
    assert node.lock_count == 1
    assert manager.direct_jobs["capacity-test"].kv_indices.tolist() == [0, 1, 2, 3, 4, 5]
    manager.cleanup_direct_transfer("capacity-test")
    assert allocator.freed == [[2, 3, 4, 5]]
    assert len(mamba.freed) == 1
    assert node.lock_count == 0


def test_reclaims_mamba_without_evicting_matching_snapshot(runtime):
    manager, allocator, mamba, node, events = manager_fixture(
        runtime, free_mamba=0, evictable_mamba=1)
    assert prepare(manager, tokens=2)["state"] == "PREPARED"
    assert events == ["pin", ("evict", 0, 1)]
    assert manager.direct_jobs["capacity-test"].mamba_index_tensor.tolist() == [1]
    manager.cleanup_direct_transfer("capacity-test")
    assert node.lock_count == 0
    assert mamba.freed == [[1]]
    assert all(0 not in freed and 1 not in freed for freed in allocator.freed)


@pytest.mark.parametrize("resource", ["kv", "mamba"])
def test_private_exhaustion_rejects_without_progress_and_cleans_pin(runtime, resource):
    manager, allocator, mamba, node, events = manager_fixture(
        runtime, free_kv=4 if resource == "mamba" else 0,
        free_mamba=0 if resource == "mamba" else 1)
    with pytest.raises(RuntimeError, match="unreserved.*capacity"):
        prepare(manager)
    assert not manager.direct_jobs
    assert node.lock_count == 0
    assert len([event for event in events if isinstance(event, tuple)]) == 1
    assert allocator.allocations == []
    assert allocator.freed == []
    assert mamba.freed == []


def test_late_prepare_error_releases_only_new_reservations(runtime):
    manager, allocator, mamba, node, events = manager_fixture(runtime, free_kv=4)

    def failed_mamba_alloc(_count):
        raise RuntimeError("injected reservation allocation failure")

    mamba.alloc = failed_mamba_alloc
    with pytest.raises(RuntimeError, match="injected"):
        prepare(manager)
    assert not manager.direct_jobs
    assert allocator.freed == [[2, 3, 4, 5]]
    assert allocator.available_size() == 4
    assert mamba.freed == []
    assert node.lock_count == 0


@pytest.mark.parametrize("resource", ["kv", "mamba"])
def test_allocator_none_never_becomes_prepared(runtime, resource):
    manager, allocator, mamba, node, _events = manager_fixture(runtime, free_kv=4)
    (allocator if resource == "kv" else mamba).alloc = lambda _count: None
    with pytest.raises(RuntimeError, match="reservation failed after reclamation"):
        prepare(manager)
    assert not manager.direct_jobs
    assert node.lock_count == 0
    assert allocator.freed == ([] if resource == "kv" else [[2, 3, 4, 5]])
    assert mamba.freed == []


def test_staging_failure_releases_both_private_reservations(runtime, monkeypatch):
    manager, allocator, mamba, node, _events = manager_fixture(runtime, free_kv=4)

    def allocation_failure(*_args, **_kwargs):
        raise RuntimeError("injected staging allocation failure")

    monkeypatch.setattr(torch, "empty", allocation_failure)
    with pytest.raises(RuntimeError, match="staging allocation"):
        prepare(manager)
    assert not manager.direct_jobs
    assert node.lock_count == 0
    assert allocator.freed == [[2, 3, 4, 5]]
    assert mamba.freed == [[1]]
