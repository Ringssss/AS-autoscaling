import asyncio
import os
from pathlib import Path

import pytest

from experiments.fluidkv.dualmap_adapter import DualMapAdapter


@pytest.mark.parametrize("mode", ["dualmap", "cache-affinity", "least-loaded", "sticky"])
def test_upstream_queue_dispatch_and_budget_lifecycle(mode):
    source = os.environ.get("FLUIDKV_DUALMAP_SOURCE")
    if not source:
        pytest.skip("set FLUIDKV_DUALMAP_SOURCE to the pinned upstream checkout")

    async def run():
        router = DualMapAdapter(Path(source), 3, 4096, 57344, 0.01, 0.0001, mode)
        assert type(router.scheduler).__module__.startswith("dualmap.")
        for ordinal in range(12):
            row = {"chat_id": ordinal, "hash_ids": [ordinal + 1, 10], "turn": 1}
            await router.enqueue(ordinal, row, [1000 + ordinal] * 32, 2)
        dispatched = []
        for _ in range(20):
            for request in await router.dispatch():
                dispatched.append(request._id)
                await router.first_token(request, 0.002)
                await router.finished(request, [10, 11], True)
            if not router.queued():
                break
        assert sorted(dispatched) == list(range(12))
        assert not router.queued()
        for replica in router.replica_budgets.values():
            assert not replica.pending_requests
            assert not replica.running_requests
            assert replica.current_budget == replica.replica_slo_budget

    asyncio.run(run())


def test_retirement_requeues_work_and_stops_new_mapping_to_removed_instance():
    source = os.environ.get("FLUIDKV_DUALMAP_SOURCE")
    if not source:
        pytest.skip("set FLUIDKV_DUALMAP_SOURCE to the pinned upstream checkout")

    async def run():
        router = DualMapAdapter(Path(source), 3, 4096, 57344, 1, 0.0001)
        for ordinal in range(15):
            row = {"chat_id": ordinal, "hash_ids": [ordinal + 1, 10], "turn": 1}
            await router.enqueue(ordinal, row, [1000 + ordinal] * 32, 2)
        assert await router.retire_last() == 2
        assert len(router.scheduler.global_request_queue.queues) == 2
        dispatched = []
        for _ in range(20):
            for request in await router.dispatch():
                assert request._primary_replica in (0, 1)
                dispatched.append(request._id)
                await router.first_token(request, 0.002)
                await router.finished(request, [10, 11], True)
            if not router.queued():
                break
        assert sorted(dispatched) == list(range(15))

    asyncio.run(run())


def test_retiring_nonfinal_engine_keeps_inflight_callbacks_on_original_engine():
    source = os.environ.get("FLUIDKV_DUALMAP_SOURCE")
    if not source:
        pytest.skip("set FLUIDKV_DUALMAP_SOURCE to the pinned upstream checkout")

    async def run():
        router = DualMapAdapter(Path(source), 3, 4096, 20480, 1, 0.0001)
        for ordinal in range(9):
            await router.enqueue(ordinal, {"chat_id": ordinal, "hash_ids": [ordinal + 1]}, [1200 + ordinal] * 32, 2)
        inflight = await router.dispatch()
        old_targets = {r._id: router.physical_target(r) for r in inflight}
        await router.retire_engine(0)
        for request in inflight:
            assert router.physical_target(request) == old_targets[request._id]
            await router.first_token(request, 0.002)
            await router.finished(request, [10, 11], True)
        while router.queued():
            for request in await router.dispatch():
                assert router.physical_target(request) in (1, 2)
                await router.first_token(request, 0.002)
                await router.finished(request, [10, 11], True)
        for replica in router.all_replicas.values():
            assert not replica.pending_requests
            assert not replica.running_requests
            assert replica.current_budget == replica.replica_slo_budget

    asyncio.run(run())


def test_hybrid_estimate_requires_a_snapshot_at_matching_branch_position():
    source = os.environ.get("FLUIDKV_DUALMAP_SOURCE")
    if not source:
        pytest.skip("set FLUIDKV_DUALMAP_SOURCE to the pinned upstream checkout")
    router = DualMapAdapter(Path(source), 3, 4096, 20480, 1, 0.0001, hybrid_snapshot_slots=2)
    replica = router.replica_budgets[0]
    parent = [1001] * 32
    replica.save_token_ids(parent + [1100] * 7)
    assert replica.get_num_recompute_token_ids(parent + [1200] * 16) == 48
    replica.save_prefill_token_ids(1, parent)
    assert replica.get_num_recompute_token_ids(parent + [1200] * 16) == 16
    replica.save_token_ids([1002] * 16)
    replica.save_token_ids([1003] * 16)
    assert replica.get_num_recompute_token_ids(parent + [1200] * 16) == 48


def test_owner_selection_uses_only_observed_requests():
    from types import SimpleNamespace
    router = object.__new__(DualMapAdapter)
    router.active_ids = [0, 1, 2]
    router.all_replicas = {
        0: SimpleNamespace(pending_requests=[], running_requests=[], snapshots={}),
        1: SimpleNamespace(pending_requests=[], running_requests=[SimpleNamespace(_input_ids=[1] * 32)], snapshots={}),
        2: SimpleNamespace(pending_requests=[SimpleNamespace(_input_ids=[2] * 64)], running_requests=[], snapshots={}),
    }
    assert router.largest_observed_prefix_owner() == 2


def test_inspected_snapshots_override_assumed_prompt_boundaries():
    source = os.environ.get('FLUIDKV_DUALMAP_SOURCE')
    if not source:
        pytest.skip('requires pinned DualMap checkout')
    router = DualMapAdapter(Path(source), 3, 131072, 20480, 1, .00003,
                            hybrid_snapshot_slots=48, inspect_snapshots=True)
    parent = list(range(1007))
    replica = router.all_replicas[0]
    replica.save_prefill_token_ids(0, parent)
    replica.save_token_ids(parent+[1500]*8)
    assert replica.get_num_recompute_token_ids(parent+[1300]*16)==1023
    response={'query_tokens':1007,'matched_tokens':960,'snapshot_boundaries':[960],
              'available_token_slots':100,'evictable_token_slots':1007,
              'available_snapshot_slots':5,'evictable_snapshot_slots':1}
    router.observe_snapshot_response(0,parent,response)
    child=parent[:992]+[1300]*32
    assert replica.get_num_recompute_token_ids(child)==len(child)-960
    assert tuple(parent) not in replica.snapshots
    router.observe_snapshot_response(0,child,{**response,'query_tokens':len(child),
        'matched_tokens':0,'snapshot_boundaries':[]})
    assert replica.get_num_recompute_token_ids(child)==len(child)
