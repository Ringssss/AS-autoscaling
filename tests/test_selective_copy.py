"""CPU-only checks of the proactive filter and its common full-copy behavior."""

import asyncio
from collections import OrderedDict
import time
from types import SimpleNamespace

import pytest

from agentshift.controller.preparation_decisions import CostCurve
from experiments.fluidkv.benchmark_decisions import PreparationManager


def fixture(policy, *, curves=None, busy=False, **overrides):
    args = dict(prefill_seconds_per_token=0.00001, warm_ttft_seconds=0.1,
                busy_copy_multiplier=1.6, copy_interference_seconds=0.006,
                minimum_gain_seconds=0.02, snapshot_slots=48,
                engines=["unused0", "unused1", "unused2"], slo=1,
                inspect_prefixes=False)
    args.update(overrides)
    replicas = {}
    for engine in range(3):
        saved = []
        replica = SimpleNamespace(pending_requests=[], running_requests=[],
                                  snapshots={}, saved=saved)
        replica.get_num_running_req = lambda r=replica: len(r.running_requests)
        replica.save_token_ids = lambda tokens, r=replica: r.saved.append(tuple(tokens))
        replicas[engine] = replica
    if busy:
        replicas[0].running_requests.append(SimpleNamespace(_input_ids=[1]))
    adapter = SimpleNamespace(active_ids=[1, 2], engine_map=[1, 2],
                              all_replicas=replicas, requests={})
    events, migrations, registered = [], [], []

    async def migrate(branch, target):
        continuation = next(row for row in registered if row.agent_id == branch)
        migrations.append({"source": continuation.owner_engine, "target": target,
                           "tokens": continuation.token_ids})
        return SimpleNamespace(bytes_transferred=len(continuation.token_ids) * 20480,
                               migration_id="mock-migration")

    manager = object.__new__(PreparationManager)
    manager.args, manager.adapter, manager.policy = SimpleNamespace(**args), adapter, policy
    manager.curves = curves or {
        "cold": CostCurve(((1088, 0.12), (8256, 0.50))),
        "warm": CostCurve(((1024, 0.10), (8192, 0.10))),
        "copy": CostCurve(((1024, 0.05), (8192, 0.05))),
    }
    manager.completed, manager.arrived = OrderedDict(), {}
    manager.tasks, manager.records = {}, []
    manager.lock = asyncio.Lock()
    manager.retired, manager.source_closed = None, False
    manager.began = time.perf_counter()
    manager.log = SimpleNamespace(write=events.append)
    manager.store = SimpleNamespace(register_agent=registered.append)
    manager.coordinator = SimpleNamespace(migrate_async=migrate, engines={})
    return manager, migrations, events


async def prepare_once(manager, length):
    tokens = tuple(range(length))
    manager.retired = 0
    manager.observe_completion({"chat_id": 1}, 0, tokens)
    await manager.finish()
    return manager.records[-1]


@pytest.mark.parametrize("length,should_copy", [(1024, False), (8192, True)])
def test_known_prefix_cost_selects_without_any_future_child(length, should_copy):
    async def check():
        manager, migrations, _ = fixture("selective-copy")
        assert manager.arrived == {}
        record = await prepare_once(manager, length)
        assert record["attempted"] is should_copy
        assert record["success"] is should_copy
        assert len(migrations) == int(should_copy)
        assert record["known_waiting_matches"] == []
        assert "no future child or gap input" in record["selection_scope"]
        assert record["target"] == 1
        if not should_copy:
            assert record["reason"] == "preparation_exceeds_one_reuse_saving"
            assert record["transfer_bytes"] == 0

    asyncio.run(check())


def test_warm_floor_is_subtracted_once_and_known_suffix_is_64():
    async def check():
        curves = {"cold": CostCurve(((8256, 0.30),)),
                  "warm": CostCurve(((8192, 0.12),)),
                  "copy": CostCurve(((8192, 0.10),))}
        manager, migrations, _ = fixture("selective-copy", curves=curves,
                                        minimum_gain_seconds=0.05)
        record = await prepare_once(manager, 8192)
        assert record["estimated_cold_ttft_seconds"] == pytest.approx(0.30)
        assert record["estimated_warm_ttft_seconds"] == pytest.approx(0.12)
        assert record["estimated_net_gain_seconds"] == pytest.approx(0.08)
        # Double charging the warm floor would turn this valid 80ms proxy gain
        # into a negative gain and incorrectly prevent the mocked transfer.
        assert record["attempted"] and len(migrations) == 1

    asyncio.run(check())


@pytest.mark.parametrize("margin,busy,multiplier,interference,selected", [
    (0.25, False, 1.0, 0.0, True),
    (0.250001, False, 1.0, 0.0, False),
    (0.15, True, 1.0, 0.0, True),
    (0.15, True, 2.0, 0.01, False),
    (0.15, True, 1.0, 0.11, False),
])
def test_margin_busy_copy_and_interference_change_filter_only(margin, busy, multiplier, interference, selected):
    async def check():
        curves = {"cold": CostCurve(((8256, 0.50),)),
                  "warm": CostCurve(((8192, 0.125),)),
                  "copy": CostCurve(((8192, 0.125),))}
        arguments = dict(curves=curves, busy=busy, minimum_gain_seconds=margin,
                         busy_copy_multiplier=multiplier, copy_interference_seconds=interference)
        selective, filtered, _ = fixture("selective-copy", **arguments)
        full, unfiltered, _ = fixture("full-copy", **arguments)
        a = await prepare_once(selective, 8192)
        b = await prepare_once(full, 8192)
        assert a["attempted"] is selected
        assert b["attempted"] is True
        assert a["target"] == b["target"] == 1
        assert a["estimated_copy_seconds"] == b["estimated_copy_seconds"]
        assert a["known_waiting_matches"] == b["known_waiting_matches"] == []
        assert len(filtered) == int(selected) and len(unfiltered) == 1

    asyncio.run(check())


def test_retirement_and_later_source_observation_share_full_copy_triggers():
    async def check():
        outcomes = {}
        for policy in ("full-copy", "selective-copy", "cost-aware"):
            manager, migrations, _ = fixture(policy)
            parent = tuple(range(8192))
            manager.observe_completion({"chat_id": 1}, 0, parent)
            manager.adapter.all_replicas[0].snapshots[parent] = None
            manager.retire(0)
            await manager.finish()
            later = tuple(range(10000, 18192))
            manager.observe_completion({"chat_id": 2}, 0, later)
            await manager.finish()
            outcomes[policy] = (manager, migrations)
        for policy in ("full-copy", "selective-copy"):
            manager, migrations = outcomes[policy]
            assert len(migrations) == 2
            assert [r["chat_id"] for r in manager.records] == [1, 2]
            assert [r["target"] for r in manager.records] == [1, 1]
        reactive, migrations = outcomes["cost-aware"]
        assert migrations == []
        assert [r["reason"] for r in reactive.records] == ["no_observed_waiting_reuse"] * 2

    asyncio.run(check())


def test_existing_routed_child_target_is_shared_across_methods():
    async def check():
        for policy in ("full-copy", "selective-copy", "cost-aware"):
            manager, migrations, _ = fixture(policy)
            parent = tuple(range(8192))
            child = (*parent, *([20000] * 64))
            manager.arrived[2] = child
            manager.adapter.requests[2] = SimpleNamespace(
                _native_session_id=2, _primary_replica=1, _input_ids=child)
            record = await prepare_once(manager, 8192)
            assert record["known_waiting_matches"] == [2]
            # Current DualMap slot1 maps to engine2, overriding the common
            # least-queued tie preference for engine1 in every method.
            assert record["target"] == 2
            assert migrations[0]["target"] == "2"

    asyncio.run(check())


def test_selective_and_full_copy_never_add_reactive_dispatch_wait():
    async def check():
        parent = tuple(range(8192))
        for policy in ("full-copy", "selective-copy", "cost-aware"):
            manager, _, events = fixture(policy)
            task = asyncio.create_task(asyncio.sleep(0.005))
            manager.tasks[parent] = {"task": task, "target": 1,
                                     "started": time.perf_counter(),
                                     "estimate_seconds": 0.005, "success": False}
            waited = await manager.before_dispatch({"chat_id": 2}, [*parent, *([20000] * 64)], 1)
            if policy == "cost-aware":
                assert waited > 0
                assert any(row["event"] == "request_waited_for_preparation" for row in events)
            else:
                assert waited == 0.0
                assert not task.done()
                assert events == []
            await task

    asyncio.run(check())


def test_source_closed_prevents_speculative_revival_and_duplicate_observations_copy_once():
    async def check():
        manager, migrations, _ = fixture("selective-copy")
        tokens = tuple(range(8192))
        manager.retired = 0
        manager.observe_completion({"chat_id": 1}, 0, tokens)
        manager.observe_completion({"chat_id": 1}, 0, tokens)
        await manager.finish()
        assert len(migrations) == 1
        manager.source_closed = True
        manager.observe_completion({"chat_id": 3}, 0, tuple(range(10000, 18192)))
        manager.observe_arrival({"chat_id": 2}, [*tokens, *([20000] * 64)])
        await manager.finish()
        assert len(migrations) == 1

    asyncio.run(check())
