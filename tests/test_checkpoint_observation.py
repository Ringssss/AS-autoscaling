"""Exercise shared checkpoint inspection through the real preparation manager.

Transport is mocked; no live endpoint, CUDA, process shutdown or engine restart
is invoked. Replica metadata updates use the production adapter method.
"""

import asyncio
from collections import OrderedDict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from agentshift.controller.preparation_decisions import CostCurve
from agentshift.controller.snapshot_directory import SnapshotDirectory, best_reusable_prefix
from experiments.fluidkv import benchmark_decisions as bench
from experiments.fluidkv.dualmap_adapter import DualMapAdapter


def inspection(tokens, boundaries=(960,), matched=None):
    return {"query_tokens": len(tokens), "matched_tokens": len(tokens) if matched is None else matched,
            "snapshot_boundaries": list(boundaries), "available_token_slots": 50000,
            "evictable_token_slots": 20000, "available_snapshot_slots": 10,
            "evictable_snapshot_slots": 20}


class Replica:
    def __init__(self):
        self.pending_requests = []
        self.running_requests = []
        self.snapshots = OrderedDict()

    def get_num_recompute_token_ids(self, tokens):
        return len(tokens) - best_reusable_prefix(tokens, self.snapshots)

    def get_num_running_req(self):
        return len(self.running_requests)

    def save_token_ids(self, _tokens):
        raise AssertionError("Inspection mode cannot fabricate cache from request completion")


class SnapshotAdapter:
    observe_snapshot_response = DualMapAdapter.observe_snapshot_response
    invalidate_snapshot_engine = DualMapAdapter.invalidate_snapshot_engine

    def __init__(self, *_args, **_kwargs):
        self.snapshot_directory = SnapshotDirectory()
        self.all_replicas = {0: Replica(), 1: Replica()}
        self.active_ids = [0, 1]
        self.engine_map = [0, 1]
        self.requests = {}
        self.waiting = []
        self.scheduler = SimpleNamespace(rebalance_cnt=0)

    async def enqueue(self, ordinal, row, tokens, output_length):
        target = min(self.active_ids, key=lambda engine: (
            self.all_replicas[engine].get_num_recompute_token_ids(tokens), engine))
        request = SimpleNamespace(_id=ordinal, _native_session_id=row["chat_id"],
            _input_ids=tokens, _output_len=output_length,
            _primary_replica=self.engine_map.index(target), _physical_engine=target,
            _actual_num_prefill_tokens=self.all_replicas[target].get_num_recompute_token_ids(tokens))
        self.requests[ordinal] = request
        self.waiting.append(request)

    async def dispatch(self):
        ready, self.waiting = self.waiting, []
        for request in ready:
            self.all_replicas[request._physical_engine].pending_requests.append(request)
        return ready

    def physical_target(self, request):
        return request._physical_engine

    async def first_token(self, request, _delay):
        replica = self.all_replicas[request._physical_engine]
        replica.pending_requests.remove(request)
        replica.running_requests.append(request)

    async def finished(self, request, _tokens, _success):
        replica = self.all_replicas[request._physical_engine]
        for collection in (replica.pending_requests, replica.running_requests):
            if request in collection:
                collection.remove(request)

    def queued(self):
        return len(self.waiting)

    async def retire_engine(self, physical):
        self.active_ids.remove(physical)
        self.engine_map = list(self.active_ids)
        return physical


def arguments(tmp_path, **overrides):
    values = dict(engines=["http://mock-source", "http://mock-target"], request_timeout=1,
                  transfer_port=32000, snapshot_slots=48, prefill_seconds_per_token=0.001,
                  busy_copy_multiplier=1, warm_ttft_seconds=0.01, copy_interference_seconds=0,
                  minimum_gain_seconds=0, slo=1, inspect_prefixes=True,
                  metadata_directory=tmp_path / "metadata", dualmap_source=None,
                  cache_tokens=131072, start=0, duration=0.08, warmup=0,
                  trace_origin=0, time_scale=1, retire_offset=None, retire_engine=0,
                  physical_release=False, run_timeout=1, state_observation="prefill")
    values.update(overrides)
    return SimpleNamespace(**values)


def curves():
    return {"cold": CostCurve(((960, 0.5), (1100, 0.6))),
            "copy": CostCurve(((960, 0.01),)), "warm": CostCurve(((960, 0.01),))}


def make_manager(tmp_path, *, policy="full-copy", adapter=None):
    output = tmp_path / "run"
    output.mkdir()
    log_rows = []
    manager = bench.PreparationManager(arguments(tmp_path), adapter or SnapshotAdapter(), None,
        policy, output, curves(), SimpleNamespace(write=log_rows.append), time.perf_counter())
    manager.test_logs = log_rows
    return manager


@pytest.mark.parametrize("policy", ["dualmap", "full-copy", "cost-aware"])
def test_unaligned_prompt_only_copies_confirmed960_and_baseline_receives_same_metadata(tmp_path, policy):
    async def check():
        manager = make_manager(tmp_path, policy=policy)
        prompt = tuple(range(1007))
        observed_migrations = []

        async def inspect(tokens):
            return inspection(tokens, (960,), matched=960)

        async def migrate(branch, target):
            observed_migrations.append((manager.store.get_agent(branch).token_ids, target))
            return SimpleNamespace(bytes_transferred=100, migration_id="mock")

        for client in manager.coordinator.engines.values():
            object.__setattr__(client, "inspect_hybrid_prefix", inspect)
        manager.coordinator.migrate_async = migrate
        manager.retired = 0
        manager.adapter.active_ids = [1]
        manager.adapter.engine_map = [1]
        manager.arrived[2] = prompt[:960] + (3000,) * 80
        await manager.observe_engine_state({"chat_id": 1}, 0, prompt)
        await manager.finish()
        assert set(manager.adapter.all_replicas[0].snapshots) == {prompt[:960]}
        assert manager.adapter.all_replicas[0].get_num_recompute_token_ids(prompt) == 47
        assert prompt not in manager.adapter.all_replicas[0].snapshots
        if policy == "dualmap":
            assert observed_migrations == []
        else:
            assert observed_migrations == [(prompt[:960], "1")]
            assert manager.records[-1]["input_tokens"] == 960
        manager.store.conn.close()

    asyncio.run(check())


def test_actual_upstream_replica_uses_observed_boundary_without_automatic_remember():
    source = os.environ.get("FLUIDKV_DUALMAP_SOURCE")
    if not source:
        pytest.skip("Set FLUIDKV_DUALMAP_SOURCE for pinned upstream replica integration")
    adapter = DualMapAdapter(Path(source), 2, 131072, 20480, 1, 0.001,
                             hybrid_snapshot_slots=48, inspect_snapshots=True)
    prompt = tuple(range(1007))
    replica = adapter.all_replicas[0]
    replica.save_prefill_token_ids(1, list(prompt))
    replica.save_token_ids(list(prompt))
    assert replica.get_num_recompute_token_ids(prompt) == 1007
    adapter.observe_snapshot_response(0, prompt, inspection(prompt, (960,), matched=960))
    assert replica.get_num_recompute_token_ids(prompt) == 47
    adapter.observe_snapshot_response(0, prompt, inspection(prompt, (), matched=0))
    assert replica.get_num_recompute_token_ids(prompt) == 1007


def test_routing_targets_use_physical_observation_and_refresh_stale_path(tmp_path):
    async def check():
        manager = make_manager(tmp_path, policy="dualmap")
        prompt = tuple(range(1007))
        answers = {"0": inspection(prompt, (), matched=0), "1": inspection(prompt, (960,))}
        for engine, client in manager.coordinator.engines.items():
            async def inspect(tokens, engine=engine):
                return answers[engine]
            object.__setattr__(client, "inspect_hybrid_prefix", inspect)
        await manager.inspect_routing_targets(prompt)
        await manager.adapter.enqueue(0, {"chat_id": 1}, prompt, 1)
        request = manager.adapter.requests[0]
        assert request._physical_engine == 1
        assert request._actual_num_prefill_tokens == 47
        await manager.adapter.dispatch()
        assert manager.queue_seconds(1) == pytest.approx(47 * manager.args.prefill_seconds_per_token)
        answers["1"] = inspection(prompt, (), matched=0)
        await manager.inspect_routing_targets(prompt)
        assert manager.adapter.all_replicas[1].get_num_recompute_token_ids(prompt) == 1007
        assert set(manager.adapter.all_replicas[0].snapshots) == set()
        manager.store.conn.close()

    asyncio.run(check())


def test_failed_inspection_falls_back_without_fabricating_state(tmp_path):
    async def check():
        manager = make_manager(tmp_path, policy="dualmap")
        async def inspect(_tokens):
            raise OSError("injected endpoint race")
        object.__setattr__(manager.coordinator.engines["0"], "inspect_hybrid_prefix", inspect)
        assert await manager.inspect_engine(0, tuple(range(1007))) == ()
        assert not manager.adapter.all_replicas[0].snapshots
        assert not manager.completed
        assert any("injected endpoint race" in str(row) for row in manager.test_logs)
        manager.store.conn.close()

    asyncio.run(check())


def test_authoritative_eviction_prevents_old_observation_from_scheduling_again(tmp_path):
    async def check():
        manager = make_manager(tmp_path, policy="cost-aware")
        prompt = tuple(range(1007))
        available = True
        async def inspect(tokens):
            return inspection(tokens, (960,) if available else (), matched=960 if available else 0)
        object.__setattr__(manager.coordinator.engines["0"], "inspect_hybrid_prefix", inspect)
        await manager.observe_engine_state({"chat_id": 1}, 0, prompt)
        assert (0, prompt[:960]) in manager.completed
        available = False
        await manager.inspect_engine(0, prompt)
        manager.retired = 0
        attempts = []
        manager.schedule = lambda *args: attempts.append(args)
        manager.observe_arrival({"chat_id": 2}, prompt[:960] + (8888,))
        assert not attempts
        manager.store.conn.close()
    asyncio.run(check())


def test_closed_source_inspection_never_launches_rpc_or_preparation(tmp_path):
    async def check():
        manager = make_manager(tmp_path)
        manager.retired = 0
        manager.source_closed = True
        async def forbidden(_tokens):
            raise AssertionError("Closed source received inspection RPC")
        object.__setattr__(manager.coordinator.engines["0"], "inspect_hybrid_prefix", forbidden)
        await manager.observe_engine_state({"chat_id": 1}, 0, tuple(range(1007)))
        assert not manager.completed and not manager.tasks
        assert manager.source_inspection_idle(0)
        manager.store.conn.close()
    asyncio.run(check())


def test_cancelled_inspection_releases_inflight_guard_without_new_metadata(tmp_path):
    async def check():
        manager = make_manager(tmp_path)
        entered = asyncio.Event()
        async def pending(_tokens):
            entered.set()
            await asyncio.Event().wait()
        object.__setattr__(manager.coordinator.engines["0"], "inspect_hybrid_prefix", pending)
        task = asyncio.create_task(manager.inspect_engine(0, tuple(range(1007))))
        await entered.wait()
        assert not manager.source_inspection_idle(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.source_inspection_idle(0)
        assert not manager.adapter.all_replicas[0].snapshots
        manager.store.conn.close()
    asyncio.run(check())


def install_run_transport(monkeypatch, *, completion_delay=0.04, fail_completion=False):
    calls = []
    source_counts = {"0": 0, "1": 0}
    inflight = set()

    async def inspect(client, tokens):
        engine = client.engine_id
        source_counts[engine] += 1
        ordinal = source_counts[engine]
        # Routing is call1 on each engine. For selected source0, first-token
        # inspection is call2 and completion inspection is call3.
        delay = completion_delay if engine == "0" and ordinal == 3 else 0.006
        row = {"engine": engine, "ordinal": ordinal, "start": time.perf_counter()}
        calls.append(row)
        inflight.add((engine, ordinal))
        await asyncio.sleep(delay)
        inflight.remove((engine, ordinal))
        row["finish"] = time.perf_counter()
        if fail_completion and engine == "0" and ordinal == 3:
            raise OSError("injected completion inspection failure")
        return inspection(tokens, (960,) if engine == "0" else (),
                          matched=960 if engine == "0" else 0)

    async def fake_flush(*_args):
        pass

    async def fake_stream(_session, _url, tokens, outputs, first_callback):
        await asyncio.sleep(0.003)
        first = time.perf_counter()
        meta = {"prompt_tokens": len(tokens), "completion_tokens": outputs, "cached_tokens": 960}
        await first_callback({"meta_info": meta}, 0.003)
        await asyncio.sleep(0.005)
        return {"first_monotonic": first, "finish_monotonic": time.perf_counter(),
                "meta_info": meta, "output_ids": [7] * outputs, "stream_chunk_count": 1,
                "inter_chunk_p50_p90_p99_seconds": None}

    monkeypatch.setattr(bench, "DualMapAdapter", SnapshotAdapter)
    monkeypatch.setattr(bench.ObservedTransferClient, "inspect_hybrid_prefix", inspect)
    monkeypatch.setattr(bench, "flush", fake_flush)
    monkeypatch.setattr(bench, "timed_stream", fake_stream)
    monkeypatch.setattr(bench, "surrogate_tokens", lambda row: tuple(range(row["input_length"])))
    return calls, inflight


def execute(tmp_path, rows=None, **overrides):
    row = {"chat_id": 1, "parent_chat_id": -1, "timestamp": 0,
           "input_length": 1007, "output_length": 1, "hash_ids": [1] * 63}
    output = tmp_path / "integration"
    result = asyncio.run(bench.run_one(arguments(tmp_path, **overrides), rows or [row],
        {"replay_start_timestamp": 0}, "dualmap", 0, output, curves()))
    record = json.loads((output / "requests.jsonl").read_text().splitlines()[0])
    return result, record


def test_routing_inspection_is_in_ttft_and_postcompletion_refresh_does_not_block_child(monkeypatch, tmp_path):
    calls, _ = install_run_transport(monkeypatch)
    rows = [{"chat_id": 1, "parent_chat_id": -1, "timestamp": 0,
             "input_length": 1007, "output_length": 1, "hash_ids": [1] * 63},
            {"chat_id": 2, "parent_chat_id": 1, "timestamp": 0.001,
             "input_length": 1008, "output_length": 1, "hash_ids": [1] * 63}]
    result, record = execute(tmp_path, rows)
    assert record["success"] and record["estimated_cached_tokens"] == 960
    assert record["ttft_seconds"] >= 0.006  # Routing inspection is in TTFT.
    completion = next(row for row in calls if row["engine"] == "0" and row["ordinal"] == 3)
    records = {row["chat_id"]: row for row in map(json.loads,
        (tmp_path / "integration/requests.jsonl").read_text().splitlines())}
    inspections = [row for row in map(json.loads,
        (tmp_path / "integration/events.jsonl").read_text().splitlines())
        if row["event"] == "prefix_inspection" and row["engine"] == 0]
    postcompletion = max(inspections, key=lambda row: row["wall_seconds"])
    # This refresh runs after foreground completion. It is separately logged
    # and counted in observation occupancy, but must not hold causal eligibility.
    assert records[2]["eligibility_seconds"] >= records[1]["finished_seconds"]
    assert records[2]["eligibility_seconds"] < postcompletion["seconds"]
    assert result["duration_seconds"] >= postcompletion["seconds"]
    assert postcompletion["wall_seconds"] >= 0.04
    assert record["arrival_e2e_seconds"] == pytest.approx(
        record["e2e_seconds"] + record["eligibility_delay_seconds"])
    assert result["measurement"]["successful"] == 2


def test_completion_inspection_failure_does_not_lose_successful_foreground(monkeypatch, tmp_path):
    install_run_transport(monkeypatch, fail_completion=True)
    result, record = execute(tmp_path)
    assert record["success"]
    assert result["measurement"]["successful"] == 1


def test_source_shutdown_waits_for_pending_completion_inspection(monkeypatch, tmp_path):
    calls, inflight = install_run_transport(monkeypatch, completion_delay=0.06)
    from experiments.fluidkv import engine_lifecycle
    monkeypatch.setattr(engine_lifecycle, "capture_owned_engine", lambda _url: {"mock": True})
    releases = []

    async def release(*_args, **_kwargs):
        releases.append({"at": time.perf_counter(), "inflight": set(inflight)})
        return {"released": True, "confirmed_monotonic": time.perf_counter()}

    monkeypatch.setattr(engine_lifecycle, "release_engine", release)
    result, record = execute(tmp_path, physical_release=True, retire_offset=0.015, duration=0.12)
    assert record["success"]
    assert result["physical_release"]["released"]
    completion = next(row for row in calls if row["engine"] == "0" and row["ordinal"] == 3)
    assert releases[0]["at"] >= completion["finish"]
    assert not any(engine == "0" for engine, _ in releases[0]["inflight"])
