"""Serving-accounting contracts exercised without engines, networking, or GPUs."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from agentshift.controller.preparation_decisions import CostCurve
from experiments.fluidkv import benchmark_decisions as bench
from experiments.fluidkv.analyze_decision_workloads import select_continuous_window


def test_runtime_automatic_rollback_missing_is_confirmed_cleanup(monkeypatch):
    async def run():
        client=bench.ObservedTransferClient('0','http://unused')
        client.live_transfers.add('rolled-back')
        async def cleanup(_self, transfer_id):
            return {'transfer_id':transfer_id,'state':'MISSING'}
        monkeypatch.setattr(bench.SGLangAgentShiftClient,'cleanup_async_hybrid_state_transfer',cleanup)
        await client.cleanup_async_hybrid_state_transfer('rolled-back')
        assert not client.live_transfers
        client.live_transfers.add('unknown')
        async def failed(_self, transfer_id):
            raise RuntimeError('connection lost')
        monkeypatch.setattr(bench.SGLangAgentShiftClient,'cleanup_async_hybrid_state_transfer',failed)
        with pytest.raises(RuntimeError):
            await client.cleanup_async_hybrid_state_transfer('unknown')
        assert 'unknown' in client.live_transfers
    asyncio.run(run())


def source_row(chat, parent, timestamp):
    return dict(chat_id=chat, parent_chat_id=parent, timestamp=timestamp,
                input_length=16, hash_ids=[chat], output_length=1, turn=1, type="text")


def arguments(**overrides):
    values = dict(engines=["http://unused:31480", "http://unused:31481"],
                  request_timeout=1, dualmap_source=None, cache_tokens=65536, slo=1,
                  prefill_seconds_per_token=0.000025, snapshot_slots=48,
                  start=0.02, duration=0.04, warmup=0.02, trace_origin=0,
                  time_scale=1, retire_offset=None, retire_engine=0,
                  physical_release=False, run_timeout=1,
                  transfer_port=32100, warm_ttft_seconds=0.001,
                  copy_interference_seconds=0, busy_copy_multiplier=1, minimum_gain_seconds=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def install_transport(monkeypatch, *, allow_dispatch=True, preparation_wait=0.015):
    class Replica:
        def __init__(self):
            self.pending_requests = []
            self.running_requests = []
            self.snapshots = {}

        def get_num_recompute_token_ids(self, tokens):
            return len(tokens)

        def get_num_running_req(self):
            return len(self.running_requests)

        def save_token_ids(self, tokens):
            self.snapshots[tuple(tokens)] = None

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.all_replicas = {0: Replica(), 1: Replica()}
            self.active_ids = [0, 1]
            self.waiting = []
            self.scheduler = SimpleNamespace(rebalance_cnt=0)

        async def enqueue(self, ordinal, row, tokens, output_length):
            self.waiting.append(SimpleNamespace(_id=ordinal, _input_ids=tokens,
                                               _output_len=output_length))

        async def dispatch(self):
            if not allow_dispatch:
                return []
            ready, self.waiting = self.waiting, []
            for request in ready:
                request._physical_engine = self.active_ids[0]
                self.all_replicas[request._physical_engine].pending_requests.append(request)
            return ready

        def physical_target(self, request):
            return request._physical_engine

        async def first_token(self, request, delay):
            replica = self.all_replicas[request._physical_engine]
            replica.pending_requests.remove(request)
            replica.running_requests.append(request)

        async def finished(self, request, tokens, success):
            replica = self.all_replicas[request._physical_engine]
            if request in replica.pending_requests:
                replica.pending_requests.remove(request)
            if request in replica.running_requests:
                replica.running_requests.remove(request)

        def queued(self):
            return len(self.waiting)

        async def retire_engine(self, physical):
            self.active_ids.remove(physical)
            return physical

    class Manager:
        def __init__(self, *args):
            async def ensure_group(*args):
                return ("mock-group", 1)

            self.tasks = {}
            self.records = []
            self.source_closed = False
            self.release_blocked = False
            self.coordinator = SimpleNamespace(_ensure_group=ensure_group)

        def observe_arrival(self, row, tokens):
            pass

        def observe_dispatch(self, row):
            pass

        def observe_completion(self, row, target, tokens):
            pass

        def retire(self, physical):
            pass

        async def before_dispatch(self, row, tokens, target):
            began = time.perf_counter()
            await asyncio.sleep(preparation_wait)
            return time.perf_counter() - began

        async def finish(self):
            pass

        def assert_transfers_clean(self):
            pass

    async def fake_flush(*args):
        pass

    async def fake_stream(session, url, tokens, outputs, first_callback):
        await asyncio.sleep(0.004)
        first = time.perf_counter()
        meta = dict(prompt_tokens=len(tokens), completion_tokens=outputs, cached_tokens=0)
        await first_callback({"meta_info": meta}, 0.004)
        await asyncio.sleep(0.015)
        return dict(first_monotonic=first, finish_monotonic=time.perf_counter(),
                    meta_info=meta, output_ids=[7] * outputs,
                    stream_chunk_count=1, inter_chunk_p50_p90_p99_seconds=None)

    monkeypatch.setattr(bench, "DualMapAdapter", Adapter)
    monkeypatch.setattr(bench, "PreparationManager", Manager)
    monkeypatch.setattr(bench, "flush", fake_flush)
    monkeypatch.setattr(bench, "timed_stream", fake_stream)


def execute(args, rows, output):
    selected, manifest = select_continuous_window(
        rows, start_seconds=args.start, duration_seconds=args.duration,
        warmup_seconds=args.warmup, time_origin=args.trace_origin)
    curve = CostCurve(((16, 0.01), (32, 0.02)))
    result = asyncio.run(bench.run_one(args, selected, manifest, "cost-aware", 0,
                                      output, {"cold": curve, "copy": curve}))
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    return result, {row["chat_id"]: row for row in records}


def test_causal_delay_and_preparation_wait_are_counted(monkeypatch, tmp_path):
    install_transport(monkeypatch)
    result, records = execute(arguments(), [source_row(1, -1, 0), source_row(2, 1, 0.025)], tmp_path / "run")
    child = records[2]
    assert child["phase"] == "measurement"
    assert child["eligibility_delay_seconds"] > 0
    assert child["preparation_wait_seconds"] >= 0.015
    assert child["router_wait_seconds"] >= child["preparation_wait_seconds"]
    assert child["ttft_seconds"] >= child["preparation_wait_seconds"]
    assert child["arrival_ttft_seconds"] == pytest.approx(child["ttft_seconds"] + child["eligibility_delay_seconds"])
    assert child["arrival_e2e_seconds"] == pytest.approx(child["e2e_seconds"] + child["eligibility_delay_seconds"])
    assert result["measurement"]["arrival_e2e_p50_p90_p99_seconds"][0] == pytest.approx(child["arrival_e2e_seconds"])


def test_online_child_input_is_hidden_until_parent_finishes(monkeypatch, tmp_path):
    install_transport(monkeypatch)
    observed = {}
    original_manager = bench.PreparationManager

    class ObservingManager(original_manager):
        def observe_arrival(self, row, tokens):
            observed[row["chat_id"]] = time.perf_counter()-self.began
            super().observe_arrival(row, tokens)

    monkeypatch.setattr(bench, "PreparationManager", ObservingManager)
    _, records = execute(arguments(duration=0.06),
        [source_row(1, -1, 0), source_row(2, 1, 0.001), source_row(3, -1, 0.055)], tmp_path / "run")
    assert set(observed) == {1, 2, 3}
    # Parent's first token occurs well before finish; prefill observation must
    # not reveal the queued child's exact future input during that interval.
    assert records[1]["first_token_seconds"] < records[1]["finished_seconds"]
    assert observed[2] >= records[1]["finished_seconds"]
    assert observed[2] >= records[2]["eligibility_seconds"]
    assert observed[2] > records[2]["scheduled_seconds"] + 0.01
    assert observed[3] >= records[3]["scheduled_seconds"]


def test_timeout_retains_queued_parent_blocked_and_future_offered_rows(monkeypatch, tmp_path):
    install_transport(monkeypatch, allow_dispatch=False)
    rows = [source_row(1, -1, 0), source_row(2, 1, 0.025), source_row(3, -1, 0.09)]
    result, records = execute(arguments(duration=0.1, run_timeout=0.04), rows, tmp_path / "run")
    assert result["timed_out"]
    assert set(records) == {1, 2, 3}
    assert all(not row["success"] for row in records.values())
    assert result["warmup"]["offered"] == 1
    assert result["measurement"]["offered"] == 2
    assert result["measurement"]["failed"] == 2
    assert result["measurement"]["slo_attainment"] == 0


def test_manifest_clock_preserves_leading_warmup_idle_time(monkeypatch, tmp_path):
    install_transport(monkeypatch)
    result, records = execute(arguments(), [source_row(1, -1, 0.01), source_row(2, -1, 0.025)], tmp_path / "run")
    assert result["measurement_start_seconds"] == pytest.approx(0.02)
    assert records[1]["scheduled_seconds"] == pytest.approx(0.01)
    assert records[2]["scheduled_seconds"] == pytest.approx(0.025)


def test_gpu_occupancy_covers_full_observation_tail(monkeypatch, tmp_path):
    install_transport(monkeypatch)
    result, _ = execute(arguments(duration=0.08), [source_row(1, -1, 0), source_row(2, -1, 0.021)], tmp_path / "run")
    assert result["duration_seconds"] >= 0.1
    assert result["model_occupied_gpu_seconds"] >= 0.2
    assert result["model_occupied_gpu_seconds"] == pytest.approx(2 * result["duration_seconds"])


def test_failed_physical_release_never_credits_gpu_savings(monkeypatch, tmp_path):
    install_transport(monkeypatch)
    from experiments.fluidkv import engine_lifecycle
    monkeypatch.setattr(engine_lifecycle, "capture_owned_engine", lambda url: {"url": url})

    async def failed_release(*args, **kwargs):
        return {"released": False, "confirmed_monotonic": None, "error": "unconfirmed release"}

    monkeypatch.setattr(engine_lifecycle, "release_engine", failed_release)
    args = arguments(duration=0.1, physical_release=True, retire_offset=0)
    result, records = execute(args, [source_row(1, -1, 0), source_row(2, -1, 0.09)], tmp_path / "run")
    assert result["fatal_error"] is not None
    assert "release" in result["fatal_error"].lower()
    assert not result["physical_release"] or not result["physical_release"]["released"]
    assert set(records) == {1, 2}
    assert result["model_occupied_gpu_seconds"] == pytest.approx(2 * result["duration_seconds"])


def test_fatal_dispatch_drains_admitted_stream_and_transfer_without_releasing_gpu(monkeypatch, tmp_path):
    install_transport(monkeypatch, preparation_wait=0)
    adapter_base = bench.DualMapAdapter
    manager_base = bench.PreparationManager
    state = {"stream_started": False, "stream_completed": False, "stream_cancelled": False,
             "transfer_completed": False, "transfer_cancelled": False, "release_calls": 0}

    class FailingDispatch(adapter_base):
        sent = False

        async def dispatch(self):
            if self.sent:
                # Let the already-admitted task start and other independent
                # producers queue before the injected router exception.
                await asyncio.sleep(0.01)
                assert state["stream_started"]
                raise RuntimeError("injected dispatch failure")
            if not self.waiting:
                return []
            request = self.waiting.pop(0)
            request._physical_engine = self.active_ids[0]
            self.all_replicas[request._physical_engine].pending_requests.append(request)
            self.sent = True
            return [request]

    class InflightTransferManager(manager_base):
        def __init__(self, *args):
            super().__init__(*args)

            async def transfer():
                try:
                    await asyncio.sleep(0.06)
                    state["transfer_completed"] = True
                except asyncio.CancelledError:
                    state["transfer_cancelled"] = True
                    raise

            self.launched = asyncio.create_task(transfer())
            # A known launched transfer prevents normal source shutdown until
            # manager.finish drains it. It must survive the router exception.
            self.tasks = {"mock-prefix": {"task": self.launched}}

        async def finish(self):
            await asyncio.shield(asyncio.gather(self.launched))
            self.assert_transfers_clean()

    async def slow_stream(session, url, tokens, outputs, first_callback):
        state["stream_started"] = True
        try:
            await asyncio.sleep(0.004)
            first = time.perf_counter()
            meta = dict(prompt_tokens=len(tokens), completion_tokens=outputs, cached_tokens=0)
            await first_callback({"meta_info": meta}, 0.004)
            await asyncio.sleep(0.025)
            state["stream_completed"] = True
            return dict(first_monotonic=first, finish_monotonic=time.perf_counter(),
                        meta_info=meta, output_ids=[7] * outputs,
                        stream_chunk_count=1, inter_chunk_p50_p90_p99_seconds=None)
        except asyncio.CancelledError:
            state["stream_cancelled"] = True
            raise

    async def forbidden_release(*args, **kwargs):
        state["release_calls"] += 1
        pytest.fail("fatal run must not begin physical release after draining")

    from experiments.fluidkv import engine_lifecycle
    monkeypatch.setattr(bench, "DualMapAdapter", FailingDispatch)
    monkeypatch.setattr(bench, "PreparationManager", InflightTransferManager)
    monkeypatch.setattr(bench, "timed_stream", slow_stream)
    monkeypatch.setattr(engine_lifecycle, "capture_owned_engine", lambda url: {"url": url})
    monkeypatch.setattr(engine_lifecycle, "release_engine", forbidden_release)
    rows = [source_row(1, -1, 0), source_row(2, -1, 0.001),
            source_row(3, 1, 0.002), source_row(4, -1, 0.09)]
    result, records = execute(arguments(duration=0.10, physical_release=True, retire_offset=-0.02),
                              rows, tmp_path / "fatal-dispatch")
    assert state["stream_completed"] and not state["stream_cancelled"]
    assert state["transfer_completed"] and not state["transfer_cancelled"]
    assert state["release_calls"] == 0
    assert "injected dispatch failure" in result["fatal_error"]
    assert result["physical_release"] is None
    assert result["retired_engine"] == 0
    assert not result["timed_out"]
    assert set(records) == {1, 2, 3, 4}
    assert records[1]["success"]
    for chat_id in (2, 3, 4):
        assert not records[chat_id]["success"]
        assert records[chat_id]["error"] == "run_aborted_before_completion"
    assert result["warmup"]["offered"] + result["measurement"]["offered"] == 4
    assert result["warmup"]["failed"] + result["measurement"]["failed"] == 3
    assert result["model_occupied_gpu_seconds"] == pytest.approx(2 * result["duration_seconds"])


def test_actual_manager_finish_waits_for_other_launched_job_before_propagating_failure():
    async def check():
        state = {"finished": False, "cancelled": False}

        async def failed_job():
            raise RuntimeError("injected preparation failure")

        async def inflight_job():
            try:
                await asyncio.sleep(0.03)
                state["finished"] = True
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        failed = asyncio.create_task(failed_job())
        launched = asyncio.create_task(inflight_job())
        manager = object.__new__(bench.PreparationManager)
        manager.tasks = {"failed": {"task": failed}, "launched": {"task": launched}}
        manager.coordinator = SimpleNamespace(engines={})
        try:
            with pytest.raises(RuntimeError, match="injected preparation failure"):
                await manager.finish()
            assert launched.done() and state["finished"]
            assert not state["cancelled"]
        finally:
            # Keep test teardown safe even while the production drain regresses.
            await asyncio.gather(failed, launched, return_exceptions=True)

    asyncio.run(check())


def test_unknown_transfer_ledger_blocks_release_and_preserves_fatal_summary(monkeypatch, tmp_path):
    install_transport(monkeypatch, allow_dispatch=False, preparation_wait=0)
    manager_base = bench.PreparationManager
    state = {"release_calls": 0, "flush_calls": 0}

    class UnconfirmedManager(manager_base):
        def assert_transfers_clean(self):
            raise RuntimeError("transfer cleanup unconfirmed: unknown-job")

        async def finish(self):
            self.assert_transfers_clean()

    async def track_flush(*args):
        state["flush_calls"] += 1

    async def forbidden_release(*args, **kwargs):
        state["release_calls"] += 1
        pytest.fail("unknown transfer ledger forbids physical release")

    from experiments.fluidkv import engine_lifecycle
    monkeypatch.setattr(bench, "PreparationManager", UnconfirmedManager)
    monkeypatch.setattr(bench, "flush", track_flush)
    monkeypatch.setattr(engine_lifecycle, "capture_owned_engine", lambda url: {"url": url})
    monkeypatch.setattr(engine_lifecycle, "release_engine", forbidden_release)
    result, records = execute(arguments(duration=0.1, physical_release=True, retire_offset=-0.02),
                              [source_row(1, -1, 0), source_row(2, -1, 0.09)], tmp_path / "unconfirmed")
    assert "unknown-job" in result["fatal_error"]
    assert set(records) == {1, 2}
    assert all(not row["success"] for row in records.values())
    assert state["release_calls"] == 0
    assert state["flush_calls"] == 1  # Initial reset only; no destructive reset after abort.
    assert result["physical_release"] is None


def test_survivor_dispatch_continues_during_slow_physical_shutdown(monkeypatch, tmp_path):
    install_transport(monkeypatch, preparation_wait=0)
    from experiments.fluidkv import engine_lifecycle
    monkeypatch.setattr(engine_lifecycle, "capture_owned_engine", lambda url: {"url": url})

    async def slow_release(*args, **kwargs):
        started = time.perf_counter()
        await asyncio.sleep(0.05)
        return {"released": True, "confirmed_monotonic": time.perf_counter(),
                "signal_monotonic": started, "graceful": True}

    monkeypatch.setattr(engine_lifecycle, "release_engine", slow_release)
    args = arguments(duration=0.12, physical_release=True, retire_offset=0)
    result, records = execute(args, [source_row(1, -1, 0), source_row(2, -1, 0.055)], tmp_path / "run")
    release = result["physical_release"]
    release_start = release["confirmed_seconds"] - (release["confirmed_monotonic"]-release["signal_monotonic"])
    survivor = records[2]
    assert survivor["success"]
    assert survivor["target"] == 1
    assert release_start < survivor["scheduled_seconds"] < release["confirmed_seconds"]
    assert survivor["dispatched_seconds"] < release["confirmed_seconds"]
    assert survivor["first_token_seconds"] < release["confirmed_seconds"]


def test_arrived_reuse_reconsiders_previous_no_demand_decision(tmp_path):
    async def check():
        args = arguments()
        target = SimpleNamespace(pending_requests=[], running_requests=[], get_num_running_req=lambda: 0,
                                 save_token_ids=lambda ids: None)
        adapter = SimpleNamespace(active_ids=[1], all_replicas={0: target, 1: target}, requests={}, engine_map=[1])
        curve_cold = CostCurve(((16, 0.3), (32, 0.6)))
        manager = bench.PreparationManager(args, adapter, None, "cost-aware", tmp_path,
            {"cold": curve_cold, "copy": CostCurve(((16, 0.001),))},
            SimpleNamespace(write=lambda row: None), time.perf_counter())
        tokens = tuple(range(16))
        manager.retired = 0
        manager.observe_completion({"chat_id": 1}, 0, tokens)
        await manager.finish()
        assert manager.records[-1]["reason"] == "no_observed_waiting_reuse"
        assert not manager.records[-1]["attempted"]
        attempts = []

        async def fake_migrate(branch, destination):
            attempts.append((branch, destination))
            return SimpleNamespace(bytes_transferred=1024, migration_id="migration")

        manager.coordinator = SimpleNamespace(migrate_async=fake_migrate, acknowledge_async_migration=lambda mid: None, engines={})
        manager.observe_arrival({"chat_id": 2}, [*tokens, 99])
        await manager.finish()
        assert len(attempts) == 1
        assert manager.records[-1]["known_waiting_matches"] == [2]

    asyncio.run(check())


def test_summary_retains_failed_slo_denominator():
    successful = dict(phase="measurement", success=True, ttft_seconds=0.1,
                      e2e_seconds=1.0, arrival_ttft_seconds=0.5, arrival_e2e_seconds=1.4,
                      router_wait_seconds=0.03, actual_cached_tokens=16,
                      input_tokens=32, completion_tokens=4)
    failure = dict(phase="measurement", success=False, input_tokens=48)
    summary = bench.summarize([successful, failure], phase="measurement", slo_seconds=0.2, observation_seconds=2)
    assert summary["offered"] == 2
    assert summary["failed"] == 1
    assert summary["slo_attainment"] == 0.5
    assert summary["slo_goodput_per_observation_second"] == 0.5
    assert summary["input_tokens"] == 80


def test_transfer_tracking_requires_explicit_cleaned_response(monkeypatch):
    from agentshift.engine.sglang import SGLangAgentShiftClient

    async def started(self, **kwargs):
        return {"state": "PREPARED"}

    cleanup_result = {"state": "FAILED"}

    async def cleanup(self, transfer_id):
        return dict(cleanup_result)

    monkeypatch.setattr(SGLangAgentShiftClient, "start_async_hybrid_state_transfer", started)
    monkeypatch.setattr(SGLangAgentShiftClient, "cleanup_async_hybrid_state_transfer", cleanup)

    async def check():
        client = bench.ObservedTransferClient("0", "http://unused:1")
        await client.start_async_hybrid_state_transfer(transfer_id="transfer")
        assert client.live_transfers == {"transfer"}
        await client.cleanup_async_hybrid_state_transfer("transfer")
        assert client.live_transfers == {"transfer"}
        cleanup_result["state"] = "CLEANED"
        await client.cleanup_async_hybrid_state_transfer("transfer")
        assert not client.live_transfers

    asyncio.run(check())


def test_preparation_wait_timeout_preserves_launched_task():
    async def check():
        manager = object.__new__(bench.PreparationManager)
        manager.args = arguments()
        manager.policy = "cost-aware"
        manager.curves = {"cold": CostCurve(((16, 0.01), (32, 0.02))),
                          "copy": CostCurve(((16, 0.001),))}
        manager.log = SimpleNamespace(write=lambda row: None)
        prefix = tuple(range(16))
        launched = asyncio.create_task(asyncio.sleep(0.2))
        manager.tasks = {prefix: {"target": 1, "started": time.perf_counter(),
                                  "estimate_seconds": 0.001, "success": False, "task": launched}}
        waited = await manager.before_dispatch({"chat_id": 1}, [*prefix, *range(16)], 1)
        assert waited >= 0.005
        assert not launched.done()
        assert not launched.cancelled()
        # This is a mocked operation, so explicit teardown cancellation is safe.
        launched.cancel()
        await asyncio.gather(launched, return_exceptions=True)

    asyncio.run(check())
