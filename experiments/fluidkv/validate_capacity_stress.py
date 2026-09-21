#!/usr/bin/env python3
"""Measure 128k-pool headroom during a 32k copy and concurrent foreground work."""

import argparse
import asyncio
import json
import random
import statistics
import threading
import time
import uuid
from pathlib import Path

import aiohttp
import pynvml

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.calibrate_decision_costs import (
    TimedClient, calibration_store, digest, finish_store, migration_stages,
)
from experiments.fluidkv.validate_eviction_admission import workload


class MemorySampler:
    def __init__(self, indices, guard_gib, hz=100):
        pynvml.nvmlInit()
        self.handles = [(index, pynvml.nvmlDeviceGetHandleByIndex(index)) for index in indices]
        def decoded(value):
            return value.decode() if isinstance(value, bytes) else value
        self.devices = [{"index": index, "uuid": decoded(pynvml.nvmlDeviceGetUUID(handle)),
                         "name": decoded(pynvml.nvmlDeviceGetName(handle))} for index, handle in self.handles]
        self.guard_bytes = guard_gib * 2**30
        self.hz = hz
        self.rows = []
        self.stop_event = threading.Event()
        self.guard_failed = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            now = time.perf_counter()
            row = {"monotonic": now, "gpus": []}
            try:
                for index, handle in self.handles:
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    row["gpus"].append({"index": index, "used": memory.used,
                                        "free": memory.free, "total": memory.total})
                    if memory.free < self.guard_bytes:
                        self.guard_failed.set()
                self.rows.append(row)
            except Exception as exc:
                self.errors.append(repr(exc))
                self.guard_failed.set()
            next_tick += 1 / self.hz
            self.stop_event.wait(max(0, next_tick - time.perf_counter()))

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        intervals = [b["monotonic"] - a["monotonic"] for a, b in zip(self.rows, self.rows[1:])]
        minimum = {str(index): min((m["free"] for row in self.rows for m in row["gpus"]
                                   if m["index"] == index), default=0) for index, _ in self.handles}
        return {"devices": self.devices, "requested_hz": self.hz,
                "effective_hz": 1 / statistics.mean(intervals) if intervals else 0,
                "maximum_interval_seconds": max(intervals, default=0),
                "sample_count": len(self.rows), "minimum_free_bytes": minimum,
                "minimum_free_gib": {index: value / 2**30 for index, value in minimum.items()},
                "guard_gib": self.guard_bytes / 2**30,
                "guard_passed": not self.guard_failed.is_set(), "errors": self.errors,
                "scope": "Driver-visible NVML free memory, requested 100Hz; discrete samples can miss shorter peaks. CUDA allocated/reserved peaks are not exposed by the serving API."}


async def engine_load(session, url):
    async with session.get(url + "/v1/loads?include=all") as response:
        response.raise_for_status()
        return await response.json()


async def wait_first_tokens(tasks, first_events, timeout=180):
    """Fail immediately if a generation ends without producing its first token."""
    ready = asyncio.gather(*(event.wait() for event in first_events))
    unfinished = set(tasks)
    deadline = time.monotonic() + timeout
    try:
        while not ready.done():
            completed, _ = await asyncio.wait({ready, *unfinished},
                timeout=max(0, deadline-time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
            if not completed:
                raise TimeoutError("Timed out waiting for foreground first tokens")
            for task in completed:
                if task is not ready:
                    task.result()  # Propagate a rejected/failed request promptly.
                    unfinished.discard(task)
        await ready
    finally:
        if not ready.done():
            ready.cancel()
        await asyncio.gather(ready, return_exceptions=True)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    events = []
    clients = {name: TimedClient(name, url, events)
               for name, url in zip(("source", "destination"), args.engines)}
    store = calibration_store(args)
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        info = []
        for url in args.engines:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        source_outputs = args.source_output_tokens or args.output_tokens
        # TpModelWorker reserves one position in max_req_len; the scheduler
        # reserves one more in init_req_max_new_tokens. Avoid silent truncation.
        source_budget = min(info[0]["context_length"], args.capacity_tokens) - 32000 - 2
        if source_outputs > source_budget:
            raise ValueError(f"Source output budget exceeds runtime-preserved limit {source_budget}")
        for url in args.engines:
            load = await engine_load(session, url)
            if any(item["max_total_num_tokens"] != args.capacity_tokens for item in load["loads"]):
                raise RuntimeError("Effective token capacity differs from stress configuration")
        await coordinator._ensure_group("source", "destination")
        with (args.output / "records.jsonl").open("w", buffering=1) as log:
            for repetition in range(-1, args.repetitions):
                count = min(3, args.background_requests) if repetition < 0 else args.background_requests
                row = {"phase": "smoke" if repetition < 0 else "measured", "repetition": repetition,
                       "background_requests": count, "background_input_tokens": 4096,
                       "output_tokens_per_request": args.output_tokens,
                       "source_output_tokens": source_outputs, "source_input_tokens": 32000,
                       "success": False, "expected_capacity_rejection": False}
                sampler = MemorySampler(args.gpu_indices, args.minimum_free_gib)
                tasks = []
                try:
                    await flush(session, args.engines)
                    sampler.start()
                    fillers, _, _ = workload(args.seed, repetition, args.capacity_tokens)
                    for tokens in fillers:
                        await stream(session, args.engines[1], tokens, 1)
                        if sampler.guard_failed.is_set():
                            raise RuntimeError("Memory guard failed during cache filling")
                    row["filled_cached_tokens"] = sum(map(len, fillers))
                    rng = random.Random(args.seed + 901 + repetition)
                    parent = [2000] + [rng.randrange(20000, 25000) for _ in range(31999)]
                    child = parent + [rng.randrange(26000, 27000) for _ in range(64)]
                    backgrounds = [[30000 + index] + [rng.randrange(1100, 19000) for _ in range(4095)]
                                   for index in range(count)]
                    row["parent_hash"] = digest(parent)
                    agent = uuid.uuid4().hex
                    store.register_agent(AgentContinuation(agent, 0, "source", 1, tuple(parent)))
                    first_events = [asyncio.Event() for _ in range(count + 1)]

                    def callback(index):
                        async def first(_payload, _latency):
                            first_events[index].set()
                        return first

                    tasks.append(asyncio.create_task(stream(session, args.engines[0], parent, source_outputs, callback(0))))
                    tasks.extend(asyncio.create_task(stream(session, args.engines[1], tokens, args.output_tokens, callback(index + 1)))
                                 for index, tokens in enumerate(backgrounds))
                    await wait_first_tokens(tasks, first_events)
                    row["source_active_before_copy"] = not tasks[0].done()
                    row["background_active_before_copy"] = sum(not task.done() for task in tasks[1:])
                    row["loads_before_copy"] = await asyncio.gather(*(engine_load(session, url) for url in args.engines))
                    if sampler.guard_failed.is_set():
                        raise RuntimeError("Memory guard failed before copy; no transfer started")
                    events.clear()
                    started = time.perf_counter()
                    migration = None
                    try:
                        migration = await coordinator.migrate_async(agent, "destination")
                    except Exception as exc:
                        row["migration_error"] = repr(exc)
                        row["expected_capacity_rejection"] = "capacity" in str(exc)
                        if not row["expected_capacity_rejection"]:
                            raise
                    finally:
                        finish = time.perf_counter()
                        row["migration_stages"] = migration_stages(events, started, finish)
                        row["migration_events"] = list(events)
                        row["source_active_after_copy"] = not tasks[0].done()
                        row["background_active_after_copy"] = sum(not task.done() for task in tasks[1:])
                    row["foreground_results"] = await asyncio.gather(*tasks)
                    if sampler.guard_failed.is_set():
                        raise RuntimeError("Memory guard failed; foreground drained, further trials stopped")
                    if migration is None:
                        row["after_rejection_retry"] = True
                        events.clear()
                        migration = await coordinator.migrate_async(agent, "destination")
                        row["retry_events"] = list(events)
                    row["transfer_bytes"] = migration.bytes_transferred
                    coordinator.acknowledge_async_migration(migration.migration_id)
                    row["sticky_child"] = await stream(session, args.engines[0], child, 8)
                    row["migrated_child"] = await stream(session, args.engines[1], child, 8)
                    cleanup = [event for event in events if "/cleanup/" in event["path"]]
                    row["checks"] = {
                        "source_active_at_launch": row["source_active_before_copy"],
                        "all_background_active_at_launch": row["background_active_before_copy"] == count,
                        "all_foreground_completed": len(row["foreground_results"]) == count + 1
                            and len(row["foreground_results"][0]["output_ids"]) == source_outputs
                            and all(len(result["output_ids"]) == args.output_tokens for result in row["foreground_results"][1:]),
                        "both_jobs_cleaned": len(cleanup) == 2 and all(
                            event.get("response", {}).get("state") == "CLEANED" for event in cleanup),
                        "source_cached_prompt": row["sticky_child"]["meta_info"]["cached_tokens"] == len(parent),
                        "destination_cached_prompt": row["migrated_child"]["meta_info"]["cached_tokens"] == len(parent),
                        "continuation_equal": row["sticky_child"]["output_ids"] == row["migrated_child"]["output_ids"],
                    }
                    row["loads_after_drain"] = await asyncio.gather(*(engine_load(session, url) for url in args.engines))
                    row["checks"]["engines_drained"] = all(
                        item["num_running_reqs"] == 0 and item["num_waiting_reqs"] == 0
                        for load in row["loads_after_drain"] for item in load["loads"])
                    # Keep all allocator residency until sampling ends; no flush
                    # can hide the late high-water mark or race a transfer job.
                    await asyncio.sleep(0.25)
                    row["success"] = all(row["checks"].values())
                    if not row["success"]:
                        raise RuntimeError("Capacity stress correctness/overlap/cleanup gate failed")
                except Exception as exc:
                    row["error"] = repr(exc)
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                finally:
                    row["memory"] = sampler.stop()
                    row["success"] = row["success"] and row["memory"]["guard_passed"]
                    (args.output / f"memory-{repetition}.jsonl").write_text("".join(json.dumps(value) + "\n" for value in sampler.rows))
                    records.append(row)
                    log.write(json.dumps(row) + "\n")
                    (args.output / "summary.json").write_text(json.dumps({
                        "scope": __doc__, "records": records, "capacity_tokens": args.capacity_tokens,
                        "minimum_free_gib_guard": args.minimum_free_gib,
                    }, indent=2) + "\n")
                    print(json.dumps({"phase": row["phase"], "repetition": repetition,
                                      "success": row["success"], "error": row.get("error"),
                                      "memory": row["memory"], "checks": row.get("checks")}), flush=True)
                if not row["success"]:
                    raise RuntimeError("Memory guard failed; stop without retuning")
                await flush(session, args.engines)
    finish_store(store, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs=2, default=["http://127.0.0.1:31480", "http://127.0.0.1:31481"])
    parser.add_argument("--capacity-tokens", type=int, default=131072)
    parser.add_argument("--gpu-indices", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--minimum-free-gib", type=float, default=2)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--source-output-tokens", type=int,
                        help="Separate source output budget so the 32k prompt stays within context")
    parser.add_argument("--background-requests", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--transfer-port", type=int, default=32402)
    parser.add_argument("--metadata-directory", type=Path, default=Path("/tmp/fluidkv-metadata-20260914/calibration"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.background_requests < 1 or args.output_tokens < 1 or (args.source_output_tokens is not None and args.source_output_tokens < 1):
        parser.error("Request counts and output budgets must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
