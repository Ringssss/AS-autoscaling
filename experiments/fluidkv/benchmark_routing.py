#!/usr/bin/env python3
"""Common-engine native Qwen hash-trace pilot for cache-aware routers."""

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np

from agentshift.workloads.production import surrogate_tokens, select_exact_parent_cohort
from experiments.fluidkv.audit_traces import sha256
from experiments.fluidkv.dualmap_adapter import DualMapAdapter, UPSTREAM_COMMIT
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.retirement_materializer import RetirementMaterializer


async def run_policy(args, policy, repetition, rows, session, calibration):
    await flush(session, args.engines)
    materialization_mode = policy in ("dualmap-full-copy", "fluidkv-planner-prototype")
    adapter = DualMapAdapter(args.dualmap_source, len(args.engines), args.cache_tokens,
                             args.bytes_per_token, args.slo, calibration,
                             mode="dualmap" if materialization_mode else policy,
                             hybrid_snapshot_slots=args.hybrid_snapshot_slots)
    materializer = RetirementMaterializer(args, adapter, session, policy,
        1 / calibration, repetition) if materialization_mode else None
    if materializer:
        await materializer.initialize()
    records, tasks = [], set()
    finished = {row["chat_id"]: asyncio.Event() for row in rows}
    source_rows = {i: row for i, row in enumerate(rows)}
    eligibility = {}
    event = asyncio.Event()
    producers_done = False
    began = time.perf_counter()
    base = rows[0]["timestamp"]

    async def serve(request):
        row = source_rows[request._id]
        target = adapter.physical_target(request)
        dispatched = time.perf_counter()
        expected_cache = len(request._input_ids) - request._actual_num_prefill_tokens
        result = None
        error = None

        async def first_token(payload, delay):
            await adapter.first_token(request, delay)
            event.set()

        try:
            result = await stream(session, args.engines[target], request._input_ids, request._output_len, on_first_token=first_token)
            if result["meta_info"]["prompt_tokens"] != row["input_length"]:
                raise ValueError("backend changed prompt length")
            if result["meta_info"]["completion_tokens"] != request._output_len:
                raise ValueError("backend changed output length")
        except Exception as exc:
            error = str(exc)
        await adapter.finished(request, result["output_ids"] if result else [], error is None)
        ready = eligibility[request._id]
        record = {
            "policy": policy, "repetition": repetition, "chat_id": row["chat_id"],
            "parent_chat_id": row["parent_chat_id"], "trace_arrival_seconds": row["timestamp"],
            "target": target, "input_tokens": row["input_length"],
            "recorded_output_tokens": row["output_length"], "requested_output_tokens": request._output_len,
            "eligibility_delay_seconds": ready - (began + (row["timestamp"] - base) / args.time_scale),
            "router_wait_seconds": dispatched - ready,
            "ttft_seconds": dispatched - ready + result["ttft_seconds"] if result and error is None else None,
            "e2e_seconds": time.perf_counter() - ready,
            "estimated_cached_tokens": expected_cache,
            "actual_cached_tokens": result["meta_info"]["cached_tokens"] if result else None,
            "output_digest": hashlib.sha256(json.dumps(result["output_ids"]).encode()).hexdigest() if result else None,
            "success": error is None, "error": error,
        }
        records.append(record)
        with (args.output / "requests.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")
        finished[row["chat_id"]].set()
        if materializer and error is None:
            materializer.observe_completion(row["chat_id"], target, request._input_ids)
        event.set()

    async def produce(ordinal, row):
        delay = began + (row["timestamp"] - base) / args.time_scale - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if args.causal and row["parent_chat_id"] in finished:
            await finished[row["parent_chat_id"]].wait()
        eligibility[ordinal] = time.perf_counter()
        count = min(row["output_length"], args.output_cap) if args.output_cap else row["output_length"]
        await adapter.enqueue(ordinal, row, surrogate_tokens(row), count)
        event.set()

    producers = [asyncio.create_task(produce(i, row)) for i, row in enumerate(rows)]

    async def production():
        nonlocal producers_done
        await asyncio.gather(*producers)
        producers_done = True
        event.set()

    production_task = asyncio.create_task(production())
    retired = False
    retired_engine = None
    while not producers_done or tasks or adapter.queued():
        if production_task.done() and production_task.exception():
            raise production_task.exception()
        event.clear()
        if args.retire_at is not None and not retired and time.perf_counter() - began >= args.retire_at:
            victim = adapter.largest_observed_prefix_owner() if args.retire_engine == -1 else args.retire_engine
            source = await adapter.retire_engine(victim if victim is not None else adapter.active_ids[-1])
            retired_engine = source
            retired = True
            if materializer:
                materializer.retire(source)
        ready = await adapter.dispatch()
        for request in ready:
            task = asyncio.create_task(serve(request))
            tasks.add(task)
            task.add_done_callback(lambda done: event.set())
        completed = {task for task in tasks if task.done()}
        for task in completed:
            task.result()
        tasks.difference_update(completed)
        if producers_done and not tasks and not adapter.queued():
            break
        try:
            await asyncio.wait_for(event.wait(), timeout=0.02)
        except asyncio.TimeoutError:
            pass
    await production_task
    serving_elapsed = time.perf_counter() - began
    materialization = await materializer.finish() if materializer else None
    elapsed = time.perf_counter() - began
    successes = [r for r in records if r["success"]]
    good = sum(r["ttft_seconds"] <= args.slo for r in successes)
    return {
        "policy": policy, "repetition": repetition, "offered": len(rows), "completed": len(records),
        "logical_retirement_triggered": retired, "materialization": materialization,
        "retired_physical_engine": retired_engine,
        "retirement_elapsed_seconds": materializer.retired_at - began if materializer and materializer.retired_at else None,
        "serving_elapsed_seconds": serving_elapsed,
        "successful": len(successes), "duration_seconds": elapsed,
        "ttft_p50_p90": np.percentile([r["ttft_seconds"] for r in successes], [50, 90]).tolist() if successes else None,
        "slo_seconds": args.slo, "slo_attainment": good / len(rows),
        "slo_goodput_requests_per_second": good / elapsed,
        "actual_cached_tokens": sum(r["actual_cached_tokens"] for r in successes),
        "cache_estimate_absolute_error_tokens": sum(abs(r["actual_cached_tokens"] - r["estimated_cached_tokens"]) for r in successes),
        "target_request_counts": dict(Counter(r["target"] for r in records)),
        "upstream_rebalanced_requests": adapter.scheduler.rebalance_cnt,
    }


async def main_async(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = [json.loads(line) for line in args.trace.open()]
    candidates = [r for r in original if args.start <= r["timestamp"] < args.start + args.duration]
    def fits(row):
        output = min(row["output_length"], args.output_cap) if args.output_cap else row["output_length"]
        return (0 < row["input_length"] <= args.max_input and output > 0
                and row["input_length"] + output <= args.context_length)

    rows = [r for r in candidates if fits(r)]
    if args.selection == "exact-parent-cohort":
        rows = select_exact_parent_cohort(rows, args.requests)
    else:
        rows = sorted(rows, key=lambda r: (r["timestamp"], r["chat_id"]))[:args.requests]
    if not rows:
        raise ValueError("empty selected trace window")
    report = {
        "scope": "Common-engine routing/logical-retirement pilot on anonymous hash-preserving inputs; no physical DRAM tier, physical GPU release, or full FluidKV scheduling claim.",
        "trace": str(args.trace), "trace_sha256": sha256(args.trace),
        "upstream_commit": UPSTREAM_COMMIT, "selected_requests": len(rows),
        "window_requests": len(candidates), "max_input_tokens": args.max_input,
        "length_filtered_requests": sum(not fits(row) for row in candidates),
        "output_cap": args.output_cap, "time_scale": args.time_scale, "causal": args.causal,
        "boundary_parents_missing_from_selection": sum(r["parent_chat_id"] >= 0 and r["parent_chat_id"] not in {x["chat_id"] for x in rows} for r in rows),
        "adaptations": ["SGLang input IDs and streaming timing", "upstream block cache is an estimate, actual hits come from engine", "actual generated IDs update cache accounting", "fixed membership and common backend for pilot"],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_chat_ids": [row["chat_id"] for row in rows],
        "selection": args.selection,
        "selected_internal_parent_links": sum(row["parent_chat_id"] in {r["chat_id"] for r in rows} for row in rows),
        "runs": [],
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        servers = []
        for url in args.engines:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                servers.append(await response.json())
        if len({server["model_path"] for server in servers}) != 1:
            raise ValueError("all comparison engines must use the same model")
        if any(args.context_length > server["context_length"] for server in servers):
            raise ValueError("harness context limit exceeds an engine context limit")
        if any(policy in ("dualmap-full-copy", "fluidkv-planner-prototype") for policy in args.policies):
            if args.retire_at is None:
                raise ValueError("materialization pilot requires an explicit retirement event")
            if any("Qwen3.5-35B-A3B" not in server["model_path"] for server in servers):
                raise ValueError("current descriptor calibration requires Qwen3.5-35B-A3B")
        (args.output / "server_info.json").write_text(json.dumps(servers, indent=2) + "\n")
        calibration = []
        for url in args.engines:
            await stream(session, url, list(range(1500, 1628)), 8)
            times = []
            for length in (1024, 4096):
                await flush(session, [url])
                result = await stream(session, url, [1000 + i % 30000 for i in range(length)], 1)
                times.append(result["ttft_seconds"])
            calibration.append(max(1e-6, (times[1] - times[0]) / 3072))
        prefill_cost = float(np.median(calibration))
        report["calibration_seconds_per_token"] = calibration
        rng = random.Random(args.seed)
        for repetition in range(args.repetitions):
            for policy in rng.sample(args.policies, len(args.policies)):
                result = await asyncio.wait_for(run_policy(args, policy, repetition, rows, session, prefill_cost), timeout=args.run_timeout)
                report["runs"].append(result)
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dualmap-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", default=["cache-affinity", "least-loaded", "dualmap"])
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--selection", choices=["chronological", "exact-parent-cohort"], default="chronological")
    parser.add_argument("--max-input", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--hybrid-snapshot-slots", type=int, default=0)
    parser.add_argument("--output-cap", type=int, default=16)
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--slo", type=float, default=1)
    parser.add_argument("--cache-tokens", type=int, default=32768)
    parser.add_argument("--bytes-per-token", type=int, default=57344)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-timeout", type=float, default=600)
    parser.add_argument("--retire-at", type=float)
    parser.add_argument("--retire-engine", type=int)
    parser.add_argument("--transfer-port", type=int, default=29940)
    parser.add_argument("--bandwidth-gib", type=float, default=4.37)
    parser.add_argument("--copy-fixed-ms", type=float, default=50)
    parser.add_argument("--drain-budget", type=float, default=60)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
