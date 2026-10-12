#!/usr/bin/env python3
"""Controlled fan-out skew: one hot executable prefix, two equal TP workers.

Both workers have the same GPU/memory budget. A parent prefix is installed only
on worker 0. A fixed fan-out arrives after a declared preparation gap. Replays
retain exact prefix identity and use fixed synthetic suffixes, not agent text.
Every request records arrival-to-first-token and arrival-to-completion, including
any exposed preparation. There is no engine inspect in the request router.

Prepared policies copy/recompute ONE prefix then reuse it across many consumers.
Their first burst waits for full readiness (the current synchronous runtime).
This is an explicit full-preparation treatment, not progressive/busy migration.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import aiohttp
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "p03", Path(__file__).with_name("p03_real_copy_e2e.py"))
p03 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p03)

POLICIES = ("sticky", "balanced", "cache_cost", "prepared_recompute", "prepared_copy")


async def generate(session, url, ids, output, *, arrival=None):
    dispatched = time.perf_counter()
    arrival = dispatched if arrival is None else arrival
    first = last = None
    previous_count = 0
    intervals, coalesced = [], 0
    final = {}
    async with session.post(url + "/generate", json={
        "input_ids": ids, "stream": True,
        "sampling_params": {"temperature": 0, "ignore_eos": True,
                            "max_new_tokens": output},
    }) as response:
        response.raise_for_status()
        async for raw in response.content:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                continue
            final = json.loads(data)
            count = final.get("meta_info", {}).get("completion_tokens", 0)
            if count > previous_count:
                now = time.perf_counter()
                if first is None:
                    first = now
                elif last is not None:
                    intervals.append((now - last) / (count - previous_count))
                if count - previous_count > 1:
                    coalesced += count - previous_count - 1
                last, previous_count = now, count
    done = time.perf_counter()
    if first is None or previous_count != output:
        raise RuntimeError(f"expected {output} output tokens, got {previous_count}")
    meta = final.get("meta_info", {})
    return {
        "ttft": first - arrival, "e2e": done - arrival,
        "router_wait": dispatched - arrival, "serve_ttft": first - dispatched,
        "service_seconds": done - dispatched,
        "tpot": (last - first) / (previous_count - 1) if previous_count > 1 else None,
        "token_intervals": intervals, "coalesced_tokens": coalesced,
        "cached_tokens": meta.get("cached_tokens", 0),
        "completion_tokens": previous_count,
        "output_ids": final.get("output_ids"),
        "output_sha256": hashlib.sha256(json.dumps(final.get("output_ids")).encode()).hexdigest(),
        "meta_info": meta, "error": None,
    }


async def json_post(session, url, path, payload=None):
    async with session.post(url + path, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def drain(session, urls, timeout=120):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        loads = []
        for url in urls:
            async with session.get(url + "/get_load") as response:
                response.raise_for_status()
                loads.extend(await response.json())
        if all(row["num_reqs"] == 0 for row in loads):
            await asyncio.sleep(0.5)
            return
        await asyncio.sleep(0.5)
    raise RuntimeError("engines failed to drain; refusing contaminated next arm")


async def reset(session, urls):
    await drain(session, urls)
    for url in urls:
        async with session.post(url + "/flush_cache?timeout=30") as response:
            response.raise_for_status()
            body = await response.text()
            # The success response includes a generic warning about busy engines.
            # Match the actual outcome, not that warning's wording.
            if not body.lstrip().startswith("Cache flushed."):
                raise RuntimeError(f"flush rejected: {body}")
    await asyncio.sleep(0.5)


class Router:
    """Reservations count immediately, and retire on every completion."""
    def __init__(self, policy, calibration, max_running, ready):
        self.policy = policy
        self.calibration = calibration
        self.max_running = max_running
        self.inflight = [0, 0]
        self.ready = list(ready)

    def reserve(self):
        if self.policy == "sticky":
            target = 0
        elif self.policy == "cache_cost":
            def score(i):
                # Simple observable cost baseline, not an upstream DualMap claim.
                # Warm requests can batch. Cold reservations also carry a prefill
                # backlog until a completed consumer establishes a usable prefix.
                warm = self.calibration["warm_ttft"]
                cold = self.calibration["cold_ttft"]
                batch_queue = (self.inflight[i] // self.max_running
                               * self.calibration["warm_service"])
                return (batch_queue + warm if self.ready[i] else
                        batch_queue + cold + self.inflight[i] * max(0., cold - warm))
            target = min(range(2), key=lambda i: (score(i), self.inflight[i], i))
        else:
            target = min(range(2), key=lambda i: (self.inflight[i], i))
        self.inflight[target] += 1
        return target

    def release(self, target, success):
        self.inflight[target] -= 1
        if self.inflight[target] < 0:
            raise RuntimeError("negative in-flight reservation")
        if success:
            self.ready[target] = True


def summarize(records, wall, ttft_slo, tpot_slo):
    ok = [r for r in records if r["error"] is None]
    good = [r for r in ok if r["ttft"] <= ttft_slo and
            (r["tpot"] is None or r["tpot"] <= tpot_slo)]
    intervals = [x for r in ok for x in r["token_intervals"]]
    def pct(field, p):
        vals = [r[field] for r in ok if r[field] is not None]
        return float(np.percentile(vals, p)) if vals else None
    return {
        "requests": len(records), "errors": len(records) - len(ok), "wall": wall,
        "ttft_p50": pct("ttft", 50), "ttft_p90": pct("ttft", 90),
        "e2e_p50": pct("e2e", 50), "e2e_p90": pct("e2e", 90),
        "tpot_p50": pct("tpot", 50), "tpot_p90": pct("tpot", 90),
        "token_interval_p99": float(np.percentile(intervals, 99)) if intervals else None,
        "coalesced_tokens": sum(r["coalesced_tokens"] for r in ok),
        "slo_attainment": len(good) / len(records), "goodput": len(good) / wall,
        "ttft_slo_attainment": sum(r["ttft"] <= ttft_slo for r in ok) / len(records),
        "cached_tokens": sum(r["cached_tokens"] for r in ok),
        "target_counts": dict(Counter(r["target"] for r in records)),
        "slo_grid": [{"ttft_limit": t, "tpot_limit": p,
                      "attainment": sum(r["ttft"] <= t and
                          (r["tpot"] is None or r["tpot"] <= p) for r in ok) / len(records),
                      "goodput": sum(r["ttft"] <= t and
                          (r["tpot"] is None or r["tpot"] <= p) for r in ok) / wall}
                     for t in (5., 10., 20.) for p in (.1, .2, .4)],
    }


async def run_trial(session, args, gc, prefix, policy, rep, calibration, log):
    await reset(session, args.urls)
    parent = await generate(session, args.urls[0], prefix, 16)
    st = await json_post(session, args.urls[0], "/agentshift/hybrid_state/inspect",
                         {"token_ids": prefix})
    matched = st["matched_tokens"]
    if matched < len(prefix) * .8:
        raise RuntimeError(f"source executable prefix unexpectedly short: {matched}")
    # Setup is identical across arms; source lookup is outside the timed preparation.
    preparation_start = time.perf_counter()
    arrival = preparation_start + args.gap
    preparation = {"method": "none", "seconds": 0, "bytes": 0}
    if policy == "prepared_copy":
        preparation = await p03.copy_state(session, gc, args.urls, 0, 1, prefix[:matched])
        preparation["method"] = "copy"
        if not preparation.get("ok"):
            raise RuntimeError(f"copy failed: {preparation}")
    elif policy == "prepared_recompute":
        result = await generate(session, args.urls[1], prefix, 16)
        preparation = {"method": "recompute", "seconds": result["service_seconds"],
                       "bytes": 0, "result": result}
    preparation_done = time.perf_counter()
    if policy.startswith("prepared_"):
        dst = await json_post(session, args.urls[1], "/agentshift/hybrid_state/inspect",
                              {"token_ids": prefix})
        if dst["matched_tokens"] < matched:
            raise RuntimeError(f"destination prefix unavailable: {dst['matched_tokens']} < {matched}")
    # Inspection above is included if it exposes a preparation tail.
    if arrival > time.perf_counter():
        await asyncio.sleep(arrival - time.perf_counter())
    router = Router(policy, calibration, args.max_running,
                    [True, policy.startswith("prepared_")])
    rows = []

    async def consumer(index):
        # Fixed replay suffixes are equal across arms/reps and append per turn.
        suffix = np.random.default_rng(args.seed + index).integers(1000, 240000,
                         args.suffix * args.turns, dtype=np.int32).tolist()
        previous_done = arrival
        for turn in range(args.turns):
            eligible = arrival if turn == 0 else previous_done + args.turn_gap
            if eligible > time.perf_counter():
                await asyncio.sleep(eligible - time.perf_counter())
            target = router.reserve()
            ids = prefix + suffix[:args.suffix * (turn + 1)]
            try:
                result = await generate(session, args.urls[target], ids, args.output_tokens,
                                        arrival=eligible)
            except Exception as exc:
                result = {"error": repr(exc), "ttft": None, "e2e": None,
                          "tpot": None, "token_intervals": [], "coalesced_tokens": 0,
                          "cached_tokens": 0}
            finally:
                previous_done = time.perf_counter()
            router.release(target, result["error"] is None)
            row = dict(policy=policy, repetition=rep, prefix_tokens=len(prefix),
                       gap=args.gap, streams=args.streams, consumer=index, turn=turn,
                       target=target, input_tokens=len(ids), output_tokens=args.output_tokens,
                       **result)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
    await asyncio.gather(*(consumer(i) for i in range(args.streams)))
    if router.inflight != [0, 0]:
        raise RuntimeError("reservations remain after all consumers")
    wall = time.perf_counter() - arrival
    summary = summarize(rows, wall, args.ttft_slo, args.tpot_slo)
    return {
        "policy": policy, "repetition": rep, "summary": summary,
        "parent": parent, "matched_source_tokens": matched, "preparation": preparation,
        "preparation_wall": preparation_done - preparation_start,
        "exposed_preparation": max(0, preparation_done - arrival),
        "preparation_inclusive_wall": time.perf_counter() - preparation_start,
        "first_wave_targets": [r["target"] for r in rows if r["turn"] == 0],
    }


async def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    toks = np.load(args.tokens, mmap_mode="r")
    prefix = toks[:args.prefix].astype(int).tolist()
    if len(prefix) != args.prefix:
        raise ValueError("token source shorter than requested prefix")
    out = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "scope": "Controlled shared-prefix fanout, fixed synthetic suffix replay; full synchronous preparation, not progressive or busy-source migration.",
           "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
           "trials": []}
    runtime = p03.ROOT / "experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py"
    out["runtime_sha256"] = hashlib.sha256(runtime.read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        out["server_info"] = []
        for url in args.urls:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                out["server_info"].append(await response.json())
        for info in out["server_info"]:
            actual = info.get("max_running_requests", info.get("server_args", {}).get("max_running_requests"))
            if actual is not None and actual != args.max_running:
                raise RuntimeError(f"router assumes max-running={args.max_running}, engine has {actual}")
        await reset(session, args.urls)
        gc = p03.GroupCache()
        started = time.perf_counter()
        group = await gc.ensure(session, args.urls, 0, 1)
        if group is None:
            raise RuntimeError(f"group init failed: {gc.last_error}")
        out["group_init_seconds"] = time.perf_counter() - started
        out["group"] = list(group)
        print("group ready", out["group_init_seconds"], flush=True)
        if args.warmup:
            # Compile comparable warm prefill/decode batch shapes on BOTH workers.
            # Trial caches are subsequently reset; kernel warmup is not cache prep.
            out["warmups"] = []
            for url in args.urls:
                await generate(session, url, prefix, 16)
                batches = []
                batch = 1
                while batch <= min(args.max_running, args.streams):
                    batches.append(batch)
                    batch *= 2
                for batch in batches:
                    prompts = [prefix + np.random.default_rng(args.seed + i).integers(
                        1000, 240000, args.suffix, dtype=np.int32).tolist() for i in range(batch)]
                    results = await asyncio.gather(*(generate(session, url, ids, 16) for ids in prompts))
                    out["warmups"].append({"url": url, "batch": batch,
                                          "wall_max": max(r["e2e"] for r in results)})
                    print("warmup", url, batch, flush=True)
            await reset(session, args.urls)
        calibration_suffix = np.random.default_rng(args.seed).integers(
            1000, 240000, args.suffix, dtype=np.int32).tolist()
        calibration_ids = prefix + calibration_suffix
        cold = await generate(session, args.urls[0], calibration_ids, args.output_tokens)
        warm = await generate(session, args.urls[0], calibration_ids, args.output_tokens)
        calibration = {"cold_ttft": cold["ttft"], "warm_ttft": warm["ttft"],
                       "warm_service": warm["service_seconds"],
                       "cold": cold, "warm": warm}
        out["calibration"] = calibration
        print("calibration", {k: calibration[k] for k in
                              ("cold_ttft", "warm_ttft", "warm_service")}, flush=True)
        rng = random.Random(args.seed)
        with (args.out / "requests.jsonl").open("w") as log:
            for rep in range(args.reps):
                order = rng.sample(args.policies, len(args.policies))
                for policy in order:
                    print("start", rep, policy, flush=True)
                    result = await run_trial(session, args, gc, prefix, policy, rep, calibration, log)
                    out["trials"].append(result)
                    (args.out / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
                    print(json.dumps(result["summary"]), flush=True)
        await drain(session, args.urls)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", nargs=2, default=["http://12.11.3.27:31881", "http://12.11.3.27:31882"])
    ap.add_argument("--tokens", type=Path, default=Path(p03.ROOT / "experiments/weka_baselines/data/tokens.npy"))
    ap.add_argument("--prefix", type=int, default=65536)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--suffix", type=int, default=64)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--gap", type=float, default=1.)
    ap.add_argument("--turn-gap", type=float, default=.1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-running", type=int, default=8)
    ap.add_argument("--ttft-slo", type=float, default=10.)
    ap.add_argument("--tpot-slo", type=float, default=.1)
    ap.add_argument("--timeout", type=float, default=300.)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.prefix + args.suffix * args.turns + args.output_tokens > 131072:
        ap.error("workload exceeds engine context length")
    if min(args.streams, args.turns, args.output_tokens, args.suffix, args.reps) < 1 or args.gap < 0:
        ap.error("positive workload sizes and nonnegative gap required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
