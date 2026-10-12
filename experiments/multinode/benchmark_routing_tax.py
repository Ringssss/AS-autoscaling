"""Measure a cache-location change; this is not a DualMap/CacheRoute ranking.

All modes share the same parent and continuation. Destination state is known
cold in this component experiment. A separate warm-on-destination control
prevents equating every routing change with a cache miss. Copy time and
controller ownership commit are included in on-return latency.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source, destination = args.source, args.destination
    clients = {name: SGLangAgentShiftClient(name, url, timeout=300)
               for name, url in (("source", source), ("destination", destination))}
    store = SQLiteStateStore(args.output / "ownership.db")
    coordinator = HybridMigrationCoordinator(store, clients, master_address=args.master_address,
                                             base_port=args.base_port)
    rng = random.Random(args.seed)
    modes = ["sticky-warm-source", "reroute-cold-destination", "reroute-warm-destination", "migrate-on-return"]
    if args.token_file:
        corpus = np.load(args.token_file)["tokens"].tolist()
        if len(corpus) < max(args.prefix_tokens) + 32:
            raise ValueError("corpus has insufficient tokens")
        provenance = {"kind": "public-text-substitution", "file": str(args.token_file),
                      "sha256": hashlib.sha256(args.token_file.read_bytes()).hexdigest()}
    else:
        corpus = [100000 + ((i * 41) % 120000) for i in range(max(args.prefix_tokens) + 32)]
        provenance = {"kind": "synthetic-token-IDs", "formula": "100000 + (i * 41) % 120000"}
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        servers = []
        for url in (source, destination):
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                servers.append(await response.json())
            # Compile every measured prefill/continuation shape before timing.
            for length in args.prefix_tokens:
                await flush(session, [url])
                await stream(session, url, corpus[:length], 1)
                await stream(session, url, corpus[:length + 32], args.output_tokens)
        (args.output / "server_info.json").write_text(json.dumps(servers, indent=2) + "\n")
        started = time.perf_counter()
        await coordinator._ensure_group("source", "destination")
        group_seconds = time.perf_counter() - started
        # RCCL initializes the actual device communicator lazily on its first
        # collective. Materialize it before steady-state paired measurements.
        await flush(session, [source, destination])
        warm_parent = corpus[:min(args.prefix_tokens)]
        await stream(session, source, warm_parent, 1)
        store.register_agent(AgentContinuation("transport-warmup", 1, "source", 1, tuple(warm_parent)))
        warm_started = time.perf_counter()
        await coordinator.migrate("transport-warmup", "destination")
        group_warm_transfer_seconds = time.perf_counter() - warm_started
        with (args.output / "requests.jsonl").open("w", buffering=1) as log:
            for repeat in range(args.repetitions):
                for length in rng.sample(args.prefix_tokens, len(args.prefix_tokens)):
                    parent, child = corpus[:length], corpus[:length + 32]
                    paired = []
                    for mode in rng.sample(modes, len(modes)):
                        await flush(session, [source, destination])
                        parent_result = await stream(session, source, parent, 1)
                        if mode == "reroute-warm-destination":
                            await stream(session, destination, parent, 1)
                        migration = None
                        start = time.perf_counter()
                        if mode == "migrate-on-return":
                            agent = uuid.uuid4().hex
                            store.register_agent(AgentContinuation(agent, 1, "source", 1, tuple(parent)))
                            migration = await coordinator.migrate(agent, "destination")
                        copy_and_commit_seconds = time.perf_counter() - start
                        target = source if mode == "sticky-warm-source" else destination
                        first_token_at = None

                        async def first_token(payload, delay):
                            nonlocal first_token_at
                            first_token_at = time.perf_counter()

                        result = await stream(session, target, child, args.output_tokens, on_first_token=first_token)
                        # One client clock includes dispatch, RPC, transfer, ownership
                        # commit, network and engine work exactly once.
                        ttft = first_token_at - start
                        row = {"repeat": repeat, "prefix_tokens": length, "mode": mode,
                               "target": target, "generation": result, "parent": parent_result,
                               "post_return_ttft_seconds": ttft,
                               "post_return_e2e_seconds": time.perf_counter() - start,
                               "migration_and_commit_seconds": copy_and_commit_seconds if migration else 0,
                               "transfer_bytes": migration.bytes_transferred if migration else 0,
                               "ownership_committed": store.get_agent(agent).owner_engine == "destination" if migration else None}
                        paired.append(row)
                    sticky = next(r for r in paired if r["mode"] == "sticky-warm-source")["generation"]["output_ids"]
                    cold = next(r for r in paired if r["mode"] == "reroute-cold-destination")["generation"]["output_ids"]
                    for row in paired:
                        result = row["generation"]
                        expected = 0 if row["mode"] == "reroute-cold-destination" else length
                        row["checks"] = {
                            "exact_sticky_output": result["output_ids"] == sticky,
                            "exact_cold_output": result["output_ids"] == cold,
                            "expected_cache_hit": result["meta_info"]["cached_tokens"] == expected,
                            "input_length": result["meta_info"]["prompt_tokens"] == len(child),
                            "output_length": result["meta_info"]["completion_tokens"] == args.output_tokens,
                            "parent_initially_cold": row["parent"]["meta_info"]["cached_tokens"] == 0,
                            "owner_committed": row["ownership_committed"] is not False,
                        }
                        log.write(json.dumps(row) + "\n")
                        rows.append(row)
                    print(json.dumps({"repeat": repeat, "prefix": length,
                                      "checks": all(all(r["checks"].values()) for r in paired)}), flush=True)
    summaries = []
    for length in args.prefix_tokens:
        for mode in modes:
            selected = [r for r in rows if r["prefix_tokens"] == length and r["mode"] == mode]
            values = [r["post_return_ttft_seconds"] for r in selected]
            summaries.append({"prefix_tokens": length, "mode": mode, "samples": len(values),
                              "ttft_median_seconds": float(np.median(values)),
                              "ttft_min_max_seconds": [min(values), max(values)],
                              "all_checks_passed": all(all(r["checks"].values()) for r in selected)})
    report = {"scope": "Isolated cache-location/copy component measurements on real multi-node engines, not a production policy ranking.",
              "content_provenance": provenance, "source": source, "destination": destination,
              "group_initialization_seconds": group_seconds, "output_tokens": args.output_tokens,
              "group_first_transfer_warmup_seconds": group_warm_transfer_seconds,
              "repetitions": args.repetitions, "seed": args.seed, "summaries": summaries,
              "cache_setup": "Independent flushed trials; all have warm source; warm-destination control additionally prefills destination outside request timing.",
              "limitations": ["No foreground contention", "No production prefix/lineage reconstruction", "No tool gap assumed or subtracted", "Not a p99/SLO-capacity campaign"]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(all(row["checks"].values()) for row in rows):
        raise SystemExit("Correctness gate failed; retain results and diagnose before comparing performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--base-port", type=int, default=30100)
    parser.add_argument("--seed", type=int, default=20260922)
    asyncio.run(run(parser.parse_args()))
