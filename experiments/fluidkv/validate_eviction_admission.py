#!/usr/bin/env python3
"""Verify async destination admission can reclaim ordinary cached hybrid state.

Disjoint 16,368-token prompts fill the configured KV pool almost completely. Before
migration, read-only source preparation probes confirm each complete prompt is
still present; probes are immediately cleaned without launching NCCL. An 8k
incoming state therefore requires eviction, unlike privately reserved capacity.
"""

import argparse
import asyncio
import json
import random
import socket
import time
import uuid
from pathlib import Path

import aiohttp

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.calibrate_decision_costs import (
    TimedClient, calibration_store, digest, finish_store, migration_stages,
)


def workload(seed, repetition, capacity_tokens=65536, filler_tokens=16368):
    rng = random.Random(seed + repetition)
    count = (capacity_tokens - 1) // filler_tokens
    fillers = [[1000 + index] + [rng.randrange(1100, 19000) for _ in range(filler_tokens - 1)]
               for index in range(count)]
    parent = [2000] + [rng.randrange(20000, 25000) for _ in range(8191)]
    child = parent + [rng.randrange(27000, 32000) for _ in range(64)]
    assert len({tokens[0] for tokens in [*fillers, parent]}) == count + 1
    assert 0 < capacity_tokens - sum(len(tokens) for tokens in fillers) < len(parent)
    return fillers, parent, child


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.transfer_port))
    events = []
    clients = {name: TimedClient(name, url, events)
               for name, url in zip(("source", "destination"), args.engines)}
    store = calibration_store(args)
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        server_info = []
        for url in args.engines:
            async with session.get(url.rstrip("/") + "/get_server_info") as response:
                response.raise_for_status()
                server_info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(server_info, indent=2) + "\n")
        if any(info.get("max_total_tokens") != args.capacity_tokens for info in server_info):
            raise RuntimeError("Configured engine token slots do not match requested capacity")
        group, port = await coordinator._ensure_group("source", "destination")
        with (args.output / "records.jsonl").open("w", buffering=1) as log:
            for repetition in range(args.repetitions):
                fillers, parent, child = workload(args.seed, repetition, args.capacity_tokens, args.filler_tokens)
                row = {"repetition": repetition, "success": False,
                       "capacity_tokens": args.capacity_tokens,
                       "filler_tokens": sum(map(len, fillers)), "incoming_tokens": len(parent),
                       "nominal_initial_free_tokens": args.capacity_tokens - sum(map(len, fillers)),
                       "filler_hashes": [digest(tokens) for tokens in fillers],
                       "parent_hash": digest(parent), "fill_results": [], "presence_probes": []}
                try:
                    await flush(session, args.engines)
                    for tokens in fillers:
                        row["fill_results"].append(await stream(session, args.engines[1], tokens, 1))
                    # These are source-role snapshot reads on the destination
                    # engine. They reserve no new KV slots and never launch a
                    # send; cleanup drops only the temporary pin/staging.
                    for tokens in fillers:
                        probe_id = "presence-" + uuid.uuid4().hex
                        prepared = await clients["destination"].start_async_hybrid_state_transfer(
                            transfer_id=probe_id, role="source", token_ids=tuple(tokens),
                            group_name=group, port=port, prepare_only=True)
                        cleaned = await clients["destination"].cleanup_async_hybrid_state_transfer(probe_id)
                        row["presence_probes"].append({"prepared": prepared, "cleanup": cleaned})
                    row["source_prepare"] = await stream(session, args.engines[0], parent, 1)
                    agent = uuid.uuid4().hex
                    store.register_agent(AgentContinuation(agent, 0, "source", 1, tuple(parent)))
                    events.clear()
                    started = time.perf_counter()
                    try:
                        migration = await coordinator.migrate_async(agent, "destination")
                    finally:
                        finish = time.perf_counter()
                        row["migration_events"] = list(events)
                        row["migration_stages"] = migration_stages(events, started, finish)
                    row["transfer_bytes"] = migration.bytes_transferred
                    coordinator.acknowledge_async_migration(migration.migration_id)
                    row["sticky_child"] = await stream(session, args.engines[0], child, 8)
                    row["migrated_child"] = await stream(session, args.engines[1], child, 8)
                    cleanup = [event for event in events if "/cleanup/" in event["path"]]
                    row["checks"] = {
                        "all_disjoint_cold_fillers": len(row["fill_results"]) == len(fillers) and all(
                            result["meta_info"]["cached_tokens"] == 0
                            and result["meta_info"]["prompt_tokens"] == args.filler_tokens
                            for result in row["fill_results"]),
                        "all_fillers_still_resident_before_migration": len(row["presence_probes"]) == len(fillers) and all(
                            probe["prepared"]["state"] == "PREPARED"
                            and probe["prepared"]["token_count"] == args.filler_tokens
                            and probe["cleanup"]["state"] == "CLEANED"
                            for probe in row["presence_probes"]),
                        "source_initially_cold": row["source_prepare"]["meta_info"]["cached_tokens"] == 0,
                        "migration_committed": store.get_agent(agent).owner_engine == "destination",
                        "both_cleanup_cleaned": len(cleanup) == 2 and all(
                            event.get("response", {}).get("state") == "CLEANED" for event in cleanup),
                        "sticky_parent_hit": row["sticky_child"]["meta_info"]["cached_tokens"] == len(parent),
                        "migrated_parent_hit": row["migrated_child"]["meta_info"]["cached_tokens"] == len(parent),
                        "continuation_equal": row["sticky_child"]["output_ids"] == row["migrated_child"]["output_ids"],
                        "output_lengths": len(row["sticky_child"]["output_ids"]) == 8 and len(row["migrated_child"]["output_ids"]) == 8,
                    }
                    row["success"] = all(row["checks"].values())
                    if not row["success"]:
                        raise RuntimeError("Live eviction admission regression failed")
                except Exception as exc:
                    row["error"] = repr(exc)
                    raise
                finally:
                    records.append(row)
                    log.write(json.dumps(row) + "\n")
                    (args.output / "summary.json").write_text(json.dumps({
                        "scope": __doc__, "records": records,
                        "successful_repetitions": sum(record["success"] for record in records),
                        "repetitions": args.repetitions,
                    }, indent=2) + "\n")
                    print(json.dumps({"repetition": repetition, "success": row["success"],
                                      "checks": row.get("checks"), "error": row.get("error")}), flush=True)
        await flush(session, args.engines)
    finish_store(store, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs=2, default=["http://127.0.0.1:31480", "http://127.0.0.1:31481"])
    parser.add_argument("--transfer-port", type=int, default=31200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--capacity-tokens", type=int, default=65536)
    parser.add_argument("--filler-tokens", type=int, default=16368)
    parser.add_argument("--metadata-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("At least one repetition is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
