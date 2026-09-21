#!/usr/bin/env python3
"""Exhaust destination capacity without launching an unmatched NCCL receive."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import aiohttp

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation, MigrationState
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.calibrate_decision_costs import calibration_store, finish_store


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    clients = {name: SGLangAgentShiftClient(name, url, timeout=180) for name, url in zip(("source", "destination"), args.engines)}
    store = (calibration_store(args) if args.metadata_directory is not None
             else SQLiteStateStore(args.output / "state.db"))
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        info = []
        for url in args.engines:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        group, port = await coordinator._ensure_group("source", "destination")
        for repetition in range(3):
            await flush(session, args.engines)
            parent = [1100 + i % 10000 for i in range(1024)]
            child = parent + [19000 + i for i in range(64)]
            await stream(session, args.engines[0], parent, 1)
            blocker = "reserve-" + uuid.uuid4().hex
            reserved = await clients["destination"].start_async_hybrid_state_transfer(
                transfer_id=blocker, role="destination", token_ids=tuple([7000] * args.capacity_tokens),
                group_name=group, port=port, prepare_only=True)
            if reserved["state"] != "PREPARED":
                raise RuntimeError("capacity blocker was not privately prepared")
            agent_id = f"blocked-{repetition}"
            store.register_agent(AgentContinuation(agent_id, 0, "source", 1, tuple(parent)))
            started = time.perf_counter()
            error = None
            try:
                await coordinator.migrate_async(agent_id, "destination")
            except Exception as exc:
                error = str(exc)
            failure_seconds = time.perf_counter() - started
            migration_id = store.conn.execute("SELECT migration_id FROM migrations WHERE agent_id=?", (agent_id,)).fetchone()[0]
            state = store.get_migration(migration_id).state
            source_after_failure = await stream(session, args.engines[0], child, 8)
            cleaned = await clients["destination"].cleanup_async_hybrid_state_transfer(blocker)
            # Reuse the same NCCL group immediately after capacity rejection.
            recovered = await coordinator.migrate_async(agent_id, "destination")
            destination_after_retry = await stream(session, args.engines[1], child, 8)
            checks = {
                "capacity_error": error is not None and "capacity" in error,
                "migration_aborted": state == MigrationState.ABORTED,
                "private_reservation_cleaned": cleaned["state"] == "CLEANED",
                "retry_committed": recovered.new_epoch == 2,
                "continuation_equal_after_retry": source_after_failure["output_ids"] == destination_after_retry["output_ids"],
                "destination_parent_hit": destination_after_retry["meta_info"]["cached_tokens"] >= len(parent),
            }
            record = {"repetition": repetition, "checks": checks, "rejection_seconds": failure_seconds,
                      "error": error, "retry_transfer_bytes": recovered.bytes_transferred}
            records.append(record)
            (args.output / "summary.json").write_text(json.dumps({"scope": "Destination capacity rejection, prepared-state cleanup and same-group retry with live Qwen3.5 engines.", "records": records}, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if not all(checks.values()):
                raise RuntimeError("reservation failure regression")
        await flush(session, args.engines)
    if args.metadata_directory is not None:
        finish_store(store, args)
    else:
        store.conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs=2, default=["http://127.0.0.1:31480", "http://127.0.0.1:31481"])
    parser.add_argument("--capacity-tokens", type=int, default=65536)
    parser.add_argument("--transfer-port", type=int, default=29970)
    parser.add_argument("--metadata-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
