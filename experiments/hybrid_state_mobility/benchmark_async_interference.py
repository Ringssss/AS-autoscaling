#!/usr/bin/env python3
"""Measure foreground interference of non-blocking hybrid-state transfer."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def flush(url: str) -> None:
    requests.post(url.rstrip("/") + "/flush_cache", timeout=30).raise_for_status()


def generate_sync(url: str, token_ids: list[int], output_tokens: int) -> dict:
    started = time.perf_counter()
    response = requests.post(
        url.rstrip("/") + "/generate",
        json={
            "input_ids": token_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    result["client_seconds"] = time.perf_counter() - started
    return result


def foreground_pair(
    source: str, destination: str, source_ids: list[int], destination_ids: list[int]
) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        source_future = executor.submit(generate_sync, source, source_ids, 96)
        destination_future = executor.submit(
            generate_sync, destination, destination_ids, 96
        )
        return source_future.result(), destination_future.result()


def summarize(rows: list[dict]) -> dict:
    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows]

    return {
        "samples": len(rows),
        "client_seconds_p50": statistics.median(values("client_seconds")),
        "client_seconds_p90": statistics.quantiles(values("client_seconds"), n=10)[8],
        "engine_e2e_p50": statistics.median(values("meta_info.e2e_latency")),
        "ttft_p50": statistics.median(values("meta_info.ttft")),
        "completion_tokens": int(rows[0]["meta_info.completion_tokens"]),
    }


def flat_result(result: dict) -> dict:
    return {
        "client_seconds": result["client_seconds"],
        "meta_info.e2e_latency": result["meta_info"]["e2e_latency"],
        "meta_info.ttft": result["meta_info"].get(
            "ttft", result["meta_info"]["e2e_latency"]
        ),
        "meta_info.completion_tokens": result["meta_info"]["completion_tokens"],
        "meta_info.cached_tokens": result["meta_info"]["cached_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--foreground-tokens", type=int, default=2048)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.manifest.open()]
    by_agent: dict[str, list[dict]] = {}
    for event in events:
        by_agent.setdefault(event["agent_id"], []).append(event)
    candidates = []
    for rows in by_agent.values():
        rows.sort(key=lambda row: row["turn_index"])
        if len(rows) >= 2 and rows[1]["prompt_tokens"] > rows[0]["prompt_tokens"]:
            candidates.append(rows[:2])
    candidates.sort(key=lambda rows: rows[0]["arrival_rel_s"])
    selected = []
    for rows in candidates:
        if len(selected) == args.repetitions:
            break
        if selected and rows[0]["agent_id"] == selected[-1][0]["agent_id"]:
            continue
        selected.append(rows)
    if len(selected) < args.repetitions:
        raise RuntimeError("not enough distinct replay agents")

    tokens = np.load(args.tokens)
    source_client = SGLangAgentShiftClient("source", args.source, timeout=300)
    destination_client = SGLangAgentShiftClient(
        "destination", args.destination, timeout=300
    )
    foreground_source_ids = [
        160000 + ((index * 101) % 40000) for index in range(args.foreground_tokens)
    ]
    foreground_destination_ids = [
        180000 + ((index * 103) % 40000) for index in range(args.foreground_tokens)
    ]

    baseline_rows = {"source": [], "destination": []}
    transfer_rows = {"source": [], "destination": []}
    migration_records = []

    with tempfile.TemporaryDirectory(prefix="agentshift-interference-") as state_dir:
        store = SQLiteStateStore(Path(state_dir) / "state.db")
        coordinator = HybridMigrationCoordinator(
            store,
            {"source": source_client, "destination": destination_client},
            base_port=args.base_port,
        )
        # Initialize the persistent group outside measured intervals.
        asyncio.run(coordinator._ensure_group("source", "destination"))

        for repetition, rows in enumerate(selected):
            first, second = rows
            first_ids = tokens[first["event_id"]].tolist()
            second_ids = tokens[second["event_id"]].tolist()

            # Baseline foreground run.
            flush(args.source)
            flush(args.destination)
            generate_sync(args.source, first_ids, 1)
            source_result, destination_result = foreground_pair(
                args.source,
                args.destination,
                foreground_source_ids,
                foreground_destination_ids,
            )
            baseline_rows["source"].append(flat_result(source_result))
            baseline_rows["destination"].append(flat_result(destination_result))

            # Warm source branch again after baseline cache flush.
            flush(args.source)
            flush(args.destination)
            generate_sync(args.source, first_ids, 1)
            agent_id = f"interference-{repetition:02d}"
            store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=first["turn_index"],
                    owner_engine="source",
                    owner_epoch=repetition + 100,
                    token_ids=tuple(first_ids),
                )
            )

            started = time.perf_counter()
            source_future_started = time.perf_counter()

            async def run_foreground_and_migrate():
                foreground = asyncio.gather(
                    asyncio.to_thread(
                        generate_sync, args.source, foreground_source_ids, 96
                    ),
                    asyncio.to_thread(
                        generate_sync,
                        args.destination,
                        foreground_destination_ids,
                        96,
                    ),
                )
                await asyncio.sleep(0.02)
                migration = await coordinator.migrate_async(agent_id, "destination")
                source_result, destination_result = await foreground
                return migration, source_result, destination_result

            migration, source_result, destination_result = asyncio.run(
                run_foreground_and_migrate()
            )
            wall_seconds = time.perf_counter() - started
            transfer_rows["source"].append(flat_result(source_result))
            transfer_rows["destination"].append(flat_result(destination_result))

            continuation = generate_sync(args.destination, second_ids, 1)
            checks = {
                "owner_committed": store.get_agent(agent_id).owner_engine
                == "destination",
                "bundle_bytes_formula": migration.bytes_transferred
                == len(first_ids) * 20480 + 64389120,
                "destination_full_parent_hit": continuation["meta_info"][
                    "cached_tokens"
                ]
                == len(first_ids),
            }
            migration_records.append(
                {
                    "repetition": repetition,
                    "agent_id": agent_id,
                    "source_replay_agent": first["agent_id"],
                    "prefix_tokens": len(first_ids),
                    "bundle_bytes": migration.bytes_transferred,
                    "migration_transfer_seconds": migration.transfer_seconds,
                    "wall_seconds": wall_seconds,
                    "checks": checks,
                }
            )
            if not all(checks.values()):
                raise SystemExit(f"interference migration failed: {agent_id}")

    def overhead(baseline: dict, transfer: dict) -> dict:
        return {
            key: transfer[key] / baseline[key]
            for key in (
                "client_seconds_p50",
                "engine_e2e_p50",
                "ttft_p50",
            )
        }

    result = {
        "repetitions": args.repetitions,
        "foreground_tokens": args.foreground_tokens,
        "baseline": {engine: summarize(rows) for engine, rows in baseline_rows.items()},
        "transfer": {engine: summarize(rows) for engine, rows in transfer_rows.items()},
        "overhead": {
            engine: overhead(
                summarize(baseline_rows[engine]),
                summarize(transfer_rows[engine]),
            )
            for engine in baseline_rows
        },
        "migrations": migration_records,
        "all_migration_checks": all(
            all(record["checks"].values()) for record in migration_records
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
