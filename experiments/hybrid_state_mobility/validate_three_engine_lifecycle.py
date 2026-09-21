#!/usr/bin/env python3
"""Run scale-out followed by scale-in on three real Qwen3.5 engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def flush(urls: list[str]) -> None:
    for url in urls:
        requests.post(url.rstrip("/") + "/flush_cache", timeout=30).raise_for_status()


def generate(url: str, token_ids: list[int]) -> dict:
    started = time.perf_counter()
    response = requests.post(
        url.rstrip("/") + "/generate",
        json={
            "input_ids": token_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    result["client_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--scaleout", default="http://127.0.0.1:31082")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=6000)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.manifest.open()]
    by_agent: dict[str, list[dict]] = {}
    for event in events:
        by_agent.setdefault(event["agent_id"], []).append(event)
    candidates = []
    for rows in by_agent.values():
        rows.sort(key=lambda row: row["turn_index"])
        if len(rows) < 3:
            continue
        if max(row["prompt_tokens"] for row in rows[:3]) > args.max_prompt_tokens:
            continue
        if not (
            rows[1]["prompt_tokens"] > rows[0]["prompt_tokens"]
            and rows[2]["prompt_tokens"] > rows[1]["prompt_tokens"]
        ):
            continue
        candidates.append(rows[:3])
    candidates.sort(key=lambda rows: rows[0]["arrival_rel_s"])
    candidates = candidates[: args.agents]
    if len(candidates) != args.agents:
        raise RuntimeError("not enough three-turn replay agents")

    tokens = np.load(args.tokens)
    clients = {
        "source": SGLangAgentShiftClient("source", args.source, timeout=300),
        "destination": SGLangAgentShiftClient(
            "destination", args.destination, timeout=300
        ),
        "scaleout": SGLangAgentShiftClient("scaleout", args.scaleout, timeout=300),
    }
    urls = [args.source, args.destination, args.scaleout]
    for url in urls:
        requests.get(url.rstrip("/") + "/health", timeout=30).raise_for_status()
    flush(urls)

    records = []
    with tempfile.TemporaryDirectory(prefix="agentshift-lifecycle-") as state_dir:
        store = SQLiteStateStore(Path(state_dir) / "state.db")
        coordinator = HybridMigrationCoordinator(
            store, clients, base_port=args.base_port
        )
        for ordinal, rows in enumerate(candidates):
            first, second, third = rows
            first_ids = tokens[first["event_id"]].tolist()
            second_ids = tokens[second["event_id"]].tolist()
            third_ids = tokens[third["event_id"]].tolist()
            agent_id = f"lifecycle-{ordinal:02d}"
            store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=first["turn_index"],
                    owner_engine="source",
                    owner_epoch=ordinal + 200,
                    token_ids=tuple(first_ids),
                )
            )

            first_generate_started = time.perf_counter()
            first_result = generate(args.source, first_ids)
            first_generate_seconds = time.perf_counter() - first_generate_started
            scaleout_started = time.perf_counter()
            scaleout_migration = asyncio.run(
                coordinator.migrate_async(agent_id, "scaleout")
            )
            coordinator.acknowledge_async_migration(scaleout_migration.migration_id)
            scaleout_wall = time.perf_counter() - scaleout_started
            time.sleep(max(0.0, first["tool_duration_s"] - scaleout_wall))
            second_result = generate(args.scaleout, second_ids)

            # The scale-out engine has now advanced the branch to turn two.
            with store.transaction() as conn:
                conn.execute(
                    "UPDATE agents SET token_ids=? WHERE agent_id=?",
                    (json.dumps(second_ids), agent_id),
                )
            scalein_started = time.perf_counter()
            scalein_migration = asyncio.run(
                coordinator.migrate_async(agent_id, "destination")
            )
            scalein_wall = time.perf_counter() - scalein_started
            time.sleep(max(0.0, second["tool_duration_s"] - scalein_wall))
            third_result = generate(args.destination, third_ids)

            checks = {
                "first_source_zero_hit": first_result["meta_info"]["cached_tokens"]
                == 0,
                "scaleout_second_full_hit": second_result["meta_info"]["cached_tokens"]
                == len(first_ids),
                "destination_third_full_hit": third_result["meta_info"]["cached_tokens"]
                == len(second_ids),
                "scaleout_owner_committed": scaleout_migration.new_epoch
                == scaleout_migration.old_epoch + 1,
                "scalein_owner_committed": scalein_migration.new_epoch
                == scalein_migration.old_epoch + 1,
                "final_owner_destination": store.get_agent(agent_id).owner_engine
                == "destination",
                "scaleout_gap_covered": scaleout_wall <= first["tool_duration_s"],
                "scalein_gap_covered": scalein_wall <= second["tool_duration_s"],
                "bundle_bytes_formula": scaleout_migration.bytes_transferred
                == len(first_ids) * 20480 + 64389120
                and scalein_migration.bytes_transferred
                == len(second_ids) * 20480 + 64389120,
            }
            record = {
                "agent_id": agent_id,
                "source_replay_agent": first["agent_id"],
                "tools": [first["tool_name"], second["tool_name"]],
                "prefix_tokens": [
                    len(first_ids),
                    len(second_ids),
                    len(third_ids),
                ],
                "first_generate_seconds": first_generate_seconds,
                "scaleout": {
                    "migration_id": scaleout_migration.migration_id,
                    "transfer_seconds": scaleout_migration.transfer_seconds,
                    "wall_seconds": scaleout_wall,
                    "bytes": scaleout_migration.bytes_transferred,
                },
                "scalein": {
                    "migration_id": scalein_migration.migration_id,
                    "transfer_seconds": scalein_migration.transfer_seconds,
                    "wall_seconds": scalein_wall,
                    "bytes": scalein_migration.bytes_transferred,
                },
                "checks": checks,
            }
            records.append(record)
            print(json.dumps(record, indent=2))
            if not all(checks.values()):
                raise SystemExit(f"lifecycle failed for {agent_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "three_engine_lifecycle.json"
    output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"artifact: {output}")


if __name__ == "__main__":
    main()
