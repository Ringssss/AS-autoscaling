#!/usr/bin/env python3
"""Run a small real-engine window from the Kimi/LooGLE/ToolBench replay."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument("--base-port", type=int, default=29810)
    parser.add_argument("--async-transfer", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.manifest.open()]
    by_agent: dict[str, list[dict]] = {}
    for event in events:
        by_agent.setdefault(event["agent_id"], []).append(event)
    candidates = []
    for agent_id, rows in by_agent.items():
        rows.sort(key=lambda row: row["turn_index"])
        if len(rows) < 2:
            continue
        if max(row["prompt_tokens"] for row in rows[:2]) > args.max_prompt_tokens:
            continue
        if rows[1]["prompt_tokens"] <= rows[0]["prompt_tokens"]:
            continue
        candidates.append(rows[:2])
    candidates.sort(key=lambda rows: rows[0]["arrival_rel_s"])
    candidates = candidates[: args.agents]
    if not candidates:
        raise RuntimeError("no replay agents satisfy the window constraints")

    tokens = np.load(args.tokens)
    source_client = SGLangAgentShiftClient("source", args.source, timeout=300)
    destination_client = SGLangAgentShiftClient(
        "destination", args.destination, timeout=300
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="agentshift-trace-window-") as state_dir:
        store = SQLiteStateStore(Path(state_dir) / "state.db")
        coordinator = HybridMigrationCoordinator(
            store,
            {"source": source_client, "destination": destination_client},
            base_port=args.base_port,
        )
        for ordinal, rows in enumerate(candidates):
            first, second = rows
            first_ids = tokens[first["event_id"]].tolist()
            second_ids = tokens[second["event_id"]].tolist()
            agent_id = f"trace-window-{ordinal:02d}"
            requests.post(
                args.source.rstrip("/") + "/flush_cache", timeout=30
            ).raise_for_status()
            requests.post(
                args.destination.rstrip("/") + "/flush_cache", timeout=30
            ).raise_for_status()

            first_generate_started = time.perf_counter()
            first_result = generate(args.source, first_ids)
            first_generate_seconds = time.perf_counter() - first_generate_started
            store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=first["turn_index"],
                    owner_engine="source",
                    owner_epoch=ordinal + 1,
                    token_ids=tuple(first_ids),
                )
            )

            migration_started = time.perf_counter()
            migrate = (
                coordinator.migrate_async
                if args.async_transfer
                else coordinator.migrate
            )
            migration = asyncio.run(migrate(agent_id, "destination"))
            migration_seconds = time.perf_counter() - migration_started
            remaining_gap = max(
                0.0,
                first["tool_duration_s"] - migration_seconds,
            )
            time.sleep(remaining_gap)
            second_result = generate(args.destination, second_ids)
            agent = store.get_agent(agent_id)
            checks = {
                "owner_committed": agent.owner_engine == "destination",
                "first_zero_hit": first_result["meta_info"]["cached_tokens"] == 0,
                "second_full_parent_hit": second_result["meta_info"]["cached_tokens"]
                == len(first_ids),
                "bundle_bytes_formula": migration.bytes_transferred
                == len(first_ids) * 20480 + 64389120,
                "gap_covers_materialization": migration_seconds
                <= first["tool_duration_s"],
            }
            record = {
                "agent_id": agent_id,
                "source_agent_id": first["agent_id"],
                "tool_name": first["tool_name"],
                "tool_duration_s": first["tool_duration_s"],
                "first_prompt_tokens": len(first_ids),
                "second_prompt_tokens": len(second_ids),
                "migration": {
                    "migration_id": migration.migration_id,
                    "transfer_seconds": migration.transfer_seconds,
                    "wall_seconds": migration_seconds,
                    "bytes_transferred": migration.bytes_transferred,
                },
                "checks": checks,
            }
            records.append(record)
            print(json.dumps(record, indent=2))
            if not all(checks.values()):
                raise SystemExit(f"trace window failed for {agent_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "trace_direct_window.json"
    output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"artifact: {output}")


if __name__ == "__main__":
    main()
