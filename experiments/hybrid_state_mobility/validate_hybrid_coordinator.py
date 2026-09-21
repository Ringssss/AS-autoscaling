#!/usr/bin/env python3
"""Drive the durable hybrid migration coordinator against two live engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import requests

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation, MigrationState
from agentshift.state.store import SQLiteStateStore


def generate(url: str, input_ids: list[int], output_tokens: int) -> dict:
    response = requests.post(
        url.rstrip("/") + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--base-port", type=int, default=29760)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for url in (args.source, args.destination):
        requests.get(url.rstrip("/") + "/health", timeout=30).raise_for_status()
        requests.post(url.rstrip("/") + "/flush_cache", timeout=30).raise_for_status()

    branch_ids = [
        100000 + ((index * 71) % 120000) for index in range(args.prefix_tokens)
    ]
    continuation_ids = [
        200000 + ((index * 73) % 40000) for index in range(args.continuation_tokens)
    ]
    continuation_input = branch_ids + continuation_ids
    generate(args.source, branch_ids, 1)

    source = SGLangAgentShiftClient("source", args.source)
    destination = SGLangAgentShiftClient("destination", args.destination)
    with tempfile.TemporaryDirectory(prefix="agentshift-hybrid-") as state_dir:
        store = SQLiteStateStore(Path(state_dir) / "state.db")
        store.register_agent(
            AgentContinuation(
                agent_id="hybrid-agent",
                committed_step=1,
                owner_engine="source",
                owner_epoch=7,
                token_ids=tuple(branch_ids),
            )
        )
        coordinator = HybridMigrationCoordinator(
            store,
            {"source": source, "destination": destination},
            base_port=args.base_port,
        )
        result = asyncio.run(coordinator.migrate("hybrid-agent", "destination"))
        migration = store.get_migration(result.migration_id)
        agent = store.get_agent("hybrid-agent")
        record = {
            "result": {
                "migration_id": result.migration_id,
                "old_epoch": result.old_epoch,
                "new_epoch": result.new_epoch,
                "token_count": result.token_count,
                "bytes_transferred": result.bytes_transferred,
                "transfer_seconds": result.transfer_seconds,
                "group_name": result.group_name,
            },
            "durable_state": {
                "migration_state": migration.state.value,
                "owner_engine": agent.owner_engine,
                "owner_epoch": agent.owner_epoch,
            },
        }

    source_result = generate(args.source, continuation_input, args.continuation_tokens)
    destination_result = generate(
        args.destination, continuation_input, args.continuation_tokens
    )
    record["source_output_ids"] = source_result["output_ids"]
    record["destination_output_ids"] = destination_result["output_ids"]
    record["source_cached_tokens"] = source_result["meta_info"]["cached_tokens"]
    record["destination_cached_tokens"] = destination_result["meta_info"][
        "cached_tokens"
    ]
    record["checks"] = {
        "owner_committed": agent.owner_engine == "destination"
        and agent.owner_epoch == 8,
        "migration_committed": migration.state == MigrationState.COMMITTED,
        "exact_output_match": source_result["output_ids"]
        == destination_result["output_ids"],
        "destination_full_hit": destination_result["meta_info"]["cached_tokens"]
        == len(branch_ids),
        "bytes_formula_match": result.bytes_transferred
        == len(branch_ids) * 20480 + 64389120,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "hybrid_coordinator_result.json"
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["checks"], indent=2))
    print(f"artifact: {output_path}")
    if not all(record["checks"].values()):
        raise SystemExit("hybrid coordinator checks failed")


if __name__ == "__main__":
    main()
