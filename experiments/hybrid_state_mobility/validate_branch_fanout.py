#!/usr/bin/env python3
"""Validate copy-on-write child branches after parent-state migration."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def generate(url: str, token_ids: list[int], output_tokens: int = 8) -> dict:
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
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--parent-tokens", type=int, default=4096)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.branches < 2:
        raise ValueError("branch fanout requires at least two branches")
    parent = [140000 + ((index * 97) % 35000) for index in range(args.parent_tokens)]
    branches = []
    for branch_index in range(args.branches):
        suffix = [
            180000 + ((branch_index * 3000 + index * 89) % 30000)
            for index in range(args.suffix_tokens)
        ]
        branches.append(parent + suffix)

    for url in (args.source, args.destination):
        requests.get(url.rstrip("/") + "/health", timeout=30).raise_for_status()
        requests.post(url.rstrip("/") + "/flush_cache", timeout=30).raise_for_status()

    parent_result = generate(args.source, parent, 1)
    with tempfile.TemporaryDirectory(prefix="agentshift-fanout-") as state_dir:
        store = SQLiteStateStore(Path(state_dir) / "state.db")
        store.register_agent(
            AgentContinuation(
                agent_id="fanout-parent",
                committed_step=0,
                owner_engine="source",
                owner_epoch=300,
                token_ids=tuple(parent),
            )
        )
        coordinator = HybridMigrationCoordinator(
            store,
            {
                "source": SGLangAgentShiftClient("source", args.source, timeout=300),
                "destination": SGLangAgentShiftClient(
                    "destination", args.destination, timeout=300
                ),
            },
            base_port=args.base_port,
        )
        migration = asyncio.run(
            coordinator.migrate_async("fanout-parent", "destination")
        )
        with ThreadPoolExecutor(max_workers=args.branches) as executor:
            futures = [
                executor.submit(generate, args.destination, branch, 8)
                for branch in branches
            ]
            branch_results = [future.result() for future in futures]

    checks = {
        "parent_zero_hit": parent_result["meta_info"]["cached_tokens"] == 0,
        "owner_committed": migration.continuation.owner_engine == "destination",
        "bundle_bytes_formula": migration.bytes_transferred
        == args.parent_tokens * 20480 + 64389120,
        "all_branches_full_parent_hit": all(
            result["meta_info"]["cached_tokens"] == args.parent_tokens
            for result in branch_results
        ),
        "all_branches_completed": all(
            result["meta_info"]["completion_tokens"] == 8 for result in branch_results
        ),
    }
    result = {
        "parent_tokens": args.parent_tokens,
        "suffix_tokens": args.suffix_tokens,
        "branches": args.branches,
        "migration": {
            "migration_id": migration.migration_id,
            "transfer_seconds": migration.transfer_seconds,
            "bytes_transferred": migration.bytes_transferred,
        },
        "branch_cached_tokens": [
            item["meta_info"]["cached_tokens"] for item in branch_results
        ],
        "branch_output_ids": [item["output_ids"] for item in branch_results],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit("branch fanout checks failed")


if __name__ == "__main__":
    main()
