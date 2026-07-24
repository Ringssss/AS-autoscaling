from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Callable

from agentshift.state.schema import (
    AgentContinuation,
    MigrationRecord,
    MigrationState,
    ToolResult,
)
from agentshift.state.store import SQLiteStateStore


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def run_parallel(
    operation: Callable[[int], None], count: int, concurrency: int
) -> dict[str, float]:
    latencies: list[float] = []

    def timed(index: int) -> float:
        started = time.perf_counter()
        operation(index)
        return time.perf_counter() - started

    wall_started = time.perf_counter()
    if concurrency == 1:
        latencies = [timed(index) for index in range(count)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            latencies = list(pool.map(timed, range(count)))
    wall = time.perf_counter() - wall_started
    return {
        "operations": count,
        "concurrency": concurrency,
        "throughput_ops_per_second": count / wall,
        "mean_ms": statistics.fmean(latencies) * 1000,
        "p50_ms": percentile(latencies, 0.5) * 1000,
        "p95_ms": percentile(latencies, 0.95) * 1000,
        "p99_ms": percentile(latencies, 0.99) * 1000,
        "wall_seconds": wall,
    }


def database_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def handoff(store: SQLiteStateStore, agent_id: str, destination: str) -> None:
    continuation = store.get_agent(agent_id)
    migration_id = uuid.uuid4().hex
    store.start_migration(
        MigrationRecord(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=continuation.owner_engine,
            destination_engine=destination,
            source_epoch=continuation.owner_epoch,
            state=MigrationState.PREPARING,
        )
    )
    store.transition_migration(
        migration_id, MigrationState.PREPARING, MigrationState.COPYING
    )
    store.transition_migration(
        migration_id, MigrationState.COPYING, MigrationState.DEST_READY
    )
    store.commit_migration(migration_id)
    store.transition_migration(
        migration_id, MigrationState.COMMITTED, MigrationState.SOURCE_RELEASED
    )


def benchmark_scale(args: argparse.Namespace, agent_count: int) -> dict:
    path = Path(args.output_dir) / f"control-{agent_count}-{time.time_ns()}.db"
    store = SQLiteStateStore(path)
    populate_started = time.perf_counter()
    for index in range(agent_count):
        store.register_agent(
            AgentContinuation(
                agent_id=f"agent-{index}",
                committed_step=0,
                owner_engine="engine-a",
                owner_epoch=1,
                token_ids=(),
                pending_tool_future=f"future-{index}",
            )
        )
    populate_seconds = time.perf_counter() - populate_started

    results: list[dict] = []
    for concurrency in args.concurrency:
        operation_count = min(args.operations, agent_count)

        read = run_parallel(
            lambda index: store.assert_lease(
                f"agent-{index % agent_count}", "engine-a", 1
            ),
            operation_count,
            concurrency,
        )
        results.append({"operation": "lease_read", **read})

        mailbox = run_parallel(
            lambda index: store.put_tool_result(
                ToolResult(
                    agent_id=f"agent-{index % agent_count}",
                    step_id=1,
                    future_id=f"mailbox-c{concurrency}-{index}",
                    payload={"ok": True},
                )
            ),
            operation_count,
            concurrency,
        )
        results.append({"operation": "mailbox_insert", **mailbox})

        def claim_and_fail(index: int) -> None:
            agent_id = f"agent-{index % agent_count}"
            rid = f"claim-c{concurrency}-{index}"
            store.claim_step(
                agent_id=agent_id,
                step_id=1,
                owner_engine="engine-a",
                owner_epoch=1,
                rid=rid,
            )
            store.fail_claimed_step(agent_id, 1, rid, outcome_unknown=False)

        claims = run_parallel(claim_and_fail, operation_count, concurrency)
        results.append({"operation": "step_claim_fail", **claims})

    segments = len(args.concurrency)
    segment_size = agent_count // segments
    for segment, concurrency in enumerate(args.concurrency):
        start = segment * segment_size
        available = segment_size if segment < segments - 1 else agent_count - start
        operation_count = min(args.operations, available)
        ownership = run_parallel(
            lambda index, base=start: handoff(
                store, f"agent-{base + index}", "engine-b"
            ),
            operation_count,
            concurrency,
        )
        results.append({"operation": "ownership_cas", **ownership})

    bytes_before_restart = database_bytes(path)
    restart_started = time.perf_counter()
    restarted = SQLiteStateStore(path)
    live = restarted.list_migrations(
        (
            MigrationState.PREPARING,
            MigrationState.COPYING,
            MigrationState.DEST_READY,
            MigrationState.COMMITTED,
        )
    )
    restart_seconds = time.perf_counter() - restart_started
    return {
        "agent_count": agent_count,
        "populate_seconds": populate_seconds,
        "populate_ops_per_second": agent_count / populate_seconds,
        "database_bytes": bytes_before_restart,
        "bytes_per_agent": bytes_before_restart / agent_count,
        "restart_scan_seconds": restart_seconds,
        "live_migrations_after_restart": len(live),
        "database": str(path),
        "operations": results,
    }


def main(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output = {
        "config": vars(args),
        "platform": {
            "cpu_count": os.cpu_count(),
            "sqlite_version": __import__("sqlite3").sqlite_version,
            "journal_mode": "WAL",
            "synchronous": "FULL",
        },
        "scales": [benchmark_scale(args, count) for count in args.agent_counts],
    }
    output_path = Path(args.output_dir) / f"control-plane-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), **output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--operations", type=int, default=1000)
    parser.add_argument("--output-dir", default="results")
    main(parser.parse_args())
