from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient, generate
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


async def main(args) -> None:
    source = SGLangAgentShiftClient("engine-a", args.source, timeout=180)
    destination = SGLangAgentShiftClient("engine-b", args.destination, timeout=180)
    prompt = [args.seed_token] + [args.fill_token] * (args.context_length - 1)

    first_started = time.perf_counter()
    first = await generate(
        source,
        prompt,
        max_new_tokens=args.output_length,
        rid=f"{args.agent_id}-turn-1",
    )
    first_seconds = time.perf_counter() - first_started
    completed_tokens = tuple(prompt + first["output_ids"])

    db_path = Path(args.state_dir) / f"{args.agent_id}-{time.time_ns()}.db"
    store = SQLiteStateStore(db_path)
    store.register_agent(
        AgentContinuation(
            agent_id=args.agent_id,
            committed_step=1,
            owner_engine="engine-a",
            owner_epoch=1,
            token_ids=completed_tokens,
            pending_tool_future="tool-1",
        )
    )
    coordinator = MigrationCoordinator(
        store,
        {"engine-a": source, "engine-b": destination},
        base_port=args.transfer_port,
        tp_size=args.tp_size,
        async_transfer=not args.sync_transfer,
    )
    migration = await coordinator.migrate(args.agent_id, "engine-b")

    tool_result_tokens = [args.tool_token] * args.tool_result_length
    second_prompt = list(completed_tokens) + tool_result_tokens
    second_started = time.perf_counter()
    second = await generate(
        destination,
        second_prompt,
        max_new_tokens=1,
        rid=f"{args.agent_id}-turn-2",
    )
    second_seconds = time.perf_counter() - second_started
    await coordinator.acknowledge_destination(migration.migration_id)

    cached_tokens = int(second["meta_info"]["cached_tokens"])
    if cached_tokens < migration.token_count:
        raise RuntimeError(
            f"target cache hit {cached_tokens} is smaller than migrated prefix "
            f"{migration.token_count}"
        )
    result = {
        "agent_id": args.agent_id,
        "migration_id": migration.migration_id,
        "context_length": args.context_length,
        "first_turn_seconds": first_seconds,
        "migration_seconds": migration.transfer_seconds,
        "worker_transfer_seconds": migration.worker_transfer_seconds,
        "transfer_queue_seconds": migration.queue_seconds,
        "migration_tokens": migration.token_count,
        "bytes_transferred": migration.bytes_transferred,
        "transfer_gib_per_second": migration.bytes_transferred
        / migration.transfer_seconds
        / (1024**3),
        "target_cached_tokens": cached_tokens,
        "next_turn_seconds": second_seconds,
        "new_owner_epoch": migration.new_epoch,
        "migration_state": store.get_migration(migration.migration_id).state.value,
        "state_db": str(db_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--output-length", type=int, default=4)
    parser.add_argument("--tool-result-length", type=int, default=16)
    parser.add_argument("--agent-id", default="smoke-agent")
    parser.add_argument("--seed-token", type=int, default=1100)
    parser.add_argument("--fill-token", type=int, default=100)
    parser.add_argument("--tool-token", type=int, default=200)
    parser.add_argument("--transfer-port", type=int, default=29600)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sync-transfer", action="store_true")
    parser.add_argument("--state-dir", default="results")
    asyncio.run(main(parser.parse_args()))
