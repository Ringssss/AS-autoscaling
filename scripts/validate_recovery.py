from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore, StaleLease


async def main(args) -> None:
    source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
    destination = SGLangAgentShiftClient("engine-b", args.destination, timeout=300)
    await asyncio.gather(flush(source), flush(destination))

    prompt = [18000] + [100] * (args.context_length - 1)
    _, first = await timed_generate(
        source, prompt, max_new_tokens=4, rid="recovery-turn-1"
    )
    completed = tuple(prompt + first["output_ids"])
    state_path = Path(args.output_dir) / f"recovery-state-{time.time_ns()}.db"
    store = SQLiteStateStore(state_path)
    store.register_agent(
        AgentContinuation(
            "recovery-agent", 1, "engine-a", 1, completed, "tool-recovery"
        )
    )
    coordinator = MigrationCoordinator(
        store,
        {"engine-a": source, "engine-b": destination},
        base_port=args.transfer_port,
    )
    migration = await coordinator.migrate("recovery-agent", "engine-b")

    recovered = await coordinator.recover_committed(migration.migration_id)
    stale_destination_rejected = False
    try:
        store.assert_lease("recovery-agent", "engine-b", migration.new_epoch)
    except StaleLease:
        stale_destination_rejected = True

    next_prompt = list(completed) + [200] * 32
    next_seconds, second = await timed_generate(
        source, next_prompt, max_new_tokens=1, rid="recovery-turn-2"
    )
    cached_tokens = int(second["meta_info"]["cached_tokens"])
    if cached_tokens < migration.token_count:
        raise RuntimeError("source shadow prefix was not preserved during recovery")
    if not stale_destination_rejected:
        raise RuntimeError("failed destination retained a valid execution lease")

    await source.release_prefix("recovery-agent", recovered.owner_epoch)
    await destination.release_prefix("recovery-agent", migration.new_epoch)
    result = {
        "migration_id": migration.migration_id,
        "migration_tokens": migration.token_count,
        "failed_destination_epoch": migration.new_epoch,
        "recovery_owner": recovered.owner_engine,
        "recovery_epoch": recovered.owner_epoch,
        "migration_state": store.get_migration(migration.migration_id).state.value,
        "stale_destination_rejected": stale_destination_rejected,
        "source_cached_tokens": cached_tokens,
        "recovered_next_turn_seconds": next_seconds,
        "state_db": str(state_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--transfer-port", type=int, default=29980)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
