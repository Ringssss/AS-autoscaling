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
from agentshift.state.store import SQLiteStateStore


async def main(args) -> None:
    source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
    destination = SGLangAgentShiftClient("engine-b", args.destination, timeout=300)
    await asyncio.gather(flush(source), flush(destination))

    prompt = [args.seed_token] + [100] * (args.context_length - 1)
    _, first = await timed_generate(
        source, prompt, max_new_tokens=4, rid="correctness-turn-1"
    )
    completed = tuple(prompt + first["output_ids"])
    second_prompt = list(completed) + [200] * args.tool_result_tokens

    _, sticky = await timed_generate(
        source,
        second_prompt,
        max_new_tokens=args.output_tokens,
        rid="correctness-sticky",
    )
    _, re_prefill = await timed_generate(
        destination,
        second_prompt,
        max_new_tokens=args.output_tokens,
        rid="correctness-reprefill",
    )
    await flush(destination)

    state_path = Path(args.output_dir) / f"correctness-state-{time.time_ns()}.db"
    store = SQLiteStateStore(state_path)
    store.register_agent(
        AgentContinuation(
            "correctness-agent", 1, "engine-a", 1, completed, "tool-1"
        )
    )
    coordinator = MigrationCoordinator(
        store,
        {"engine-a": source, "engine-b": destination},
        base_port=args.transfer_port,
        tp_size=args.tp_size,
        async_transfer=not args.sync_transfer,
    )
    migration = await coordinator.migrate("correctness-agent", "engine-b")
    _, migrated = await timed_generate(
        destination,
        second_prompt,
        max_new_tokens=args.output_tokens,
        rid="correctness-migrated",
    )
    await coordinator.acknowledge_destination(migration.migration_id)
    await destination.release_prefix("correctness-agent", migration.new_epoch)

    outputs_equal = (
        sticky["output_ids"] == re_prefill["output_ids"] == migrated["output_ids"]
    )
    result = {
        "outputs_equal": outputs_equal,
        "output_tokens_compared": len(migrated["output_ids"]),
        "sticky_cached_tokens": sticky["meta_info"]["cached_tokens"],
        "reprefill_cached_tokens": re_prefill["meta_info"]["cached_tokens"],
        "migrated_cached_tokens": migrated["meta_info"]["cached_tokens"],
        "migration_tokens": migration.token_count,
        "migration_seconds": migration.transfer_seconds,
        "worker_transfer_seconds": migration.worker_transfer_seconds,
        "transfer_queue_seconds": migration.queue_seconds,
        "state_db": str(state_path),
    }
    if not outputs_equal:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--tool-result-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed-token", type=int, default=15000)
    parser.add_argument("--transfer-port", type=int, default=29800)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sync-transfer", action="store_true")
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
