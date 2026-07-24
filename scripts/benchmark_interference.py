from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import (
    SGLangAgentShiftClient,
    generate,
    stream_generate,
)
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


class InterferenceBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = (
            Path(args.output_dir) / f"interference-state-{time.time_ns()}.db"
        )
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            async_transfer=True,
            transfer_poll_interval=args.poll_interval,
        )
        self.run_id = time.time_ns()
        self.serial = 0

    def next_id(self, prefix: str) -> str:
        self.serial += 1
        return f"{prefix}-{self.run_id}-{self.serial}"

    @staticmethod
    def prompt(length: int, salt: int) -> list[int]:
        return [12000 + salt] + [100] * (length - 1)

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def run_one(
        self, mode: str, prefix_length: int, concurrency: int, repeat: int
    ) -> dict[str, Any]:
        await self.clean()
        agent_id = self.next_id(f"interference-{mode}-{prefix_length}-{concurrency}")
        first = await generate(
            self.source,
            self.prompt(prefix_length, self.serial),
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(
            self.prompt(prefix_length, self.serial) + first["output_ids"]
        )
        self.store.register_agent(
            AgentContinuation(
                agent_id=agent_id,
                committed_step=1,
                owner_engine="engine-a",
                owner_epoch=1,
                token_ids=completed,
                pending_tool_future=f"tool-{agent_id}",
            )
        )
        await self.source.pin_prefix(agent_id, 1, completed)
        if mode != "no-migration":
            await self.coordinator._ensure_group("engine-a", "engine-b")

        first_token_events = [asyncio.Event() for _ in range(concurrency)]
        foreground_tasks = [
            asyncio.create_task(
                stream_generate(
                    self.source,
                    self.prompt(self.args.foreground_prompt_tokens, self.serial + i + 1),
                    max_new_tokens=self.args.foreground_output_tokens,
                    rid=f"{agent_id}-foreground-{i}",
                    first_token_event=first_token_events[i],
                )
            )
            for i in range(concurrency)
        ]
        if first_token_events:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in first_token_events)),
                timeout=60,
            )

        migration = None
        migration_started = time.perf_counter()
        if mode != "no-migration":
            self.coordinator.async_transfer = mode == "async"
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
            await asyncio.sleep(self.args.arrival_probe_delay_ms / 1000.0)
        else:
            migration_task = None
            await asyncio.sleep(self.args.arrival_probe_delay_ms / 1000.0)
        probe = await stream_generate(
            self.source,
            self.prompt(self.args.foreground_prompt_tokens, self.serial + concurrency + 1),
            max_new_tokens=1,
            rid=f"{agent_id}-arrival-probe",
        )
        if migration_task is not None:
            migration = await migration_task
        migration_finished = time.perf_counter()
        foreground = await asyncio.gather(*foreground_tasks)

        token_intervals = [
            interval
            for result in foreground
            for interval in result["token_intervals_seconds"]
        ]
        ttfts = [result["ttft_seconds"] for result in foreground]
        if foreground:
            foreground_wall = max(result["finished"] for result in foreground) - min(
                result["request_started"] for result in foreground
            )
            total_tokens = sum(
                result["response"]["meta_info"]["completion_tokens"]
                for result in foreground
            )
        else:
            foreground_wall = 0.0
            total_tokens = 0

        record = {
            "mode": mode,
            "prefix_length": prefix_length,
            "concurrency": concurrency,
            "repeat": repeat,
            "foreground_tokens": total_tokens,
            "foreground_throughput_tokens_per_second": (
                total_tokens / foreground_wall if foreground_wall else None
            ),
            "ttft_p50_seconds": percentile(ttfts, 0.50),
            "ttft_p95_seconds": percentile(ttfts, 0.95),
            "arrival_probe_ttft_seconds": probe["ttft_seconds"],
            "tpot_mean_seconds": statistics.mean(token_intervals)
            if token_intervals
            else None,
            "tpot_p50_seconds": percentile(token_intervals, 0.50),
            "tpot_p95_seconds": percentile(token_intervals, 0.95),
            "tpot_p99_seconds": percentile(token_intervals, 0.99),
            "max_token_gap_seconds": max(token_intervals) if token_intervals else None,
            "migration_wall_seconds": migration_finished - migration_started
            if migration
            else 0.0,
            "migration_worker_seconds": migration.worker_transfer_seconds
            if migration
            else 0.0,
            "migration_queue_seconds": migration.queue_seconds if migration else 0.0,
            "migration_bytes": migration.bytes_transferred if migration else 0,
        }

        if migration is None:
            await self.source.release_prefix(agent_id, 1)
        else:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(agent_id, migration.new_epoch)
        await self.clean()
        return record


async def main(args) -> None:
    benchmark = InterferenceBenchmark(args)
    records = []
    for prefix_length in args.prefix_lengths:
        for concurrency in args.foreground_concurrencies:
            for repeat in range(args.repeats):
                mode_order = list(args.modes)
                random.Random(
                    args.seed + prefix_length * 1009 + concurrency * 17 + repeat
                ).shuffle(mode_order)
                for mode in mode_order:
                    record = await benchmark.run_one(
                        mode, prefix_length, concurrency, repeat
                    )
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "baseline_implementation": "same-testbed mechanism ablation",
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"interference-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--modes", nargs="+", default=["no-migration", "sync", "async"]
    )
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument(
        "--foreground-concurrencies", type=int, nargs="+", default=[1, 4]
    )
    parser.add_argument("--foreground-prompt-tokens", type=int, default=128)
    parser.add_argument("--foreground-output-tokens", type=int, default=256)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--arrival-probe-delay-ms", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--transfer-port", type=int, default=29900)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
