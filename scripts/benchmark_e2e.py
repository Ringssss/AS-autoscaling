from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient, generate, stream_generate
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


async def flush(client: SGLangAgentShiftClient) -> None:
    def request() -> None:
        with urllib.request.urlopen(
            f"{client.base_url}/flush_cache?timeout=30", timeout=40
        ) as response:
            response.read()

    await asyncio.to_thread(request)


async def timed_generate(
    client: SGLangAgentShiftClient,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    rid: str,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    result = await generate(
        client, token_ids, max_new_tokens=max_new_tokens, rid=rid
    )
    return time.perf_counter() - started, result


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        timestamp = time.time_ns()
        self.state_path = Path(args.output_dir) / f"benchmark-state-{timestamp}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=not args.sync_transfer,
        )
        self.serial = 0

    def next_agent_id(self, prefix: str) -> str:
        self.serial += 1
        return f"{prefix}-{self.serial}"

    def prompt(self, context_length: int, salt: int) -> list[int]:
        return [10000 + salt] + [100] * (context_length - 1)

    async def clean_caches(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def run_single(
        self, scenario: str, context_length: int, repeat: int
    ) -> dict[str, Any]:
        await self.clean_caches()
        agent_id = self.next_agent_id(f"single-{scenario}-{context_length}-{repeat}")
        prompt = self.prompt(context_length, self.serial)
        first_seconds, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        migration = None
        if scenario == "agentshift":
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
            migration = await self.coordinator.migrate(agent_id, "engine-b")

        engine = self.source if scenario == "sticky" else self.destination
        second_prompt = list(completed) + [200] * self.args.tool_result_tokens
        next_seconds, second = await timed_generate(
            engine,
            second_prompt,
            max_new_tokens=1,
            rid=f"{agent_id}-turn-2",
        )

        migration_seconds = migration.transfer_seconds if migration else 0.0
        record = {
            "kind": "single",
            "scenario": scenario,
            "context_length": context_length,
            "repeat": repeat,
            "first_turn_seconds": first_seconds,
            "next_turn_seconds": next_seconds,
            "cached_tokens": int(second["meta_info"]["cached_tokens"]),
            "migration_seconds": migration_seconds,
            "migration_tokens": migration.token_count if migration else 0,
            "bytes_transferred": migration.bytes_transferred if migration else 0,
            "migration_gib_per_second": (
                migration.bytes_transferred / migration_seconds / (1024**3)
                if migration_seconds
                else 0.0
            ),
            "tool_gap_seconds": self.args.tool_gap_seconds,
            "exposed_migration_seconds": max(
                0.0, migration_seconds - self.args.tool_gap_seconds
            ),
        }
        if migration:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(
                agent_id=agent_id, owner_epoch=migration.new_epoch
            )
        await self.clean_caches()
        return record

    async def _prepare_burst(
        self, scenario: str, context_length: int, concurrency: int, repeat: int
    ) -> tuple[list[str], list[tuple[int, ...]], list]:
        await self.clean_caches()
        agent_ids = [
            self.next_agent_id(f"burst-{scenario}-{context_length}-{repeat}")
            for _ in range(concurrency)
        ]
        first_results = await asyncio.gather(
            *[
                timed_generate(
                    self.source,
                    self.prompt(context_length, self.serial + index + 1),
                    max_new_tokens=self.args.first_output_tokens,
                    rid=f"{agent_id}-turn-1",
                )
                for index, agent_id in enumerate(agent_ids)
            ]
        )
        completed = []
        for index, (_, result) in enumerate(first_results):
            prompt = self.prompt(context_length, self.serial + index + 1)
            completed.append(tuple(prompt + result["output_ids"]))

        migrations = []
        if scenario == "agentshift":
            for agent_id, token_ids in zip(
                agent_ids[: concurrency // 2], completed[: concurrency // 2]
            ):
                self.store.register_agent(
                    AgentContinuation(
                        agent_id=agent_id,
                        committed_step=1,
                        owner_engine="engine-a",
                        owner_epoch=1,
                        token_ids=token_ids,
                        pending_tool_future=f"tool-{agent_id}",
                    )
                )
                migrations.append(
                    await self.coordinator.migrate(agent_id, "engine-b")
                )
        return agent_ids, completed, migrations

    async def run_burst(
        self, scenario: str, context_length: int, concurrency: int, repeat: int
    ) -> dict[str, Any]:
        agent_ids, completed, migrations = await self._prepare_burst(
            scenario, context_length, concurrency, repeat
        )
        background_events = [
            asyncio.Event() for _ in range(self.args.background_concurrency)
        ]
        background_tasks = [
            asyncio.create_task(
                stream_generate(
                    self.source,
                    self.prompt(
                        self.args.background_prompt_tokens,
                        self.serial + 1000 + index,
                    ),
                    max_new_tokens=self.args.background_output_tokens,
                    rid=f"background-{scenario}-{context_length}-{repeat}-{index}",
                    first_token_event=background_events[index],
                )
            )
            for index in range(self.args.background_concurrency)
        ]
        if background_events:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in background_events)),
                timeout=60,
            )
        clients = []
        for index in range(concurrency):
            if scenario == "sticky":
                clients.append(self.source)
            else:
                clients.append(
                    self.destination if index < concurrency // 2 else self.source
                )

        burst_started = time.perf_counter()
        results = await asyncio.gather(
            *[
                timed_generate(
                    client,
                    list(tokens) + [200] * self.args.tool_result_tokens,
                    max_new_tokens=self.args.burst_output_tokens,
                    rid=f"{agent_id}-turn-2",
                )
                for client, tokens, agent_id in zip(clients, completed, agent_ids)
            ]
        )
        makespan = time.perf_counter() - burst_started
        await asyncio.gather(*background_tasks)
        latencies = [elapsed for elapsed, _ in results]
        cache_hits = [int(result["meta_info"]["cached_tokens"]) for _, result in results]
        historical_tokens = sum(len(tokens) for tokens in completed)
        historical_cache_hits = sum(
            min(len(tokens), cache_hit)
            for tokens, cache_hit in zip(completed, cache_hits)
        )
        migration_total = sum(item.transfer_seconds for item in migrations)
        exposed_migration = max(
            0.0, migration_total - self.args.tool_gap_seconds
        )
        record = {
            "kind": "burst",
            "scenario": scenario,
            "context_length": context_length,
            "concurrency": concurrency,
            "repeat": repeat,
            "output_tokens": self.args.burst_output_tokens,
            "background_concurrency": self.args.background_concurrency,
            "makespan_seconds": makespan,
            "mean_request_seconds": statistics.mean(latencies),
            "p95_request_seconds": sorted(latencies)[
                max(0, int(0.95 * len(latencies)) - 1)
            ],
            "mean_cached_tokens": statistics.mean(cache_hits),
            "historical_tokens": historical_tokens,
            "historical_cache_hits": historical_cache_hits,
            "historical_reprefilled_tokens": historical_tokens
            - historical_cache_hits,
            "migration_total_seconds": migration_total,
            "exposed_migration_seconds": exposed_migration,
            "post_tool_makespan_seconds": makespan + exposed_migration,
            "on_return_post_tool_makespan_seconds": makespan + migration_total,
        }
        for migration in migrations:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(
                migration.agent_id, migration.new_epoch
            )
        await self.clean_caches()
        return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"single": {}, "burst": {}}
    single_keys = sorted(
        {(row["context_length"], row["scenario"]) for row in records if row["kind"] == "single"}
    )
    for context, scenario in single_keys:
        rows = [
            row
            for row in records
            if row["kind"] == "single"
            and row["context_length"] == context
            and row["scenario"] == scenario
        ]
        summary["single"][f"{context}:{scenario}"] = {
            "next_turn_mean_seconds": statistics.mean(
                row["next_turn_seconds"] for row in rows
            ),
            "cached_tokens_mean": statistics.mean(row["cached_tokens"] for row in rows),
            "migration_mean_seconds": statistics.mean(
                row["migration_seconds"] for row in rows
            ),
        }
    for context in sorted(
        {row["context_length"] for row in records if row["kind"] == "single"}
    ):
        values = {
            scenario: summary["single"][f"{context}:{scenario}"][
                "next_turn_mean_seconds"
            ]
            for scenario in ("sticky", "reroute", "agentshift")
        }
        summary["single"][f"{context}:speedup"] = {
            "vs_reroute": values["reroute"] / values["agentshift"],
            "vs_sticky": values["sticky"] / values["agentshift"],
        }
    for scenario in ("sticky", "reroute", "agentshift"):
        rows = [
            row
            for row in records
            if row["kind"] == "burst" and row["scenario"] == scenario
        ]
        if rows:
            summary["burst"][scenario] = {
                "makespan_mean_seconds": statistics.mean(
                    row["makespan_seconds"] for row in rows
                ),
                "p95_mean_seconds": statistics.mean(
                    row["p95_request_seconds"] for row in rows
                ),
            }
    if "agentshift" in summary["burst"]:
        ours = summary["burst"]["agentshift"]["makespan_mean_seconds"]
        summary["burst"]["speedup"] = {
            scenario: summary["burst"][scenario]["makespan_mean_seconds"] / ours
            for scenario in ("sticky", "reroute")
        }
    return summary


async def main(args) -> None:
    benchmark = Benchmark(args)
    records: list[dict[str, Any]] = []
    for context_length in args.context_lengths:
        for repeat in range(args.repeats):
            for scenario in ("sticky", "reroute", "agentshift"):
                record = await benchmark.run_single(scenario, context_length, repeat)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    for repeat in range(args.burst_repeats):
        for scenario in ("sticky", "reroute", "agentshift"):
            record = await benchmark.run_burst(
                scenario, args.burst_context_length, args.burst_concurrency, repeat
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    output = {
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
        "summary": summarize(records),
    }
    output_path = Path(args.output_dir) / f"e2e-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--tool-gap-seconds", type=float, default=0.25)
    parser.add_argument("--burst-context-length", type=int, default=4096)
    parser.add_argument("--burst-concurrency", type=int, default=8)
    parser.add_argument("--burst-output-tokens", type=int, default=64)
    parser.add_argument("--burst-repeats", type=int, default=3)
    parser.add_argument("--background-concurrency", type=int, default=0)
    parser.add_argument("--background-prompt-tokens", type=int, default=128)
    parser.add_argument("--background-output-tokens", type=int, default=256)
    parser.add_argument("--transfer-port", type=int, default=29700)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sync-transfer", action="store_true")
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
