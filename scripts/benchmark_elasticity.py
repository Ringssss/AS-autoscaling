from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator, MigrationResult
from agentshift.engine.sglang import SGLangAgentShiftClient, stream_generate
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


class ElasticityBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = Path(args.output_dir) / f"elasticity-state-{time.time_ns()}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
        )
        self.serial = 0

    def prompt(self, index: int) -> list[int]:
        self.serial += 1
        return [30000 + self.serial + index] + [100] * (self.args.prefix_length - 1)

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def prepare(
        self, mode: str, scenario: str, repeat: int
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        await self.clean()
        agent_ids = [
            f"elastic-{mode}-{scenario}-{repeat}-{self.serial}-{index}"
            for index in range(self.args.agent_count)
        ]
        prompts = [self.prompt(index) for index in range(self.args.agent_count)]
        first = await asyncio.gather(
            *[
                timed_generate(
                    self.source,
                    prompt,
                    max_new_tokens=self.args.first_output_tokens,
                    rid=f"{agent_id}-turn-1",
                )
                for agent_id, prompt in zip(agent_ids, prompts)
            ]
        )
        completed = [
            tuple(prompt + result["output_ids"])
            for prompt, (_, result) in zip(prompts, first)
        ]
        for agent_id, tokens in zip(agent_ids, completed):
            self.store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=1,
                    owner_engine="engine-a",
                    owner_epoch=1,
                    token_ids=tokens,
                    pending_tool_future=f"tool-{agent_id}",
                )
            )
        return agent_ids, completed

    def semantic_handoff(self, agent_id: str) -> float:
        continuation = self.store.get_agent(agent_id)
        migration_id = uuid.uuid4().hex
        self.store.start_migration(
            MigrationRecord(
                migration_id=migration_id,
                agent_id=agent_id,
                source_engine="engine-a",
                destination_engine="engine-b",
                source_epoch=continuation.owner_epoch,
                state=MigrationState.PREPARING,
            )
        )
        self.store.transition_migration(
            migration_id, MigrationState.PREPARING, MigrationState.COPYING
        )
        self.store.transition_migration(
            migration_id, MigrationState.COPYING, MigrationState.DEST_READY
        )
        self.store.commit_migration(migration_id)
        self.store.transition_migration(
            migration_id, MigrationState.COMMITTED, MigrationState.SOURCE_RELEASED
        )
        return time.perf_counter()

    async def migrate_selected(
        self, selected: list[str], start: float
    ) -> tuple[list[MigrationResult], list[float]]:
        migrations = []
        ownership_events = []
        for agent_id in selected:
            migrations.append(await self.coordinator.migrate(agent_id, "engine-b"))
            ownership_events.append(time.perf_counter() - start)
        return migrations, ownership_events

    async def run_case(self, mode: str, scenario: str, repeat: int) -> dict[str, Any]:
        agent_ids, completed = await self.prepare(mode, scenario, repeat)
        selected_count = self.args.agent_count // 2 if mode == "scale-out" else self.args.agent_count
        selected_ids = agent_ids[:selected_count]
        start = time.perf_counter()
        gap_deadline = start + self.args.tool_gap_ms / 1000.0
        migrations: list[MigrationResult] = []
        ownership_events: list[float] = []

        if scenario == "agentshift":
            migrations, ownership_events = await self.migrate_selected(selected_ids, start)
            await asyncio.sleep(max(0.0, gap_deadline - time.perf_counter()))
        else:
            await asyncio.sleep(max(0.0, gap_deadline - time.perf_counter()))
            if scenario == "on-return":
                migrations, ownership_events = await self.migrate_selected(selected_ids, start)
            elif scenario == "semantic-reroute":
                ownership_events = [
                    self.semantic_handoff(agent_id) - start for agent_id in selected_ids
                ]

        tool_return = gap_deadline
        if scenario in ("on-return", "agentshift"):
            tool_return = gap_deadline
        clients = []
        for index in range(self.args.agent_count):
            moved = index < selected_count and scenario != "sticky"
            clients.append(self.destination if moved else self.source)

        events = [asyncio.Event() for _ in range(self.args.agent_count)]
        next_tasks = [
            asyncio.create_task(
                stream_generate(
                    client,
                    list(tokens) + [200] * self.args.tool_result_tokens,
                    max_new_tokens=self.args.output_tokens,
                    rid=f"{agent_id}-turn-2",
                    first_token_event=event,
                )
            )
            for client, tokens, agent_id, event in zip(
                clients, completed, agent_ids, events
            )
        ]

        release_times: list[float] = []
        if migrations:
            by_agent = {migration.agent_id: migration for migration in migrations}

            async def acknowledge(index: int, agent_id: str) -> None:
                await events[index].wait()
                await self.coordinator.acknowledge_destination(
                    by_agent[agent_id].migration_id
                )
                release_times.append(time.perf_counter() - start)

            ack_tasks = [
                asyncio.create_task(acknowledge(index, agent_id))
                for index, agent_id in enumerate(agent_ids[:selected_count])
            ]
        else:
            ack_tasks = []

        if scenario == "semantic-reroute" and mode == "scale-in":
            await asyncio.gather(*(events[index].wait() for index in range(selected_count)))
            await flush(self.source)
            release_times.append(time.perf_counter() - start)

        results = await asyncio.gather(*next_tasks)
        if ack_tasks:
            await asyncio.gather(*ack_tasks)
        finished = time.perf_counter()

        cache_hits = [
            int(result["response"]["meta_info"]["cached_tokens"]) for result in results
        ]
        first_tokens = [result["token_timestamps"][0] for result in results]
        historical_tokens = sum(len(tokens) for tokens in completed)
        historical_hits = sum(
            min(len(tokens), hit) for tokens, hit in zip(completed, cache_hits)
        )
        source_owners = sum(
            self.store.get_agent(agent_id).owner_engine == "engine-a"
            for agent_id in agent_ids
        )
        destination_indexes = [
            index
            for index in range(self.args.agent_count)
            if clients[index] is self.destination
        ]
        destination_full_hits = [
            cache_hits[index] >= len(completed[index]) for index in destination_indexes
        ]
        drain_complete = mode == "scale-in" and source_owners == 0

        record = {
            "mode": mode,
            "scenario": scenario,
            "repeat": repeat,
            "agent_count": self.args.agent_count,
            "selected_count": selected_count,
            "prefix_length": self.args.prefix_length,
            "tool_gap_ms": self.args.tool_gap_ms,
            "post_tool_makespan_seconds": finished - tool_return,
            "first_destination_token_after_tool_seconds": (
                min(first_tokens[index] for index in destination_indexes) - tool_return
                if destination_indexes
                else None
            ),
            "time_to_first_state_ready_seconds": (
                ownership_events[0] if migrations else None
            ),
            "time_to_first_authority_ready_seconds": (
                ownership_events[0] if ownership_events else None
            ),
            "time_to_target_redistribution_seconds": (
                ownership_events[-1] if ownership_events else None
            ),
            "source_hbm_relief_seconds": max(release_times) if release_times else None,
            "owner_relocated_fraction": (self.args.agent_count - source_owners)
            / self.args.agent_count,
            "remaining_source_owners": source_owners,
            "drain_complete": drain_complete,
            "destination_full_hit_rate": (
                statistics.fmean(destination_full_hits) if destination_full_hits else None
            ),
            "historical_reprefilled_tokens": historical_tokens - historical_hits,
            "transfer_bytes": sum(item.bytes_transferred for item in migrations),
            "transfer_wall_seconds": sum(item.transfer_seconds for item in migrations),
            "ownership_events_seconds": ownership_events,
        }

        for migration in migrations:
            await self.destination.release_prefix(
                migration.agent_id, migration.new_epoch, allow_missing=True
            )
        await self.clean()
        return record


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for mode in sorted({record["mode"] for record in records}):
        for scenario in sorted(
            record["scenario"] for record in records if record["mode"] == mode
        ):
            rows = [
                record
                for record in records
                if record["mode"] == mode and record["scenario"] == scenario
            ]
            summary[f"{mode}:{scenario}"] = {
                "post_tool_makespan_mean_seconds": statistics.fmean(
                    row["post_tool_makespan_seconds"] for row in rows
                ),
                "owner_relocated_fraction_mean": statistics.fmean(
                    row["owner_relocated_fraction"] for row in rows
                ),
                "destination_full_hit_rate_mean": (
                    statistics.fmean(
                        row["destination_full_hit_rate"]
                        for row in rows
                        if row["destination_full_hit_rate"] is not None
                    )
                    if any(row["destination_full_hit_rate"] is not None for row in rows)
                    else None
                ),
                "historical_reprefilled_tokens_mean": statistics.fmean(
                    row["historical_reprefilled_tokens"] for row in rows
                ),
                "drain_success_rate": statistics.fmean(
                    row["drain_complete"] for row in rows
                ),
            }
    return summary


async def main(args: argparse.Namespace) -> None:
    benchmark = ElasticityBenchmark(args)
    # Transfer groups are persistent runtime infrastructure, not per-agent work.
    # Establish them before timing either migration policy.
    await benchmark.coordinator.initialize_transfer_pair("engine-a", "engine-b")
    records = []
    for repeat in range(args.repeats):
        for mode in args.modes:
            for scenario in args.scenarios:
                record = await benchmark.run_case(mode, scenario, repeat)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
        "summary": summarize(records),
    }
    output_path = Path(args.output_dir) / f"elasticity-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--modes", nargs="+", default=["scale-out", "scale-in"])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["sticky", "semantic-reroute", "on-return", "agentshift"],
    )
    parser.add_argument("--agent-count", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=16384)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--tool-gap-ms", type=float, default=500.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--transfer-port", type=int, default=30700)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
