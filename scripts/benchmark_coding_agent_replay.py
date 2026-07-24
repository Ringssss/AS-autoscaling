from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.benchmark_e2e import flush, timed_generate
from scripts.benchmark_real_tools_all import run_tool

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def coding_tools(python: str) -> list[tuple[str, list[str]]]:
    return [
        ("git-status", ["git", "status", "--short"]),
        (
            "state-store-tests",
            [python, "-m", "pytest", "-q", "tests/test_state_store.py"],
        ),
        (
            "migration-tests",
            [python, "-m", "pytest", "-q", "tests/test_migration_coordinator.py"],
        ),
        (
            "control-plane-tests",
            [python, "-m", "pytest", "-q", "tests"],
        ),
    ]


class CodingAgentReplay:
    def __init__(self, args):
        self.args = args
        self.engines = {
            "engine-a": SGLangAgentShiftClient("engine-a", args.source, timeout=300),
            "engine-b": SGLangAgentShiftClient(
                "engine-b", args.destination, timeout=300
            ),
        }
        self.state_path = (
            Path(args.output_dir) / f"coding-replay-state-{time.time_ns()}.db"
        )
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            self.engines,
            base_port=args.transfer_port,
            async_transfer=True,
            transfer_poll_interval=args.poll_interval,
        )
        self.serial = 0
        self.tool_predictions: dict[str, float] = {}

    async def clean(self) -> None:
        await asyncio.gather(*(flush(engine) for engine in self.engines.values()))

    async def calibrate_tools(self) -> dict[str, float]:
        for name, command in coding_tools(self.args.python):
            samples = []
            for _ in range(self.args.tool_calibration_repeats):
                elapsed, _, _, _ = await run_tool(command, self.args.tool_cwd)
                samples.append(elapsed)
            self.tool_predictions[name] = statistics.median(samples)
        return dict(self.tool_predictions)

    def prompt(self, length: int, salt: int) -> list[int]:
        return [28000 + salt] + [100] * (length - 1)

    @staticmethod
    def opposite(engine_id: str) -> str:
        return "engine-b" if engine_id == "engine-a" else "engine-a"

    async def prepare_sessions(self, scenario: str) -> list[dict[str, Any]]:
        await self.clean()
        sessions = []
        selected_count = math.ceil(self.args.agents * self.args.relocation_fraction)
        selected_indexes = list(range(0, self.args.agents, 2))[:selected_count]
        if len(selected_indexes) < selected_count:
            selected_indexes.extend(
                index
                for index in range(self.args.agents)
                if index not in selected_indexes
            )
            selected_indexes = selected_indexes[:selected_count]
        for index in range(self.args.agents):
            self.serial += 1
            agent_id = f"coding-{scenario}-{self.serial}-{index}"
            prompt = self.prompt(self.args.initial_prefix, self.serial)
            _, first = await timed_generate(
                self.engines["engine-a"],
                prompt,
                max_new_tokens=self.args.output_tokens,
                rid=f"{agent_id}-initial",
            )
            completed = tuple(prompt + first["output_ids"])
            if scenario in ("on-return", "agentshift"):
                self.store.register_agent(
                    AgentContinuation(
                        agent_id=agent_id,
                        committed_step=1,
                        owner_engine="engine-a",
                        owner_epoch=1,
                        token_ids=completed,
                        pending_tool_future=f"tool-{agent_id}-1",
                    )
                )
            if scenario in ("sticky", "on-return", "agentshift"):
                pin = await self.engines["engine-a"].pin_prefix(
                    agent_id, 1, completed
                )
                if int(pin["token_count"]) != len(completed):
                    raise RuntimeError(
                        f"source retained only {pin['token_count']} of "
                        f"{len(completed)} tokens for {agent_id}"
                    )
            sessions.append(
                {
                    "agent_id": agent_id,
                    "engine": "engine-a",
                    "epoch": 1,
                    "step": 1,
                    "completed": completed,
                    "selected": index in selected_indexes,
                }
            )
        if scenario in ("on-return", "agentshift"):
            await self.coordinator.initialize_transfer_pair("engine-a", "engine-b")
            await self.coordinator.initialize_transfer_pair("engine-b", "engine-a")
        return sessions

    async def run_session_turn(
        self,
        scenario: str,
        session: dict[str, Any],
        turn: int,
        tool_name: str,
        command: list[str],
    ) -> dict[str, Any]:
        owner_before = session["engine"]
        epoch_before = session["epoch"]
        destination = self.opposite(owner_before)
        should_move = turn == 1 and session["selected"] and scenario != "sticky"
        tool_started = time.perf_counter()
        tool_task = asyncio.create_task(
            run_tool(command, self.args.tool_cwd)
        )
        migration_task = None
        migration = None
        if scenario == "agentshift" and should_move:
            migration_task = asyncio.create_task(
                self.coordinator.migrate(session["agent_id"], destination)
            )

        tool_seconds, tool_completed, stdout, stderr = await tool_task
        if scenario == "on-return" and should_move:
            migration_task = asyncio.create_task(
                self.coordinator.migrate(session["agent_id"], destination)
            )
        if migration_task is not None:
            migration = await migration_task
            ready = time.perf_counter()
            next_engine = destination
            next_epoch = migration.new_epoch
        elif scenario == "reroute" and should_move:
            ready = tool_completed
            next_engine = destination
            next_epoch = session["epoch"] + 1
        else:
            ready = tool_completed
            next_engine = owner_before
            next_epoch = session["epoch"]

        exposed = max(0.0, ready - tool_completed)
        tool_result_tokens = max(
            16,
            min(
                self.args.max_tool_result_tokens,
                (len(stdout) + len(stderr) + 3) // 4,
            ),
        )
        target_prefix = min(
            self.args.max_prefix,
            self.args.initial_prefix + turn * self.args.turn_growth,
        )
        additions = max(
            tool_result_tokens,
            target_prefix - len(session["completed"]),
        )
        next_prompt = list(session["completed"]) + [200] * additions
        previous_prefix_tokens = len(session["completed"])
        next_seconds, result = await timed_generate(
            self.engines[next_engine],
            next_prompt,
            max_new_tokens=self.args.output_tokens,
            rid=f"{session['agent_id']}-turn-{turn + 1}",
        )
        finished = time.perf_counter()
        new_completed = tuple(next_prompt + result["output_ids"])

        if scenario in ("on-return", "agentshift"):
            current = self.store.get_agent(session["agent_id"])
            self.store.commit_step(
                replace(
                    current,
                    committed_step=session["step"] + 1,
                    token_ids=new_completed,
                    pending_tool_future=f"tool-{session['agent_id']}-{turn + 1}",
                ),
                expected_epoch=next_epoch,
            )

        session.update(
            engine=next_engine,
            epoch=next_epoch,
            step=session["step"] + 1,
            completed=new_completed,
        )
        cached = int(result["meta_info"]["cached_tokens"])
        reusable_history = max(0, previous_prefix_tokens - 1)
        return {
            "agent_id": session["agent_id"],
            "scenario": scenario,
            "turn": turn,
            "tool": tool_name,
            "tool_seconds": tool_seconds,
            "tool_output_bytes": len(stdout) + len(stderr),
            "owner_before": owner_before,
            "owner_epoch_before": epoch_before,
            "owner_after": next_engine,
            "owner_moved": next_engine != owner_before,
            "selected_for_relocation": session["selected"],
            "prefix_tokens": len(next_prompt),
            "cached_tokens": cached,
            "full_hit": cached >= reusable_history,
            "historical_reprefilled_tokens": max(
                0, reusable_history - cached
            ),
            "migration_seconds": migration.transfer_seconds if migration else 0.0,
            "migration_bytes": migration.bytes_transferred if migration else 0,
            "exposed_migration_seconds": exposed,
            "post_tool_seconds": exposed + next_seconds,
            "turn_wall_seconds": finished - tool_started,
            "_migration_id": migration.migration_id if migration else None,
        }

    async def run_one(self, scenario: str, repeat: int) -> dict[str, Any]:
        sessions = await self.prepare_sessions(scenario)
        tools = coding_tools(self.args.python)
        started = time.perf_counter()
        records = []
        for turn in range(1, self.args.turns + 1):
            jobs = []
            for index, session in enumerate(sessions):
                tool_name, command = tools[(index + turn - 1) % len(tools)]
                jobs.append((session, tool_name, command))
            if scenario == "agentshift":
                jobs.sort(
                    key=lambda item: (
                        self.tool_predictions[item[1]],
                        len(item[0]["completed"]),
                    )
                )
            tasks = []
            for session, tool_name, command in jobs:
                tasks.append(
                    asyncio.create_task(
                        self.run_session_turn(
                            scenario, session, turn, tool_name, command
                        )
                    )
                )
            turn_records = await asyncio.gather(*tasks)
            sessions_by_id = {
                session["agent_id"]: session for session in sessions
            }
            for record in turn_records:
                migration_id = record.pop("_migration_id")
                if migration_id is not None:
                    await self.coordinator.finalize_destination(
                        migration_id, keep_cached=True
                    )
                elif scenario in ("sticky", "on-return", "agentshift"):
                    await self.engines[record["owner_before"]].release_prefix(
                        record["agent_id"],
                        record["owner_epoch_before"],
                        evict_after_release=False,
                    )
                if turn < self.args.turns and scenario in (
                    "sticky",
                    "on-return",
                    "agentshift",
                ):
                    session = sessions_by_id[record["agent_id"]]
                    pin = await self.engines[session["engine"]].pin_prefix(
                        session["agent_id"],
                        session["epoch"],
                        session["completed"],
                    )
                    if int(pin["token_count"]) != len(session["completed"]):
                        raise RuntimeError(
                            f"owner retained only {pin['token_count']} of "
                            f"{len(session['completed'])} tokens for "
                            f"{session['agent_id']}"
                        )
            records.extend(turn_records)
        makespan = time.perf_counter() - started
        post_tool = [record["post_tool_seconds"] for record in records]
        output = {
            "scenario": scenario,
            "repeat": repeat,
            "agents": len(sessions),
            "turns": self.args.turns,
            "requests": len(records),
            "makespan_seconds": makespan,
            "post_tool_mean_seconds": statistics.mean(post_tool),
            "post_tool_p95_seconds": percentile(post_tool, 0.95),
            "owner_moved_fraction": statistics.mean(
                record["owner_moved"] for record in records
            ),
            "agents_relocated_fraction": statistics.mean(
                any(
                    record["agent_id"] == session["agent_id"]
                    and record["owner_moved"]
                    for record in records
                )
                for session in sessions
            ),
            "full_hit_fraction": statistics.mean(
                record["full_hit"] for record in records
            ),
            "historical_reprefilled_tokens": sum(
                record["historical_reprefilled_tokens"] for record in records
            ),
            "transfer_bytes": sum(record["migration_bytes"] for record in records),
            "turn_records": records,
        }
        if scenario in ("on-return", "agentshift"):
            for session in sessions:
                await self.engines[session["engine"]].release_prefix(
                    session["agent_id"], session["epoch"], allow_missing=True
                )
        await self.clean()
        return output


async def main(args) -> None:
    benchmark = CodingAgentReplay(args)
    tool_predictions = await benchmark.calibrate_tools()
    records = []
    for repeat in range(args.repeats):
        order = list(args.scenarios)
        random.Random(args.seed + repeat).shuffle(order)
        for scenario in order:
            record = await benchmark.run_one(scenario, repeat)
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "turn_records"}, sort_keys=True), flush=True)
    output = {
        "workload": "controlled multi-turn coding-agent replay with real subprocess tools",
        "config": vars(args),
        "tool_predictions_seconds": tool_predictions,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"coding-agent-replay-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["sticky", "reroute", "on-return", "agentshift"],
    )
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--relocation-fraction", type=float, default=0.5)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--initial-prefix", type=int, default=8192)
    parser.add_argument("--turn-growth", type=int, default=8192)
    parser.add_argument("--max-prefix", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--max-tool-result-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tool-calibration-repeats", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.002)
    parser.add_argument("--transfer-port", type=int, default=31100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--tool-cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
