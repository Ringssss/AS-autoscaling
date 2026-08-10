#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentshift.controller.autoscaling import (
    EngineLifecycle,
    EngineObservation,
    WarmPoolAutoscaler,
)
from agentshift.controller.migration import MigrationCoordinator, MigrationResult
from agentshift.controller.placement import (
    CostBenefitPlacementPolicy,
    EngineLoad,
    MobilityCandidate,
)
from agentshift.engine.sglang import SGLangAgentShiftClient, stream_generate
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


@dataclass(slots=True)
class LiveAgent:
    agent_id: str
    event: dict[str, Any]
    state: str = "ARRIVED"
    owner_engine: str | None = None
    completed_prefix: tuple[int, ...] = ()
    tool_ready_at: float = 0.0
    migration_task: asyncio.Task | None = None
    migration: MigrationResult | None = None
    proactive_pin_engine: str | None = None
    proactive_pin_epoch: int | None = None
    proactive_pin_completed_at: float | None = None
    proactive_pin_active: bool = False
    result: dict[str, Any] = field(default_factory=dict)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
    }


class TraceAutoscalingBenchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest = json.loads(args.manifest.read_text())
        self.events = sorted(
            self.manifest["events"], key=lambda item: item["arrival_seconds"]
        )
        self.engine_ids = [f"engine-{index}" for index in range(len(args.engine_urls))]
        self.engines = {
            engine_id: SGLangAgentShiftClient(engine_id, url, timeout=args.request_timeout)
            for engine_id, url in zip(self.engine_ids, args.engine_urls)
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.time_ns()
        self.run_dir = args.output_dir / f"{args.run_name}-{timestamp}"
        self.run_dir.mkdir(parents=True)
        self.state_path = self.run_dir / "state.db"
        self.telemetry_path = self.run_dir / "telemetry.jsonl"
        self.requests_path = self.run_dir / "requests.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            self.engines,
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=True,
            transfer_timeout=args.transfer_timeout,
        )
        self.autoscaler = WarmPoolAutoscaler(
            self.engine_ids,
            min_active=args.min_active,
            scale_out_waiting=args.scale_out_waiting,
            scale_out_requests_per_engine=args.scale_out_requests_per_engine,
            scale_in_requests_per_engine=args.scale_in_requests_per_engine,
            scale_out_windows=args.scale_out_windows,
            scale_in_windows=args.scale_in_windows,
            cooldown_seconds=args.cooldown_seconds,
        )
        self.placement = CostBenefitPlacementPolicy(
            bandwidth_bytes_per_second=args.migration_bandwidth_gib_s * 1024**3,
            acceptable_exposure_seconds=args.acceptable_exposure_seconds,
            max_concurrent_migrations=max(1, len(self.engine_ids) // 2),
        )
        self.live_agents: dict[str, LiveAgent] = {}
        self.agent_tasks: set[asyncio.Task] = set()
        self.client_inflight = {engine_id: 0 for engine_id in self.engine_ids}
        self.latest_loads: dict[str, dict[str, Any]] = {
            engine_id: {} for engine_id in self.engine_ids
        }
        self.scaling_events: list[dict[str, Any]] = []
        self.migration_events: list[dict[str, Any]] = []
        self.request_records: list[dict[str, Any]] = []
        self.transfer_group_init_seconds = 0.0
        self.preinitialized_transfer_pairs = 0
        self._start = 0.0
        self._stop_scheduler = asyncio.Event()
        self._telemetry = self.telemetry_path.open("w")
        self._requests = self.requests_path.open("w")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def log_telemetry(self, kind: str, **fields: Any) -> None:
        row = {"kind": kind, "elapsed_seconds": self.elapsed, **fields}
        self._telemetry.write(json.dumps(row, sort_keys=True) + "\n")
        self._telemetry.flush()

    def log_request(self, row: dict[str, Any]) -> None:
        self.request_records.append(row)
        self._requests.write(json.dumps(row, sort_keys=True) + "\n")
        self._requests.flush()

    async def get_load(self, engine_id: str) -> dict[str, Any]:
        url = f"{self.engines[engine_id].base_url.rstrip('/')}/v1/loads?include=core"

        def fetch() -> dict[str, Any]:
            with urllib.request.urlopen(url, timeout=self.args.load_timeout) as response:
                return json.loads(response.read())

        try:
            payload = await asyncio.to_thread(fetch)
            load = payload["aggregate"]
            max_tokens = sum(
                int(item["max_total_num_tokens"]) for item in payload["loads"]
            )
            return {
                "running_requests": int(load["total_running_reqs"]),
                "waiting_requests": int(load["total_waiting_reqs"]),
                "used_tokens": int(load["total_used_tokens"]),
                "max_tokens": max_tokens,
                "token_usage": float(load["avg_token_usage"]),
            }
        except Exception as exc:
            self.log_telemetry("load_error", engine_id=engine_id, error=str(exc))
            previous = self.latest_loads.get(engine_id, {})
            return {
                "running_requests": self.client_inflight[engine_id],
                "waiting_requests": 0,
                "used_tokens": int(previous.get("used_tokens", 0)),
                "max_tokens": int(previous.get("max_tokens", self.args.max_total_tokens)),
                "token_usage": float(previous.get("token_usage", 0.0)),
            }

    async def flush_engines(self) -> None:
        async def flush(engine_id: str) -> None:
            url = (
                f"{self.engines[engine_id].base_url.rstrip('/')}"
                "/flush_cache?timeout=60"
            )

            def request() -> None:
                with urllib.request.urlopen(url, timeout=70) as response:
                    response.read()

            await asyncio.to_thread(request)

        await asyncio.gather(*(flush(engine_id) for engine_id in self.engine_ids))

    async def initialize_transfer_groups(self) -> None:
        if self.args.policy not in ("agentshift", "on-return"):
            return
        started = time.perf_counter()
        for source in self.engine_ids:
            for destination in self.engine_ids:
                if source == destination:
                    continue
                await self.coordinator.initialize_transfer_pair(source, destination)
                self.preinitialized_transfer_pairs += 1
        self.transfer_group_init_seconds = time.perf_counter() - started

    def owner_counts(self) -> dict[str, int]:
        counts = {engine_id: 0 for engine_id in self.engine_ids}
        for agent in self.live_agents.values():
            if (
                agent.owner_engine is not None
                and agent.state not in ("FINISHED", "FAILED")
            ):
                counts[agent.owner_engine] += 1
        return counts

    def due_counts(self) -> dict[str, int]:
        counts = {engine_id: 0 for engine_id in self.engine_ids}
        horizon = time.monotonic() + self.args.return_horizon_seconds
        for agent in self.live_agents.values():
            if (
                agent.owner_engine is not None
                and agent.state == "BLOCKED"
                and agent.tool_ready_at <= horizon
            ):
                counts[agent.owner_engine] += 1
        return counts

    def choose_new_owner(self) -> str:
        active = self.autoscaler.active_engines()
        if not active:
            raise RuntimeError("autoscaler has no active engine")
        owners = self.owner_counts()
        return min(
            active,
            key=lambda engine_id: (
                self.client_inflight[engine_id],
                int(self.latest_loads[engine_id].get("waiting_requests", 0)),
                owners[engine_id],
                engine_id,
            ),
        )

    def choose_relocation_destination(self, source_engine: str) -> str | None:
        candidates = [
            engine_id
            for engine_id in self.autoscaler.active_engines()
            if engine_id != source_engine
        ]
        if not candidates:
            return None
        owners = self.owner_counts()
        destination = min(
            candidates,
            key=lambda engine_id: (
                self.client_inflight[engine_id],
                owners[engine_id],
                engine_id,
            ),
        )
        source_draining = (
            self.autoscaler.states[source_engine] == EngineLifecycle.DRAINING
        )
        if not source_draining and owners[source_engine] <= owners[destination] + 1:
            return None
        return destination

    async def generate(
        self,
        engine_id: str,
        token_ids: list[int],
        *,
        max_new_tokens: int,
        rid: str,
        first_token_event: asyncio.Event | None = None,
        agentshift_pin_agent_id: str | None = None,
        agentshift_pin_owner_epoch: int | None = None,
    ) -> dict[str, Any]:
        self.client_inflight[engine_id] += 1
        try:
            return await stream_generate(
                self.engines[engine_id],
                token_ids,
                max_new_tokens=max_new_tokens,
                rid=rid,
                first_token_event=first_token_event,
                ignore_eos=True,
                agentshift_pin_agent_id=agentshift_pin_agent_id,
                agentshift_pin_owner_epoch=agentshift_pin_owner_epoch,
            )
        finally:
            self.client_inflight[engine_id] -= 1

    def semantic_handoff(self, agent_id: str, destination_engine: str) -> None:
        continuation = self.store.get_agent(agent_id)
        migration_id = f"reroute-{uuid.uuid4().hex}"
        self.store.start_migration(
            MigrationRecord(
                migration_id,
                agent_id,
                continuation.owner_engine,
                destination_engine,
                continuation.owner_epoch,
                MigrationState.PREPARING,
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

    async def migrate_agent(
        self, agent: LiveAgent, destination_engine: str, trigger: str
    ) -> MigrationResult | None:
        source_engine = agent.owner_engine
        started = time.perf_counter()
        started_monotonic = time.monotonic()
        try:
            result = await self.coordinator.migrate(agent.agent_id, destination_engine)
            agent.owner_engine = result.destination_engine
            agent.migration = result
            finished = time.monotonic()
            record = {
                "agent_id": agent.agent_id,
                "source_engine": result.source_engine,
                "destination_engine": result.destination_engine,
                "trigger": trigger,
                "token_count": result.token_count,
                "bytes_transferred": result.bytes_transferred,
                "transfer_seconds": result.transfer_seconds,
                "worker_transfer_seconds": result.worker_transfer_seconds,
                "queue_seconds": result.queue_seconds,
                "wall_seconds": time.perf_counter() - started,
                "proactively_pinned": agent.proactive_pin_active,
                "pin_to_migration_seconds": (
                    started_monotonic - agent.proactive_pin_completed_at
                    if agent.proactive_pin_completed_at is not None
                    else None
                ),
                "completed_before_tool_return": finished <= agent.tool_ready_at,
                "elapsed_seconds": self.elapsed,
            }
            self.migration_events.append(record)
            self.log_telemetry("migration", **record)
            return result
        except Exception as exc:
            self.log_telemetry(
                "migration_error",
                agent_id=agent.agent_id,
                source_engine=source_engine,
                destination_engine=destination_engine,
                trigger=trigger,
                error=str(exc),
            )
            return None

    async def release_proactive_source_pin(
        self, agent: LiveAgent, *, reason: str
    ) -> None:
        if (
            not agent.proactive_pin_active
            or agent.proactive_pin_engine is None
            or agent.proactive_pin_epoch is None
        ):
            return
        residency_seconds = (
            time.monotonic() - agent.proactive_pin_completed_at
            if agent.proactive_pin_completed_at is not None
            else None
        )
        try:
            release = await self.engines[agent.proactive_pin_engine].release_prefix(
                agent.agent_id,
                agent.proactive_pin_epoch,
                evict_after_release=False,
                allow_missing=True,
            )
            agent.proactive_pin_active = False
            released_tokens = int(release["token_count"])
            self.log_telemetry(
                "proactive_pin_release",
                agent_id=agent.agent_id,
                engine_id=agent.proactive_pin_engine,
                owner_epoch=agent.proactive_pin_epoch,
                reason=reason,
                residency_seconds=residency_seconds,
                token_count=released_tokens,
                success=released_tokens > 0,
            )
        except Exception as exc:
            self.log_telemetry(
                "proactive_pin_release_error",
                agent_id=agent.agent_id,
                engine_id=agent.proactive_pin_engine,
                owner_epoch=agent.proactive_pin_epoch,
                reason=reason,
                residency_seconds=residency_seconds,
                error=str(exc),
            )

    async def finish_migration_after_first_token(
        self, agent: LiveAgent, first_token_event: asyncio.Event
    ) -> None:
        await first_token_event.wait()
        if agent.migration is not None:
            await self.coordinator.finalize_destination(
                agent.migration.migration_id, keep_cached=True
            )
            residency_seconds = (
                time.monotonic() - agent.proactive_pin_completed_at
                if agent.proactive_pin_completed_at is not None
                else None
            )
            agent.proactive_pin_active = False
            self.log_telemetry(
                "proactive_pin_release",
                agent_id=agent.agent_id,
                engine_id=agent.migration.source_engine,
                owner_epoch=agent.migration.old_epoch,
                reason="migration_first_token",
                residency_seconds=residency_seconds,
            )

    async def run_agent(self, event: dict[str, Any]) -> None:
        agent_id = f"trace-agent-{event['event_id']}"
        agent = LiveAgent(agent_id, event)
        self.live_agents[agent_id] = agent
        owner = self.choose_new_owner()
        agent.owner_engine = owner
        prompt = [30000 + (event["event_id"] % 10000)] + [100] * (
            event["context_tokens"] - 1
        )
        arrival = self.elapsed
        try:
            agent.state = "FIRST_TURN"
            first = await self.generate(
                owner,
                prompt,
                max_new_tokens=event["first_output_tokens"],
                rid=f"{agent_id}-turn-1",
                agentshift_pin_agent_id=(
                    agent_id if self.args.policy == "agentshift" else None
                ),
                agentshift_pin_owner_epoch=(
                    1 if self.args.policy == "agentshift" else None
                ),
            )
            tool_started_at = time.monotonic()
            completed = tuple(prompt + first["response"]["output_ids"])
            agent.completed_prefix = completed
            self.store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=1,
                    owner_engine=owner,
                    owner_epoch=1,
                    token_ids=completed,
                    pending_tool_future=f"tool-{agent_id}",
                    metadata={"trace_source": event["source"]},
                )
            )
            agent.tool_ready_at = tool_started_at + event["tool_gap_seconds"]
            if self.args.policy == "agentshift":
                agent.proactive_pin_engine = owner
                agent.proactive_pin_epoch = 1
                agent.proactive_pin_completed_at = tool_started_at
                agent.proactive_pin_active = True
                self.log_telemetry(
                    "proactive_pin",
                    agent_id=agent_id,
                    engine_id=owner,
                    owner_epoch=1,
                    mode="scheduler_completion_hook",
                    pin_seconds=0.0,
                    success=True,
                )
            agent.state = "BLOCKED"
            await asyncio.sleep(max(0.0, agent.tool_ready_at - time.monotonic()))

            if agent.migration_task is not None:
                await agent.migration_task
            current = self.store.get_agent(agent_id)
            destination = self.choose_relocation_destination(current.owner_engine)
            if self.args.policy == "on-return" and destination is not None:
                await self.migrate_agent(agent, destination, "on-return")
            elif self.args.policy == "reroute" and destination is not None:
                self.semantic_handoff(agent_id, destination)
                agent.owner_engine = destination
            elif (
                self.args.policy == "agentshift"
                and self.autoscaler.states[current.owner_engine]
                == EngineLifecycle.DRAINING
                and destination is not None
                and agent.migration is None
            ):
                await self.migrate_agent(agent, destination, "late-drain")

            continuation = self.store.get_agent(agent_id)
            agent.owner_engine = continuation.owner_engine
            rid = f"{agent_id}-turn-2-e{continuation.owner_epoch}"
            self.store.claim_step(
                agent_id=agent_id,
                step_id=2,
                owner_engine=continuation.owner_engine,
                owner_epoch=continuation.owner_epoch,
                rid=rid,
            )
            agent.state = "SECOND_TURN"
            first_token_event = asyncio.Event()
            acknowledge_task = None
            if agent.migration is not None:
                acknowledge_task = asyncio.create_task(
                    self.finish_migration_after_first_token(agent, first_token_event)
                )
            second_prompt = list(completed) + [200] * event["tool_result_tokens"]
            try:
                second = await self.generate(
                    continuation.owner_engine,
                    second_prompt,
                    max_new_tokens=event["second_output_tokens"],
                    rid=rid,
                    first_token_event=first_token_event,
                )
                if acknowledge_task is not None:
                    await acknowledge_task
                elif agent.proactive_pin_active:
                    await self.release_proactive_source_pin(
                        agent, reason="unmigrated_second_turn"
                    )
            except Exception:
                if acknowledge_task is not None and not acknowledge_task.done():
                    acknowledge_task.cancel()
                raise
            cached_tokens = int(second["response"]["meta_info"]["cached_tokens"])
            # The final sampled token has not itself gone through a forward
            # pass, so SGLang cannot reuse KV for that token on resumption.
            reusable_prefix_tokens = max(0, len(completed) - 1)
            historical_hits = min(reusable_prefix_tokens, cached_tokens)
            finished_continuation = AgentContinuation(
                agent_id=agent_id,
                committed_step=2,
                owner_engine=continuation.owner_engine,
                owner_epoch=continuation.owner_epoch,
                token_ids=(),
                pending_tool_future=None,
                metadata={"finished": True, "trace_source": event["source"]},
            )
            self.store.commit_claimed_step(
                finished_continuation,
                expected_epoch=continuation.owner_epoch,
                rid=rid,
            )
            record = {
                "agent_id": agent_id,
                "event_id": event["event_id"],
                "source": event["source"],
                "phase": event["phase"],
                "arrival_seconds": arrival,
                "completion_seconds": self.elapsed,
                "initial_engine": owner,
                "final_engine": continuation.owner_engine,
                "migrated": agent.migration is not None,
                "context_tokens": event["context_tokens"],
                "completed_prefix_tokens": len(completed),
                "reusable_prefix_tokens": reusable_prefix_tokens,
                "cached_tokens": cached_tokens,
                "historical_reprefilled_tokens": (
                    reusable_prefix_tokens - historical_hits
                ),
                "full_prefix_hit": historical_hits == reusable_prefix_tokens,
                "first_turn_ttft_seconds": first["ttft_seconds"],
                "first_turn_e2e_seconds": first["e2e_seconds"],
                "post_tool_ttft_seconds": second["ttft_seconds"],
                "second_turn_e2e_seconds": second["e2e_seconds"],
                "tool_gap_seconds": event["tool_gap_seconds"],
            }
            agent.result = record
            agent.state = "FINISHED"
            agent.completed_prefix = ()
            self.log_request(record)
        except Exception as exc:
            await self.release_proactive_source_pin(agent, reason="request_failure")
            agent.state = "FAILED"
            record = {
                "agent_id": agent_id,
                "event_id": event["event_id"],
                "source": event["source"],
                "phase": event["phase"],
                "arrival_seconds": arrival,
                "completion_seconds": self.elapsed,
                "initial_engine": owner,
                "final_engine": agent.owner_engine,
                "error": str(exc),
            }
            self.log_request(record)

    async def schedule_migrations(self) -> None:
        if self.args.policy != "agentshift":
            return
        owners = self.owner_counts()
        due = self.due_counts()
        loads: list[EngineLoad] = []
        for engine_id in self.engine_ids:
            raw = self.latest_loads[engine_id]
            max_tokens = int(raw.get("max_tokens", self.args.max_total_tokens))
            used_tokens = int(raw.get("used_tokens", 0))
            loads.append(
                EngineLoad(
                    engine_id=engine_id,
                    queue_depth=(
                        int(raw.get("waiting_requests", 0))
                        + due[engine_id]
                        + owners[engine_id]
                        + (
                            100
                            if self.autoscaler.states[engine_id]
                            == EngineLifecycle.DRAINING
                            else 0
                        )
                    ),
                    active_requests=int(raw.get("running_requests", 0)),
                    free_kv_tokens=max(0, max_tokens - used_tokens),
                    kv_pressure=(used_tokens / max_tokens if max_tokens else 0.0),
                )
            )
        now = time.monotonic()
        candidates = []
        by_id = {}
        for agent in self.live_agents.values():
            if (
                agent.state != "BLOCKED"
                or agent.owner_engine is None
                or agent.migration_task is not None
            ):
                continue
            remaining = agent.tool_ready_at - now
            if remaining <= 0:
                continue
            candidate = MobilityCandidate(
                agent_id=agent.agent_id,
                source_engine=agent.owner_engine,
                prefix_tokens=len(agent.completed_prefix),
                kv_bytes=len(agent.completed_prefix) * self.args.kv_bytes_per_token,
                remaining_gap_seconds=remaining,
                recompute_seconds=len(agent.completed_prefix)
                / self.args.reprefill_tokens_per_second,
                future_service_seconds=self.args.future_service_seconds,
            )
            candidates.append(candidate)
            by_id[agent.agent_id] = agent
        selected = self.placement.select(
            candidates,
            loads,
            destination_engine_ids=set(self.autoscaler.active_engines()),
        )
        for score in selected:
            agent = by_id[score.agent_id]
            trigger = (
                "scale-in"
                if self.autoscaler.states[score.source_engine]
                == EngineLifecycle.DRAINING
                else "balance"
            )
            agent.migration_task = asyncio.create_task(
                self.migrate_agent(agent, score.destination_engine, trigger)
            )

    async def scheduler_loop(self) -> None:
        while not self._stop_scheduler.is_set():
            loads = await asyncio.gather(
                *(self.get_load(engine_id) for engine_id in self.engine_ids)
            )
            self.latest_loads = dict(zip(self.engine_ids, loads))
            owners = self.owner_counts()
            observations = [
                EngineObservation(
                    engine_id,
                    load["running_requests"],
                    load["waiting_requests"],
                    owners[engine_id],
                    load["used_tokens"],
                    load["max_tokens"],
                )
                for engine_id, load in zip(self.engine_ids, loads)
            ]
            actions = self.autoscaler.tick(self.elapsed, observations)
            for action in actions:
                row = {
                    "action": action.action,
                    "engine_id": action.engine_id,
                    "reason": action.reason,
                    "elapsed_seconds": self.elapsed,
                }
                self.scaling_events.append(row)
                self.log_telemetry("scaling", **row)
            await self.schedule_migrations()
            self.log_telemetry(
                "sample",
                phase=min(2, int(self.elapsed // self.args.phase_seconds)),
                states={key: value.value for key, value in self.autoscaler.states.items()},
                owners=owners,
                client_inflight=dict(self.client_inflight),
                loads=self.latest_loads,
                live_agents=sum(
                    agent.state not in ("FINISHED", "FAILED")
                    for agent in self.live_agents.values()
                ),
            )
            try:
                await asyncio.wait_for(
                    self._stop_scheduler.wait(), timeout=self.args.scheduler_interval
                )
            except asyncio.TimeoutError:
                pass

    async def run(self) -> dict[str, Any]:
        await self.initialize_transfer_groups()
        await self.flush_engines()
        self._start = time.monotonic()
        scheduler_task = asyncio.create_task(self.scheduler_loop())
        for event in self.events:
            deadline = self._start + event["arrival_seconds"]
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            task = asyncio.create_task(self.run_agent(event))
            self.agent_tasks.add(task)
            task.add_done_callback(self.agent_tasks.discard)
        if self.agent_tasks:
            await asyncio.wait_for(
                asyncio.gather(*list(self.agent_tasks)), timeout=self.args.drain_timeout
            )
        self._stop_scheduler.set()
        await scheduler_task
        self._telemetry.close()
        self._requests.close()
        successes = [row for row in self.request_records if "error" not in row]
        failures = [row for row in self.request_records if "error" in row]
        post_tool = [row["post_tool_ttft_seconds"] for row in successes]
        first_turn = [row["first_turn_ttft_seconds"] for row in successes]
        migrated = [row for row in successes if row["migrated"]]
        summary = {
            "run_name": self.args.run_name,
            "policy": self.args.policy,
            "manifest": str(self.args.manifest.resolve()),
            "duration_seconds": self.elapsed,
            "engine_urls": self.args.engine_urls,
            "tp_size": self.args.tp_size,
            "preinitialized_transfer_pairs": self.preinitialized_transfer_pairs,
            "transfer_group_init_seconds": self.transfer_group_init_seconds,
            "events": len(self.events),
            "successful_agents": len(successes),
            "failed_agents": len(failures),
            "first_turn_ttft_seconds": quantiles(first_turn),
            "post_tool_ttft_seconds": quantiles(post_tool),
            "migrations": len(self.migration_events),
            "migration_bytes": sum(
                row["bytes_transferred"] for row in self.migration_events
            ),
            "migration_transfer_seconds": quantiles(
                [row["transfer_seconds"] for row in self.migration_events]
            ),
            "migration_hidden_fraction": (
                statistics.fmean(
                    row["completed_before_tool_return"]
                    for row in self.migration_events
                )
                if self.migration_events
                else None
            ),
            "migrated_full_prefix_hit_fraction": (
                statistics.fmean(row["full_prefix_hit"] for row in migrated)
                if migrated
                else None
            ),
            "historical_reprefilled_tokens": sum(
                row["historical_reprefilled_tokens"] for row in successes
            ),
            "scaling_events": self.scaling_events,
            "run_dir": str(self.run_dir),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


async def main(args: argparse.Namespace) -> None:
    benchmark = TraceAutoscalingBenchmark(args)
    await benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine-urls", nargs="+", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--policy", choices=("sticky", "reroute", "on-return", "agentshift"), default="agentshift"
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--min-active", type=int, default=1)
    parser.add_argument("--phase-seconds", type=float, default=600.0)
    parser.add_argument("--scheduler-interval", type=float, default=1.0)
    parser.add_argument("--scale-out-waiting", type=int, default=1)
    parser.add_argument("--scale-out-requests-per-engine", type=float, default=2.0)
    parser.add_argument("--scale-in-requests-per-engine", type=float, default=0.25)
    parser.add_argument("--scale-out-windows", type=int, default=2)
    parser.add_argument("--scale-in-windows", type=int, default=20)
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    parser.add_argument("--return-horizon-seconds", type=float, default=2.0)
    parser.add_argument("--migration-bandwidth-gib-s", type=float, default=50.0)
    parser.add_argument("--acceptable-exposure-seconds", type=float, default=0.1)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--reprefill-tokens-per-second", type=float, default=20000.0)
    parser.add_argument("--future-service-seconds", type=float, default=0.1)
    parser.add_argument("--max-total-tokens", type=int, default=270000)
    parser.add_argument("--transfer-port", type=int, default=33000)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--load-timeout", type=float, default=2.0)
    parser.add_argument("--transfer-timeout", type=float, default=120.0)
    parser.add_argument("--drain-timeout", type=float, default=600.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/autoscaling/runs")
    )
    asyncio.run(main(parser.parse_args()))
