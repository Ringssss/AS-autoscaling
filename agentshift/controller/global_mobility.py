from __future__ import annotations

from dataclasses import dataclass

from agentshift.controller.materialization import (
    BranchMaterializationRequest,
    BranchStatePlanner,
    MaterializationPlan,
)
from agentshift.state.descriptor import StateDescriptor


@dataclass(frozen=True, slots=True)
class EngineMobilityState:
    engine_id: str
    free_token_slots: int
    free_state_snapshots: int
    active_requests: int = 0
    warm_states: int = 0
    retiring: bool = False

    @property
    def load(self) -> int:
        return self.active_requests + self.warm_states


@dataclass(frozen=True, slots=True)
class GlobalMobilityCandidate:
    agent_id: str
    source_engine: str
    descriptor: StateDescriptor
    remaining_gap_seconds: float
    future_service_seconds: float = 0.0
    recompute_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class GlobalMobilityDecision:
    agent_id: str
    source_engine: str
    destination_engine: str
    plan: MaterializationPlan
    score: float
    drain_urgent: bool


class GlobalMobilityPolicy:
    """Event-driven placement for branch-valid heterogeneous state bundles."""

    def __init__(
        self,
        planner: BranchStatePlanner,
        *,
        load_threshold: int = 1,
        drain_bonus: float = 1000.0,
    ):
        if load_threshold < 0:
            raise ValueError("load_threshold must be non-negative")
        self.planner = planner
        self.load_threshold = load_threshold
        self.drain_bonus = drain_bonus

    def select(
        self,
        candidate: GlobalMobilityCandidate,
        engines: list[EngineMobilityState],
    ) -> GlobalMobilityDecision | None:
        by_id = {engine.engine_id: engine for engine in engines}
        if candidate.source_engine not in by_id:
            raise KeyError(candidate.source_engine)
        source = by_id[candidate.source_engine]
        drain_urgent = source.retiring

        ranked: list[GlobalMobilityDecision] = []
        for destination in engines:
            if destination.engine_id == source.engine_id or destination.retiring:
                continue
            if (
                not drain_urgent
                and source.load < destination.load + self.load_threshold
            ):
                continue

            request = BranchMaterializationRequest(
                descriptor=candidate.descriptor,
                destination_engine=destination.engine_id,
                remaining_gap_seconds=candidate.remaining_gap_seconds,
                destination_free_token_slots=destination.free_token_slots,
                destination_free_snapshot_slots=destination.free_state_snapshots,
                source_snapshot_available=True,
                replay_tokens_per_second=10_000.0,
            )
            plan = self.planner.plan(request, require_within_gap=True)
            if plan is None:
                continue

            load_relief = max(0, source.load - destination.load)
            score = (
                load_relief
                + candidate.future_service_seconds
                + (self.drain_bonus if drain_urgent else 0.0)
                - plan.selected.completion_seconds
            )
            if score <= 0:
                continue
            ranked.append(
                GlobalMobilityDecision(
                    agent_id=candidate.agent_id,
                    source_engine=source.engine_id,
                    destination_engine=destination.engine_id,
                    plan=plan,
                    score=score,
                    drain_urgent=drain_urgent,
                )
            )

        if not ranked:
            return None
        return max(
            ranked,
            key=lambda decision: (
                decision.score,
                -decision.plan.selected.completion_seconds,
                decision.destination_engine,
            ),
        )
