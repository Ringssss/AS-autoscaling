from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


def order_admissible_first(
    items: list[T],
    *,
    deadline_seconds: Callable[[T], float],
    duration_seconds: Callable[[T], float],
    size: Callable[[T], int],
) -> list[T]:
    """Order a serial handoff channel to maximize predicted on-time jobs."""
    admitted: list[T] = []
    deferred: list[T] = []
    elapsed = 0.0
    for item in sorted(items, key=deadline_seconds):
        admitted.append(item)
        elapsed += duration_seconds(item)
        if elapsed > deadline_seconds(item):
            longest = max(
                admitted,
                key=lambda candidate: (
                    duration_seconds(candidate),
                    size(candidate),
                ),
            )
            admitted.remove(longest)
            elapsed -= duration_seconds(longest)
            deferred.append(longest)
    return admitted + sorted(
        deferred,
        key=lambda item: (
            deadline_seconds(item),
            duration_seconds(item),
        ),
    )


@dataclass(frozen=True, slots=True)
class EngineLoad:
    engine_id: str
    queue_depth: int
    free_kv_tokens: int
    active_requests: int = 0
    kv_pressure: float = 0.0


@dataclass(frozen=True, slots=True)
class MobilityCandidate:
    agent_id: str
    source_engine: str
    prefix_tokens: int
    kv_bytes: int
    remaining_gap_seconds: float
    recompute_seconds: float
    future_service_seconds: float
    unknown_effect: bool = False


@dataclass(frozen=True, slots=True)
class MobilityScore:
    agent_id: str
    source_engine: str
    destination_engine: str
    score: float
    estimated_copy_seconds: float
    exposed_seconds: float
    queue_relief_seconds: float
    recompute_saved_seconds: float
    hbm_relief_gib_seconds: float
    interference_seconds: float


class PlacementPolicy:
    def choose_destination(
        self,
        *,
        source_engine: str,
        prefix_tokens: int,
        loads: list[EngineLoad],
    ) -> str | None:
        candidates = [
            load
            for load in loads
            if load.engine_id != source_engine and load.free_kv_tokens >= prefix_tokens
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: (x.queue_depth, x.active_requests)).engine_id

    def should_move(
        self,
        *,
        source: EngineLoad,
        destination: EngineLoad,
        migration_seconds: float,
        tool_gap_seconds: float,
    ) -> bool:
        queue_relief = source.queue_depth - destination.queue_depth
        exposed_seconds = max(0.0, migration_seconds - tool_gap_seconds)
        return queue_relief > 0 and exposed_seconds < max(0.05, 0.01 * queue_relief)


class CostBenefitPlacementPolicy:
    """Explainable, bounded scheduler for blocked-agent handoffs."""

    def __init__(
        self,
        *,
        bandwidth_bytes_per_second: float,
        fixed_copy_seconds: float = 0.0,
        acceptable_exposure_seconds: float = 0.01,
        recompute_weight: float = 1.0,
        hbm_weight: float = 0.1,
        interference_weight: float = 1.0,
        exposure_weight: float = 2.0,
        max_concurrent_migrations: int = 2,
    ):
        if bandwidth_bytes_per_second <= 0:
            raise ValueError("migration bandwidth must be positive")
        if max_concurrent_migrations <= 0:
            raise ValueError("max concurrent migrations must be positive")
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self.fixed_copy_seconds = fixed_copy_seconds
        self.acceptable_exposure_seconds = acceptable_exposure_seconds
        self.recompute_weight = recompute_weight
        self.hbm_weight = hbm_weight
        self.interference_weight = interference_weight
        self.exposure_weight = exposure_weight
        self.max_concurrent_migrations = max_concurrent_migrations

    def estimate_copy_seconds(self, candidate: MobilityCandidate) -> float:
        return self.fixed_copy_seconds + (
            candidate.kv_bytes / self.bandwidth_bytes_per_second
        )

    def score(
        self,
        candidate: MobilityCandidate,
        destination: EngineLoad,
        loads: dict[str, EngineLoad],
    ) -> MobilityScore | None:
        if candidate.unknown_effect or destination.engine_id == candidate.source_engine:
            return None
        if destination.free_kv_tokens < candidate.prefix_tokens:
            return None
        try:
            source = loads[candidate.source_engine]
        except KeyError as exc:
            raise KeyError(f"missing source load: {candidate.source_engine}") from exc
        copy_seconds = self.estimate_copy_seconds(candidate)
        exposed_seconds = max(0.0, copy_seconds - candidate.remaining_gap_seconds)
        if exposed_seconds > self.acceptable_exposure_seconds:
            return None
        queue_delta = max(0, source.queue_depth - destination.queue_depth)
        queue_relief_seconds = queue_delta * candidate.future_service_seconds
        pressure_delta = max(0.0, source.kv_pressure - destination.kv_pressure)
        hbm_relief = (
            candidate.kv_bytes
            / (1024**3)
            * candidate.remaining_gap_seconds
            * pressure_delta
        )
        if queue_relief_seconds == 0 and hbm_relief == 0:
            return None
        interference = copy_seconds * (
            1 + source.active_requests + destination.active_requests
        )
        score = (
            queue_relief_seconds
            + self.recompute_weight * candidate.recompute_seconds
            + self.hbm_weight * hbm_relief
            - self.interference_weight * interference
            - self.exposure_weight * exposed_seconds
        )
        return MobilityScore(
            agent_id=candidate.agent_id,
            source_engine=candidate.source_engine,
            destination_engine=destination.engine_id,
            score=score,
            estimated_copy_seconds=copy_seconds,
            exposed_seconds=exposed_seconds,
            queue_relief_seconds=queue_relief_seconds,
            recompute_saved_seconds=candidate.recompute_seconds,
            hbm_relief_gib_seconds=hbm_relief,
            interference_seconds=interference,
        )

    def select(
        self,
        candidates: list[MobilityCandidate],
        loads: list[EngineLoad],
    ) -> list[MobilityScore]:
        load_map = {load.engine_id: load for load in loads}
        ranked = []
        for candidate in candidates:
            for destination in loads:
                result = self.score(candidate, destination, load_map)
                if result is not None and result.score > 0:
                    ranked.append(result)
        ranked.sort(key=lambda item: item.score, reverse=True)
        selected = []
        selected_agents = set()
        reserved_tokens = {load.engine_id: 0 for load in loads}
        candidates_by_id = {candidate.agent_id: candidate for candidate in candidates}
        for result in ranked:
            if len(selected) >= self.max_concurrent_migrations:
                break
            if result.agent_id in selected_agents:
                continue
            candidate = candidates_by_id[result.agent_id]
            destination = load_map[result.destination_engine]
            if (
                reserved_tokens[result.destination_engine] + candidate.prefix_tokens
                > destination.free_kv_tokens
            ):
                continue
            selected.append(result)
            selected_agents.add(result.agent_id)
            reserved_tokens[result.destination_engine] += candidate.prefix_tokens
        return selected
