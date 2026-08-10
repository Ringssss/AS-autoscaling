from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EngineLifecycle(str, Enum):
    STANDBY = "STANDBY"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"


@dataclass(frozen=True, slots=True)
class EngineObservation:
    engine_id: str
    running_requests: int
    waiting_requests: int
    owner_count: int
    used_tokens: int = 0
    max_tokens: int = 0

    @property
    def request_count(self) -> int:
        return self.running_requests + self.waiting_requests


@dataclass(frozen=True, slots=True)
class ScalingAction:
    action: str
    engine_id: str
    reason: str


class WarmPoolAutoscaler:
    """Hysteretic autoscaler for model-ready serving engines."""

    def __init__(
        self,
        engine_ids: list[str],
        *,
        min_active: int,
        scale_out_waiting: int = 1,
        scale_out_requests_per_engine: float = 2.0,
        scale_in_requests_per_engine: float = 0.25,
        scale_out_windows: int = 2,
        scale_in_windows: int = 20,
        cooldown_seconds: float = 15.0,
    ) -> None:
        if not engine_ids:
            raise ValueError("at least one engine is required")
        if not 1 <= min_active <= len(engine_ids):
            raise ValueError("min_active must be within the engine pool")
        if scale_out_windows <= 0 or scale_in_windows <= 0:
            raise ValueError("window counts must be positive")
        self.engine_ids = tuple(engine_ids)
        self.min_active = min_active
        self.scale_out_waiting = scale_out_waiting
        self.scale_out_requests_per_engine = scale_out_requests_per_engine
        self.scale_in_requests_per_engine = scale_in_requests_per_engine
        self.scale_out_windows = scale_out_windows
        self.scale_in_windows = scale_in_windows
        self.cooldown_seconds = cooldown_seconds
        self.states = {
            engine_id: (
                EngineLifecycle.ACTIVE
                if index < min_active
                else EngineLifecycle.STANDBY
            )
            for index, engine_id in enumerate(self.engine_ids)
        }
        self._scale_out_streak = 0
        self._scale_in_streak = 0
        self._last_action_seconds = float("-inf")

    def active_engines(self) -> list[str]:
        return [
            engine_id
            for engine_id in self.engine_ids
            if self.states[engine_id] == EngineLifecycle.ACTIVE
        ]

    def draining_engines(self) -> list[str]:
        return [
            engine_id
            for engine_id in self.engine_ids
            if self.states[engine_id] == EngineLifecycle.DRAINING
        ]

    def tick(
        self, now_seconds: float, observations: list[EngineObservation]
    ) -> list[ScalingAction]:
        by_id = {item.engine_id: item for item in observations}
        missing = set(self.engine_ids) - set(by_id)
        if missing:
            raise KeyError(f"missing engine observations: {sorted(missing)}")

        actions: list[ScalingAction] = []
        for engine_id in self.draining_engines():
            observation = by_id[engine_id]
            if observation.owner_count == 0 and observation.request_count == 0:
                self.states[engine_id] = EngineLifecycle.STANDBY
                actions.append(
                    ScalingAction("SCALE_IN_COMPLETE", engine_id, "drain empty")
                )

        active = self.active_engines()
        total_waiting = sum(by_id[engine_id].waiting_requests for engine_id in active)
        total_requests = sum(by_id[engine_id].request_count for engine_id in active)
        requests_per_engine = total_requests / max(1, len(active))

        wants_scale_out = (
            total_waiting >= self.scale_out_waiting
            or requests_per_engine >= self.scale_out_requests_per_engine
        )
        wants_scale_in = (
            total_waiting == 0
            and requests_per_engine <= self.scale_in_requests_per_engine
        )
        self._scale_out_streak = self._scale_out_streak + 1 if wants_scale_out else 0
        self._scale_in_streak = self._scale_in_streak + 1 if wants_scale_in else 0

        cooldown_ready = (
            now_seconds - self._last_action_seconds >= self.cooldown_seconds
        )
        standby = [
            engine_id
            for engine_id in self.engine_ids
            if self.states[engine_id] == EngineLifecycle.STANDBY
        ]
        if (
            cooldown_ready
            and standby
            and self._scale_out_streak >= self.scale_out_windows
        ):
            engine_id = standby[0]
            self.states[engine_id] = EngineLifecycle.ACTIVE
            self._last_action_seconds = now_seconds
            self._scale_out_streak = 0
            self._scale_in_streak = 0
            actions.append(
                ScalingAction(
                    "SCALE_OUT",
                    engine_id,
                    f"waiting={total_waiting}, requests_per_engine={requests_per_engine:.2f}",
                )
            )
            return actions

        can_drain = (
            len(active) > self.min_active and not self.draining_engines()
        )
        if (
            cooldown_ready
            and can_drain
            and self._scale_in_streak >= self.scale_in_windows
        ):
            engine_id = min(
                active,
                key=lambda item: (
                    by_id[item].owner_count,
                    by_id[item].request_count,
                    -self.engine_ids.index(item),
                ),
            )
            self.states[engine_id] = EngineLifecycle.DRAINING
            self._last_action_seconds = now_seconds
            self._scale_out_streak = 0
            self._scale_in_streak = 0
            actions.append(
                ScalingAction(
                    "SCALE_IN_START",
                    engine_id,
                    f"requests_per_engine={requests_per_engine:.2f}",
                )
            )
        return actions
