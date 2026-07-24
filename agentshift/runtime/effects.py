from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentshift.state.schema import EffectRecord, EffectStatus
from agentshift.state.store import SQLiteStateStore, StateConflict


class ManagedEffectProxy:
    """At-most-once submission for tools routed through AgentShift."""

    def __init__(self, store: SQLiteStateStore):
        self.store = store

    def execute(
        self,
        *,
        agent_id: str,
        step_id: int,
        operation_id: str,
        owner_epoch: int,
        payload: dict[str, Any],
        submit: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        record = self.store.prepare_effect(
            EffectRecord(
                agent_id=agent_id,
                step_id=step_id,
                operation_id=operation_id,
                owner_epoch=owner_epoch,
                status=EffectStatus.PREPARED,
                payload=payload,
            )
        )
        if record.status == EffectStatus.COMPLETED:
            return record.result or {}
        if record.status != EffectStatus.PREPARED:
            raise StateConflict(f"effect cannot be replayed from {record.status}")
        self.store.transition_effect(
            agent_id,
            step_id,
            operation_id,
            EffectStatus.PREPARED,
            EffectStatus.SUBMITTED,
            owner_epoch,
        )
        try:
            result = submit(payload)
        except Exception:
            self.store.transition_effect(
                agent_id,
                step_id,
                operation_id,
                EffectStatus.SUBMITTED,
                EffectStatus.UNKNOWN,
                owner_epoch,
            )
            raise
        self.store.transition_effect(
            agent_id,
            step_id,
            operation_id,
            EffectStatus.SUBMITTED,
            EffectStatus.COMPLETED,
            owner_epoch,
            result,
        )
        return result
