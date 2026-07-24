from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MigrationState(str, Enum):
    PREPARING = "PREPARING"
    COPYING = "COPYING"
    DEST_READY = "DEST_READY"
    COMMITTED = "COMMITTED"
    SOURCE_RELEASED = "SOURCE_RELEASED"
    RECOVERED = "RECOVERED"
    ABORTED = "ABORTED"


class EffectStatus(str, Enum):
    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"


class StepStatus(str, Enum):
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class AgentContinuation:
    agent_id: str
    committed_step: int
    owner_engine: str
    owner_epoch: int
    token_ids: tuple[int, ...] = ()
    pending_tool_future: str | None = None
    workspace_ref: str | None = None
    stream_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    migration_id: str
    agent_id: str
    source_engine: str
    destination_engine: str
    source_epoch: int
    state: MigrationState
    token_count: int = 0
    bytes_transferred: int = 0
    transfer_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    agent_id: str
    step_id: int
    future_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectRecord:
    agent_id: str
    step_id: int
    operation_id: str
    owner_epoch: int
    status: EffectStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
