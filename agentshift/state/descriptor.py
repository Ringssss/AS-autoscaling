from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StateComponentType(str, Enum):
    TOKEN_KV = "token_kv"
    RECURRENT_SNAPSHOT = "recurrent_snapshot"
    CONTROL_METADATA = "control_metadata"


@dataclass(frozen=True, slots=True)
class StateComponent:
    component_type: StateComponentType
    size_bytes: int
    token_slots: int = 0
    snapshot_slots: int = 0
    validity_scope: str = "exact-prefix-branch"
    layout: str = ""
    capability: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.size_bytes < 0 or self.token_slots < 0 or self.snapshot_slots < 0:
            raise ValueError("state component footprints must be non-negative")
        if self.component_type == StateComponentType.TOKEN_KV and self.snapshot_slots:
            raise ValueError("token KV cannot consume recurrent snapshot slots")
        if (
            self.component_type == StateComponentType.RECURRENT_SNAPSHOT
            and self.token_slots
        ):
            raise ValueError("recurrent snapshots cannot consume token slots")


@dataclass(frozen=True, slots=True)
class StateDescriptor:
    descriptor_id: str
    model_identity: str
    branch_id: str
    prefix_tokens: int
    components: tuple[StateComponent, ...]
    token_fingerprint: str = ""
    causal_epoch: int = 0

    def __post_init__(self):
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")

    @classmethod
    def homogeneous_mha(
        cls,
        *,
        descriptor_id: str,
        model_identity: str,
        branch_id: str,
        prefix_tokens: int,
        bytes_per_token: int,
    ) -> "StateDescriptor":
        return cls(
            descriptor_id=descriptor_id,
            model_identity=model_identity,
            branch_id=branch_id,
            prefix_tokens=prefix_tokens,
            components=(
                StateComponent(
                    component_type=StateComponentType.TOKEN_KV,
                    size_bytes=prefix_tokens * bytes_per_token,
                    token_slots=prefix_tokens,
                    layout="layer-first-kv",
                ),
            ),
        )

    @classmethod
    def hybrid_qwen35(
        cls,
        *,
        descriptor_id: str,
        model_identity: str,
        branch_id: str,
        prefix_tokens: int,
        kv_bytes_per_token: int = 20_480,
        recurrent_snapshot_bytes: int = 64_389_120,
        snapshot_count: int = 1,
    ) -> "StateDescriptor":
        return cls(
            descriptor_id=descriptor_id,
            model_identity=model_identity,
            branch_id=branch_id,
            prefix_tokens=prefix_tokens,
            components=(
                StateComponent(
                    component_type=StateComponentType.TOKEN_KV,
                    size_bytes=prefix_tokens * kv_bytes_per_token,
                    token_slots=prefix_tokens,
                    layout="layer-first-full-attention-kv",
                ),
                StateComponent(
                    component_type=StateComponentType.RECURRENT_SNAPSHOT,
                    size_bytes=recurrent_snapshot_bytes * snapshot_count,
                    snapshot_slots=snapshot_count,
                    layout="layer-first-mamba-conv-temporal",
                ),
            ),
        )

    @property
    def total_bytes(self) -> int:
        return sum(component.size_bytes for component in self.components)

    @property
    def token_slots(self) -> int:
        return sum(component.token_slots for component in self.components)

    @property
    def snapshot_slots(self) -> int:
        return sum(component.snapshot_slots for component in self.components)

    def component(self, component_type: StateComponentType) -> StateComponent | None:
        return next(
            (
                component
                for component in self.components
                if component.component_type == component_type
            ),
            None,
        )

    def valid_for(self, *, branch_id: str, causal_epoch: int) -> bool:
        return self.branch_id == branch_id and self.causal_epoch <= causal_epoch
