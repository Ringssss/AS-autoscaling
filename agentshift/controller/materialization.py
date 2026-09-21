from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agentshift.state.descriptor import StateDescriptor


class MaterializationMethod(str, Enum):
    COPY_FULL_STATE = "copy_full_state"
    REPLAY_FROM_ANCESTOR = "replay_from_ancestor"
    RECONSTRUCT_FROM_TOKENS = "reconstruct_from_tokens"


@dataclass(frozen=True, slots=True)
class MaterializationOption:
    method: MaterializationMethod
    completion_seconds: float
    transfer_bytes: int
    replay_tokens: int
    destination_token_slots: int
    destination_snapshot_slots: int
    branch_valid: bool = True


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    selected: MaterializationOption
    options: tuple[MaterializationOption, ...]
    remaining_gap_seconds: float
    exposed_seconds: float
    fits_gap: bool


@dataclass(frozen=True, slots=True)
class BranchMaterializationRequest:
    descriptor: StateDescriptor
    destination_engine: str
    remaining_gap_seconds: float
    destination_free_token_slots: int
    destination_free_snapshot_slots: int
    source_snapshot_available: bool = True
    ancestor_snapshot_available: bool = False
    ancestor_state_bytes: int = 0
    ancestor_prefix_tokens: int = 0
    replay_tokens_per_second: float = 10_000.0
    reconstructible: bool = True
    causal_epoch: int = 0


class BranchStatePlanner:
    """Choose how one branch-valid execution-state bundle becomes runnable."""

    def __init__(
        self,
        *,
        bandwidth_bytes_per_second: float,
        fixed_copy_seconds: float = 0.0,
    ):
        if bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth must be positive")
        if fixed_copy_seconds < 0:
            raise ValueError("fixed copy overhead must be non-negative")
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self.fixed_copy_seconds = fixed_copy_seconds

    def options(
        self, request: BranchMaterializationRequest
    ) -> tuple[MaterializationOption, ...]:
        descriptor = request.descriptor
        if not descriptor.valid_for(
            branch_id=descriptor.branch_id, causal_epoch=request.causal_epoch
        ):
            return ()

        capacity_ok = (
            request.destination_free_token_slots >= descriptor.token_slots
            and request.destination_free_snapshot_slots >= descriptor.snapshot_slots
        )
        options: list[MaterializationOption] = []

        if request.source_snapshot_available and capacity_ok:
            options.append(
                MaterializationOption(
                    method=MaterializationMethod.COPY_FULL_STATE,
                    completion_seconds=self.fixed_copy_seconds
                    + descriptor.total_bytes / self.bandwidth_bytes_per_second,
                    transfer_bytes=descriptor.total_bytes,
                    replay_tokens=0,
                    destination_token_slots=descriptor.token_slots,
                    destination_snapshot_slots=descriptor.snapshot_slots,
                )
            )

        if (
            request.ancestor_snapshot_available
            and 0 <= request.ancestor_prefix_tokens <= descriptor.prefix_tokens
            and capacity_ok
        ):
            suffix_tokens = descriptor.prefix_tokens - request.ancestor_prefix_tokens
            options.append(
                MaterializationOption(
                    method=MaterializationMethod.REPLAY_FROM_ANCESTOR,
                    completion_seconds=self.fixed_copy_seconds
                    + request.ancestor_state_bytes / self.bandwidth_bytes_per_second
                    + suffix_tokens / request.replay_tokens_per_second,
                    transfer_bytes=request.ancestor_state_bytes,
                    replay_tokens=suffix_tokens,
                    destination_token_slots=descriptor.token_slots,
                    destination_snapshot_slots=descriptor.snapshot_slots,
                )
            )

        if request.reconstructible and capacity_ok:
            options.append(
                MaterializationOption(
                    method=MaterializationMethod.RECONSTRUCT_FROM_TOKENS,
                    completion_seconds=descriptor.prefix_tokens
                    / request.replay_tokens_per_second,
                    transfer_bytes=0,
                    replay_tokens=descriptor.prefix_tokens,
                    destination_token_slots=descriptor.token_slots,
                    destination_snapshot_slots=descriptor.snapshot_slots,
                )
            )

        return tuple(options)

    def plan(
        self,
        request: BranchMaterializationRequest,
        *,
        require_within_gap: bool = False,
    ) -> MaterializationPlan | None:
        options = self.options(request)
        if not options:
            return None

        ordered = sorted(
            options,
            key=lambda option: (
                option.completion_seconds,
                option.transfer_bytes,
                option.method.value,
            ),
        )
        fitting = [
            option
            for option in ordered
            if option.completion_seconds <= request.remaining_gap_seconds
        ]
        if require_within_gap and not fitting:
            return None
        selected = fitting[0] if fitting else ordered[0]
        return MaterializationPlan(
            selected=selected,
            options=tuple(ordered),
            remaining_gap_seconds=request.remaining_gap_seconds,
            exposed_seconds=max(
                0.0, selected.completion_seconds - request.remaining_gap_seconds
            ),
            fits_gap=bool(fitting),
        )
