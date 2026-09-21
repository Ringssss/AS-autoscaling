import pytest

from agentshift.controller.materialization import (
    BranchMaterializationRequest,
    BranchStatePlanner,
    MaterializationMethod,
)
from agentshift.state.descriptor import (
    StateComponent,
    StateComponentType,
    StateDescriptor,
)


def hybrid(prefix_tokens=512):
    return StateDescriptor.hybrid_qwen35(
        descriptor_id="state",
        model_identity="qwen35-h100-bf16-tp1",
        branch_id="branch",
        prefix_tokens=prefix_tokens,
    )


def test_hybrid_descriptor_has_independent_footprints():
    descriptor = hybrid()
    assert descriptor.total_bytes == 512 * 20_480 + 64_389_120
    assert descriptor.token_slots == 512
    assert descriptor.snapshot_slots == 1
    assert descriptor.component(StateComponentType.TOKEN_KV).size_bytes == 10_485_760
    assert (
        descriptor.component(StateComponentType.RECURRENT_SNAPSHOT).size_bytes
        == 64_389_120
    )
    assert descriptor.valid_for(branch_id="branch", causal_epoch=0)
    assert not descriptor.valid_for(branch_id="other", causal_epoch=0)


def test_component_footprints_are_type_checked():
    with pytest.raises(ValueError):
        StateComponent(
            component_type=StateComponentType.TOKEN_KV,
            size_bytes=1,
            snapshot_slots=1,
        )
    with pytest.raises(ValueError):
        StateComponent(
            component_type=StateComponentType.RECURRENT_SNAPSHOT,
            size_bytes=1,
            token_slots=1,
        )


def request(**overrides):
    values = {
        "descriptor": hybrid(),
        "destination_engine": "destination",
        "remaining_gap_seconds": 1.0,
        "destination_free_token_slots": 512,
        "destination_free_snapshot_slots": 1,
        "replay_tokens_per_second": 1_000,
    }
    values.update(overrides)
    return BranchMaterializationRequest(**values)


def test_planner_prefers_fastest_valid_materialization():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    plan = planner.plan(request())
    assert plan.selected.method == MaterializationMethod.COPY_FULL_STATE
    assert plan.fits_gap
    assert plan.exposed_seconds == 0


def test_planner_uses_ancestor_replay_when_full_snapshot_is_unavailable():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    plan = planner.plan(
        request(
            source_snapshot_available=False,
            ancestor_snapshot_available=True,
            ancestor_state_bytes=64_389_120,
            ancestor_prefix_tokens=256,
        )
    )
    assert plan.selected.method == MaterializationMethod.REPLAY_FROM_ANCESTOR
    assert plan.selected.replay_tokens == 256
    assert plan.selected.transfer_bytes == 64_389_120


def test_planner_reconstructs_when_no_snapshot_can_move():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    plan = planner.plan(
        request(
            source_snapshot_available=False,
            ancestor_snapshot_available=False,
            replay_tokens_per_second=10_000,
        )
    )
    assert plan.selected.method == MaterializationMethod.RECONSTRUCT_FROM_TOKENS
    assert plan.selected.transfer_bytes == 0
    assert plan.selected.replay_tokens == 512


def test_planner_enforces_destination_dual_capacity():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    assert (
        planner.plan(
            request(
                destination_free_snapshot_slots=0,
                source_snapshot_available=False,
                ancestor_snapshot_available=False,
                reconstructible=False,
            )
        )
        is None
    )


def test_planner_can_require_gap_coverage():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    constrained = request(
        remaining_gap_seconds=0.01,
        source_snapshot_available=False,
        ancestor_snapshot_available=False,
        replay_tokens_per_second=100,
    )
    assert planner.plan(constrained, require_within_gap=True) is None
    fallback = planner.plan(constrained)
    assert fallback is not None
    assert not fallback.fits_gap
    assert fallback.exposed_seconds > 0
