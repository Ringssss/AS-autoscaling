from agentshift.controller.global_mobility import (
    EngineMobilityState,
    GlobalMobilityCandidate,
    GlobalMobilityPolicy,
)
from agentshift.controller.materialization import (
    BranchStatePlanner,
    MaterializationMethod,
)
from agentshift.state.descriptor import StateDescriptor


def descriptor(prefix=2048):
    return StateDescriptor.hybrid_qwen35(
        descriptor_id="state",
        model_identity="qwen35",
        branch_id="branch",
        prefix_tokens=prefix,
    )


def test_selects_less_loaded_compatible_destination():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    policy = GlobalMobilityPolicy(planner)
    decision = policy.select(
        GlobalMobilityCandidate("a", "source", descriptor(), 1.0),
        [
            EngineMobilityState("source", 4096, 1, active_requests=8, warm_states=2),
            EngineMobilityState("dest", 4096, 1, active_requests=1),
        ],
    )
    assert decision is not None
    assert decision.destination_engine == "dest"
    assert decision.plan.selected.method == MaterializationMethod.COPY_FULL_STATE


def test_drain_urgency_overrides_source_load_but_keeps_hard_capacity():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    policy = GlobalMobilityPolicy(planner)
    candidate = GlobalMobilityCandidate("a", "retiring", descriptor(), 1.0)
    engines = [
        EngineMobilityState("retiring", 4096, 1, retiring=True),
        EngineMobilityState("full-snapshots", 4096, 0),
        EngineMobilityState("dest", 4096, 1),
    ]
    decision = policy.select(candidate, engines)
    assert decision is not None
    assert decision.drain_urgent
    assert decision.destination_engine == "dest"


def test_gap_and_dual_capacity_are_hard_gates():
    planner = BranchStatePlanner(bandwidth_bytes_per_second=1024**3)
    policy = GlobalMobilityPolicy(planner)
    candidate = GlobalMobilityCandidate("a", "source", descriptor(), 0.001)
    engines = [
        EngineMobilityState("source", 4096, 1, active_requests=9),
        EngineMobilityState("dest", 4096, 1),
    ]
    assert policy.select(candidate, engines) is None
