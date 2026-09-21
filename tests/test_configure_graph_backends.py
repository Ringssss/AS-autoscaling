import pytest

from experiments.fluidkv.configure_graph_backends import graph_arguments


BASE = ["/python", "-m", "sglang.launch_server", "--port", "31482",
        "--max-total-tokens", "131072", "--max-mamba-cache-size", "64",
        "--context-length", "32768", "--max-running-requests", "16"]


def test_default_preserves_existing_strategy_and_overlap():
    args = BASE + ["--mamba-scheduler-strategy", "extra_buffer", "--disable-overlap-schedule"]
    result = graph_arguments(args)
    assert result[:len(args)] == args


def test_explicit_extra_buffer_enables_normal_overlap_and_preserves_budgets():
    args = BASE + ["--mamba-scheduler-strategy=no_buffer", "--disable-overlap-schedule"]
    result = graph_arguments(args, mamba_scheduler_strategy="extra_buffer")
    assert result[:len(BASE)] == BASE
    assert result.count("--mamba-scheduler-strategy") == 1
    assert result[result.index("--mamba-scheduler-strategy") + 1] == "extra_buffer"
    assert "--disable-overlap-schedule" not in result
    assert not any(x.startswith("--mamba-scheduler-strategy=") for x in result)


@pytest.mark.parametrize("overlap_disabled", [False, True])
def test_explicit_overlap_setting_wins_and_is_idempotent(overlap_disabled):
    result = graph_arguments(BASE, mamba_scheduler_strategy="extra_buffer",
                             disable_overlap_schedule=overlap_disabled)
    assert ("--disable-overlap-schedule" in result) == overlap_disabled
    assert graph_arguments(result, mamba_scheduler_strategy="extra_buffer",
                           disable_overlap_schedule=overlap_disabled) == result


def test_explicit_enable_overlap_without_strategy_preserves_existing_strategy():
    args = BASE + ["--mamba-scheduler-strategy", "extra_buffer", "--disable-overlap-schedule"]
    result = graph_arguments(args, disable_overlap_schedule=False)
    assert result[result.index("--mamba-scheduler-strategy") + 1] == "extra_buffer"
    assert "--disable-overlap-schedule" not in result


def test_capacity_strategy_and_graph_overrides_do_not_duplicate_flags():
    args = BASE + ["--cuda-graph-bs", "1", "4", "--mamba-scheduler-strategy", "no_buffer"]
    result = graph_arguments(args, 65536, mamba_scheduler_strategy="extra_buffer")
    assert result.count("--max-total-tokens") == 1
    assert result[result.index("--max-total-tokens") + 1] == "65536"
    assert result.count("--cuda-graph-bs") == 1
    assert result[result.index("--max-mamba-cache-size") + 1] == "64"


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        graph_arguments(BASE, mamba_scheduler_strategy="invented")
