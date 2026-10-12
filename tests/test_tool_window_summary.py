import json

import pytest

from experiments.multinode.summarize_tool_window import summarize


def write_trials(tmp_path):
    rows = []
    for mode, ttft in (("sticky-hot-owner", 2.0), ("route-cold-idle", 3.0), ("copy-during-tool", 0.5)):
        rows.append({"repeat": 0, "prefix_tokens": 1024, "controlled_tool_seconds": 0.5,
                     "mode": mode, "checks": {"exact_output": True},
                     "timeline": {"tool_return": 0.5, "continuation_dispatch": 0.6,
                                  "continuation_first_token": 0.5 + ttft, "continuation_finished": 4.0,
                                  "foreground_scheduled_arrival": 0.2, "foreground_dispatch": 0.21,
                                  "foreground_first_token": 0.4, "foreground_finished": 5.0,
                                  "copy_started": 0.01, "copy_committed": 0.4},
                     "post_tool_ttft_seconds": ttft, "post_tool_e2e_seconds": 3.5,
                     "tool_start_to_completion_seconds": 4.0,
                     "foreground_arrival_to_completion_seconds": 4.8,
                     "copy_exposed_seconds": 0.0, "foreground_active_at_tool_return": True})
    save(tmp_path, rows)
    return rows


def save(directory, rows):
    (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_uses_paired_ratios_and_keeps_foreground_cost(tmp_path):
    write_trials(tmp_path)
    result = summarize(tmp_path)
    case = result["cases"][0]
    assert case["median_paired_ttft_ratio_vs_sticky-hot-owner"] == 4.0
    assert case["median_paired_ttft_ratio_vs_route-cold-idle"] == 6.0
    assert case["median_paired_foreground_delta_seconds_vs_sticky-hot-owner"] == 0.0
    assert result["completed_continuations"] == 3


@pytest.mark.parametrize("problem", ["missing", "duplicate", "bad_output", "early_dispatch", "late_commit", "failure"])
def test_summary_rejects_invalid_or_incomplete_evidence(tmp_path, problem):
    rows = write_trials(tmp_path)
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "bad_output":
        rows[0]["checks"]["exact_output"] = False
    elif problem == "early_dispatch":
        rows[0]["timeline"]["continuation_dispatch"] = 0.1
    elif problem == "late_commit":
        rows[2]["timeline"]["copy_committed"] = 2.0
    else:
        rows[0]["failed"] = True
    save(tmp_path, rows)
    with pytest.raises(ValueError):
        summarize(tmp_path)
