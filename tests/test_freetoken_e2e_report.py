import json

from experiments.freetoken_elastic.report_e2e_comparison import summarize


def make_campaign(policy_rows):
    runs = []
    for index, (pair, label, cycle) in enumerate(policy_rows):
        runs.append({"pair_id": pair, "policy": label, "condition": label,
                     "repetition": index % 2, "success": True,
                     "metrics": {"intervention_to_completion_seconds": cycle,
                                 "intervention_to_first_text_seconds": cycle / 2,
                                 "first_child_cached_tokens": 1}})
    return {"complete": True, "runs": runs}


def test_e2e_summary_pairs_preserve_and_drop(tmp_path):
    preservation = tmp_path/"preserve"; preservation.mkdir()
    alternatives = tmp_path/"alternatives"; alternatives.mkdir()
    rows = [("prefix-8192", "drop", 8.0), ("prefix-8192", "preserve", 4.0)]
    (preservation/"campaign.json").write_text(json.dumps(make_campaign(rows)))
    alternative_rows = [("prefix-8192", "expert-only", 3.0), ("prefix-8192", "no-resize", 2.0)]
    (alternatives/"campaign.json").write_text(json.dumps(make_campaign(alternative_rows)))
    result = summarize(preservation, alternatives)[0]
    assert result["vs_freetoken_drop"]["speedup"] == 2.0
    assert result["vs_freetoken_drop"]["cycle_delta_percent"] == -50.0
    assert result["freetoken_expert_only"]["preserve_delta_percent"] > 0
