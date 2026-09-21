import hashlib

import pytest

from experiments.freetoken_elastic.report_repartition import audit_trial


def fixture():
    config = dict(initial_pages=256, final_pages=128, initial_moe=20, expert_only_moe=16,
                  initial_mamba=12, final_mamba=8, parent_output=16, child_output=64, cache_report_enabled=True)
    pair = dict(parent_text='parent', child_text='child', expected_parent_tokens=8, expected_child_tokens=12)
    spec = dict(index=0, pair_id='p', condition='expert-only', repetition=0)
    initial = dict(num_pages=256, moe_cache_size=20, num_mamba_slots=12)
    final = {**initial, 'moe_cache_size': 16}
    ops = []
    for label, prompt, count, output, cache in [('cold-parent', 'parent', 8, 16, 0), ('first-child', 'child', 12, 64, 4), ('repeated-child', 'child', 12, 64, 8)]:
        ops.append(dict(label=label, success=True, payload=dict(prompt=prompt, max_tokens=output),
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), usage=dict(prompt_tokens=count, completion_tokens=output),
                        stream_done=True, cached_tokens=cache, text='answer', first_text_seconds=1., elapsed_seconds=2.))
    for label, geo in [('initial-status', initial), ('before-status', initial), ('after-first-status', final), ('final-status', final)]:
        ops.append(dict(label=label, success=True, body=dict(state='serving', geometry=geo)))
    for label in ['before-stats', 'after-first-stats', 'final-stats']:
        ops.append(dict(label=label, success=True, body=dict(instance_id='engine1')))
    ops.append(dict(label='measured-rebuild', success=True, payload=dict(mode='if_idle', moe_cache_size=16), elapsed_seconds=.5))
    metrics = dict(rebuild_seconds=.5, first_child_text_seconds=1., first_child_e2e_seconds=2., first_child_cached_tokens=4,
                   repeated_child_cached_tokens=8, repeated_child_first_text_seconds=1., repeated_child_e2e_seconds=2.,
                   rebuild_plus_first_text_seconds=1.5, rebuild_plus_child_e2e_seconds=2.5,
                   intervention_to_first_text_seconds=1.51, intervention_to_completion_seconds=2.51)
    return dict(**spec, success=True, operations=ops, metrics=metrics), spec, pair, config


def test_complete_record_passes_exact_audit():
    trial, spec, pair, config = fixture()
    result = audit_trial(trial, spec, pair, config)
    assert result['audit_pass'] and result['first_repeat_text_agreement']


@pytest.mark.parametrize('defect,expected', [('payload', 'unchanged pool'), ('tokens', 'output token'), ('text', 'exact prompt'), ('instance', 'instance identity'), ('metric', 'metric mismatch')])
def test_audit_catches_treatment_and_measurement_contamination(defect, expected):
    trial, spec, pair, config = fixture()
    by_label = {r['label']: r for r in trial['operations']}
    if defect == 'payload':
        by_label['measured-rebuild']['payload']['num_pages'] = 256
    elif defect == 'tokens':
        by_label['first-child']['usage']['completion_tokens'] = 63
    elif defect == 'text':
        by_label['first-child']['payload']['prompt'] = 'changed'
    elif defect == 'instance':
        by_label['final-stats']['body']['instance_id'] = 'engine2'
    else:
        trial['metrics']['rebuild_plus_first_text_seconds'] = 1.
    result = audit_trial(trial, spec, pair, config)
    assert not result['audit_pass']
    assert any(expected in error for error in result['errors'])


def test_partial_failed_trial_is_reported_not_dropped_or_crashed():
    trial, spec, pair, config = fixture()
    trial['success'] = False
    trial['error'] = 'HTTP503'
    trial['operations'] = [dict(label='cold-parent', success=False, usage=None, payload=None, error='HTTP503')]
    trial['metrics'] = {}
    result = audit_trial(trial, spec, pair, config)
    assert not result['audit_pass'] and not result['recorded_success']
    assert any('HTTP503' in error for error in result['errors'])
