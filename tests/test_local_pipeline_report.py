import copy
import json

import pytest

from experiments.local_pipeline.metrics import summarize_emissions
from experiments.local_pipeline.report_local_copy import analyze_campaign, digest, summarize_run


def build_campaign(tmp_path):
    args = dict(foreground_targets=['source'], concurrency=[1], prefix_tokens=[8192],
                repetitions=1, modes=['no-copy', 'copy'], input_tokens=4, output_tokens=3)
    backend = dict(pp_size=1, tp_size=1, disable_cuda_graph=False,
                   mamba_ssm_dtype='float32', max_total_tokens=131072,
                   max_mamba_cache_size=64, stream_interval=1, attention_backend='fa3',
                   linear_attn_backend='triton', disable_overlap_schedule=False,
                   mamba_scheduler_strategy='extra_buffer')
    emissions = [{'time': .1, 'tokens': 1}, {'time': .11, 'tokens': 2}, {'time': .17, 'tokens': 3}]
    request = dict(start=0., first=.1, finish=.18, input_tokens=4,
                   input_sha256='foreground', requested_output_tokens=3,
                   emissions=emissions, output_ids=[9, 8, 7], output_sha256=digest([9, 8, 7]),
                   success=True, metrics=summarize_emissions(emissions, 3),
                   meta_info=dict(prompt_tokens=4, completion_tokens=3))
    records = []
    for i, mode in enumerate(args['modes']):
        row = dict(index=i, target='source', concurrency=1, prefix_tokens=8192, repetition=0,
                   mode=mode, start=0., finish=.18, trigger=.12, success=True,
                   active_at_trigger=1, decoding_at_trigger=1, cleanup_confirmed=True,
                   parent_input_sha256='parent', source_build={'meta_info': {'prompt_tokens': 8192}, 'output_ids': [1]},
                   requests=[copy.deepcopy(request)], control_events=[])
        if mode == 'copy':
            continuation = dict(output_ids=list(range(8)), meta_info=dict(cached_tokens=8192, prompt_tokens=8193, completion_tokens=8))
            row['migration'] = dict(start=.12, finish=.15, bytes=100)
            row['validation'] = dict(source=continuation, destination=copy.deepcopy(continuation), cache_hit=True, cached_output_equal=True)
            row['control_events'] = [dict(engine='destination', path='status', start=.12, finish=.15, response={'state': 'COMPLETE'})]
        records.append(row)
    manifest = [{k: row[k] for k in ('target', 'concurrency', 'prefix_tokens', 'repetition', 'mode')} for row in records]
    for name, value in [('arguments.json', args), ('server-info.json', [backend, backend]),
                        ('fixtures.json', dict(parent_hashes={'8192': 'parent'}, foreground_hashes=['foreground'], suffix=[2])),
                        ('manifest.json', manifest), ('complete.json', dict(complete=True, runs=2))]:
        (tmp_path/name).write_text(json.dumps(value))
    save_records(tmp_path, records)
    return records


def save_records(path, records):
    (path/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))


def test_valid_design_recomputed_metrics_and_paired_deltas(tmp_path):
    build_campaign(tmp_path)
    report = analyze_campaign(tmp_path)
    assert report['audit_passed'], report['errors']
    assert not report['full_48_design']
    assert len(report['pairs']) == 1
    assert report['pairs'][0]['deltas']['fixed_window_max_ms']['delta_ms'] == 0
    assert report['runs'][1]['fixed_window_max_ms'] == pytest.approx(60)
    assert report['runs'][1]['longest_gaps'][0]['windows'][0]['overlap_seconds'] == pytest.approx(.03)


@pytest.mark.parametrize('mutation, expected', [
    (lambda rows: rows[1].update(index=0), 'index mismatch'),
    (lambda rows: rows[1].update(cleanup_confirmed=False), 'cleanup not confirmed'),
    (lambda rows: rows[1].update(parent_input_sha256='wrong'), 'parent input identity'),
    (lambda rows: rows[1]['requests'][0]['metrics'].update(gap_max_seconds=0), 'stored metrics mismatch'),
    (lambda rows: rows[1]['validation']['destination']['meta_info'].update(cached_tokens=0), 'checkpoint not consumed'),
    (lambda rows: rows[1]['requests'][0]['emissions'].pop(), 'stream incomplete'),
])
def test_corrupt_evidence_fails_audit(tmp_path, mutation, expected):
    rows = build_campaign(tmp_path)
    mutation(rows)
    save_records(tmp_path, rows)
    report = analyze_campaign(tmp_path)
    assert not report['audit_passed']
    assert any(expected in error for error in report['errors'])


def test_output_mismatch_is_visible_not_erased_by_length_parity(tmp_path):
    rows = build_campaign(tmp_path)
    rows[1]['requests'][0]['output_ids'][0] = 100
    rows[1]['requests'][0]['output_sha256'] = digest(rows[1]['requests'][0]['output_ids'])
    save_records(tmp_path, rows)
    report = analyze_campaign(tmp_path)
    assert report['audit_passed']
    assert report['warnings']
    assert report['pairs'][0]['foreground_output_equal'] == [False]


def test_backend_and_missing_completion_are_not_assumed(tmp_path):
    build_campaign(tmp_path)
    info = json.loads((tmp_path/'server-info.json').read_text())
    info[0]['pp_size'] = 2
    (tmp_path/'server-info.json').write_text(json.dumps(info))
    (tmp_path/'complete.json').unlink()
    report = analyze_campaign(tmp_path)
    assert not report['audit_passed']
    assert any('pp_size' in error for error in report['errors'])
    assert any('completion marker' in error for error in report['errors'])


def test_phase_window_captures_rare_stall_hidden_by_whole_run_p99():
    emissions = [{'time': i * .01, 'tokens': i + 1} for i in range(1000)]
    for emission in emissions[11:]:
        emission['time'] += .09
    request = dict(start=0, finish=10.1, requested_output_tokens=1000, emissions=emissions)
    row = dict(start=0, trigger=.12, requests=[request], control_events=[])
    summary = summarize_run(row)
    assert summary['gap_p99_ms'] == pytest.approx(10)
    assert summary['fixed_window_max_ms'] == pytest.approx(100)
    assert summary['gaps_over_50ms'] == 1
