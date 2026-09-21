import hashlib
import json

from experiments.agentic_profile.report_profile_campaign import audit_run, telemetry_metrics, delta_percent


def fixture():
    tool_text = 'AST result'
    sha = hashlib.sha256(tool_text.encode()).hexdigest()
    ids = [30000, 10, 11, 12, 55]
    context = {'input_ids': ids, 'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               'tool_sha256': sha, 'tool_bytes': len(tool_text)}
    args = {'agents': 1, 'output_tokens': 2, 'prompt_tokens': 2, 'cpu_start': 0, 'cores': 8, 'tool_workers': 1}
    workers = [{'pid': 20+i, 'affinity': list(range(8))} for i in range(3)]
    summary = {'offered': 1, 'successful': 1, 'policy': 'shared', 'concurrency': 1, 'repetition': 0,
               'union_cpus': list(range(8)), 'controller_cpus': [64,65,66,67], 'gpu_budget': 1,
               'worker_pings': workers, 'tool_cpus': list(range(8)), 'context_cpus': list(range(8))}
    request = {'agent': 0, 'policy': 'shared', 'concurrency': 1, 'repetition': 0, 'success': True,
               'first': {'input_tokens': 2, 'output_tokens': 2, 'output_ids': [11,12]},
               'second': {'input_tokens': 5, 'output_tokens': 2, 'output_ids': [13,14]},
               'context': context, 'tool': {'success': True, 'text': tool_text, 'metrics': {'output_sha256': sha}}}
    return args, summary, request


def test_audit_detects_consumption_and_length_changes(tmp_path):
    args, summary, request = fixture()
    assert audit_run(tmp_path, summary, args, [request]) == []
    request['context']['input_ids'][2] = 99
    errors = audit_run(tmp_path, summary, args, [request])
    assert any('context hash invalid' in e for e in errors)
    assert any('generated output not preserved' in e for e in errors)
    request['second']['output_ids'].pop()
    assert any('output length mismatch' in e for e in audit_run(tmp_path, summary, args, [request]))


def test_audit_failed_agent_stays_in_offered_count(tmp_path):
    args, summary, request = fixture()
    request['success'] = False
    summary['successful'] = 0
    assert audit_run(tmp_path, summary, args, [request]) == []
    assert any('count differs' in e for e in audit_run(tmp_path, summary, args, []))


def test_telemetry_categories_and_unpinned_thread_cpu(tmp_path):
    summary = {'engine_affinity': [{'pid': 10, 'affinity_exceptions': [{'tid': 11, 'name': 'iou-sqp', 'cpu_seconds': 3}]}],
               'engine_affinity_end': [{'pid': 10, 'affinity_exceptions': [{'tid': 11, 'name': 'iou-sqp', 'cpu_seconds': 3.25}]}],
               'worker_pings': [{'pid': 20}]}
    records = [{'event': 'sample', 'processes': [{'pid': 10, 'cpu_seconds_delta': 2}, {'pid': 20, 'cpu_seconds_delta': 3}, {'pid': 30, 'cpu_seconds_delta': 1}],
                'rss_sum_bytes': 1024, 'sampling_wall_seconds': .1},
               {'event': 'summary', 'sampled_cpu_seconds': 6, 'error': None}]
    (tmp_path/'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    result = telemetry_metrics(tmp_path, summary)
    assert result['cpu_seconds_by_category'] == {'engine': 2, 'workers': 3, 'controller_and_helpers': 1}
    assert result['affinity_exception_observed_cpu_seconds'] == .25
    assert result['sampled_cpu_seconds'] == 6  # exception CPU is a subset, never added again


def test_delta_zero_baseline_not_infinite():
    assert delta_percent(10, 0) is None
    assert delta_percent(None, 10) is None
    assert abs(delta_percent(8, 10) + 20) < 1e-10


def test_reserved_two_engine_cores_and_six_frontend_cores(tmp_path):
    args, summary, request = fixture()
    summary['policy'] = request['policy'] = 'frontend-reserved2'
    summary['tool_cpus'] = summary['context_cpus'] = list(range(2,8))
    summary['engine_affinity'] = [{'pid': 10, 'thread_affinity': {'10': [0,1]}, 'affinity_exceptions': []}]
    summary['engine_affinity_end'] = summary['engine_affinity']
    for worker in summary['worker_pings']:
        worker['affinity'] = list(range(2,8))
    assert audit_run(tmp_path, summary, args, [request]) == []
    summary['context_cpus'] = list(range(4,8))
    assert any('policy contract' in e for e in audit_run(tmp_path, summary, args, [request]))
    summary['context_cpus'] = list(range(2,8))
    summary['engine_affinity'][0]['thread_affinity']['10'] = [0,1,2,3]
    assert any('unexpected affinity' in e for e in audit_run(tmp_path, summary, args, [request]))
