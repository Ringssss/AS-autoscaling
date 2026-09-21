import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.agentic_profile.cpu_workloads import execute_context, execute_tool, freeze_manifest


def test_frozen_ast_real_output_and_cpu_accounting(tmp_path):
    source = tmp_path / 'a.py'
    source.write_text('import json\nclass Thing:\n    def evaluate(self):\n        return 42\n')
    manifest = freeze_manifest([source])
    a = execute_tool({'manifest': manifest})
    b = execute_tool({'manifest': manifest, 'passes': 2})
    assert a['success'] and b['success']
    assert a['text'] == b['text']
    output = json.loads(a['text'])['files'][0]
    assert output['imports'] == ['json']
    assert [d['name'] for d in output['definitions']] == ['Thing', 'evaluate']
    assert a['metrics']['output_sha256'] == hashlib.sha256(a['text'].encode()).hexdigest()
    assert b['metrics']['input_bytes'] == 2*a['metrics']['input_bytes']
    assert a['metrics']['wall_seconds'] >= 0 and a['metrics']['cpu_seconds'] >= 0
    source.write_text('pass\n')
    assert 'corpus changed' in execute_tool({'manifest': manifest})['error']


def test_json_and_actual_context_preservation(tmp_path):
    source = tmp_path / 'a.json'
    source.write_text('[{"result":"真实输出", "n": 3}, {"result":"next"}]')
    result = execute_tool({'operation': 'json', 'manifest': freeze_manifest([source])})
    assert result['success']
    assert json.loads(result['text'])['files'][0]['key_frequency'] == {'n': 1, 'result': 2}
    context = execute_context({'messages': [{'role': 'user', 'content': 'analyze'}], 'tool_text': result['text']})
    assert context['success']
    assert json.loads(context['text'].splitlines()[1])['content'] == result['text']


def test_bounds_and_manifest_tamper(tmp_path):
    source = tmp_path / 'a.py'
    source.write_text('x = 1')
    manifest = freeze_manifest([source])
    assert not execute_tool({'manifest': manifest, 'passes': 0})['success']
    result = execute_tool({'manifest': manifest, 'max_output_chars': 8})
    assert result['success'] and result['metrics']['output_truncated']
    manifest['files'][0]['sha256'] = 'bad'
    assert 'manifest identity' in execute_tool({'manifest': manifest})['error']


def test_worker_survives_invalid_request_and_reuses_pid():
    inputs = ['not-json', json.dumps({'action': 'context', 'spec': {'tool_text': 'one'}}),
              json.dumps({'action': 'context', 'spec': {'tool_text': 'two'}})]
    run = subprocess.run([sys.executable, '-m', 'experiments.agentic_profile.cpu_workloads', '--worker'],
                         input='\n'.join(inputs)+'\n', text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert not rows[0]['success']
    assert rows[1]['success'] and rows[2]['success']
    assert rows[1]['metrics']['pid'] == rows[2]['metrics']['pid']
