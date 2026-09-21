import asyncio
import json
from types import SimpleNamespace

import pytest

from experiments.freetoken_elastic.benchmark_preservation import PolicyClient, run
from experiments.freetoken_elastic.client import FreeTokenClient


@pytest.mark.parametrize('geometry,expected', [
    ({'num_pages': 128, 'num_mamba_slots': 8, 'moe_cache_size': 16}, 'drop'),
    ({'num_pages': 128}, 'preserve'),
    ({'num_mamba_slots': 8}, 'preserve'),
    ({'moe_cache_size': 16}, 'drop'),
])
def test_policy_marks_only_measured_state_rebuild_preserve(tmp_path, monkeypatch, geometry, expected):
    policy_path = tmp_path / 'policy.json'
    hook = tmp_path / 'hook.jsonl'
    async def backend(self, **payload):
        policy = json.loads(policy_path.read_text())
        assert policy['mode'] == expected
        assert policy['verify_bytes'] == (expected == 'preserve')
        hook.write_text(json.dumps({'id': policy['id'], 'success': True, 'restore': {'restored': True}, 'bytes_equal': True}) + '\n')
        return {'success': True}
    monkeypatch.setattr(FreeTokenClient, 'rebuild', backend)
    client = PolicyClient(None, 'http://unused', policy_path=policy_path, hook_log=hook, verify_bytes=True)
    client.preservation_mode = 'preserve'
    result = asyncio.run(client.rebuild(**geometry))
    assert result['success'] and len(result['hook_records']) == 1


@pytest.mark.parametrize('mode', ['absent', 'stale', 'duplicate', 'restore-failed'])
def test_hook_mapping_rejects_missing_ambiguous_or_failed_restore(tmp_path, monkeypatch, mode):
    policy_path = tmp_path / 'policy.json'
    hook = tmp_path / 'hook.jsonl'
    async def backend(self, **payload):
        policy = json.loads(policy_path.read_text())
        record = {'id': 'stale' if mode == 'stale' else policy['id'], 'success': True, 'restore': {'restored': mode != 'restore-failed'}}
        if mode != 'absent':
            hook.write_text((json.dumps(record) + '\n') * (2 if mode == 'duplicate' else 1))
        return {'success': True}
    monkeypatch.setattr(FreeTokenClient, 'rebuild', backend)
    client = PolicyClient(None, 'http://unused', policy_path=policy_path, hook_log=hook)
    client.preservation_mode = 'preserve'
    result = asyncio.run(client.rebuild(num_pages=128))
    assert not result['success'] and result['error']


def test_policy_matrix_indices_and_records_preserve_identity(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'pairs': [dict(id=p, parent_text='a', child_text='b', expected_parent_tokens=1, expected_child_tokens=1) for p in ['short', 'long']]}))
    args = SimpleNamespace(manifest=manifest, output=tmp_path / 'out', seed=1516, repetitions=3,
                           conditions=['kv-resize', 'mamba-resize'], engine='http://unused', timeout=1,
                           policy_file=tmp_path/'policy.json', hook_log=tmp_path/'hook.jsonl', verify_bytes=False)
    async def fake_trial(client, args, pair, spec, output):
        assert client.preservation_mode == spec['policy']
        assert pair['id'] == spec['pair_id']
        return {**spec, 'success': True, 'operations': [{'label': 'measured-rebuild', 'hook_records': [{'id': spec['index'], 'export': {'boundary': 64}}]}], 'metrics': {'first_child_cached_tokens': 64 if spec['policy']=='preserve' else 0}}
    monkeypatch.setattr('experiments.freetoken_elastic.benchmark_preservation.run_trial', fake_trial)
    assert asyncio.run(run(args)) == 0
    doc = json.loads((args.output/'campaign.json').read_text())
    assert doc['complete'] and len(doc['runs']) == 24
    assert [r['index'] for r in doc['matrix']] == list(range(24))
    assert [r['index'] for r in doc['runs']] == list(range(24))
    assert all(r['hook']['id'] == r['index'] for r in doc['runs'])
    for n in range(0, 24, 2):
        assert {r['policy'] for r in doc['matrix'][n:n+2]} == {'drop', 'preserve'}
