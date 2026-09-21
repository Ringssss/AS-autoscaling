import asyncio
from collections import Counter
import json
from types import SimpleNamespace

import pytest

from experiments.freetoken_elastic.benchmark_repartition import (
    check_generation, check_geometry, run_trial, trial_matrix,
)


def args():
    return SimpleNamespace(initial_pages=256, final_pages=128, initial_moe=20,
                           expert_only_moe=16, initial_mamba=12, final_mamba=8,
                           reset_page_delta=64, parent_output=1, child_output=64, model='model')


def test_matrix_balanced_reproducible_pairwise():
    pairs = [{'id': 'short'}, {'id': 'long'}]
    conditions = ['no-resize', 'expert-only', 'kv-resize', 'mamba-resize']
    rows = trial_matrix(pairs, conditions, 3, 1516)
    assert rows == trial_matrix(pairs, conditions, 3, 1516)
    assert len(rows) == 24
    assert all(n == 1 for n in Counter((r['pair_id'], r['condition'], r['repetition']) for r in rows).values())


def test_geometry_and_token_gates_reject_false_success():
    with pytest.raises(ValueError, match='geometry mismatch'):
        check_geometry({'success': True, 'body': {'state': 'serving', 'geometry': {'num_pages': 4}}}, {'num_pages': 8})
    with pytest.raises(ValueError, match='cold parent'):
        check_generation({'success': True, 'usage': {'prompt_tokens': 8, 'completion_tokens': 1}, 'cached_tokens': 4}, 8, 1, cold=True)
    with pytest.raises(ValueError, match='length mismatch'):
        check_generation({'success': True, 'usage': {'prompt_tokens': 8, 'completion_tokens': 1}}, 9, 1)


class FakeClient:
    def __init__(self, fail_measured=False):
        self.geometry = {}
        self.rebuilds = []
        self.prompts = []
        self.fail_measured = fail_measured

    async def rebuild(self, **geometry):
        self.rebuilds.append(geometry)
        if self.fail_measured and len(self.rebuilds) == 3:
            return {'success': False, 'http_status': 503, 'error': 'maintenance rejected', 'elapsed_seconds': 0.25}
        self.geometry.update(geometry)
        return {'success': True, 'elapsed_seconds': 0.25}

    async def status(self):
        return {'success': True, 'body': {'state': 'serving', 'geometry': dict(self.geometry)}}

    async def stats(self):
        return {'success': True, 'body': {}}

    async def generate(self, prompt, count, model):
        self.prompts.append(prompt)
        return {'success': True, 'started_monotonic': 10000000, 'elapsed_seconds': 1,
                'first_text_seconds': 0.1, 'cached_tokens': 0 if len(self.prompts) == 1 else 4,
                'text': 'answer', 'usage': {'prompt_tokens': 8 if prompt == 'parent' else 12, 'completion_tokens': count}}


def test_trial_resets_then_measures_exact_child_twice(tmp_path):
    client = FakeClient()
    spec = {'index': 0, 'pair_id': 'p', 'condition': 'kv-resize', 'repetition': 0}
    pair = {'parent_text': 'parent', 'child_text': 'child', 'expected_parent_tokens': 8, 'expected_child_tokens': 12}
    result = asyncio.run(run_trial(client, args(), pair, spec, tmp_path))
    assert result['success']
    assert [r['num_pages'] for r in client.rebuilds] == [320, 256, 128]
    assert client.prompts == ['parent', 'child', 'child']
    assert result['metrics']['rebuild_plus_first_text_seconds'] == 0.35
    assert json.loads((tmp_path / 'trial-000.json').read_text())['success']


def test_failed_rebuild_saved_and_child_not_submitted(tmp_path):
    client = FakeClient(fail_measured=True)
    spec = {'index': 0, 'pair_id': 'p', 'condition': 'kv-resize', 'repetition': 0}
    pair = {'parent_text': 'parent', 'child_text': 'child', 'expected_parent_tokens': 8, 'expected_child_tokens': 12}
    result = asyncio.run(run_trial(client, args(), pair, spec, tmp_path))
    assert not result['success'] and 'maintenance rejected' in result['error']
    assert result['operations'][-1]['http_status'] == 503
    assert client.prompts == ['parent']
    assert len(client.rebuilds) == 3


@pytest.mark.parametrize('condition,payload', [
    ('expert-only', {'moe_cache_size': 16}),
    ('kv-resize', {'num_pages': 128}),
    ('mamba-resize', {'num_mamba_slots': 8}),
    ('no-resize', None),
])
def test_measured_payload_excludes_unchanged_destructive_pool_fields(tmp_path, condition, payload):
    # FreeToken dispatches pool reconstruction on non-None keys, even at the same size.
    # Sending unchanged KV/mamba values would make the expert-only control destructive.
    client = FakeClient()
    spec = {'index': 0, 'pair_id': 'p', 'condition': condition, 'repetition': 0}
    pair = {'parent_text': 'parent', 'child_text': 'child', 'expected_parent_tokens': 8, 'expected_child_tokens': 12}
    result = asyncio.run(run_trial(client, args(), pair, spec, tmp_path))
    assert result['success']
    assert client.rebuilds[2:] == ([] if payload is None else [payload])
    assert result['final_geometry']['num_pages'] == (128 if condition == 'kv-resize' else 256)
    assert result['final_geometry']['num_mamba_slots'] == (8 if condition == 'mamba-resize' else 12)


def test_reset_delta_can_shrink_before_restoring_full_budget(tmp_path):
    client = FakeClient()
    config = args()
    config.reset_page_delta = -64
    spec = {'index': 0, 'pair_id': 'p', 'condition': 'no-resize', 'repetition': 0}
    pair = {'parent_text': 'parent', 'child_text': 'child', 'expected_parent_tokens': 8, 'expected_child_tokens': 12}
    result = asyncio.run(run_trial(client, config, pair, spec, tmp_path))
    assert result['success']
    assert [payload['num_pages'] for payload in client.rebuilds] == [192, 256]
