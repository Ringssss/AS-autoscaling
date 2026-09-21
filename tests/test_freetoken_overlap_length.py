from pathlib import Path

import pytest

from experiments.freetoken_elastic.diagnose_overlap_length import reproduce


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/freetoken_elastic/environment/FreeToken'
pytestmark = pytest.mark.skipif(not (SOURCE / 'python/freetoken/core.py').exists(), reason='isolated FreeToken research checkout unavailable')


@pytest.mark.parametrize('requested', [1, 2, 16, 64])
@pytest.mark.parametrize('overlap', [False, True])
def test_budget_uses_emitted_tokens_not_next_forward_position(requested, overlap):
    result = reproduce(SOURCE, requested, overlap)
    assert result['emitted_token_replies'] == requested
    assert result['host_generated_tokens'] == requested
    assert result['forward_calls'] == requested
    assert result['freed_at_host_counts'] == [requested]
    assert result['last_finish_reason'] == 'length'
    assert result['terminal_replies'] == 1


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('eos_at', [1, 3, 16])
def test_eos_still_wins_and_pending_overlap_token_is_not_emitted(overlap, eos_at):
    result = reproduce(SOURCE, 16, overlap, eos_at=eos_at)
    assert result['emitted_token_replies'] == eos_at
    assert result['host_generated_tokens'] == eos_at
    assert result['freed_at_host_counts'] == [eos_at]
    assert result['last_finish_reason'] == 'stop'
    assert result['terminal_replies'] == 1
    assert result['emitted_token_ids'][-1] == 99
    assert result['forward_calls'] == min(eos_at + int(overlap), 16)


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('stop_at', [1, 16])
def test_stop_string_wins_over_budget_and_frees_once(overlap, stop_at):
    result = reproduce(SOURCE, 16, overlap, stop_at=stop_at)
    assert result['emitted_token_replies'] == stop_at
    assert result['last_finish_reason'] == 'stop'
    assert result['terminal_replies'] == 1
    assert result['freed_at_host_counts'] == [stop_at]
