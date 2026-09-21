import asyncio

from experiments.agentic_profile.profile_state_work import first_or_done, overlap_seconds


def test_overlap_union_counts_shared_foreground_once():
    preparations = [{'start': 2, 'finish': 5}, {'start': 8, 'finish': 10}]
    foreground = [{'start': 0, 'finish': 4}, {'start': 3, 'finish': 9}]
    assert overlap_seconds(preparations, foreground) == 4
    assert overlap_seconds(preparations, [{'start': 11, 'finish': 12}]) == 0


def test_missing_first_token_does_not_hang():
    async def check():
        event = asyncio.Event()
        async def finished():
            return {'success': False}
        try:
            await first_or_done(event, [asyncio.create_task(finished())])
        except RuntimeError as exc:
            assert 'no first' in str(exc)
        else:
            raise AssertionError('failed requests must not trigger preparation')
    asyncio.run(check())


def test_serial_latency_includes_preparation_from_common_ready_time():
    from experiments.agentic_profile.profile_state_work import summarize
    row = {'start': 10., 'finish': 15., 'preparations': [{'start': 10., 'finish': 12.}],
           'foreground': [{'start': 12., 'first': 13., 'finish': 15., 'success': True,
                           'ttft_seconds': 1., 'e2e_seconds': 3., 'mean_decode_seconds_per_token': .2}]}
    result = summarize(row)
    assert result['foreground_ttft_p50_seconds'] == 1.
    assert result['foreground_ready_to_first_p50_seconds'] == 3.
    assert result['foreground_ready_to_finish_p50_seconds'] == 5.
    assert result['charged_gpu_seconds'] == 10.
    assert result['request_lifetime_overlap_seconds'] == 0.


def test_generation_rejects_truncated_output_and_retains_first(monkeypatch):
    from experiments.agentic_profile import profile_state_work as subject
    async def truncated(session, url, tokens, outputs, callback):
        await callback({}, .01)
        return {'ttft_seconds': .01, 'e2e_seconds': .02,
                'meta_info': {'prompt_tokens': len(tokens), 'completion_tokens': outputs},
                'output_ids': [1]}
    monkeypatch.setattr(subject, 'stream', truncated)
    result = asyncio.run(subject.generation(None, 'fake', [1, 2], 3))
    assert not result['success']
    assert result['requested_output_tokens'] == 3
    assert 'first' in result
    assert 'length mismatch' in result['error']


def test_generation_failure_is_accounted(monkeypatch):
    from experiments.agentic_profile import profile_state_work as subject
    async def failed(*args):
        raise RuntimeError('transport failed')
    monkeypatch.setattr(subject, 'stream', failed)
    result = asyncio.run(subject.generation(None, 'fake', [1], 3))
    assert not result['success']
    assert result['finish'] >= result['start']
    assert result['mean_decode_seconds_per_token'] is None
    assert 'transport failed' in result['error']
