"""HTTP observation contract: no GPU or engine process needed."""
import asyncio
import json

from experiments.local_pipeline.benchmark_local_copy import observe_generate


class Content:
    def __init__(self, lines):
        self.lines = iter(lines)

    async def readline(self):
        value = next(self.lines, b'')
        if isinstance(value, Exception):
            raise value
        return value


class Response:
    def __init__(self, lines):
        self.content = Content(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, lines):
        self.lines = lines
        self.request = None

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return Response(self.lines)


def event(count, ids=None):
    return b'data: ' + json.dumps({
        'meta_info': {'prompt_tokens': 2, 'completion_tokens': count},
        'output_ids': list(range(count)) if ids is None else ids,
    }).encode() + b'\n'


def test_observer_preserves_coalescing_duplicate_metadata_and_first_event():
    async def run():
        first = asyncio.Event()
        session = Session([b': comment\n', event(1), b'\n', event(3),
                           event(4), event(4), b'data: [DONE]\n'])
        row = await observe_generate(session, 'http://fixture', [11, 12], 4, first)
        assert first.is_set() and row['success']
        assert row['metrics']['complete']
        assert row['metrics']['coalesced_emission_count'] == 1
        assert row['metrics']['duplicate_count'] == 1
        assert row['metrics']['valid_gap_count'] == 1
        assert row['first'] == row['emissions'][0]['time']
        assert session.request[1]['json']['sampling_params']['ignore_eos']
    asyncio.run(run())


def test_observer_retains_partial_emissions_on_transport_failure():
    session = Session([event(1), TimeoutError('fixture timeout')])
    row = asyncio.run(observe_generate(session, 'http://fixture', [11, 12], 4))
    assert not row['success']
    assert 'fixture timeout' in row['error']
    assert len(row['emissions']) == 1
    assert row['metrics']['missing_tokens'] == 3


def test_observer_rejects_truncated_output_even_if_metadata_claims_completion():
    session = Session([event(1), event(4, ids=[0, 1])])
    row = asyncio.run(observe_generate(session, 'http://fixture', [11, 12], 4))
    assert not row['success']
    assert 'length mismatch' in row['error']


def test_observer_rejects_cumulative_count_reset():
    session = Session([event(2), event(1), event(4)])
    row = asyncio.run(observe_generate(session, 'http://fixture', [11, 12], 4))
    assert row['metrics']['reset_count'] == 1
    assert not row['metrics']['complete']
    assert not row['success']
