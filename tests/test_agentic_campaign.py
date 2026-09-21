"""Exercise HTTP stream accounting and real context composition without serving."""
import asyncio
import hashlib
import json

import pytest

from experiments.agentic_profile import run_profile_campaign as subject


class Response:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.content = self
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    def raise_for_status(self):
        pass
    async def readline(self):
        return next(self.rows, b'')


class Session:
    def __init__(self, rows):
        self.rows = rows
        self.request = None
    def post(self, url, json):
        self.request = json
        return Response(self.rows)


def event(count, ids, prompt=2):
    return b'data: ' + json.dumps({'output_ids': ids, 'meta_info': {
        'completion_tokens': count, 'prompt_tokens': prompt, 'cached_tokens': 0}}).encode() + b'\n'


def test_stream_chunks_are_not_assumed_single_tokens():
    session = Session([b': keepalive\n', b'\n', event(0, []), event(2, [7, 8]),
                       event(2, [7, 8]), event(3, [7, 8, 9]), b'data: [DONE]\n'])
    result = asyncio.run(subject.generate(session, 'fake', [1, 2], 3, [], 'test'))
    assert [delta for ts, delta in result['emissions']] == [2, 1]
    assert result['output_ids'] == [7, 8, 9]
    assert session.request['input_ids'] == [1, 2]
    assert session.request['sampling_params']['ignore_eos'] is True


def test_no_output_fails_instead_of_successful_zero_latency():
    with pytest.raises(RuntimeError, match='missing generated tokens'):
        asyncio.run(subject.generate(Session([event(0, []), b'data: [DONE]\n']), 'fake', [1, 2], 3, [], 'test'))


def test_runtime_truncation_metadata_is_rejected():
    with pytest.raises(RuntimeError, match='length changed'):
        asyncio.run(subject.generate(Session([event(2, [7, 8])]), 'fake', [1, 2], 3, [], 'test'))


def test_context_keeps_generated_ids_and_complete_unicode_tool(monkeypatch):
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            assert '\u4e2d\u6587 tool result' in text
            return [101, 102]
    monkeypatch.setattr(subject, 'TOKENIZER', Tokenizer())
    base, output = [1, 2], [3, 4]
    result = subject.context_job(base, output, '\u4e2d\u6587 tool result')
    assert result['input_ids'] == [1, 2, 3, 4, 101, 102]
    assert base == [1, 2] and output == [3, 4]
    assert result['tool_bytes'] == len('\u4e2d\u6587 tool result'.encode())
    assert result['tool_sha256'] == hashlib.sha256('\u4e2d\u6587 tool result'.encode()).hexdigest()
