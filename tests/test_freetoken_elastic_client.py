import asyncio
import json

import aiohttp
from aiohttp import web
import pytest

from experiments.freetoken_elastic.client import FreeTokenClient


async def with_server(handler, action):
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            return await action(FreeTokenClient(session, f'http://127.0.0.1:{port}', cache_report_enabled=True))
    finally:
        await runner.cleanup()


def test_completion_fragmented_utf8_usage_and_exact_text():
    async def handler(request):
        body = await request.json()
        assert body['prompt'] == '原始\ntext'
        assert body['temperature'] == 0
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        frames = [': heartbeat\r\n\r\n',
                  'data: ' + json.dumps({'choices': [{'text': '你好'}]}, ensure_ascii=False) + '\r\n\r\n',
                  'data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 8}}}) + '\n\n',
                  'data: [DONE]\n\n']
        for byte in ''.join(frames).encode():
            await response.write(bytes([byte]))
        return response
    result = asyncio.run(with_server(handler, lambda c: c.generate('原始\ntext', 2)))
    assert result['success'] and result['text'] == '你好'
    assert result['cached_tokens'] == 8
    assert 0 <= result['first_text_seconds'] <= result['elapsed_seconds']
    assert len(result['text_chunk_times_seconds']) == 1  # chunks are not token IDs


@pytest.mark.parametrize('status', [409, 503, 504])
def test_rebuild_rejections_remain_failures_without_retry(status):
    calls = []
    async def handler(request):
        calls.append(await request.json())
        return web.json_response({'status': 'busy', 'error': 'maintenance'}, status=status)
    result = asyncio.run(with_server(handler, lambda c: c.rebuild(num_pages=64)))
    assert len(calls) == 1 and calls[0]['mode'] == 'if_idle'
    assert not result['success'] and result['http_status'] == status
    assert result['elapsed_seconds'] >= 0 and result['body']['error'] == 'maintenance'


@pytest.mark.parametrize('body', [b'data: {bad}\n\n', b'data: {"choices":[]}\n\n', b'data: {"error":"worker failed"}\n\ndata: [DONE]\n\n'])
def test_malformed_truncated_and_worker_error_streams(body):
    async def handler(request):
        return web.Response(body=body, content_type='text/event-stream')
    result = asyncio.run(with_server(handler, lambda c: c.generate('text', 1)))
    assert not result['success'] and result['error']


def test_missing_usage_is_not_complete_measurement():
    async def handler(request):
        return web.Response(text='data: [DONE]\n\n', content_type='text/event-stream')
    result = asyncio.run(with_server(handler, lambda c: c.generate('text', 1)))
    assert not result['success'] and result['cached_tokens'] is None


def test_non_json_success_rebuild_is_invalid():
    async def handler(request):
        return web.Response(text='unexpected')
    result = asyncio.run(with_server(handler, lambda c: c.rebuild(num_pages=64)))
    assert not result['success']


def test_raw_token_input_rejected_locally():
    async def action():
        with pytest.raises(TypeError, match='raw token IDs'):
            await FreeTokenClient(None, 'http://unused').generate([1, 2], 1)
    asyncio.run(action())


def test_cache_reporting_not_assumed_from_missing_detail():
    async def handler(request):
        return web.Response(text='data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\ndata: [DONE]\n\n', content_type='text/event-stream')
    async def action(client):
        client.cache_report_enabled = False
        first = await client.generate('text', 1)
        client.cache_report_enabled = True
        second = await client.generate('text', 1)
        return first, second
    first, second = asyncio.run(with_server(handler, action))
    assert first['success'] and first['cached_tokens'] is None
    assert second['success'] and second['cached_tokens'] == 0


def test_client_timeout_retains_attempt():
    async def handler(request):
        await asyncio.sleep(0.03)
        return web.json_response({'status': 'ok'})
    async def action(client):
        client.timeout = aiohttp.ClientTimeout(total=0.005)
        return await client.rebuild(num_pages=64)
    result = asyncio.run(with_server(handler, action))
    assert not result['success'] and 'TimeoutError' in result['error']
    assert result['elapsed_seconds'] > 0
