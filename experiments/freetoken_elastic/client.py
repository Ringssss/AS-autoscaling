"""External FreeToken measurement client. No automatic retries hide maintenance failures."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time

import aiohttp


async def sse_events(content):
    """Parse SSE frames independent of TCP chunk boundaries and UTF-8 splits."""
    data = []
    async for raw in content:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
    if data:
        yield '\n'.join(data)


class FreeTokenClient:
    def __init__(self, session, base_url, timeout=600, cache_report_enabled=False):
        self.session = session
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.cache_report_enabled = cache_report_enabled

    async def request(self, method, path, payload=None, stream=False):
        record = {'method': method, 'path': path, 'payload': payload,
                  'started_unix': time.time(), 'started_monotonic': time.perf_counter(),
                  'http_status': None, 'success': False, 'error': None, 'events': []}
        try:
            async with self.session.request(method, self.base_url + path, json=payload,
                                            timeout=self.timeout) as response:
                record['http_status'] = response.status
                record['headers_elapsed_seconds'] = time.perf_counter() - record['started_monotonic']
                if stream and response.status == 200:
                    done = False
                    async for data in sse_events(response.content):
                        elapsed = time.perf_counter() - record['started_monotonic']
                        if data == '[DONE]':
                            done = True
                            break
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            raise ValueError('SSE payload must be a JSON object')
                        record['events'].append({'elapsed_seconds': elapsed, 'data': event})
                        if 'error' in event:
                            record['error'] = event['error']
                    record['stream_done'] = done
                    if not done and record['error'] is None:
                        record['error'] = 'stream ended without [DONE]'
                else:
                    body = await response.text()
                    record['body_text'] = body
                    try:
                        record['body'] = json.loads(body)
                    except ValueError:
                        record['body'] = None
                    if not 200 <= response.status < 300:
                        record['error'] = record['body'] or body or f'HTTP {response.status}'
                record['success'] = 200 <= response.status < 300 and record['error'] is None
                if path == '/v1/cache/rebuild' and (
                    not isinstance(record.get('body'), dict) or record['body'].get('status') != 'ok'
                ):
                    record['success'] = False
                    record['error'] = record.get('error') or record.get('body') or 'invalid rebuild reply'
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, UnicodeError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            record['finished_unix'] = time.time()
            record['elapsed_seconds'] = time.perf_counter() - record['started_monotonic']
        return record

    async def generate(self, prompt, max_tokens, model='default', ignore_eos=True):
        if not isinstance(prompt, str):
            raise TypeError('Pinned FreeToken rejects raw token IDs; pass the exact text prompt')
        payload = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
                   'temperature': 0, 'ignore_eos': ignore_eos, 'stream': True,
                   'stream_options': {'include_usage': True}}
        record = await self.request('POST', '/v1/completions', payload, stream=True)
        record.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      text='', first_text_seconds=None, usage=None,
                      cached_tokens=None, text_chunk_times_seconds=[])
        for event in record['events']:
            data = event['data']
            if data.get('usage') is not None:
                record['usage'] = data['usage']
            for choice in data.get('choices', []):
                text = choice.get('text', '')
                if text:
                    record['text'] += text
                    record['text_chunk_times_seconds'].append(event['elapsed_seconds'])
                    if record['first_text_seconds'] is None:
                        record['first_text_seconds'] = event['elapsed_seconds']
        if record['usage'] is not None and self.cache_report_enabled:
            record['cached_tokens'] = record['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if record['success'] and record['usage'] is None:
            record['success'] = False
            record['error'] = 'missing final usage: token counts cannot be verified'
        return record

    async def rebuild(self, **geometry):
        return await self.request('POST', '/v1/cache/rebuild', {'mode': 'if_idle', **geometry})

    async def status(self):
        return await self.request('GET', '/v1/cache/status')

    async def stats(self):
        return await self.request('GET', '/v1/stats')

    async def requests(self, since=0, limit=100):
        return await self.request('GET', f'/v1/requests?since={int(since)}&limit={int(limit)}')
