"""Bounded, corpus-backed CPU operations; no implicit filesystem traversal.

CLI: python -m experiments.agentic_profile.cpu_workloads --worker
Each stdin line is {"action": "tool"|"context", "spec": {...}}.
Output hashes cover the actual returned UTF-8 text, not timing metadata.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze_manifest(paths, output=None):
    """Hash an explicit file list; caller selects a bounded stable corpus."""
    entries = []
    for raw in sorted(set(str(Path(p).resolve()) for p in paths)):
        data = Path(raw).read_bytes()
        entries.append({'path': raw, 'bytes': len(data), 'sha256': _sha(data)})
    if not entries:
        raise ValueError('empty corpus')
    result = {'version': 1, 'files': entries}
    result['sha256'] = _sha(json.dumps(entries, sort_keys=True).encode())
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def _read_manifest(value):
    manifest = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(manifest, dict) or not manifest.get('files'):
        raise ValueError('nonempty frozen manifest required')
    expected = _sha(json.dumps(manifest['files'], sort_keys=True).encode())
    if expected != manifest.get('sha256'):
        raise ValueError('manifest identity mismatch')
    return manifest


def _measure(fn):
    start, cpu = time.monotonic(), time.process_time()
    try:
        text, extra = fn()
        error = None
    except Exception as exc:
        text, extra = '', {}
        error = f'{type(exc).__name__}: {exc}'
    end, cpu_end = time.monotonic(), time.process_time()
    encoded = text.encode('utf-8')
    return {'success': error is None, 'text': text, 'error': error,
            'metrics': {'started_monotonic': start, 'finished_monotonic': end,
                        'wall_seconds': end-start, 'cpu_seconds': cpu_end-cpu,
                        'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                        'output_bytes': len(encoded), 'output_sha256': _sha(encoded), **extra}}


def execute_tool(spec):
    """Parse every frozen input on each pass, returning useful deterministic text.

    ast: per-file classes/functions/imports and node-type counts.
    json: top-level schema/key frequencies plus actual sorted record excerpts.
    Repeated passes represent repeated analysis and are explicitly recorded.
    """
    def run():
        manifest = _read_manifest(spec['manifest'])
        passes = int(spec.get('passes', 1))
        cap = int(spec.get('max_output_chars', 32768))
        if not 1 <= passes <= 100 or not 1 <= cap <= 4_000_000:
            raise ValueError('passes/cap outside bounded range')
        operation = spec.get('operation', 'ast')
        if operation not in ('ast', 'json'):
            raise ValueError('operation must be ast or json')
        total_bytes = sum(entry['bytes'] for entry in manifest['files'])
        if total_bytes > 64*1024*1024 or len(manifest['files']) > 4096:
            raise ValueError('corpus exceeds 64MiB/4096 file limit')
        for _ in range(passes):
            reports = []
            for entry in manifest['files']:
                data = Path(entry['path']).read_bytes()
                if len(data) != entry['bytes'] or _sha(data) != entry['sha256']:
                    raise ValueError(f"corpus changed: {entry['path']}")
                if operation == 'ast':
                    tree = ast.parse(data, filename=entry['path'])
                    nodes = list(ast.walk(tree))
                    report = {'file': entry['path'], 'node_types': dict(sorted(Counter(type(n).__name__ for n in nodes).items())),
                              'definitions': [{'name': n.name, 'line': n.lineno} for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                              'imports': sorted({alias.name for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names})}
                else:
                    payload = json.loads(data)
                    rows = payload if isinstance(payload, list) else [payload]
                    report = {'file': entry['path'], 'records': len(rows),
                              'key_frequency': dict(sorted(Counter(k for row in rows if isinstance(row, dict) for k in row).items())),
                              'records_excerpt': rows[:8]}
                reports.append(report)
        full = json.dumps({'operation': operation, 'files': reports}, sort_keys=True, ensure_ascii=False)
        text = full[:cap]
        return text, {'operation': operation, 'passes': passes, 'manifest_sha256': manifest['sha256'],
                      'input_bytes': total_bytes*passes, 'corpus_bytes': total_bytes,
                      'corpus_files': len(manifest['files']), 'untruncated_output_chars': len(full),
                      'output_truncated': len(full) > cap}
    return _measure(run)


def execute_context(spec):
    """Serialize real message/tool text; does not claim chat-template tokenization."""
    def run():
        messages = list(spec.get('messages', []))
        if 'tool_text' in spec:
            messages.append({'role': 'tool', 'content': spec['tool_text']})
        if not messages:
            raise ValueError('messages or tool_text required')
        input_bytes = 0
        pieces = []
        for message in messages:
            role, content = message['role'], message['content']
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError('roles/content must be strings')
            input_bytes += len(content.encode('utf-8'))
            if input_bytes > 16*1024*1024:
                raise ValueError('context exceeds 16MiB')
            pieces.append(json.dumps({'role': role, 'content': content}, ensure_ascii=False, sort_keys=True))
        return '\n'.join(pieces), {'operation': 'context', 'input_bytes': input_bytes, 'messages': len(messages)}
    return _measure(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', required=True)
    parser.parse_args()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get('action', 'tool')
            if action not in ('tool', 'context'):
                raise ValueError('unknown worker action')
            result = (execute_tool if action == 'tool' else execute_context)(request['spec'])
        except Exception as exc:
            result = {'success': False, 'text': '', 'error': f'{type(exc).__name__}: {exc}', 'metrics': {}}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
