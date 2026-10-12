"""Two discriminators in one migration.

D1 (TP-intersection): read the per-rank snapshot_boundaries of the IMPORTED tree.
    Native reports 8 on both ranks at 32K, so an intersection cannot collapse
    anything there. If the import shows >1 on any rank while the top-level value
    is 1, the "single node" premise is a reporting artifact, not a tree shape.

D2 (granularity vs node count): serve a request of n+1 tokens after the import,
    so the truncated match key is exactly n. The two competing stories predict
    different outcomes:
      - node-count story: chain present -> reuse lands high (near n)
      - alignment story: reuse lands at floor((n-1)/64)*64 = n-64 regardless
    Also serve plain n to reproduce the known first-consumer miss.

Runs on 27:31781 -> 27:31782 with a bind-verified port span.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

SRC = 'http://12.11.3.27:31781'
DST = 'http://12.11.3.27:31782'
HOST = '12.11.3.27'
BASE = 44600
N = 32768
ARR = np.load('experiments/weka_baselines/data/tokens.npy')


def post(url, path, payload, timeout=900):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return time.perf_counter() - t0, json.loads(fh.read())
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, {
            'http': exc.code, 'b': exc.read().decode('utf-8', 'replace')[:200]}
    except Exception as exc:
        return time.perf_counter() - t0, {'exc': repr(exc)[:130]}


def inspect(url, toks):
    _, r = post(url, '/agentshift/hybrid_state/inspect',
                {'token_ids': [int(x) for x in toks]})
    return r if isinstance(r, dict) else {}


def flush(url):
    for i in range(14):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.2 + 1.0 * i)
        s = inspect(url, [1, 2, 3, 4])
        if s.get('matched_tokens') == 0 and s.get('evictable_token_slots') == 0:
            return True
    return False


def serve(url, toks, max_new=8):
    req = urllib.request.Request(
        url + '/generate',
        data=json.dumps({'input_ids': [int(x) for x in toks],
                         'sampling_params': {'max_new_tokens': max_new},
                         'stream': True}).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    meta = {}
    with urllib.request.urlopen(req, timeout=900) as fh:
        for raw in fh:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            body = line[5:].strip()
            if not body or body == '[DONE]':
                continue
            try:
                obj = json.loads(body)
            except Exception:
                continue
            if obj.get('meta_info'):
                meta = obj['meta_info']
            if (obj.get('text') or obj.get('output_ids')) and first is None:
                first = time.perf_counter() - t0
    return first, meta


def per_rank(url, toks, tag):
    s = inspect(url, toks)
    top = s.get('snapshot_boundaries') or []
    print(f'  {tag}: top-level boundaries = {len(top)}  {top[:6]}', flush=True)
    tp = s.get('tp_ranks')
    if isinstance(tp, list):
        for i, r in enumerate(tp):
            b = r.get('snapshot_boundaries') or []
            print(f'    rank{i}: {len(b)}  {b[:6]}', flush=True)


def main():
    base_tokens = [int(x) for x in ARR[5_000_000:5_000_000 + N]]
    print('=== flush ===', flush(SRC), flush(DST), flush=True)

    # source materialises natively (this gives the reference boundary count)
    serve(SRC, base_tokens)
    per_rank(SRC, base_tokens, 'SRC native')

    print('=== migrate ===', flush=True)
    name = f'disc-{int(time.time()) % 100000}'
    r1 = post(SRC, '/agentshift/hybrid_state/direct_group/init',
              {'group_rank': 0, 'group_name': name, 'master_address': HOST,
               'port': BASE, 'world_size': 2, 'backend': 'nccl'}, 300)
    r2 = post(DST, '/agentshift/hybrid_state/direct_group/init',
              {'group_rank': 1, 'group_name': name, 'master_address': HOST,
               'port': BASE, 'world_size': 2, 'backend': 'nccl'}, 300)
    ok = all(isinstance(v[1], dict) and v[1].get('success') is True
             for v in (r1, r2))
    print(f'  init base{BASE}: src {r1[0]:.1f}s dst {r2[0]:.1f}s ok={ok}', flush=True)
    if not ok:
        print(f'   src={str(r1[1])[:180]}', flush=True)
        print(f'   dst={str(r2[1])[:180]}', flush=True)
        return

    out = {}

    def mv(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer',
                        {'group_name': name, 'role': role, 'port': BASE,
                         'token_ids': base_tokens}, 900)

    th = [threading.Thread(target=mv, args=('src', SRC, 'source')),
          threading.Thread(target=mv, args=('dst', DST, 'destination'))]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f'  transfer {time.perf_counter() - t0:.2f}s '
          f'states={[v[1].get("state") if isinstance(v[1], dict) else None for v in out.values()]}',
          flush=True)

    # ---------- D1: per-rank boundaries of the IMPORTED tree ----------
    print('=== D1: imported tree, per rank ===', flush=True)
    per_rank(DST, base_tokens, 'DST imported (before any serve)')

    # ---------- D2: serve n, then n+1 ----------
    print('=== D2: serve n tokens (known miss) ===', flush=True)
    t_n, m_n = serve(DST, base_tokens)
    print(f'  serve(n={N}): {t_n if t_n else -1:.2f}s '
          f'cached_tokens={m_n.get("cached_tokens") or 0}', flush=True)

    print('=== D2: serve n+1 tokens (truncated key == n) ===', flush=True)
    longer = [int(x) for x in ARR[5_000_000:5_000_000 + N + 1]]
    t_p, m_p = serve(DST, longer)
    print(f'  serve(n+1={N + 1}): {t_p if t_p else -1:.2f}s '
          f'cached_tokens={m_p.get("cached_tokens") or 0}', flush=True)
    print(f'  -> n-64 = {N - 64};  n = {N}', flush=True)
    print('  alignment story predicts cached_tokens = n-64 on serve(n);', flush=True)
    print('  node-count story predicts a large value on BOTH serves.', flush=True)


if __name__ == '__main__':
    main()
