"""Verify the chained-import fix.

The defect this tests: the import path wrote ONE radix node with a single
end snapshot, so the request path's one-token-short match key forced a split
that orphaned the only snapshot and produced a full prefill (cached_tokens=0)
on the first consumer. The fix writes a chain, one node per source boundary.

Three assertions, in order of strength:

  A. the imported tree reports the SAME boundary count as the source
     (expect 8 at 32K, 32 at 130K; previously 1)
  B. the first serve on the destination reuses a large prefix
     (expect ~n-64 or better; previously 0)
  C. output equivalence: destination continuation matches source continuation

Run against a refreshed pair. Cross-node (27 -> 28) so the two engines are
distinct processes on distinct cards.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

SRC = 'http://12.11.3.27:31791'
DST = 'http://12.11.3.28:31791'
HOST = '12.11.3.27'
BASE = 48040          # bind-verified free on node27 AND node28
N = 32768


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
            'http': exc.code, 'b': exc.read().decode('utf-8', 'replace')[:300]}
    except Exception as exc:
        return time.perf_counter() - t0, {'exc': repr(exc)[:150]}


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


def serve(url, toks, max_new=16):
    req = urllib.request.Request(
        url + '/generate',
        data=json.dumps({'input_ids': [int(x) for x in toks],
                         'sampling_params': {'max_new_tokens': max_new},
                         'stream': True}).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    meta = {}
    outs = []
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
            if obj.get('output_ids'):
                outs.append(list(obj['output_ids']))
            if (obj.get('text') or obj.get('output_ids')) and first is None:
                first = time.perf_counter() - t0
    return first, meta, outs


def main():
    ARR = np.load('experiments/weka_baselines/data/tokens.npy')
    toks = [int(x) for x in ARR[7_000_000:7_000_000 + N]]

    ok_s, ok_d = flush(SRC), flush(DST)
    print(f'flush src={ok_s} dst={ok_d}', flush=True)

    # ---- reference: native on the source ----
    serve(SRC, toks)
    s_ref = inspect(SRC, toks)
    n_ref = len(s_ref.get('snapshot_boundaries') or [])
    print(f'SRC native: boundaries={n_ref} full={s_ref.get("full_prefix_available")}',
          flush=True)

    # ---- import ----
    # Issue BOTH sides CONCURRENTLY. TCPStore's server waits 90 s for its peer,
    # and a client that fails its first attempt backs off ~66 s before retrying,
    # so a sequential pair of calls cannot complete: the source's store times
    # out ("1/2 clients joined") before the destination's client gets a second
    # chance. Every working sweep script issues these two calls from threads.
    name = f'chain-{int(time.time()) % 100000}'
    init_out = {}

    def do_init(tag, url, rank):
        init_out[tag] = post(url, '/agentshift/hybrid_state/direct_group/init',
                             {'group_rank': rank, 'group_name': name,
                              'master_address': HOST, 'port': BASE,
                              'world_size': 2, 'backend': 'nccl'}, 300)

    th_init = [threading.Thread(target=do_init, args=('src', SRC, 0)),
               threading.Thread(target=do_init, args=('dst', DST, 1))]
    for t in th_init:
        t.start()
    for t in th_init:
        t.join()
    r1, r2 = init_out['src'], init_out['dst']
    ok = all(isinstance(v[1], dict) and v[1].get('success') is True
             for v in (r1, r2))
    print(f'  init base{BASE}: src {r1[0]:.1f}s dst {r2[0]:.1f}s ok={ok}', flush=True)
    if not ok:
        print(f'   src={str(r1[1])[:220]}', flush=True)
        print(f'   dst={str(r2[1])[:220]}', flush=True)
        return

    out = {}

    def mv(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer',
                        {'group_name': name, 'role': role, 'port': BASE,
                         'token_ids': toks}, 900)

    th = [threading.Thread(target=mv, args=('src', SRC, 'source')),
          threading.Thread(target=mv, args=('dst', DST, 'destination'))]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f'  transfer {time.perf_counter() - t0:.2f}s', flush=True)
    for tag, (wall, body) in out.items():
        st = body.get('state') if isinstance(body, dict) else None
        by = body.get('bytes_transferred') if isinstance(body, dict) else None
        msg = str(body.get('message', ''))[:150] if isinstance(body, dict) else str(body)[:150]
        print(f'     {tag}: {wall:.2f}s state={st} bytes={by} {msg}', flush=True)

    # ---- ASSERTION A: boundary count matches ----
    s_imp = inspect(DST, toks)
    n_imp = len(s_imp.get('snapshot_boundaries') or [])
    print(f'\n[A] imported boundaries = {n_imp}  (native {n_ref}, was 1 before fix)',
          flush=True)
    print(f'    full={s_imp.get("full_prefix_available")} '
          f'mamba={s_imp.get("has_mamba_state")}', flush=True)
    print(f'    -> {"PASS" if n_imp == n_ref and n_imp > 1 else "FAIL"}', flush=True)

    # ---- ASSERTION B: first consumer reuses ----
    t1, m1, o1 = serve(DST, toks)
    c1 = m1.get('cached_tokens') or 0
    print(f'\n[B] first serve on destination: {t1 if t1 else -1:.3f}s '
          f'cached_tokens={c1}', flush=True)
    print(f'    -> {"PASS" if c1 > 0 else "FAIL"}  (was 0 before fix)', flush=True)

    t2, m2, _ = serve(DST, toks)
    c2 = m2.get('cached_tokens') or 0
    print(f'    second serve: {t2 if t2 else -1:.3f}s cached_tokens={c2}', flush=True)

    # ---- ASSERTION C: output equivalence ----
    ts, ms, osrc = serve(SRC, toks)
    same = bool(osrc and o1 and osrc[-1] == o1[-1])
    print(f'\n[C] destination continuation == source continuation: {same}', flush=True)
    if osrc and o1:
        print(f'    src ids: {osrc[-1][:8]}', flush=True)
        print(f'    dst ids: {o1[-1][:8]}', flush=True)

    print('\n=== SUMMARY ===', flush=True)
    print(f'  boundaries  native={n_ref}  imported={n_imp}', flush=True)
    print(f'  first-serve cached_tokens: before fix = 0, now = {c1}', flush=True)


if __name__ == '__main__':
    main()
