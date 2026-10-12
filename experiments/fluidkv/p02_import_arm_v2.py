"""Third repetition of the same-endpoint comparison, import arm.

Why this retry exists: four earlier attempts failed at group init, not at the
transfer. The port arithmetic in init_group is

    port = base + tp_rank                (used by the process group)
    store  = TCPStore(master, base + tp_rank + tp_size, ...)

so with TP=2 one group occupies base..base+3 on BOTH hosts. Binding only `base`
is therefore not a sufficient availability test, which is what earlier probes did.

This run uses 44308-44311, verified free on both hosts with a real bind test.
The pair is cross-node (27:31781 -> 29:31761), which is the topology the
historically successful sweep line D used, and both engines are currently clean.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

SRC = 'http://12.11.3.27:31781'
DST = 'http://12.11.3.29:31761'
SRC_HOST = '12.11.3.27'
BASE_PORT = 44308          # base..base+3 verified free on both hosts
N = 130048
ARR = np.load('experiments/weka_baselines/data/tokens.npy')


def post(url, path, payload, timeout=1200):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return time.perf_counter() - t0, json.loads(fh.read())
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, {
            'http': exc.code, 'b': exc.read().decode('utf-8', 'replace')[:220]}
    except Exception as exc:
        return time.perf_counter() - t0, {'exc': repr(exc)[:140]}


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
    with urllib.request.urlopen(req, timeout=1200) as fh:
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


def ok(resp):
    return isinstance(resp[1], dict) and resp[1].get('success') is True


def main():
    # disjoint from every other experiment's range (36M used by the last attempt)
    toks = [int(x) for x in ARR[44_000_000:44_000_000 + N]]

    print(f'flush src={flush(SRC)} dst={flush(DST)}', flush=True)

    # cold reference on the destination
    tc, _ = serve(DST, toks)
    print(f'cold(dst)   {(tc if tc else -1):8.3f}s', flush=True)

    # local reference on the source
    flush(SRC)
    serve(SRC, toks)
    tl, _ = serve(SRC, toks)
    print(f'local(src)  {(tl if tl else -1):8.3f}s', flush=True)

    st = inspect(SRC, toks)
    print(f"SRC matched={st.get('matched_tokens')} "
          f"full={st.get('full_prefix_available')}", flush=True)

    # import onto the destination
    flush(DST)
    name = f'r3c-{int(time.time()) % 100000}'
    r_src = post(SRC, '/agentshift/hybrid_state/direct_group/init',
                 {'group_rank': 0, 'group_name': name,
                  'master_address': SRC_HOST, 'port': BASE_PORT,
                  'world_size': 2, 'backend': 'nccl'}, 300)
    r_dst = post(DST, '/agentshift/hybrid_state/direct_group/init',
                 {'group_rank': 1, 'group_name': name,
                  'master_address': SRC_HOST, 'port': BASE_PORT,
                  'world_size': 2, 'backend': 'nccl'}, 300)
    print(f'  init base{BASE_PORT}: src {r_src[0]:6.1f}s dst {r_dst[0]:6.1f}s '
          f'ok={ok(r_src) and ok(r_dst)}', flush=True)
    if not (ok(r_src) and ok(r_dst)):
        print(f'   src: {str(r_src[1])[:200]}', flush=True)
        print(f'   dst: {str(r_dst[1])[:200]}', flush=True)
        return

    out = {}

    def mv(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer',
                        {'group_name': name, 'role': role,
                         'port': BASE_PORT, 'token_ids': toks}, 1200)

    threads = [threading.Thread(target=mv, args=('src', SRC, 'source')),
               threading.Thread(target=mv, args=('dst', DST, 'destination'))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'  transfer {time.perf_counter() - t0:.2f}s', flush=True)
    for tag, (wall, body) in out.items():
        state = body.get('state') if isinstance(body, dict) else None
        byts = body.get('bytes_transferred') if isinstance(body, dict) else None
        print(f'     {tag}: {wall:.2f}s state={state} bytes={byts}', flush=True)

    d = inspect(DST, toks)
    nb = len(d.get('snapshot_boundaries') or [])
    print(f"DST published: full={d.get('full_prefix_available')} "
          f"mamba={d.get('has_mamba_state')} bounds={nb}", flush=True)

    ti, mi = serve(DST, toks)
    print(f'import(dst) {(ti if ti else -1):8.3f}s  '
          f'cached_tokens={mi.get("cached_tokens") or 0}', flush=True)

    print('\n=== same-endpoint, third repetition ===', flush=True)
    for label, val in (('cold', tc), ('local', tl), ('import', ti)):
        print(f'  {label:7s} {(val if val else float("nan")):8.3f}s', flush=True)
    if tc and ti:
        print(f'  cold/import = {tc / ti:.2f}x', flush=True)


if __name__ == '__main__':
    main()
