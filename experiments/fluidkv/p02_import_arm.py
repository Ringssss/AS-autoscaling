"""Third repetition, import arm only.

cold (28.250s) and local (1.454s) for this pair were already recorded before the
script aborted on a KeyError in its init-check helper. This version checks the
init response explicitly and reports each phase, so the import-to-first-token
number lands even if a port collides.

The prior run's failure was a bug in the checker, not in the engine: the helper
indexed a dict by 1 instead of pulling `success` off each response body.
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
ARR = np.load('experiments/weka_baselines/data/tokens.npy')
N = 130048
PORTS = (44016, 44024, 44032, 44040, 44048)


def post(url, path, payload, timeout=1200):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            body = fh.read()
        return time.perf_counter() - t0, json.loads(body)
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, {
            'http': exc.code, 'body': exc.read().decode('utf-8', 'replace')[:250]}
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


def init_ok(resp):
    _, body = resp
    return isinstance(body, dict) and body.get('success') is True


def main():
    toks = [int(x) for x in ARR[36_000_000:36_000_000 + N]]
    print(f'flush src={flush(SRC)} dst={flush(DST)}', flush=True)
    serve(SRC, toks)
    st = inspect(SRC, toks)
    print(f"SRC matched={st.get('matched_tokens')} "
          f"full={st.get('full_prefix_available')}", flush=True)

    for port in PORTS:
        name = f'r3b-{port}-{int(time.time()) % 100000}'
        r_src = post(SRC, '/agentshift/hybrid_state/direct_group/init',
                     {'group_rank': 0, 'group_name': name,
                      'master_address': HOST, 'port': port,
                      'world_size': 2, 'backend': 'nccl'}, 600)
        r_dst = post(DST, '/agentshift/hybrid_state/direct_group/init',
                     {'group_rank': 1, 'group_name': name,
                      'master_address': HOST, 'port': port,
                      'world_size': 2, 'backend': 'nccl'}, 600)
        good = init_ok(r_src) and init_ok(r_dst)
        print(f'  init port{port}: src {r_src[0]:.1f}s dst {r_dst[0]:.1f}s ok={good}',
              flush=True)
        if not good:
            print(f'     src={str(r_src[1])[:170]}', flush=True)
            print(f'     dst={str(r_dst[1])[:170]}', flush=True)
            continue

        out = {}

        def mv(tag, url, role):
            out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer',
                            {'group_name': name, 'role': role, 'port': port,
                             'token_ids': toks}, 1200)

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

        t_import, meta = serve(DST, toks)
        d = inspect(DST, toks)
        nb = len(d.get('snapshot_boundaries') or [])
        print(f"  import(dst) {t_import if t_import else -1:.3f}s  "
              f"full={d.get('full_prefix_available')} bounds={nb} "
              f"cached_tokens={meta.get('cached_tokens') or 0}", flush=True)
        break


if __name__ == '__main__':
    main()
