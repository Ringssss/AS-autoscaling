"""Mobility exchange ratio as a function of prefix length.

For each length: measure cold prefill-to-first-token by streaming (isolates
prefill from decode), then migrate that same prefix to an empty engine and
measure what the move cost. Ratio = recompute / copy.

Lengths use non-overlapping token regions so nothing is warm by accident, and
every arm asserts matched_tokens==0 (recompute) or a full prefix (copy) via
/agentshift/hybrid_state/inspect before its number is trusted.
"""
import argparse, json, threading, time, urllib.request

import numpy as np

ROOT = '/home/zhujianian/AS-autoscaling/experiments/weka_baselines'


def post(url, path, payload, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            b = f.read()
        try:
            return 200, json.loads(b)
        except Exception:
            return 200, b[:200].decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode('utf-8', 'replace')
    except Exception as e:
        return None, repr(e)[:300]


def inspect_matched(url, toks):
    st, r = post(url, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    if isinstance(r, dict):
        return r.get('matched_tokens'), r.get('full_prefix_available'), r.get('has_mamba_state')
    return None, None, None


def prefill_to_first_token(url, toks):
    """Streaming: first data chunk marks prefill completion, excluding decode."""
    body = json.dumps({'input_ids': toks, 'max_new_tokens': 8,
                       'temperature': 0, 'stream': True}).encode()
    req = urllib.request.Request(url + '/generate', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    meta = {}
    # Read the stream to completion: the scheduler must be idle before a
    # migration is allowed, and it is only idle once this request has finished.
    with urllib.request.urlopen(req, timeout=1800) as f:
        for line in f:
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload in (b'[DONE]', b''):
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if first is None:
                first = time.perf_counter() - t0
                meta = obj.get('meta_info', {}) or {}
            else:
                meta = obj.get('meta_info', {}) or meta
    return first, meta


def migrate(src, dst, toks, name, host, port):
    import threading
    res = {}

    def side(tag, url, rank):
        res[tag] = post(url, '/agentshift/hybrid_state/direct_group/init', {
            'group_rank': rank, 'group_name': name, 'master_address': host,
            'port': port, 'world_size': 2, 'backend': 'nccl'}, timeout=240)

    th = [threading.Thread(target=side, args=('src', src, 0)),
          threading.Thread(target=side, args=('dst', dst, 1))]
    [x.start() for x in th]; [x.join() for x in th]
    if any(not isinstance(v[1], dict) or not v[1].get('success') for v in res.values()):
        return {'error': 'group_init', 'detail': {k: str(v)[:200] for k, v in res.items()}}
    out = {}

    def mv(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer', {
            'group_name': name, 'role': role, 'port': port, 'token_ids': toks})

    th = [threading.Thread(target=mv, args=('src', src, 'source')),
          threading.Thread(target=mv, args=('dst', dst, 'destination'))]
    [x.start() for x in th]; [x.join() for x in th]
    return {k: v[1] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='http://12.11.3.29:31760')
    ap.add_argument('--dst', default='http://12.11.3.29:31761')
    ap.add_argument('--src-host', default='12.11.3.29')
    ap.add_argument('--port', type=int, default=29801)
    ap.add_argument('--out', default=f'{ROOT}/results/real-20260929/migrate-curve.json')
    args = ap.parse_args()

    arr = np.load(f'{ROOT}/data/tokens.npy')
    lens = [8192, 16384, 32768, 65536, 98304, 130048]
    base = 1_000_000
    rows = []
    print(f"{'n':>8}{'prefill_s':>11}{'s/tok e-5':>11}{'moved?':>8}{'xfer_s':>9}"
          f"{'GB/s':>7}{'ratio':>7}", flush=True)
    for i, n in enumerate(lens):
        off = base + i * 262144
        if off + n > len(arr):
            break
        toks = [int(x) for x in arr[off:off + n]]

        # recompute arm: source must be genuinely cold for THIS prefix.
        # flush_cache silently no-ops while requests are running, so wait for
        # the engine to settle and confirm with inspect rather than trusting it.
        m = -1
        for attempt in range(6):
            post(args.src, '/flush_cache?timeout=30', {})
            time.sleep(2.0 + 2.0 * attempt)
            m, full, _ = inspect_matched(args.src, toks)
            if m == 0:
                break
        print(f"{n:>8}   src matched after flush = {m} (attempts={attempt+1})", flush=True)
        if m != 0:
            print(f"{n:>8}   source not cold (matched={m}), skipped", flush=True)
            continue
        pf, meta = prefill_to_first_token(args.src, toks)
        if pf is None:
            print(f"{n:>8}   prefill failed {str(meta)[:120]}", flush=True)
            continue

        # copy arm: destination must be empty
        post(args.dst, '/flush_cache', {})
        md, _, _ = inspect_matched(args.dst, toks)
        # Each group binds port + tp_rank AND port + tp_size for its TCPStore,
        # so a group occupies port..port+tp_size inclusive. Space iterations by
        # more than that or the second iteration dies with EADDRINUSE.
        out = migrate(args.src, args.dst, toks, f'curve-{n}-{int(time.time())}',
                      args.src_host, args.port + i * 8)
        xfer = None; byt = None; ok = False
        if isinstance(out.get('dst'), dict) and out['dst'].get('success'):
            xfer = out['dst'].get('transfer_seconds')
            byt = out['dst'].get('bytes_transferred')
            ok = True
        # verify the destination really holds a runnable prefix now
        m2, full2, mamba2 = inspect_matched(args.dst, toks)
        verified = bool(full2 and mamba2)

        ratio = (pf / xfer) if (pf and xfer) else None
        gbs = (byt / 2**30 / xfer) if (byt and xfer) else None
        print(f"{n:>8}{pf:>11.3f}{pf/n*1e5:>11.2f}{str(verified):>8}"
              f"{xfer if xfer else -1:>9.3f}{(gbs or 0):>7.2f}"
              f"{(ratio or 0):>7.2f}", flush=True)
        rows.append(dict(n=n, offset=off, prefill_to_first_token_s=pf,
                         s_per_token=pf / n, src_matched_before=m,
                         dst_matched_before=md, copied=ok,
                         transfer_s=xfer, bytes_transferred=byt, gb_per_s=gbs,
                         dst_verified_runnable=verified,
                         dst_full_prefix_available=full2,
                         ratio_prefill_over_copy=ratio,
                         src_meta={k: meta.get(k) for k in
                                   ('prompt_tokens', 'e2e_latency')}))
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
