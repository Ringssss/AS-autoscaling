"""Copy-based mobility vs recompute: a real A/B on the same prefix.

Moves a runnable hybrid prefix (KV blocks + conv state + ssm temporal state)
from a source engine to an empty destination engine over RCCL, then measures
what the destination gained: its prefix hit and its TTFT for that same request.
The control arm is the same request arriving cold at the same destination.

Both engines must be idle. Only zhujianian-* containers are touched.
"""
import argparse, json, time, urllib.request

import numpy as np

ROOT = '/home/zhujianian/AS-autoscaling/experiments/weka_baselines'


def post(url, path, payload, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body[:600]
    except Exception as e:
        return None, repr(e)[:400]


def inspect(url, tokens, digest=False):
    return post(url, '/agentshift/hybrid_state/inspect',
                {'token_ids': list(tokens), 'return_digest': digest})


def migrate(src, dst, tokens, name, src_host, port):
    """RCCL broadcast is collective: both ranks must be inside concurrently."""
    import threading
    results = {}

    def side(tag, url, rank, role):
        results[tag] = post(url, '/agentshift/hybrid_state/direct_group/init', {
            'group_rank': rank, 'group_name': name, 'master_address': src_host,
            'port': port, 'world_size': 2, 'backend': 'nccl'}, timeout=180)

    t = [threading.Thread(target=side, args=('src', src, 0, 'source')),
         threading.Thread(target=side, args=('dst', dst, 1, 'destination'))]
    [x.start() for x in t]; [x.join() for x in t]
    if any(r[1] is None or not (isinstance(r[1], dict) and r[1].get('success'))
           for r in results.values()):
        return {'group_init_failed': results}

    out = {}

    def move(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer', {
            'group_name': name, 'role': role, 'port': port,
            'token_ids': list(tokens)}, timeout=900)

    t = [threading.Thread(target=move, args=('src', src, 'source')),
         threading.Thread(target=move, args=('dst', dst, 'destination'))]
    [x.start() for x in t]; [x.join() for x in t]
    return out


def ttft(url, tokens, out_tokens=1):
    """End-to-end latency for this exact prefix.

    /generate is not SSE here: it returns one JSON object after decoding, so the
    client's first byte is the whole request. With out_tokens=1 that is
    prefill + one decode step, which is the same measure on both arms and is
    dominated by prefill for a 130K prefix.
    """
    body = json.dumps({'input_ids': list(tokens), 'max_new_tokens': out_tokens,
                       'temperature': 0}).encode()
    req = urllib.request.Request(url + '/generate', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as f:
            raw = f.read()
        return time.perf_counter() - t0, json.loads(raw)
    except urllib.error.HTTPError as e:
        return None, f'HTTP {e.code}: {e.read()[:300].decode("utf-8","replace")}'
    except Exception as e:
        return None, repr(e)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='http://12.11.3.29:31760')
    ap.add_argument('--dst', default='http://12.11.3.29:31761')
    ap.add_argument('--src-host', default='12.11.3.29')
    ap.add_argument('--port', type=int, default=29700)
    ap.add_argument('--tokens', type=int, default=130048)
    ap.add_argument('--offset', type=int, default=17867264)
    ap.add_argument('--tag', default='ab')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    arr = np.load(f'{ROOT}/data/tokens.npy')
    tokens = [int(x) for x in arr[args.offset:args.offset + args.tokens]]
    print(f"prefix: {len(tokens)} tokens  offset={args.offset}", flush=True)

    rec = {'tag': args.tag, 'src': args.src, 'dst': args.dst,
           'n_tokens': len(tokens), 'offset': args.offset}

    # --- 0. both idle, both empty
    for tag, url in (('src', args.src), ('dst', args.dst)):
        st, r = post(url, '/flush_cache', {})
        rec[f'{tag}_flush'] = (st, str(r)[:120])

    # --- 1. warm the SOURCE, measure what that costs (this is the recompute arm)
    t0 = time.perf_counter()
    got, _ = ttft(args.src, tokens)
    rec['source_prefill_s'] = time.perf_counter() - t0 if got else None
    rec['source_ttft_s'] = got
    print(f"[1] source cold prefill for {len(tokens)} tok: {rec['source_prefill_s']}", flush=True)

    st, r = inspect(args.src, tokens)
    if isinstance(r, dict):
        rec['src_inspect'] = {k: r.get(k) for k in
                              ('matched_tokens', 'full_prefix_available', 'has_mamba_state',
                               'snapshot_boundaries', 'available_token_slots',
                               'available_snapshot_slots')}
        if rec['src_inspect'].get('snapshot_boundaries'):
            rec['src_inspect']['snapshot_boundaries'] = rec['src_inspect']['snapshot_boundaries'][-3:]
        rec['src_layout_bytes'] = sum(
            int(np.prod(b['shape'])) * (2 if b['dtype'].endswith('bf16') else 4)
            for b in r.get('layout', []))
    print(f"[2] source inspect: {rec.get('src_inspect')}", flush=True)
    print(f"    state bytes at this prefix: {rec.get('src_layout_bytes')}", flush=True)

    # --- 2. the migrate arm
    name = f'{args.tag}-{int(time.time())}'
    t0 = time.perf_counter()
    out = migrate(args.src, args.dst, tokens, name, args.src_host, args.port)
    rec['migrate_wall_s'] = time.perf_counter() - t0
    rec['migrate'] = out
    print(f"[3] migrate wall: {rec['migrate_wall_s']:.3f}s", flush=True)
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"    {k}: success={v.get('success')} "
                  f"bytes={v.get('bytes_transferred')} xfer_s={v.get('transfer_seconds')} "
                  f"transport={v.get('transport')} msg={str(v.get('message'))[:120]}", flush=True)
        else:
            print(f"    {k}: {str(v)[:200]}", flush=True)

    # --- 3. did the destination actually gain a runnable prefix?
    st, r = inspect(args.dst, tokens)
    if isinstance(r, dict):
        rec['dst_inspect'] = {k: r.get(k) for k in
                              ('matched_tokens', 'full_prefix_available', 'has_mamba_state')}
    print(f"[4] destination inspect: {rec.get('dst_inspect')}", flush=True)

    # --- 4. the payoff: TTFT at the destination with the copied prefix
    got, _ = ttft(args.dst, tokens)
    rec['dst_ttft_with_copied_state_s'] = got
    print(f"[5] dst TTFT with copied state: {got}", flush=True)

    # --- 5. the control arm: same prefix cold at the destination
    post(args.dst, '/flush_cache', {})
    got, _ = ttft(args.dst, tokens)
    rec['dst_ttft_cold_s'] = got
    print(f"[6] dst TTFT cold (recompute arm): {got}", flush=True)

    if rec.get('dst_ttft_cold_s') and rec.get('migrate_wall_s'):
        rec['speedup_vs_recompute'] = rec['dst_ttft_cold_s'] / rec['migrate_wall_s']
        print(f"\n>>> migrate {rec['migrate_wall_s']:.3f}s vs recompute "
              f"{rec['dst_ttft_cold_s']:.3f}s  =  {rec['speedup_vs_recompute']:.1f}x", flush=True)

    outp = args.out or f'{ROOT}/results/real-20260929/migrate-{args.tag}.json'
    with open(outp, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {outp}", flush=True)


if __name__ == '__main__':
    main()
