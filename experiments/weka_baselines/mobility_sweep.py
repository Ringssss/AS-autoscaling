"""Repeated mobility-vs-recompute sweep over one engine pair.

One line = one (src, dst) pair. Non-overlapping lines can run concurrently;
overlapping pairs contend on the same engines and must not.

Per repetition and length: flush the source, confirm with inspect that the
prefix is genuinely cold, time prefill-to-first-token by streaming, flush the
destination, then migrate that same prefix and confirm on the destination that
a full runnable prefix landed. Nothing is recorded unless both gates pass.
"""
import argparse, json, statistics, threading, time, urllib.request

import numpy as np

ROOT = '/home/zhujianian/AS-autoscaling/experiments/weka_baselines'


def post(url, path, payload, timeout=1200):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            body = f.read()
        try:
            return 200, json.loads(body)
        except Exception:
            return 200, body[:300].decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:600].decode('utf-8', 'replace')
    except Exception as e:
        return None, repr(e)[:300]


def inspect_state(url, toks):
    st, r = post(url, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    if not isinstance(r, dict):
        return None, None, None
    return r.get('matched_tokens'), r.get('full_prefix_available'), r.get('has_mamba_state')


def flush_and_confirm_cold(url, toks, budget=8):
    """flush_cache silently no-ops while the engine is busy, so retry and verify."""
    for attempt in range(budget):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.5 + 1.5 * attempt)
        m, _, _ = inspect_state(url, toks)
        if m == 0:
            return True, attempt + 1
    return False, budget


def flush_and_confirm_empty(url, budget=8):
    """Destination must have no children in its radix tree for migration."""
    for attempt in range(budget):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.5 + 1.5 * attempt)
        m, _, _ = inspect_state(url, [int(x) for x in np.load(f'{ROOT}/data/tokens.npy')[:64]])
        if m == 0:
            return True, attempt + 1
    return False, budget


def prefill_to_first_token(url, toks, max_new=8):
    """Streaming: the first data chunk marks prefill completion, excluding decode."""
    body = json.dumps({'input_ids': toks, 'max_new_tokens': max_new,
                       'temperature': 0, 'stream': True}).encode()
    req = urllib.request.Request(url + '/generate', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    meta = {}
    # Drain the stream: the scheduler must be idle before a migration is allowed.
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
    return first, meta


def migrate(src, dst, toks, name, host, port):
    if src == dst:
        return {'error': 'src == dst'}
    res = {}

    def init_side(tag, url, rank):
        res[tag] = post(url, '/agentshift/hybrid_state/direct_group/init', {
            'group_rank': rank, 'group_name': name, 'master_address': host,
            'port': port, 'world_size': 2, 'backend': 'nccl'}, timeout=300)

    th = [threading.Thread(target=init_side, args=('src', src, 0)),
          threading.Thread(target=init_side, args=('dst', dst, 1))]
    [t.start() for t in th]; [t.join() for t in th]
    bad = {k: str(v[1])[:300] for k, v in res.items()
           if not (isinstance(v[1], dict) and v[1].get('success'))}
    if bad:
        return {'error': 'group_init', 'detail': bad}

    out = {}

    def move(tag, url, role):
        out[tag] = post(url, '/agentshift/hybrid_state/direct_transfer', {
            'group_name': name, 'role': role, 'port': port, 'token_ids': toks})

    th = [threading.Thread(target=move, args=('src', src, 'source')),
          threading.Thread(target=move, args=('dst', dst, 'destination'))]
    [t.start() for t in th]; [t.join() for t in th]
    return {k: v[1] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--src-host', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--lens', default='8192,16384,32768,65536,98304,130048')
    ap.add_argument('--seed-base', type=int, default=1_000_000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(',')]
    arr = np.load(f'{ROOT}/data/tokens.npy')
    kind = 'same-node' if args.src.split('/')[-1].split(':')[0] == args.src_host else 'cross-node'
    src_ip = args.src.split('//')[1].split(':')[0]
    dst_ip = args.dst.split('//')[1].split(':')[0]
    kind = 'same-node' if src_ip == dst_ip else 'cross-node'
    print(f"[{args.label}] {kind}  {args.src} -> {args.dst}  reps={args.reps}", flush=True)

    rows = []
    n_port = args.port
    for rep in range(args.reps):
        for i, n in enumerate(lens):
            # disjoint token regions: shift by a large stride each (rep, len)
            off = args.seed_base + (rep * 7 + i) * 262144
            if off + n > len(arr):
                continue
            toks = [int(x) for x in arr[off:off + n]]
            rec = dict(label=args.label, kind=kind, src=args.src, dst=args.dst,
                       rep=rep, n=n, offset=off)

            cold, att = flush_and_confirm_cold(args.src, toks)
            if not cold:
                rec['skipped'] = f'source never cold after {att} flushes'
                rows.append(rec)
                print(f"  [{args.label}] rep{rep} n={n} SKIP (source warm)", flush=True)
                continue
            rec['src_flush_attempts'] = att

            t0 = time.perf_counter()
            first, meta = prefill_to_first_token(args.src, toks)
            rec['prefill_s'] = first
            rec['prefill_s_per_token'] = (first / n) if first else None
            rec['prefill_e2e_meta'] = {k: meta.get(k) for k in ('prompt_tokens', 'e2e_latency')}

            ok, att2 = flush_and_confirm_empty(args.dst)
            rec['dst_flush_attempts'] = att2
            if not ok:
                rec['skipped'] = 'destination never empty'
                rows.append(rec)
                print(f"  [{args.label}] rep{rep} n={n} SKIP (dst not empty)", flush=True)
                continue

            out = migrate(args.src, args.dst, toks, f'{args.label}-r{rep}-n{n}',
                          args.src_host, n_port)
            n_port += 8
            if 'error' in out:
                rec['transfer_error'] = out
                rows.append(rec)
                print(f"  [{args.label}] rep{rep} n={n} XFER FAIL {str(out)[:160]}", flush=True)
                continue

            d = out.get('dst') if isinstance(out.get('dst'), dict) else None
            rec['transfer_s'] = d.get('transfer_seconds') if d else None
            rec['bytes_transferred'] = d.get('bytes_transferred') if d else None
            src = out.get('src') if isinstance(out.get('src'), dict) else None
            rec['src_reported_s'] = src.get('transfer_seconds') if src else None
            if not d or not rec['transfer_s']:
                rec['transfer_error'] = {k: str(v)[:300] for k, v in out.items()}
            if rec['transfer_s'] and rec['bytes_transferred']:
                rec['gb_per_s'] = rec['bytes_transferred'] / 2**30 / rec['transfer_s']

            m2, full2, mamba2 = inspect_state(args.dst, toks)
            rec['dst_matched_after'] = m2
            rec['dst_verified_runnable'] = bool(full2 and mamba2)

            if first and rec['transfer_s'] and rec['dst_verified_runnable']:
                rec['ratio'] = first / rec['transfer_s']
            rows.append(rec)
            print(f"  [{args.label}] rep{rep} n={n:>7} prefill={first if first else -1:>7.3f} "
                  f"xfer={rec['transfer_s'] if rec['transfer_s'] else -1:>6.3f} "
                  f"ratio={rec.get('ratio', 0):>6.2f} gbs={rec.get('gb_per_s', 0):>5.2f} "
                  f"verified={rec['dst_verified_runnable']}", flush=True)

    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)

    # summary: median per length, only over fully verified rows
    print(f"\n[{args.label}] {kind} medians:", flush=True)
    print(f"  {'n':>8}{'prefill_med':>13}{'xfer_med':>10}{'ratio_med':>11}{'gbs_med':>9}{'valid':>7}")
    for n in lens:
        v = [r for r in rows if r.get('n') == n and r.get('dst_verified_runnable')
             and r.get('prefill_s') and r.get('transfer_s')]
        if not v:
            print(f"  {n:>8}   no valid rows"); continue
        pf = statistics.median(r['prefill_s'] for r in v)
        xf = statistics.median(r['transfer_s'] for r in v)
        gb = statistics.median(r['gb_per_s'] for r in v)
        print(f"  {n:>8}{pf:>13.3f}{xf:>10.3f}{pf/xf:>11.2f}{gb:>9.2f}{len(v):>7}")
    print(f"wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
