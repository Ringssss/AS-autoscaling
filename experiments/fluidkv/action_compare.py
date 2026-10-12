"""P0-2: the three actions compared at one common endpoint.

The earlier exchange ratio divided a *client* cold TTFT by an *engine-side*
transfer phase. Those are different endpoints and different clocks, so the
ratio cannot be read as a serving speedup. This script fixes the endpoint.

Definitions (all t=0 at the moment the client issues the request, all end at
the first streamed output event):

    T_cold   = Q_dst + R_dst(n)                  request served on an engine
                                                 that lacks the prefix
    T_local  = Q_src + H_src(n)                  prefix already resident here
    T_import = Q_dst + C_ready(n) + H_dst_rem    prefix carried over first

`C_ready` is decomposed by re-issuing the transfer with timestamps around each
phase, so the paper can name init / ready / install separately instead of
quoting one undifferentiated number.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKENS = ROOT / 'experiments/weka_baselines/data/tokens.npy'


def post(url, path, payload, timeout=2400):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            body = f.read()
        try:
            return 200, json.loads(body)
        except Exception:
            return 200, body[:400].decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:600].decode('utf-8', 'replace')
    except Exception as e:
        return None, repr(e)[:300]


def inspect(url, toks):
    _, r = post(url, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    return r if isinstance(r, dict) else None


def wait_cold(url, budget=12):
    probe = [1, 2, 3, 4, 5, 6, 7, 8]
    for attempt in range(budget):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.2 + 1.3 * attempt)
        s = inspect(url, probe)
        if s and s.get('matched_tokens') == 0:
            return True, attempt + 1
    return False, budget


def wait_empty(url, budget=12, tolerance=64):
    """Destination must have no children: migration inserts into a fresh tree.

    Emptiness is judged on the cold-match predicate and on capacity being
    within `tolerance` of the engine's maximum. The tolerance exists because
    the allocator's free-slot reading can sit a few blocks below the configured
    maximum even on an idle engine; an exact equality test would stall forever
    on such an engine.
    """
    probe = [9, 10, 11, 12, 13, 14, 15, 16]
    for attempt in range(budget):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.2 + 1.3 * attempt)
        s = inspect(url, probe)
        if not s or s.get('matched_tokens') != 0:
            continue
        cap = s.get('max_token_slots') or s.get('available_token_slots')
        if s.get('available_token_slots', 0) >= cap - tolerance:
            return True, attempt + 1
    return False, budget


def ttft_stream(url, toks, max_new_tokens=16, timeout=2400):
    """First streamed event, then drain. Returns (seconds, meta)."""
    payload = {'input_ids': [int(t) for t in toks],
               'sampling_params': {'max_new_tokens': max_new_tokens},
               'stream': True}
    req = urllib.request.Request(
        url + '/generate', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    meta = {}
    with urllib.request.urlopen(req, timeout=timeout) as f:
        for raw in f:
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


def migrate_timed(src, dst, toks, name, host, port):
    """One transfer with per-phase timestamps.

    Returns wall-clock seconds for: group init (two-sided handshake), the
    transfer call itself, and the destination's own reported transfer_seconds.
    """
    res = {}

    def init_side(tag, url, rank):
        t = time.perf_counter()
        code, body = post(url, '/agentshift/hybrid_state/direct_group/init', {
            'group_rank': rank, 'group_name': name, 'master_address': host,
            'port': port, 'world_size': 2, 'backend': 'nccl'}, timeout=600)
        res[tag] = dict(seconds=time.perf_counter() - t, code=code, body=body)

    th = [threading.Thread(target=init_side, args=('src', src, 0)),
          threading.Thread(target=init_side, args=('dst', dst, 1))]
    [t.start() for t in th]; [t.join() for t in th]
    bad = {k: str(v['body'])[:300] for k, v in res.items()
           if not (isinstance(v['body'], dict) and v['body'].get('success'))}
    if bad:
        return dict(error='group_init', detail=bad,
                    init_src=res['src']['seconds'], init_dst=res['dst']['seconds'])

    out = {}

    def move(tag, url, role):
        t = time.perf_counter()
        code, body = post(url, '/agentshift/hybrid_state/direct_transfer',
                          {'group_name': name, 'role': role, 'port': port,
                           'token_ids': [int(x) for x in toks]}, timeout=2400)
        out[tag] = dict(seconds=time.perf_counter() - t, code=code, body=body)

    th = [threading.Thread(target=move, args=('src', src, 'source')),
          threading.Thread(target=move, args=('dst', dst, 'destination'))]
    [t.start() for t in th]; [t.join() for t in th]

    dst_body = out['dst']['body'] if isinstance(out['dst']['body'], dict) else {}
    # The transfer response carries success/bytes/state but NOT the prefix
    # predicates; those live in the inspect response. Reading them from the
    # transfer body always yields False and manufactures a false negative.
    probe = inspect(dst, toks)
    return dict(
        init_seconds=max(res['src']['seconds'], res['dst']['seconds']),
        xfer_wall_src=out['src']['seconds'],
        xfer_wall_dst=out['dst']['seconds'],
        transport_success=bool(dst_body.get('success')),
        engine_transfer_seconds=dst_body.get('transfer_seconds'),
        dst_verified=bool(probe and probe.get('full_prefix_available') and probe.get('has_mamba_state')),
        dst_matched_tokens=(probe or {}).get('matched_tokens'),
        bytes_transferred=dst_body.get('bytes_transferred'),
        src=out['src']['body'] if isinstance(out['src']['body'], dict) else str(out['src']['body'])[:200],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--src-host', required=True)
    ap.add_argument('--length', type=int, default=130048)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--seed-base', type=int, default=20_000_000,
                    help='token offset for the first repetition; keep this clear '
                         'of other experiments so prefix identity cannot collide')
    ap.add_argument('--base-port', type=int, default=29600)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ARR = np.load(TOKENS)
    N = args.length
    rows = []

    for rep in range(args.reps):
        off = args.seed_base + rep * (N + 8192)
        toks = [int(x) for x in ARR[off:off + N]]

        # --- T_cold : destination lacks the prefix ------------------------
        ok, _ = wait_cold(args.dst)
        if not ok:
            print(f'rep{rep} cold: dst flush failed', flush=True); continue
        cold_t, cold_meta = ttft_stream(args.dst, toks)
        print(f'rep{rep} cold       {cold_t:.3f}s', flush=True)

        # --- T_local : prefix already resident on the source --------------
        ok, _ = wait_cold(args.src)
        if not ok:
            print(f'rep{rep} local: src flush failed', flush=True); continue
        ttft_stream(args.src, toks)                 # materialise
        s = inspect(args.src, toks)
        if not (s and s['full_prefix_available']):
            print(f'rep{rep} local: prefix did not materialise', flush=True); continue
        local_t, _ = ttft_stream(args.src, toks)
        print(f'rep{rep} local      {local_t:.3f}s', flush=True)

        # --- T_import : carry it over, then serve from the destination ----
        ok, _ = wait_empty(args.dst)
        if not ok:
            print(f'rep{rep} import: dst not empty', flush=True); continue
        name = f'p02-{rep}-{int(time.time()) % 100000}'
        port = args.base_port + rep * 8
        mig = migrate_timed(args.src, args.dst, toks, name, args.src_host, port)
        if mig.get('error') or not mig.get('dst_verified'):
            print(f'rep{rep} import: transfer failed {str(mig)[:200]}', flush=True); continue
        s = inspect(args.dst, toks)
        if not (s and s['full_prefix_available'] and s['has_mamba_state']):
            print(f'rep{rep} import: dst not runnable', flush=True); continue
        import_t, _ = ttft_stream(args.dst, toks)
        print(f'rep{rep} import     {import_t:.3f}s  '
              f'(init {mig["init_seconds"]:.3f} + engine {mig["engine_transfer_seconds"]:.3f})',
              flush=True)

        rows.append(dict(rep=rep, n=N,
                         cold_ttft_s=cold_t, local_ttft_s=local_t, import_ttft_s=import_t,
                         init_seconds=mig['init_seconds'],
                         engine_transfer_seconds=mig['engine_transfer_seconds'],
                         bytes_transferred=mig['bytes_transferred'],
                         dst_verified=mig['dst_verified'],
                         cold_cached_tokens=cold_meta.get('cached_tokens')))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')

    print('\n=== same-endpoint comparison, n=%d ===' % N)
    for k, label in (('cold_ttft_s', 'cold (dst recompute)'),
                     ('local_ttft_s', 'local (resident)'),
                     ('import_ttft_s', 'import (moved)')):
        v = [r[k] for r in rows if r.get(k)]
        if v:
            print(f'  {label:24s} median {statistics.median(v):7.3f}s  n={len(v)}')
    if len(rows) >= 1 and rows[0].get('cold_ttft_s') and rows[0].get('import_ttft_s'):
        c = statistics.median(r['cold_ttft_s'] for r in rows if r.get('cold_ttft_s'))
        i = statistics.median(r['import_ttft_s'] for r in rows if r.get('import_ttft_s'))
        print(f'  cold/import at the same endpoint: {c / i:.2f}x')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
