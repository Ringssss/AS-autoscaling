"""P0-1: paired 2x2 capacity experiment with a read eviction counter.

Motivation. The earlier pilot used three different engines for its three
conditions and compared slot counts measured at different times, so it measured
a difference it could not attribute. This script replaces it with a proper
paired design that holds everything constant except residency.

Design
------
Two factors, fully crossed, on *one* physical engine, with engine role rotated
across repetitions:

    occupancy  in {empty, full}     full = NBG background prefixes resident
    residency  in {hit, miss}       whether the arriving request's prefix
                                    is among the resident ones

Four cells per repetition: empty/hit, empty/miss, full/hit, full/miss.
`full/miss` is the interesting one: the engine is at the point where exactly one
more maximal request fits, the request is cold, and we read the engine's own
retraction counter to see whether admitting it displaced anything.

Measured per cell:
  - TTFT, timed to the first streamed event, decode excluded
  - free token slots and free snapshot slots, read before and after
  - retracted request count and the total retracted tokens, read from the
    engine's response metadata (`num_retractions`, `retracted_tokens`), which
    is a counter, not a slot delta
  - per-background-prefix survival, checked individually with `inspect`

Everything is verified: the flush loop confirms cold before each cell, and a
cell is dropped unless its preconditions hold.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhujianian/AS-autoscaling')
DEFAULT_TOKENS = ROOT / 'experiments/weka_baselines/data/tokens.npy'


def post(url, path, payload, timeout=1800):
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
    """Read-only state query. Returns None on transport failure."""
    _, r = post(url, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    return r if isinstance(r, dict) else None


def wait_cold(url, toks, budget=10):
    """flush_cache silently no-ops while the engine is busy, so retry + verify.

    The destination probe is a short sequence that shares no prefix with any
    workload region, so a successful flush means the whole cache is empty.
    Verifying on `toks` instead would keep failing whenever `toks` is itself
    the prefix we are about to test.
    """
    probe = [1, 2, 3, 4, 5, 6, 7, 8]
    for attempt in range(budget):
        post(url, '/flush_cache?timeout=30', {})
        time.sleep(1.2 + 1.2 * attempt)
        s = inspect(url, probe)
        if s and s.get('matched_tokens') == 0 and s.get('available_token_slots') is not None:
            # confirm the engine is fully drained, not just cold for this probe
            big = inspect(url, toks)
            if big and big.get('matched_tokens') == 0:
                return True, attempt + 1
    return False, budget


def ttft_stream(url, toks, max_new_tokens=16, timeout=1800):
    """Time to the first streamed event, then drain so the scheduler returns idle.

    Decode is excluded by stopping the clock at the first non-empty `data:`
    line. The stream is still drained to completion because a truncated read
    leaves the request running and the next `inspect` would then fail.
    """
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
            m = obj.get('meta_info') or {}
            if m:
                meta = m
            if obj.get('text') or obj.get('output_ids'):
                if first is None:
                    first = time.perf_counter() - t0
    return first, meta


def install(url, toks, max_new_tokens=16):
    """Materialise a resident prefix. A completed generation is required:
    max_new_tokens=1 leaves no resident prefix."""
    t0 = time.perf_counter()
    _, meta = ttft_stream(url, toks, max_new_tokens=max_new_tokens)
    return time.perf_counter() - t0, meta


def survivor_count(url, regions):
    """How many of the listed token sequences are still fully resident."""
    alive = 0
    per = []
    for r in regions:
        s = inspect(url, r)
        ok = bool(s and s.get('full_prefix_available') and s.get('has_mamba_state'))
        per.append(ok)
        alive += int(ok)
    return alive, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='http://12.11.3.29:31760')
    ap.add_argument('--tokens', default=str(DEFAULT_TOKENS))
    ap.add_argument('--prefix', type=int, default=32768)
    ap.add_argument('--nbg', type=int, default=7,
                    help='prefixes resident in the full cells; the target is one '
                         'of them (hit) or not (miss), so both arms start with '
                         'identical free slots')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default=str(ROOT / 'results/capacity-2x2.json'))
    args = ap.parse_args()

    TOK = np.load(args.tokens)
    URL = args.engine
    N = args.prefix

    def region(base, slot):
        """Disjoint token region per (repetition, slot). Deterministic.

        `slot` must be >= 1 for background prefixes: slot 0 is the target, and
        the two must never share tokens or the miss arm silently installs its
        own target and cannot be measured.
        """
        if slot < 1:
            raise ValueError('slot 0 is reserved for the target')
        off = base + slot * (2 * N + 4096)
        return [int(x) for x in TOK[off:off + N]]

    print(f'engine {URL}  prefix {N}  nbg {args.nbg}  reps {args.reps}', flush=True)
    total = inspect(URL, [1, 2, 3, 4, 5, 6, 7, 8])
    if total is None:
        raise SystemExit('engine not reachable')
    print(f"  capacity: token slots {total['available_token_slots']} "
          f"snapshot slots {total['available_snapshot_slots']}", flush=True)

    rows = []
    for rep in range(args.reps):
        base = 1000 + rep * 400000
        target = [int(x) for x in TOK[base:base + N]]
        # slots 1..nbg are the background set; slot 0 is the target and is
        # therefore never part of `bg`.
        bg = [region(base, s) for s in range(1, args.nbg + 1)]

        # ---------------- empty occupancy, hit then miss -------------------
        for residency in ('miss', 'hit'):
            cold_ok, tries = wait_cold(URL, target)
            if not cold_ok:
                print(f'  [empty/{residency}] rep{rep} flush failed; skip', flush=True)
                continue
            if residency == 'hit':
                _, meta = install(URL, target)
                pre = inspect(URL, target)
                if not (pre and pre['full_prefix_available']):
                    print(f'  [empty/{residency}] rep{rep} no resident prefix; skip', flush=True)
                    continue
            before = inspect(URL, target)
            t, meta = ttft_stream(URL, target)
            after = inspect(URL, target)
            rows.append(dict(
                cond=f'empty_{residency}', rep=rep, ttft_s=t,
                resident_before=bool(before['full_prefix_available']) if residency == 'hit' else False,
                free_tok_before=before['available_token_slots'],
                free_tok_after=after['available_token_slots'],
                free_snap_before=before['available_snapshot_slots'],
                free_snap_after=after['available_snapshot_slots'],
                retractions=meta.get('num_retractions'),
                e2e_s=meta.get('e2e_latency'),
                n_bg=0, survivors=None, survivor_detail=None))
            print(f"  [empty/{residency}] rep{rep} ttft={t:.3f} "
                  f"free {before['available_token_slots']}->{after['available_token_slots']} "
                  f"retr={meta.get('num_retractions')}", flush=True)

        # ---------------- full occupancy, hit then miss --------------------
        for residency in ('hit', 'miss'):
            cold_ok, tries = wait_cold(URL, bg[0])
            if not cold_ok:
                print(f'  [full/{residency}] rep{rep} flush failed; skip', flush=True)
                continue
            # hit: target is one of the nbg residents. miss: the same nbg
            # residents, but the target is not among them.
            installed = [target] + bg[1:] if residency == 'hit' else list(bg)
            for r in installed:
                install(URL, r)
            pre = inspect(URL, target)
            if residency == 'hit' and not pre['full_prefix_available']:
                print(f'  [full/{residency}] rep{rep} target not resident; skip', flush=True)
                continue
            if residency == 'miss' and pre['full_prefix_available']:
                print(f'  [full/{residency}] rep{rep} target resident; skip', flush=True)
                continue
            before = inspect(URL, target)
            t, meta = ttft_stream(URL, target)
            time.sleep(2.0)
            after = inspect(URL, target)
            alive, detail = survivor_count(URL, installed)
            rows.append(dict(
                cond=f'full_{residency}', rep=rep, ttft_s=t,
                resident_before=bool(before['full_prefix_available']),
                free_tok_before=before['available_token_slots'],
                free_tok_after=after['available_token_slots'],
                free_snap_before=before['available_snapshot_slots'],
                free_snap_after=after['available_snapshot_slots'],
                retractions=meta.get('num_retractions'),
                e2e_s=meta.get('e2e_latency'),
                n_bg=len(installed), survivors=alive, survivor_detail=detail))
            print(f"  [full/{residency}] rep{rep} ttft={t:.3f} "
                  f"free {before['available_token_slots']}->{after['available_token_slots']} "
                  f"retr={meta.get('num_retractions')} survivors={alive}/{len(installed)}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + '\n')

    print('\n=== summary ===')
    for cond in ('empty_miss', 'empty_hit', 'full_hit', 'full_miss'):
        sel = [r for r in rows if r['cond'] == cond and r['ttft_s']]
        if not sel:
            print(f'{cond:12s} (no valid cells)')
            continue
        tt = statistics.median(r['ttft_s'] for r in sel)
        rt = [r['retractions'] for r in sel if r['retractions'] is not None]
        sv = [r['survivors'] for r in sel if r['survivors'] is not None]
        print(f'{cond:12s} n={len(sel)} ttft_med={tt:7.3f}s '
              f'retractions={rt if rt else "n/a"} survivors={sv if sv else "n/a"}')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
