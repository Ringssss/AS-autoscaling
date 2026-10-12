#!/usr/bin/env python3
"""P0-5: hit-vs-miss gap as a function of prefix length, at FIXED occupancy.

The 32K capacity result (O5) is a single point. The paper needs the curve, and
specifically the length at which the hit/miss gap collapses -- that is the
threshold below which residency stops paying for the slots it holds.

Design, per length L:

  resident = (262144 // L) - 1   prefixes, so exactly ONE more L-length request
                                 can be admitted without eviction
  cold      engine empty, serve L                      (pure recompute)
  hit       the resident set INCLUDES the target, serve it
  miss      the resident set excludes the target, serve it

The hit and miss arms therefore carry IDENTICAL occupancy and differ in exactly
one thing: whether the arriving request's prefix is already resident.

Two deliberate corrections to the earlier capacity run:

  * Survivor counts are collected for the MISS arm too. The earlier script
    only counted neighbours for the hit arm and the miss number was then
    inferred from a slot delta, which cannot distinguish one long victim from
    several short ones.
  * No `flush` is used to sculpt a partially occupied engine. A flush that
    cannot complete silently leaves the engine warm, and the arm then measures
    something other than what it claims.
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import importlib.util
import sys

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location(
    'p03', Path(__file__).parent / 'p03_real_copy_e2e.py')
p03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p03)

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKS = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
MAX_TOKENS = 262144

ENGINES = {
    'cold': '12.11.3.27:31781',
    'hit':  '12.11.3.27:31782',
    'miss': '12.11.3.27:31783',
}


async def post(s, ep, path, payload, to=600):
    try:
        async with s.post(f'http://{ep}{path}', json=payload,
                          timeout=aiohttp.ClientTimeout(total=to)) as r:
            return await r.json()
    except Exception as exc:
        return {'ERR': repr(exc)[:140]}


async def flush(s, ep):
    try:
        async with s.post(f'http://{ep}/flush_cache?timeout=60') as r:
            await r.text()
    except Exception:
        pass


async def inspect(s, ep, ids):
    r = await post(s, ep, '/agentshift/hybrid_state/inspect', {'token_ids': ids})
    return r if isinstance(r, dict) else {}


async def install(s, ep, ids):
    """A prefix becomes inspectable as resident only after a COMPLETED request."""
    r = await post(s, ep, '/generate',
                   {'input_ids': ids,
                    'sampling_params': {'temperature': 0, 'max_new_tokens': 16,
                                        'ignore_eos': True}})
    return (r.get('meta_info') or {}).get('cached_tokens')


async def ttft(s, ep, ids):
    """Streaming first-token latency; decode is excluded by construction.

    Delegates to p03.serve, which waits for completion_tokens>=1 rather than
    for the first SSE chunk: sglang emits chunks that carry no generated token,
    and treating those as "first token" silently returns None.
    """
    ttft_s, _total, _meta = await p03.serve(s, 'http://' + ep, ids)
    return ttft_s


# Distinct 262144-token regions so no experiment reuses another's bytes.
# A band is 1 + resident blocks: block 0 is the target (installed for `hit`,
# deliberately absent for `miss`) and 1..resident are the neighbours. Nine
# bands of at most 16 blocks each stay well inside the 231 available.
def band_base(band, resident):
    return band * (1 + resident + 2)


def region(idx, length):
    base = idx * MAX_TOKENS
    return [int(x) for x in TOKS[base:base + length]]


async def run_length(s, L, reps, band):
    resident = (MAX_TOKENS // L) - 1
    print(f'\n===== L={L}  resident={resident}  free={MAX_TOKENS - resident * L} =====',
          flush=True)
    rows = []
    for rep in range(reps):
        # Each repetition gets its own band so no two arms share bytes.
        rb = band_base(band * reps + rep, resident)

        # ---- cold ----
        await flush(s, ENGINES['cold'])
        await asyncio.sleep(2)
        tgt = region(rb, L)
        t = await ttft(s, ENGINES['cold'], tgt)
        st = await inspect(s, ENGINES['cold'], tgt)
        rows.append(dict(L=L, rep=rep, cond='cold', ttft=t,
                         avail_after=st.get('available_token_slots'),
                         matched=st.get('matched_tokens')))
        print(f'  cold  rep{rep} ttft={t:7.2f}s', flush=True)

        for cond in ('hit', 'miss'):
            ep = ENGINES[cond]
            await flush(s, ep)
            await asyncio.sleep(2)
            # Install the resident set; slot 0 is the target for `hit`, and for
            # `miss` the target lives in a region that is never installed.
            t0 = time.perf_counter()
            for slot in range(resident):
                await install(s, ep, region(rb + 1 + slot, L))
            load_s = time.perf_counter() - t0
            tgt = region(rb + 1, L) if cond == 'hit' else region(rb + 40, L)
            before = await inspect(s, ep, tgt)
            t = await ttft(s, ep, tgt)
            await asyncio.sleep(2)
            after = await inspect(s, ep, tgt)
            # Survivors are counted for BOTH arms -- that is the fix.
            surv = 0
            for slot in range(resident):
                st = await inspect(s, ep, region(rb + 1 + slot, L))
                if st.get('full_prefix_available'):
                    surv += 1
            rows.append(dict(L=L, rep=rep, cond=cond, ttft=t, load_s=load_s,
                             n_resident=resident,
                             resident_before=bool(before.get('full_prefix_available')),
                             avail_before=before.get('available_token_slots'),
                             avail_after=after.get('available_token_slots'),
                             survivors=surv))
            print(f'  {cond:5s} rep{rep} load={load_s:6.1f}s '
                  f'resident_before={before.get("full_prefix_available")} '
                  f'avail={before.get("available_token_slots")}->'
                  f'{after.get("available_token_slots")} '
                  f'ttft={t:7.2f}s survivors={surv}/{resident}', flush=True)
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[16384, 32768, 65536])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default=str(ROOT / 'results/p05-length-sweep.json'))
    a = ap.parse_args()

    all_rows = []
    async with aiohttp.ClientSession() as s:
        for k, L in enumerate(a.lengths):
            all_rows += await run_length(s, L, a.reps, k)

    Path(a.out).write_text(json.dumps(all_rows, indent=2) + '\n')
    print(f'\nwrote {a.out}\n')
    print(f'{"L":>7}{"cond":>7}{"ttft_med":>10}{"avail_after":>13}'
          f'{"survivors":>11}')
    by = defaultdict(list)
    for r in all_rows:
        by[(r['L'], r['cond'])].append(r)
    for L in a.lengths:
        for cond in ('cold', 'hit', 'miss'):
            v = by.get((L, cond), [])
            if not v:
                continue
            tts = [x['ttft'] for x in v if x['ttft'] is not None]
            av = [x['avail_after'] for x in v if x.get('avail_after') is not None]
            sv = [x['survivors'] for x in v if x.get('survivors') is not None]
            n = v[0].get('n_resident')
            print(f'{L:7d}{cond:>7}{np.median(tts):10.2f}'
                  f'{(int(np.median(av)) if av else -1):13d}'
                  f'{(f"{int(np.median(sv))}/{n}" if sv else "-"):>11}')


if __name__ == '__main__':
    asyncio.run(main())
