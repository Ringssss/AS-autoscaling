#!/usr/bin/env python3
"""Minimal test of hypothesis 3: is `inspect` expensive when the engine is busy?

The mobility arm performs two `inspect` calls per request INSIDE its routing
lock, while the reroute arm performs none. If those calls are cheap, the wall
inflation in the E2E runs is entirely control-plane timeouts and hypothesis 3
is dead. If they are expensive under load, the lock then serialises every
stream on a slow network call -- a defect of my harness, not of the mechanism.

This needs NO migration, so it is unaffected by the destination-empty
constraint that blocked every earlier end-to-end attempt.

Measures inspect round-trip time against one engine in two states:

  idle      nothing else running
  busy      background generation saturating the same engine

The ratio is the quantity of interest. A few milliseconds either way means the
hypothesis is false; a hundred-fold increase means it is the dominant cost.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'p03', Path(__file__).parent / 'p03_real_copy_e2e.py')
p03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p03)

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKS = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
TARGET = 'http://12.11.3.27:31781'
LOADER = 'http://12.11.3.27:31782'
N_INSPECT = 30
PROBE_LEN = 130048


def tokens(off, n):
    return [int(x) for x in TOKS[off:off + n]]


async def measure(s, url, ids, n):
    """Round-trip time of n sequential inspect calls."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        await p03.inspect(s, url, ids)
        out.append(time.perf_counter() - t0)
    return out


async def background_load(s, dur):
    """Keep several long requests running on the loader engine."""
    stop = time.perf_counter() + dur
    tasks = []
    while time.perf_counter() < stop:
        for k in range(4):
            off = 3_000_000 + k * 400_000
            tasks.append(asyncio.create_task(
                p03.serve(s, LOADER, tokens(off, 90000))))
        await asyncio.sleep(0.2)
        tasks = [t for t in tasks if not t.done()]
    return tasks


async def main():
    async with aiohttp.ClientSession() as s:
        probe = tokens(200_000_000, PROBE_LEN)
        # The probe prefix must exist on the target or inspect trivially returns;
        # install it once so the lookup does real radix-tree work.
        await p03.flush(s, ['http://12.11.3.27:31781', LOADER])
        await asyncio.sleep(2)
        print('installing probe prefix on target...', flush=True)
        await p03.serve(s, TARGET, probe)

        print('\n[1] inspect on IDLE engine', flush=True)
        idle = await measure(s, TARGET, probe, N_INSPECT)
        print(f'    n={len(idle)} p50={statistics.median(idle) * 1e3:8.2f}ms '
              f'p90={sorted(idle)[int(.9 * len(idle))] * 1e3:8.2f}ms '
              f'max={max(idle) * 1e3:8.2f}ms', flush=True)

        print('\n[2] inspect on BUSY engine (background generation)', flush=True)
        bg = asyncio.create_task(background_load(s, 120))
        await asyncio.sleep(5)
        busy = await measure(s, TARGET, probe, N_INSPECT)
        print(f'    n={len(busy)} p50={statistics.median(busy) * 1e3:8.2f}ms '
              f'p90={sorted(busy)[int(.9 * len(busy))] * 1e3:8.2f}ms '
              f'max={max(busy) * 1e3:8.2f}ms', flush=True)
        bg.cancel()
        try:
            pending = await bg
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            pass

        i_med = statistics.median(idle)
        b_med = statistics.median(busy)
        print('\n===== VERDICT =====')
        print(f'idle  p50 = {i_med * 1e3:.2f} ms')
        print(f'busy  p50 = {b_med * 1e3:.2f} ms   ratio = {b_med / i_med:.1f}x')
        print(f'per-request serialised routing cost = 2 x {b_med * 1e3:.2f} ms '
              f'= {2 * b_med * 1e3:.2f} ms')
        print(f'over 90 requests                    = '
              f'{90 * 2 * b_med:.2f} s of pure serialisation')
        if b_med < 0.05:
            print('-> hypothesis 3 FALSE: inspect is cheap; wall inflation is '
                  'control-plane timeouts')
        else:
            print('-> hypothesis 3 TRUE: refine the harness before any more E2E')


if __name__ == '__main__':
    asyncio.run(main())
