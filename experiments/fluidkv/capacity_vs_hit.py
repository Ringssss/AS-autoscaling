#!/usr/bin/env python3
"""Hit vs miss at identical occupancy: does residing cost more than it saves?

Three conditions, one 32K target prefix, three engines (TP2 each).

  cold        engine empty, target absent        -> pure 32K prefill cost
  hit_tight   7 disjoint 32K prefixes resident   -> the target IS one of them
  miss_tight  7 disjoint 32K prefixes resident   -> the target is absent

`hit_tight` and `miss_tight` carry the *same* residency (7 x 32,768 =
229,376 of 262,144, i.e. 32,768 free). The only difference is whether the
arriving request finds its prefix already there. So the gap between them is
the cost of residency itself, with allocation pressure held constant.

A prefix only becomes inspectable as resident after a *completed* request has
run over it, so every prefix here is installed with a real generate call.
`flush_cache` is used only to reach a known-empty engine, never to sculpt a
partially-occupied one.
"""
import json, statistics, sys, time, urllib.request
import numpy as np

ROOT = '/home/zhujianian/AS-autoscaling'
TOK = np.load(f'{ROOT}/experiments/weka_baselines/data/tokens.npy')
UNIT = 32768
NBG = 7                       # 7 * 32K = 229,376 resident, 32,768 free
SETTLE = 30.0                 # prefill throughput ~8.4K tok/s -> ~28s per 32K

ENGINES = {'clean': 'http://12.11.3.29:31760',
           'hit':   'http://12.11.3.29:31761',
           'miss':  'http://12.11.3.29:31762'}


def post(url, path, payload, timeout=2400):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return 200, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode('utf-8', 'replace')
    except Exception as e:
        return None, repr(e)[:300]


def insp(url, toks):
    _, r = post(url, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    return r if isinstance(r, dict) else {}


def state(url, toks):
    s = insp(url, toks)
    return {'matched': s.get('matched_tokens'),
            'full': bool(s.get('full_prefix_available')),
            'mamba': bool(s.get('has_mamba_state')),
            'avail': s.get('available_token_slots'),
            'evict': s.get('evictable_token_slots')}


def fresh(url):
    for _ in range(6):
        post(url, '/flush_cache?timeout=60', {})
        time.sleep(4)
        if state(url, [1, 2, 3, 4]).get('avail') == 262144:
            return True
    return False


def install(url, toks):
    """A prefix becomes resident only once a completed request has run over it."""
    code, _ = post(url, '/generate', {'input_ids': toks,
                                      'sampling_params': {'max_new_tokens': 16,
                                                          'temperature': 0}})
    return code


def ttft(url, toks, max_new=16):
    """Streaming: first data chunk is prefill done, so decode is excluded."""
    body = json.dumps({'input_ids': toks, 'max_new_tokens': max_new,
                       'temperature': 0, 'stream': True}).encode()
    req = urllib.request.Request(url + '/generate', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=2400) as f:
        for line in f:
            if not line.startswith(b'data:'):
                continue
            if line[5:].strip() in (b'[DONE]', b''):
                continue
            if first is None:
                first = time.perf_counter() - t0
    return first


def region(rep, slot, base):
    return [int(x) for x in TOK[base + (rep * 32 + slot) * 262144:
                                base + (rep * 32 + slot) * 262144 + UNIT]]


def run_cold(url, rows):
    for rep in range(5):
        assert fresh(url)
        target = region(rep, 0, 4_000_000)
        t = ttft(url, target)
        after = state(url, target)
        rows.append({'cond': 'cold', 'rep': rep, 'ttft_s': t,
                     'target_full_after': after['full'],
                     'survivors': None, 'avail_after': after['avail']})
        print(f'  [cold] rep{rep} ttft={t:.3f}s avail_after={after["avail"]}', flush=True)


def run_tight(url, rows, cond, base):
    """7 resident 32K prefixes; the target is one of them (hit) or not (miss)."""
    for rep in range(3):
        assert fresh(url)
        # slot 0 is the target for `hit`; for `miss` the target is a slot that
        # is never installed.
        installed = list(range(NBG))
        t_load = time.perf_counter()
        for slot in installed:
            code = install(url, region(rep, slot, base))
            if code != 200:
                raise RuntimeError(f'install slot {slot} -> {code}')
        load_s = time.perf_counter() - t_load

        target = region(rep, 0, base) if cond == 'hit_tight' else region(rep, 20, base)
        before = state(url, target)
        t = ttft(url, target)
        time.sleep(2)
        after = state(url, target)

        # who survived? check every resident slot for the hit condition
        surv = None
        if cond == 'hit_tight':
            surv = sum(1 for s in installed if state(url, region(rep, s, base))['full'])
        rows.append({'cond': cond, 'rep': rep, 'ttft_s': t, 'load_s': load_s,
                     'n_bg': NBG, 'target_resident_before': before['full'],
                     'avail_before': before['avail'], 'avail_after': after['avail'],
                     'survivors': surv})
        print(f'  [{cond}] rep{rep} load={load_s:.1f}s resident_before={before["full"]} '
              f'avail={before["avail"]} ttft={t:.3f}s avail_after={after["avail"]} '
              f'survivors={surv}', flush=True)


def main():
    rows = []
    print('[cold] engine 31760, empty', flush=True)
    run_cold(ENGINES['clean'], rows)
    print('[hit_tight] engine 31761, 7 resident incl. target', flush=True)
    run_tight(ENGINES['hit'], rows, 'hit_tight', 8_000_000)
    print('[miss_tight] engine 31762, 7 resident, target absent', flush=True)
    run_tight(ENGINES['miss'], rows, 'miss_tight', 12_000_000)

    out = f'{ROOT}/results/capacity-vs-hit.json'
    json.dump(rows, open(out, 'w'), indent=2)
    print('\n=== 32,768-token target ===')
    print(f'{"condition":12s}{"ttft_med":>10}{"min":>9}{"max":>9}{"reps":>6}{"avail_after":>13}')
    med = {}
    for cond in ('cold', 'hit_tight', 'miss_tight'):
        v = [r for r in rows if r['cond'] == cond]
        if not v:
            continue
        med[cond] = statistics.median(r['ttft_s'] for r in v)
        print(f'{cond:12s}{med[cond]:>10.3f}{min(r["ttft_s"] for r in v):>9.3f}'
              f'{max(r["ttft_s"] for r in v):>9.3f}{len(v):>6}'
              f'{v[-1]["avail_after"] if v[-1]["avail_after"] is not None else -1:>13}')
    if 'hit_tight' in med and 'miss_tight' in med:
        d = med['miss_tight'] - med['hit_tight']
        print(f'\nresidency cost at equal occupancy (miss_tight - hit_tight) = {d:.3f}s'
              f'   ratio {med["miss_tight"]/med["hit_tight"]:.2f}x')
    if 'cold' in med and 'miss_tight' in med:
        print(f'pressure penalty vs clean prefill (miss_tight - cold) = '
              f'{med["miss_tight"] - med["cold"]:.3f}s'
              f'   ratio {med["miss_tight"]/med["cold"]:.2f}x')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
