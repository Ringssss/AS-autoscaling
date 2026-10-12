#!/usr/bin/env python3
"""Where does capacity actually bind?

Loads disjoint 32K prefixes onto one engine one at a time and records the
engine's own slot accounting after each. No policy, no migration: just the
curve that tells us what "tight VRAM" means in this configuration.
"""
import json, time, urllib.request
import numpy as np

ENGINE = 'http://12.11.3.29:31762'
TOK = np.load('/home/zhujianian/AS-autoscaling/experiments/weka_baselines/data/tokens.npy')
N = 32768


def post(url, path, payload, timeout=1200):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return 200, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode('utf-8', 'replace')
    except Exception as e:
        return None, repr(e)[:300]


def insp(toks):
    st, r = post(ENGINE, '/agentshift/hybrid_state/inspect', {'token_ids': toks})
    if not isinstance(r, dict):
        return {'error': r}
    return {k: r.get(k) for k in ('matched_tokens', 'full_prefix_available', 'has_mamba_state',
                                  'available_token_slots', 'evictable_token_slots',
                                  'available_snapshot_slots', 'evictable_snapshot_slots',
                                  'snapshot_boundaries')}


def prefix(i):
    off = 2_000_000 + i * 262144
    return [int(x) for x in TOK[off:off + N]]


post(ENGINE, '/flush_cache?timeout=60', {})
time.sleep(4)
rows = []
print('%5s %9s %9s %9s %7s' % ('#pref', 'avail_tok', 'evict_tok', 'avail_snap', 'ok'))
for i in range(6):
    toks = prefix(i)
    t0 = time.perf_counter()
    code, r = post(ENGINE, '/generate', {'input_ids': toks,
                                         'sampling_params': {'max_new_tokens': 1, 'temperature': 0}})
    dt = time.perf_counter() - t0
    s = insp(toks)
    rows.append({'loaded': i + 1, 'prefill_s': dt, 'http': code, 'inspect': s})
    print('%5d %9s %9s %9s %7s  (%.1fs)' % (
        i + 1, s.get('available_token_slots'), s.get('evictable_token_slots'),
        s.get('available_snapshot_slots'), code, dt))

# do the early prefixes survive, or has the engine already evicted them?
surv = [insp(prefix(i)).get('full_prefix_available') for i in range(6)]
print('\nsurvival after all 6 loaded:', surv)
json.dump(rows, open('/home/zhujianian/AS-autoscaling/results/capacity-knee.json', 'w'), indent=2)
