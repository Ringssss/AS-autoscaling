"""Select genuinely overlapping, model-separated AgentX fragments before testing.

Only input prefix identities are recoverable. Token content is deterministic
surrogate content; outputs are a declared fixed-length generation workload.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE = Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')
OUT = ROOT / 'results/routing-space-20261010'


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i * 64
    return min(len(a), len(b)) * 64


def main():
    from experiments.weka_gravity.audit import flatten
    candidates = {2: [], 6: []}
    digest = hashlib.sha256()
    for line, raw in enumerate(TRACE.open('rb'), 1):
        digest.update(raw)
        trace = json.loads(raw)
        rows, _ = flatten(trace)
        streams = defaultdict(list)
        for r in rows:
            streams[r['stream'], r['model']].append(r)
        eligible = {}
        for key, rs in streams.items():
            for j in range(1, len(rs) - 2):
                fragment = rs[j-1:j+3]
                if not all(32768 <= r['in'] <= 110000 and r['out'] > 0
                           and r['in'] + r['out'] <= 131072 for r in fragment):
                    continue
                if any(b['t'] < a['t'] + a['api_time'] - 1e-6
                       or lcp(a['hash_ids'], b['hash_ids']) < 32768
                       for a, b in zip(fragment, fragment[1:])):
                    continue
                eligible[rs[j]['location']] = fragment
        active = defaultdict(list)
        for r in rows:
            m = r['model']
            active[m] = [x for x in active[m] if x['t'] + x['api_time'] > r['t']]
            active[m].append(r)
            pool = [x for x in active[m] if x['location'] in eligible]
            # Distinct stream / same model only. Avoid repeated sampling of one wave.
            pool = list({x['stream']: x for x in pool}.values())
            for n in (2, 6):
                if len(pool) < n:
                    continue
                chosen = sorted(pool, key=lambda x: x['t'])[-n:]
                fragments = [eligible[x['location']] for x in chosen]
                parents = [f[0]['hash_ids'] for f in fragments]
                common = min(lcp(parents[0], h) for h in parents)
                unique = len({h for f in fragments for x in f for h in x['hash_ids']}) * 64
                spread = max(x['t'] for x in chosen) - min(x['t'] for x in chosen)
                if common < 32768 or unique > 220000 or spread > 20:
                    continue
                tag = ':'.join([trace['id'], m] + [x['location'] for x in chosen])
                rank = hashlib.sha256(('20261010:' + tag).encode()).hexdigest()
                candidates[n].append(dict(rank=rank, line=line, conversation=trace['id'],
                    model=m, common_parent_tokens=common, unique_input_tokens=unique,
                    first_arrival_spread=spread, fragments=fragments))
        if line % 50 == 0:
            print('scan', line, {n: len(v) for n, v in candidates.items()}, flush=True)
    selected = []
    token_parts, offset = [], 0
    for n, pool in candidates.items():
        if not pool:
            continue
        c = min(pool, key=lambda x: x['rank'])
        cache = {}
        namespace = c['conversation'] + '/' + c['model']
        first = min(f[1]['t'] for f in c['fragments'])
        streams = []
        for f in c.pop('fragments'):
            saved = []
            for r in f:
                parts = []
                for h in r['hash_ids']:
                    if h not in cache:
                        seed = int.from_bytes(hashlib.sha256(f'{namespace}:{h}'.encode()).digest()[:8], 'little')
                        cache[h] = np.random.default_rng(seed).integers(1000, 240000, 64, dtype=np.int32)
                    parts.append(cache[h])
                ids = np.concatenate(parts)
                token_parts.append(ids)
                saved.append(dict(location=r['location'], native_t=r['t'], native_api_time=r['api_time'],
                    input_tokens=r['in'], native_output_tokens=r['out'], output_tokens=min(r['out'], 128),
                    token_offset=offset, hash_ids=r['hash_ids'],
                    token_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
                offset += len(ids)
            streams.append(dict(key=f[0]['stream'], requests=saved,
                first_offset=saved[1]['native_t']-first,
                gaps=[max(0., b['native_t']-a['native_t']-a['native_api_time'])
                      for a, b in zip(saved[1:], saved[2:])]))
        c.update(name=f'native-n{n}', streams=streams)
        selected.append(c)
    if not selected:
        raise RuntimeError('no feasible genuinely overlapping cohorts')
    np.save(OUT/'cohort-tokens.npy', np.concatenate(token_parts))
    report = dict(trace=str(TRACE), trace_sha256=digest.hexdigest(),
        candidate_counts={n: len(v) for n, v in candidates.items()}, cohorts=selected,
        scope='Selected overlapping same-model native fragments with completed input reuse. '
              'Surrogate token content; Qwen3.6 model substitutes original models. '
              'Original input lengths, input identities, initial relative arrivals and subsequent '
              'completion-to-arrival gaps retained; output lengths capped at 128 for the pilot. '
              'Parents prefilled in setup, not included in consumer service window. '
              'Not full corpus or semantic task replay; no original global worker placement.')
    (OUT/'cohorts.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cohorts'}, indent=2))
    for c in selected:
        print(c['name'], c['conversation'], c['common_parent_tokens'], c['first_arrival_spread'])


if __name__ == '__main__':
    main()
