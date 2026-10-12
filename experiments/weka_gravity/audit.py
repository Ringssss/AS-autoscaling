"""Causal, model-separated prefix reuse audit; never infer fleet placement.

Only completed INPUT blocks enter the unlimited cache. Generated output token
identities are unavailable. Hash IDs are conversation-local. TTL sensitivity is
an analytical cache with expiry since last completed use, not recorded eviction.
Observed inter-request gaps are not asserted to be tool execution durations.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {'n': len(a), **({str(q): float(np.percentile(a, q))
                            for q in [0, 25, 50, 75, 90, 95, 99, 100]} if len(a) else {})}


def prefix_length(mask):
    misses = np.flatnonzero(~mask)
    return int(misses[0]) if len(misses) else len(mask)


def flatten(trace):
    flat = []
    groups = []
    for outer, r in enumerate(trace['requests']):
        if 'requests' in r:
            stream = r.get('agent_id', f'subagent-{outer}')
            groups.append({k: v for k, v in r.items() if k != 'requests'})
            for inner, child in enumerate(r['requests']):
                flat.append(dict(child, stream=stream, location=f'{outer}/{inner}', subagent=True))
        else:
            flat.append(dict(r, stream='main', location=str(outer), subagent=False))
    return sorted(flat, key=lambda r: (r['t'], r['location'])), groups


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows, convs, samples = [], [], []
    field_counts, anomalies = Counter(), Counter()
    digest = hashlib.sha256()
    with args.trace.open('rb') as source:
        for line_no, raw in enumerate(source, 1):
            digest.update(raw)
            trace = json.loads(raw)
            assert trace['hash_id_scope'] == 'local'
            assert trace['block_size'] == 64
            reqs, groups = flatten(trace)
            hashes = [np.asarray(r['hash_ids'], dtype=np.int32) for r in reqs]
            maxid = max(int(a.max()) for a in hashes if len(a)) + 1
            models = sorted({r['model'] for r in reqs})
            caches = {m: np.full(maxid, -np.inf) for m in models}
            # Confirm a block ID always has the same predecessor before using
            # direct ID lookups as prefix membership, including across models.
            parent = np.full(maxid, -2, dtype=np.int32)
            previous = {}
            events = []
            for i, r in enumerate(reqs):
                field_counts.update(r.keys())
                assert len(hashes[i]) * 64 == r['in']
                assert r['api_time'] >= 0 and r['t'] >= 0
                h = hashes[i]
                p = np.concatenate((np.array([-1], dtype=np.int32), h[:-1])) if len(h) else h
                old = parent[h]
                anomalies['hash_predecessor_conflicts'] += int(np.sum((old != -2) & (old != p)))
                parent[h] = p
                events.extend([(r['t'], 1, i), (r['t'] + r['api_time'], 0, i)])
            events.sort()
            active, peak, active_integral, last_t = 0, 0, 0., 0.
            for t, kind, i in events:
                r, h = reqs[i], hashes[i]
                active_integral += active * (t - last_t)
                last_t = t
                if kind == 0:
                    caches[r['model']][h] = t
                    active -= 1
                    continue
                active += 1
                peak = max(peak, active)
                ages = t - caches[r['model']][h]
                hits = {str(ttl): prefix_length(ages <= ttl) * 64
                        for ttl in [60, 300, 1800]}
                hits['unlimited'] = prefix_length(np.isfinite(ages)) * 64
                stream_key = (r['stream'], r['model'])
                j = previous.get(stream_key)
                lcp, gap, parent_tokens, parent_out = 0, None, 0, 0
                if j is not None:
                    old = reqs[j]
                    n = min(len(h), len(hashes[j]))
                    lcp = prefix_length(h[:n] == hashes[j][:n]) * 64
                    gap = t - old['t'] - old['api_time']
                    parent_tokens, parent_out = old['in'], old['out']
                    if gap < -1e-6:
                        anomalies['overlapping_previous_same_stream_model'] += 1
                previous[stream_key] = i
                row = dict(conversation=trace['id'], line=line_no, location=r['location'],
                           stream=r['stream'], subagent=r['subagent'], model=r['model'],
                           t=t, api_time=r['api_time'], input=r['in'], output=r['out'],
                           ttft=r.get('ttft'), think_time=r.get('think_time'),
                           hit=hits['unlimited'], hit60=hits['60'], hit300=hits['300'], hit1800=hits['1800'],
                           previous_lcp=lcp, gap=gap, previous_input=parent_tokens,
                           previous_output=parent_out, previous_location=reqs[j]['location'] if j is not None else None)
                rows.append(row)
                if gap is not None and gap >= 0 and lcp >= 8192:
                    samples.append(row)
            convs.append(dict(id=trace['id'], line=line_no, requests=len(reqs), groups=len(groups),
                              models=models, first_t=reqs[0]['t'], end=last_t,
                              sum_api_seconds=active_integral, peak_active=peak,
                              input=sum(r['in'] for r in reqs), output=sum(r['out'] for r in reqs),
                              unique_blocks=int(np.sum(parent != -2))))
            if line_no % 25 == 0:
                print(f'audited {line_no} conversations / {len(rows)} requests', flush=True)
    assert anomalies['hash_predecessor_conflicts'] == 0, anomalies
    with (args.output / 'requests.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'conversations.json').write_text(json.dumps(convs, indent=2)+'\n')
    paired = [r for r in rows if r['gap'] is not None]
    eligible = [r for r in paired if r['gap'] >= 0 and r['previous_lcp'] >= 32768]
    total_in = sum(r['input'] for r in rows)
    def subset(rs):
        return dict(requests=len(rs), input=stats([r['input'] for r in rs]),
                    output=stats([r['output'] for r in rs]),
                    causal_hit_fraction=sum(r['hit'] for r in rs)/sum(r['input'] for r in rs),
                    causal_hit_tokens=stats([r['hit'] for r in rs]))
    summary = dict(source=str(args.trace), sha256=digest.hexdigest(), conversations=len(convs),
                   requests=len(rows), main_requests=sum(not r['subagent'] for r in rows),
                   subagent_requests=sum(r['subagent'] for r in rows),
                   subagent_groups=sum(c['groups'] for c in convs),
                   total_input=total_in, total_output=sum(r['output'] for r in rows),
                   model_counts=dict(Counter(r['model'] for r in rows)), fields=dict(field_counts),
                   anomalies=dict(anomalies), all_first_t_zero=all(c['first_t']==0 for c in convs),
                   all=subset(rows), main=subset([r for r in rows if not r['subagent']]),
                   subagent=subset([r for r in rows if r['subagent']]),
                   reuse_by_ttl={k:sum(r[k] for r in rows)/total_in for k in ['hit','hit60','hit300','hit1800']},
                   context_coverage={str(n):dict(request_fraction=np.mean([r['input']+r['output']<=n for r in rows]).item(),
                       input_fraction=sum(r['input'] for r in rows if r['input']+r['output']<=n)/total_in)
                       for n in [32768,65536,131072,262144,524288,1048576]},
                   gap_seconds=stats([r['gap'] for r in paired if r['gap']>=0]),
                   paired_requests=len(paired), eligible_32k_pairs=len(eligible),
                   eligible_gap_seconds=stats([r['gap'] for r in eligible]),
                   previous_lcp=stats([r['previous_lcp'] for r in paired]),
                   long_gap_thresholds={str(t):sum(r['gap']>=t for r in eligible) for t in [.05,.1,.2,.5,1,2,5,10,30,60]},
                   request_counts_per_conversation=stats([c['requests'] for c in convs]),
                   conversation_duration=stats([c['end'] for c in convs]),
                   conversation_peak_active=stats([c['peak_active'] for c in convs]),
                   conversation_unique_tokens=stats([c['unique_blocks']*64 for c in convs]),
                   top10_conversation_output_fraction=sum(sorted([c['output'] for c in convs],reverse=True)[:10])/sum(c['output'] for c in convs),
                   assumptions=['Unlimited capacity, completed input blocks only, same model and same conversation',
                                'No cross-conversation reuse inferable from local hash IDs',
                                'No absolute cross-conversation timeline, engine ID or original token IDs',
                                'Gaps are observed request gaps, not measured pure tool durations',
                                'api_time includes original serving and network; not local GPU service time'])
    # State-size/transfer sensitivity, not a claim about Claude architecture.
    summary['copy_window_sensitivity'] = []
    for bytes_per_token in [20480,131072,327680]:
        for bandwidth_gbs in [5,10,25]:
            durations = [0.02+(r['previous_lcp']*bytes_per_token+64389120)/(bandwidth_gbs*1e9) for r in eligible]
            summary['copy_window_sensitivity'].append(dict(bytes_per_token=bytes_per_token,
                bandwidth_GB_s=bandwidth_gbs, fixed_overhead_s=.02,
                fixed_state_bytes=64389120, pairs=len(eligible),
                fit_fraction=float(np.mean([t<=r['gap'] for t,r in zip(durations,eligible)])),
                copy_seconds=stats(durations)))
    selected = []
    for target in [8192,32768,65536,98304]:
        pool = [r for r in samples if r['previous_input']+r['previous_output']<=131072 and r['input']+r['output']<=131072]
        best = min(pool, key=lambda r: abs(r['previous_lcp']-target))
        selected.append(dict(target=target, **best))
    (args.output/'selected_pairs.json').write_text(json.dumps(selected,indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['conversations','requests','reuse_by_ttl','eligible_32k_pairs','anomalies']},indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, default=Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl'))
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    run(p.parse_args())
