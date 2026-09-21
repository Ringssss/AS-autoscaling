"""Audit observed cross-agent phase overlap and paired input identity, without GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path


def union_duration(intervals):
    merged = []
    for start, finish in sorted(intervals):
        if finish <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], finish)
        else:
            merged.append([start, finish])
    return sum(b-a for a, b in merged)


def intersection(interval, others):
    a, b = interval
    return union_duration((max(a, c), min(b, d)) for c, d in others)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(campaign):
    state = json.loads((campaign/'campaign.json').read_text())
    runs, identity = [], {}
    for folder in sorted(campaign.iterdir()):
        if not (folder/'summary.json').exists():
            continue
        summary = json.loads((folder/'summary.json').read_text())
        records = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        stages = []
        output_length_errors = []
        for row in records:
            for name in ('first', 'second'):
                if name in row:
                    item = row[name]
                    if len(item['output_ids']) != item['output_tokens']:
                        output_length_errors.append({'agent': row['agent'], 'stage': name, 'actual': len(item['output_ids']), 'expected': item['output_tokens']})
                    stages.append((row['agent'], item['start'], item['first'], item['finish']))
            key = (summary['concurrency'], summary['repetition'], row['agent'])
            identity.setdefault(key, {})[summary['policy']] = {
                'first_output': signature(row['first']['output_ids']) if 'first' in row else None,
                'context_input': row.get('context', {}).get('input_sha256'),
                'tool_output': row.get('tool', {}).get('metrics', {}).get('output_sha256'),
                'success': row.get('success', False)}
        report = {k: summary[k] for k in ('policy', 'concurrency', 'repetition')}
        report['run'] = folder.name
        report['output_length_errors'] = output_length_errors
        for phase in ('tool', 'context'):
            pieces = []
            for row in records:
                if phase not in row or (phase == 'tool' and summary['policy'] == 'precomputed'):
                    continue
                value = row[phase]
                interval = ((value['metrics']['started_monotonic'], value['metrics']['finished_monotonic'])
                            if phase == 'tool' else (value['start'], value['finish']))
                other = [(start, finish) for agent, start, first, finish in stages if agent != row['agent']]
                decode = [(first, finish) for agent, start, first, finish in stages if agent != row['agent']]
                pieces.append({'agent': row['agent'], 'wall_seconds': interval[1]-interval[0],
                               'other_request_overlap_seconds': intersection(interval, other),
                               'other_decode_lifetime_overlap_seconds': intersection(interval, decode)})
            wall = sum(p['wall_seconds'] for p in pieces)
            report[phase] = {'operations': len(pieces), 'summed_operation_wall_seconds': wall,
                'operations_overlapping_other_requests': sum(p['other_request_overlap_seconds'] > 0 for p in pieces),
                'operations_overlapping_other_decode_lifetimes': sum(p['other_decode_lifetime_overlap_seconds'] > 0 for p in pieces),
                'operation_wall_fraction_under_other_requests': sum(p['other_request_overlap_seconds'] for p in pieces)/wall if wall else None,
                'operation_wall_fraction_under_other_decode_lifetimes': sum(p['other_decode_lifetime_overlap_seconds'] for p in pieces)/wall if wall else None,
                'per_operation': pieces}
        runs.append(report)
    comparisons = []
    for key, variants in sorted(identity.items()):
        mismatches = {field: len({v[field] for v in variants.values() if v[field] is not None}) > 1
                      for field in ('first_output', 'context_input', 'tool_output')}
        comparisons.append({'concurrency': key[0], 'repetition': key[1], 'agent': key[2],
                            'available_policies': sorted(variants), 'mismatches': mismatches,
                            'all_available_successful': all(v['success'] for v in variants.values())})
    result = {'campaign_complete': state['complete'], 'completed_runs': len(runs),
              'scope': 'Execution intervals intersect union of OTHER-agent request/decode lifetimes; not CUDA simultaneity. Fractions are operation-wall weighted, not wallclock GPU utilization. Precomputed tools excluded.',
              'runs': runs, 'identity_comparisons': comparisons}
    (campaign/'cpu-overlap-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Workflow overlap and identity audit', '', f"Campaign complete: {state['complete']}; completed runs: {len(runs)}.", '', result['scope'], '',
             '| Run | Tool operations with request overlap | Tool wall overlap | Tool wall during other decode lifetime | Context wall overlap |',
             '| --- | ---: | ---: | ---: | ---: |']
    def pct(value):
        return 'n/a' if value is None else f'{100*value:.2f}%'
    for r in runs:
        t, c = r['tool'], r['context']
        lines.append(f"| {r['run']} | {t['operations_overlapping_other_requests']}/{t['operations']} | {pct(t['operation_wall_fraction_under_other_requests'])} | {pct(t['operation_wall_fraction_under_other_decode_lifetimes'])} | {pct(c['operation_wall_fraction_under_other_requests'])} |")
    drift = [c for c in comparisons if any(c['mismatches'].values())]
    lines += ['', f'Paired agent identities with any observed cross-policy drift: {len(drift)}/{len(comparisons)}. Incomplete groups compare available variants only.', '',
              'Nonzero request overlap establishes an opportunity for host interference, not evidence of CUDA simultaneity or resource contention. Near-zero overlap cannot disprove contention under a staggered workflow; it instead limits this phase-synchronized fixture.']
    (campaign/'cpu-overlap-audit.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    print(json.dumps({'completed_runs': result['completed_runs'], 'complete': result['campaign_complete']}))
