"""Audit and summarize local copy experiments without treating tokens as repeats."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from experiments.local_pipeline.metrics import (
    align_gaps_with_windows, percentile, summarize_emissions,
)

SPEC_KEYS = ('target', 'concurrency', 'prefix_tokens', 'repetition', 'mode')
METRICS = ('gap_p50_ms', 'gap_p95_ms', 'gap_p99_ms', 'gap_max_ms',
           'median_request_max_ms', 'fixed_window_p99_ms', 'fixed_window_max_ms',
           'mean_request_finish_ms', 'foreground_total_ms')


def digest(tokens):
    # Same stable representation as calibrate_decision_costs.digest.
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def key(row):
    return tuple(row.get(k) for k in SPEC_KEYS)


def phase_windows(events):
    windows = []
    for event in events:
        response = event.get('response') or {}
        state = response.get('state', response.get('status', '')) if isinstance(response, dict) else ''
        windows.append({'start': event['start'], 'end': event['finish'],
                        'name': f"{event.get('engine')}:{event.get('path')}:{state}",
                        'response_state': state, 'role': event.get('role')})
    return windows


def summarize_run(row):
    requests = row.get('requests', [])
    recalculated = [summarize_emissions(r['emissions'], r['requested_output_tokens']) for r in requests]
    gaps = [dict(g, request_index=i) for i, m in enumerate(recalculated) for g in m['gaps']]
    seconds = [g['seconds'] for g in gaps]
    trigger = row.get('trigger')
    window_gaps = [g for g in gaps if trigger is not None and
                   g['start'] < trigger + .35 and g['end'] > trigger - .05]
    window_seconds = [g['seconds'] for g in window_gaps]
    maxima = [m['gap_max_seconds'] for m in recalculated if m['gap_max_seconds'] is not None]
    result = {k: row.get(k) for k in ('index', *SPEC_KEYS, 'success', 'error')}
    result.update({f'gap_p{p}_ms': 1000 * percentile(seconds, p) if seconds else None for p in (50, 95, 99)})
    result.update({
        'gap_max_ms': max(seconds) * 1000 if seconds else None,
        'median_request_max_ms': statistics.median(maxima) * 1000 if maxima else None,
        'observed_gap_count': len(gaps),
        'gaps_over_20ms': sum(g > .02 for g in seconds),
        'gaps_over_50ms': sum(g > .05 for g in seconds),
        'coalesced_emissions': sum(m['coalesced_emission_count'] for m in recalculated),
        'duplicate_emissions': sum(m['duplicate_count'] for m in recalculated),
        'fixed_window_gap_count': len(window_gaps),
        'fixed_window_p99_ms': percentile(window_seconds, 99) * 1000 if window_seconds else None,
        'fixed_window_max_ms': max(window_seconds) * 1000 if window_seconds else None,
        'fixed_window_gaps_over_20ms': sum(g > .02 for g in window_seconds),
        'fixed_window_gaps_over_50ms': sum(g > .05 for g in window_seconds),
        'mean_request_finish_ms': statistics.mean(r['finish'] - r['start'] for r in requests) * 1000 if requests else None,
        'foreground_total_ms': (max(r['finish'] for r in requests) - row['start']) * 1000 if requests else None,
        'decoding_at_trigger': row.get('decoding_at_trigger'),
        'active_at_trigger': row.get('active_at_trigger'),
        'migration_ms': (row['migration']['finish'] - row['migration']['start']) * 1000 if row.get('migration') else None,
        'migration_bytes': row.get('migration', {}).get('bytes', 0),
        'per_request_metrics': recalculated,
    })
    windows = phase_windows(row.get('control_events', []))
    longest = sorted(gaps, key=lambda g: g['seconds'], reverse=True)[:12]
    result['longest_gaps'] = [dict(g, start_relative_trigger_ms=(g['start'] - trigger) * 1000,
                                  end_relative_trigger_ms=(g['end'] - trigger) * 1000)
                              for g in align_gaps_with_windows(longest, windows)] if trigger is not None else []
    result['phase_windows'] = windows
    result['phase_gap_associations'] = [g for g in align_gaps_with_windows(window_gaps, windows) if g['windows']]
    return result


def analyze_campaign(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    args = json.loads((path / 'arguments.json').read_text())
    fixtures = json.loads((path / 'fixtures.json').read_text())
    info = json.loads((path / 'server-info.json').read_text())
    records = [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines() if line.strip()]
    complete = json.loads((path / 'complete.json').read_text()) if (path / 'complete.json').exists() else {}
    errors, warnings = [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    expected = list(itertools.product(args['foreground_targets'], args['concurrency'], args['prefix_tokens'],
                                      range(args['repetitions']), args['modes']))
    check(Counter(map(key, manifest)) == Counter(expected), 'manifest differs from arguments Cartesian design')
    check(len(records) == len(manifest), 'records count differs from manifest')
    check(complete.get('complete') is True and complete.get('runs') == len(manifest), 'completion marker missing or inconsistent')
    check(len(info) == 2, 'requires two engine information records')
    backend_keys = ('pp_size', 'tp_size', 'disable_cuda_graph', 'mamba_scheduler_strategy',
                    'mamba_ssm_dtype', 'max_total_tokens', 'max_mamba_cache_size',
                    'attention_backend', 'linear_attn_backend', 'disable_overlap_schedule', 'stream_interval')
    backends = [{k: engine.get('server_args', engine).get(k) for k in backend_keys} for engine in info]
    for i, backend in enumerate(backends):
        for name, expected_value in [('pp_size', 1), ('tp_size', 1), ('disable_cuda_graph', False),
                                     ('mamba_ssm_dtype', 'float32'), ('max_total_tokens', 131072),
                                     ('max_mamba_cache_size', 64), ('stream_interval', 1),
                                     ('attention_backend', 'fa3'), ('linear_attn_backend', 'triton'),
                                     ('disable_overlap_schedule', False)]:
            check(backend.get(name) == expected_value, f'engine {i}: backend {name}={backend.get(name)!r}, expected {expected_value!r}')
        # Runtime versions expose this under either spelling; report unknown explicitly.
        strategy = info[i].get('server_args', info[i]).get('mamba_scheduler_strategy')
        if strategy is None:
            strategy = info[i].get('server_args', info[i]).get('mamba_cache_strategy')
        check(strategy == 'extra_buffer', f'engine {i}: checkpoint strategy {strategy!r}, expected extra_buffer')
    summaries = []
    by_pair = defaultdict(dict)
    for position, row in enumerate(records):
        label = f'run {position}'
        check(row.get('index') == position, f'{label}: index mismatch')
        check(position < len(manifest) and key(row) == key(manifest[position]), f'{label}: manifest mismatch')
        check(row.get('success') is True, f'{label}: unsuccessful run')
        check(row.get('cleanup_confirmed') is True, f'{label}: cleanup not confirmed')
        check(row.get('active_at_trigger', 0) > 0, f'{label}: no active foreground at trigger')
        check(len(row.get('requests', [])) == row['concurrency'], f'{label}: request count mismatch')
        parent_hash = row.get('parent_input_sha256')
        check(parent_hash == fixtures['parent_hashes'].get(str(row['prefix_tokens'])) and parent_hash is not None,
              f'{label}: parent input identity unverified')
        source_build = row.get('source_build', {})
        check(source_build.get('meta_info', {}).get('prompt_tokens') == row['prefix_tokens'], f'{label}: source build prompt length')
        for j, request in enumerate(row.get('requests', [])):
            rlabel = f'{label} request {j}'
            check(request.get('success') is True, f'{rlabel}: unsuccessful request')
            check(request.get('input_sha256') == fixtures['foreground_hashes'][j], f'{rlabel}: fixture mismatch')
            check(request.get('output_sha256') == digest(request.get('output_ids', [])), f'{rlabel}: output digest mismatch')
            check(request.get('input_tokens') == args['input_tokens'], f'{rlabel}: input length mismatch')
            expected_output = args['output_tokens']
            check(request.get('requested_output_tokens') == expected_output, f'{rlabel}: requested output mismatch')
            check(len(request.get('output_ids', [])) == expected_output, f'{rlabel}: output IDs length mismatch')
            check(request.get('meta_info', {}).get('completion_tokens') == expected_output, f'{rlabel}: completion metadata mismatch')
            check(request.get('meta_info', {}).get('prompt_tokens') == args['input_tokens'], f'{rlabel}: prompt metadata mismatch')
            recomputed = summarize_emissions(request['emissions'], expected_output)
            check(recomputed['complete'], f'{rlabel}: stream incomplete/reset/nonmonotonic')
            check(recomputed == request.get('metrics'), f'{rlabel}: stored metrics mismatch')
            check(all(request['start'] <= e['time'] <= request['finish'] for e in request['emissions']), f'{rlabel}: emission outside request envelope')
        if row['mode'] == 'copy':
            validation = row.get('validation', {})
            src, dst = validation.get('source', {}), validation.get('destination', {})
            check(row.get('migration', {}).get('bytes', 0) > 0, f'{label}: no actual copied bytes')
            check(validation.get('cache_hit') is True and dst.get('meta_info', {}).get('cached_tokens') == row['prefix_tokens'], f'{label}: copied checkpoint not consumed')
            check(validation.get('cached_output_equal') is True and src.get('output_ids') == dst.get('output_ids') and bool(src.get('output_ids')), f'{label}: cached continuation mismatch')
            check(src.get('meta_info', {}).get('cached_tokens', 0) >= row['prefix_tokens'], f'{label}: source continuation not cached')
            for endpoint, continuation in [('source', src), ('destination', dst)]:
                check(len(continuation.get('output_ids', [])) == 8 and continuation.get('meta_info', {}).get('completion_tokens') == 8, f'{label}: {endpoint} validation output length')
                check(continuation.get('meta_info', {}).get('prompt_tokens') == row['prefix_tokens'] + len(fixtures['suffix']), f'{label}: {endpoint} validation prompt length')
        else:
            check(not row.get('migration'), f'{label}: no-copy contains migration')
        summaries.append(summarize_run(row))
        pair_key = tuple(row[k] for k in SPEC_KEYS[:-1])
        check(row['mode'] not in by_pair[pair_key], f'{label}: duplicate paired mode')
        by_pair[pair_key][row['mode']] = (row, summaries[-1])
    pairs = []
    for group, variants in sorted(by_pair.items()):
        check(set(variants) == {'no-copy', 'copy'}, f'pair {group}: missing mode')
        if set(variants) != {'no-copy', 'copy'}:
            continue
        (baseline, base), (copy, treatment) = variants['no-copy'], variants['copy']
        check(baseline.get('parent_input_sha256') == copy.get('parent_input_sha256'), f'pair {group}: parent differs')
        source_equal = baseline.get('source_build', {}).get('output_ids') == copy.get('source_build', {}).get('output_ids')
        output_equal = [a.get('output_ids') == b.get('output_ids') for a, b in zip(baseline['requests'], copy['requests'])]
        if not source_equal or not all(output_equal):
            warnings.append(f'pair {group}: source build parity={source_equal}; foreground output parity={output_equal}')
        deltas = {metric: {'baseline': base[metric], 'copy': treatment[metric],
                          'delta_ms': treatment[metric] - base[metric] if treatment[metric] is not None and base[metric] is not None else None,
                          'delta_percent': (treatment[metric] / base[metric] - 1) * 100 if treatment[metric] is not None and base[metric] else None}
                  for metric in METRICS}
        p99 = deltas['gap_p99_ms']
        pairs.append({'target': group[0], 'concurrency': group[1], 'prefix_tokens': group[2], 'repetition': group[3],
                      'source_build_output_equal': source_equal, 'foreground_output_equal': output_equal,
                      'baseline_index': baseline['index'], 'copy_index': copy['index'], 'deltas': deltas,
                      'global_p99_screen': p99['delta_ms'] is not None and p99['delta_ms'] > 2 and p99['delta_percent'] > 10})
    return {'campaign': str(path.resolve()), 'audit_passed': not errors, 'errors': errors, 'warnings': warnings,
            'planned_runs': len(manifest), 'observed_runs': len(records), 'complete': complete,
            'full_48_design': len(manifest) == 48 and set(args['foreground_targets']) == {'source', 'destination'} and set(args['concurrency']) == {2, 8} and set(args['prefix_tokens']) == {8192, 24576} and args['repetitions'] == 3 and set(args['modes']) == {'copy', 'no-copy'},
            'backends': backends, 'runs': summaries, 'pairs': pairs,
            'interpretation': 'Client HTTP emission intervals, not GPU per-token times. Coalesced emissions never expanded. Quantiles pool valid gaps within each run only. Three runs, not tokens, are repetitions. RPC envelopes associate host timing only; COMPLETE status response can encompass destination installation but does not timestamp its exact kernel/lock interval. Fixed marker window is [-50,+350]ms; intersecting full gaps are retained without clipping. No-copy measures interference, not equal-work throughput. Output parity warnings are retained; no semantic-equivalence claim from a length-only pass.'}


def render_markdown(report):
    lines = ['# Local state-copy serving cadence', '', f"Audit: **{'PASS' if report['audit_passed'] else 'FAIL'}**; {report['observed_runs']}/{report['planned_runs']} runs. Full 48-run design: {report['full_48_design']}.", '', report['interpretation'], '']
    if report['errors'] or report['warnings']:
        lines += ['## Audit findings', '']
        lines += [f'- ERROR: {x}' for x in report['errors']]
        lines += [f'- WARNING: {x}' for x in report['warnings']]
        lines += ['']
    lines += ['## Paired changes', '', 'Positive means copy is slower. Every repetition is shown; no token-level significance test.', '',
              '| Target / concurrency / prefix | Rep | Global gap p99 Δ ms | Fixed-window max Δ ms | Fixed-window p99 Δ ms | Foreground finish Δ ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):
        return f'{value:.3f}' if value is not None else 'N/A'
    for pair in report['pairs']:
        values = [pair['deltas'][m]['delta_ms'] for m in ('gap_p99_ms', 'fixed_window_max_ms', 'fixed_window_p99_ms', 'foreground_total_ms')]
        lines.append(f"| {pair['target']} / {pair['concurrency']} / {pair['prefix_tokens']} | {pair['repetition']} | " + ' | '.join(map(fmt, values)) + ' |')
    lines += ['', '## Per-run observed gaps', '', '| Index | Target | C | Prefix | Mode | Rep | p99 ms | Max ms | Window max ms | >20 ms | >50 ms | Coalesced | Copy ms |', '| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in report['runs']:
        lines.append(f"| {run['index']} | {run['target']} | {run['concurrency']} | {run['prefix_tokens']} | {run['mode']} | {run['repetition']} | {fmt(run['gap_p99_ms'])} | {fmt(run['gap_max_ms'])} | {fmt(run['fixed_window_max_ms'])} | {run['gaps_over_20ms']} | {run['gaps_over_50ms']} | {run['coalesced_emissions']} | {fmt(run['migration_ms'])} |")
    lines += ['', '## Timing evidence', '', 'The JSON retains the 12 longest gaps per run with request identity, marker-relative timestamps and overlapping RPC envelopes, plus every fixed-window gap that intersects a control envelope. PREPARED and COMPLETE are response states, not exact staging/installation timestamps. This association alone does not establish causality.', '', '![Paired gap metrics](paired-gap-metrics.png)', '']
    return '\n'.join(lines)


def make_plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = sorted({(p['target'], p['concurrency'], p['prefix_tokens']) for p in report['pairs']})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, metric, label in zip(axes, ['gap_p99_ms', 'fixed_window_max_ms'], ['Whole-run observed gap p99 (ms)', 'Marker-window maximum gap (ms)']):
        for p in report['pairs']:
            x = groups.index((p['target'], p['concurrency'], p['prefix_tokens'])) + (p['repetition'] - 1) * .14
            d = p['deltas'][metric]
            if d['baseline'] is None or d['copy'] is None:
                continue
            ax.plot([x-.025, x+.025], [d['baseline'], d['copy']], color='0.6', linewidth=.7)
            ax.scatter(x-.025, d['baseline'], c='#2166ac', marker='o', s=22)
            ax.scatter(x+.025, d['copy'], c='#b2182b', marker='x', s=25)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.25)
    axes[0].scatter([], [], c='#2166ac', label='No copy')
    axes[0].scatter([], [], c='#b2182b', marker='x', label='Copy')
    axes[0].legend()
    axes[-1].set_xticks(range(len(groups)), [f'{r}\nC={c}, prefix={n}' for r, c, n in groups])
    fig.suptitle('Paired observations; each point is one run, three repetitions per condition')
    fig.tight_layout()
    fig.savefig(output/'paired-gap-metrics.png', dpi=180)
    fig.savefig(output/'paired-gap-metrics.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    report = analyze_campaign(args.campaign)
    (args.campaign/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.campaign/'RESULTS.md').write_text(render_markdown(report))
    make_plot(report, args.campaign)
    print(json.dumps({'audit_passed': report['audit_passed'], 'errors': report['errors'], 'warnings': report['warnings']}))
    raise SystemExit(0 if report['audit_passed'] else 2)


if __name__ == '__main__':
    main()
