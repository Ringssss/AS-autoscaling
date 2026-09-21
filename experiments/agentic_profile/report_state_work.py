#!/usr/bin/env python3
"""Independent offline accounting for the bounded two-GPU state-work experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def median(values):
    return statistics.median(values) if values else None


def change(value, baseline):
    return 100 * (value / baseline - 1) if value is not None and baseline not in (None, 0) else None


def close(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a-b) < 1e-6 * max(1, abs(b))


def union_overlap(preparations, foreground):
    intersections = sorted((max(p['start'], f['start']), min(p['finish'], f['finish']))
                           for p in preparations for f in foreground
                           if max(p['start'], f['start']) < min(p['finish'], f['finish']))
    merged = []
    for start, finish in intersections:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], finish)
        else:
            merged.append([start, finish])
    return sum(b-a for a,b in merged)


def valid_generation(row, inputs, outputs):
    meta = row.get('meta_info', {})
    return (row.get('success') is True and row.get('input_tokens') == inputs
            and row.get('requested_output_tokens') == outputs
            and meta.get('prompt_tokens') == inputs and meta.get('completion_tokens') == outputs
            and len(row.get('output_ids', [])) == outputs)


def telemetry_summary(folder):
    path = Path(folder)/'telemetry.jsonl'
    if not path.exists():
        return {'available':False}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    samples = [r for r in records if r.get('event')=='sample']
    ending = next((r for r in reversed(records) if r.get('event')=='summary'),None)
    metadata = next((r for r in records if r.get('event')=='metadata'),{})
    gpus = {}
    for sample in samples:
        for gpu in sample.get('gpu',{}).get('devices',[]):
            if gpu.get('memory_used_mib') is None or gpu.get('memory_total_mib') is None:
                continue
            free = gpu['memory_total_mib']-gpu['memory_used_mib']
            item = gpus.setdefault(str(gpu['index']), {'uuid':gpu['uuid'],'samples':0,'minimum_free_mib':free,'maximum_used_mib':gpu['memory_used_mib']})
            item['samples'] += 1
            if free < item['minimum_free_mib']:
                item['minimum_free_mib'] = free
                item['minimum_free_sample_monotonic'] = sample['monotonic_seconds']
            item['maximum_used_mib'] = max(item['maximum_used_mib'],gpu['memory_used_mib'])
    return {'available':True,'complete':ending is not None,'samples':len(samples),
            'sampled_cpu_seconds':ending['sampled_cpu_seconds'] if ending else sum(s.get('cpu_seconds_delta',0) for s in samples),
            'root_pids':metadata.get('root_pids'), 'sampling_wall_seconds':sum(s['sampling_wall_seconds'] for s in samples),
            'sampling_wall_max_seconds':max((s['sampling_wall_seconds'] for s in samples),default=None),
            'sampler_error':ending.get('error') if ending else 'missing final observer summary',
            'gpu_memory':gpus,
            'scope':'Whole observer lifetime, including excluded checkpoint builds/warmups/validation, not only measured trial intervals. CPU sums own process counters and includes observer/client; short-lived children and exit tails can be missed. GPU free memory is driver-sampled MiB, not allocator headroom or a guarantee against between-sample peaks.'}


def evaluate_row(row, config):
    errors = []
    foreground = row.get('foreground', [])
    prep = row.get('preparations', [])
    validation = row.get('validation', [])
    offered = 0 if row['mode'] == 'preparation-only' else row['concurrency']
    expected_prep = 0 if row['method'] == 'none' else config['preparations']
    if len(foreground) != offered:
        errors.append('foreground row count differs from offered work')
    if len(prep) != expected_prep:
        errors.append('preparation count differs from offered work')
    successful = [f for f in foreground if valid_generation(f, config['foreground_input'], config['foreground_output'])]
    if row.get('foreground_offered', len(foreground)) != offered or row.get('foreground_successful', len(successful)) != len(successful):
        errors.append('foreground summary counts disagree with independently checked work')
    if len(successful) != len(foreground):
        errors.append('foreground failure or input/output length mismatch')
    if [f.get('input_sha256') for f in foreground] != config['foreground_hashes'][:offered]:
        errors.append('foreground input identities differ from frozen fixture')
    if [p.get('input_sha256') for p in prep] != config['parent_hashes'][:expected_prep]:
        errors.append('preparation identities differ from frozen fixture')
    for p in prep:
        if not p.get('success') or p.get('prefix_tokens') != config['prefix_tokens']:
            errors.append('preparation failed or prefix length mismatch')
        if row['method'] == 'replay' and not valid_generation(p.get('generation', {}), config['prefix_tokens'], 1):
            errors.append('replay input/output length or execution failure')
        if row['method'] == 'copy' and p.get('transfer_bytes', 0) <= 0:
            errors.append('copy did not record positive transfer bytes')
    if len(validation) != expected_prep:
        errors.append('prepared-prefix validation count mismatch')
    consumed, parity, validation_ok = [], [], []
    for v in validation:
        source, target = v.get('source', {}), v.get('destination', {})
        okay = all(valid_generation(g, config['prefix_tokens']+64, 8) for g in (source,target))
        hit = target.get('meta_info', {}).get('cached_tokens') == config['prefix_tokens']
        equal = okay and source.get('output_ids') == target.get('output_ids')
        if hit != v.get('destination_hit') or equal != v.get('cached_continuation_equal'):
            errors.append('validation flags disagree with actual runtime counts/outputs')
        validation_ok.append(okay); consumed.append(hit); parity.append(equal)
    execution = (len(successful) == offered and len(prep) == expected_prep and all(p.get('success') for p in prep))
    mechanism = execution and row.get('cleanup_confirmed', False) and (row['method'] == 'none' or
                len(validation) == expected_prep and all(validation_ok) and all(consumed) and (row['method'] != 'copy' or all(parity)))
    # Recomputed flags never substitute cold/cached replay quality for copy parity.
    if row.get('mechanism_valid') != mechanism:
        errors.append('mechanism_valid differs from independently recomputed gate')
    if row.get('execution_success') != execution:
        errors.append('execution_success differs from offered/terminal work')
    if not row.get('cleanup_confirmed'):
        errors.append('transfer cleanup not confirmed')
    start, finish = row['start'], row['finish']
    metrics = {'measured_makespan_seconds': finish-start, 'charged_gpu_seconds': 2*(finish-start),
               'foreground_ready_to_first_p50_seconds': median([f['first']-start for f in successful]),
               'foreground_ready_to_finish_p50_seconds': median([f['finish']-start for f in successful]),
               'foreground_ttft_p50_seconds': median([f['ttft_seconds'] for f in successful]),
               'foreground_e2e_p50_seconds': median([f['e2e_seconds'] for f in successful]),
               'foreground_decode_p50_seconds_per_token': median([f['mean_decode_seconds_per_token'] for f in successful if f.get('mean_decode_seconds_per_token') is not None]),
               'preparation_total_seconds': sum(p['finish']-p['start'] for p in prep),
               'request_lifetime_overlap_seconds': union_overlap(prep,foreground)}
    for name, value in metrics.items():
        if not close(row.get(name), value):
            errors.append(f'{name} does not match raw timestamps')
    for timed in foreground+prep:
        if timed['start'] < start-1e-6 or timed['finish'] > finish+1e-6 or timed['finish'] < timed['start']:
            errors.append('measured stage outside common interval')
    for built in row.get('source_checkpoint_build', []):
        if built['finish'] > start+1e-6:
            errors.append('source checkpoint build overlaps measured interval')
    for v in validation:
        if any(g.get('start', finish) < finish-1e-6 for g in (v.get('source',{}), v.get('destination',{}))):
            errors.append('consumption validation overlaps measured interval')
    output = {k: row[k] for k in ('index','method','mode','foreground_target','concurrency','repetition')}
    output.update(metrics, offered_foreground=offered, successful_foreground=len(successful),
                  offered_preparations=expected_prep, successful_preparations=sum(bool(p.get('success')) for p in prep),
                  execution_success=execution, mechanism_valid=mechanism,
                  consumed_count=sum(consumed), validation_count=len(validation), cached_parity_count=sum(parity),
                  replay_quality_gate_passed=all(parity) if validation and row['method']=='replay' else None,
                  copy_bytes=sum(p.get('transfer_bytes',0) for p in prep),
                  reconstructed_tokens=sum(p.get('reconstructed_tokens',0) for p in prep),
                  errors=errors, harness_error=row.get('error'))
    return output


def build_report(folder):
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    raw = [json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines() if line.strip()] if (folder/'records.jsonl').exists() else []
    rows = [evaluate_row(row, config) for row in raw]
    fields = ('foreground_target','concurrency','repetition','method','mode')
    key = lambda r: tuple(r[f] for f in fields)
    expected = Counter(key(r) for r in manifest)
    observed = Counter(key(r) for r in rows)
    errors = []
    configured = Counter()
    for target in config['foreground_targets']:
        for concurrency in config['concurrency']:
            for repetition in range(config['repetitions']):
                conditions = [('none','foreground-only')]
                conditions += [(method,mode) for method in config['methods'] for mode in ('preparation-only','serial','concurrent')]
                for method,mode in conditions:
                    if mode in config['modes']:
                        configured[(target,concurrency,repetition,method,mode)] += 1
    if expected != configured: errors.append('manifest condition matrix differs from configured factorial experiment')
    if any(count!=1 for count in expected.values()): errors.append('manifest has duplicate conditions')
    if any(count!=1 for count in observed.values()): errors.append('records have duplicate conditions')
    if set(observed)-set(expected): errors.append('records include unexpected conditions')
    for row in rows:
        errors.extend(f'row {row["index"]}: {error}' for error in row['errors'])
        if row['index'] >= len(manifest) or key(row) != key(manifest[row['index']]): errors.append(f'row {row["index"]}: manifest index mismatch')
    if config.get('gpu_budget') != 2: errors.append('GPU budget is not two')
    complete_record = json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else {}
    complete = complete_record.get('complete', False)
    missing = list((expected-observed).elements())
    if complete and (missing or complete_record.get('runs') != len(raw)): errors.append('complete marker count mismatch')
    lookup = {key(row): row for row in rows}
    pairs = []
    service_metrics = ('foreground_ttft_p50_seconds','foreground_e2e_p50_seconds','foreground_decode_p50_seconds_per_token')
    common_metrics = ('foreground_ready_to_first_p50_seconds','foreground_ready_to_finish_p50_seconds','measured_makespan_seconds','charged_gpu_seconds','preparation_total_seconds')
    for row in rows:
        if row['mode'] != 'concurrent': continue
        prefix = (row['foreground_target'], row['concurrency'], row['repetition'])
        serial = lookup.get((*prefix,row['method'],'serial'))
        baseline = lookup.get((*prefix,'none','foreground-only'))
        preparation = lookup.get((*prefix,row['method'],'preparation-only'))
        pairs.append({k:row[k] for k in fields[:-1]} | {
            'concurrent_vs_serial_delta_percent': {m:change(row[m],serial[m]) for m in common_metrics+service_metrics} if serial else None,
            'concurrent_vs_foreground_only_delta_percent': {m:change(row[m],baseline[m]) for m in service_metrics} if baseline else None,
            'concurrent_vs_preparation_only_delta_percent': change(row['preparation_total_seconds'],preparation['preparation_total_seconds']) if preparation else None,
            'serial_and_concurrent_mechanism_valid': bool(serial and serial['mechanism_valid'] and row['mechanism_valid']),
            'serial_and_concurrent_accounting_valid': bool(serial and not serial['errors'] and not row['errors']),
            'replay_quality_passed_both': bool(serial and serial['replay_quality_gate_passed'] and row['replay_quality_gate_passed']) if row['method']=='replay' else None})
    return {'campaign':str(folder.resolve()), 'config':config, 'complete':complete, 'expected_runs':len(manifest),
            'observed_runs':len(rows),'missing_conditions':missing,'audit_errors':errors,'runs':rows,'paired':pairs,
            'telemetry':telemetry_summary(folder),
            'limits':[
                'Two GPUs are reserved and charged as 2 × measured makespan in every condition, including single-work controls. This is phase-window resource occupancy, not complete GPU cost or an elastic release experiment.',
                'Source checkpoint creation, cache flushing, group setup, graph/path warmup, metadata setup and post-run consumption validation are outside the measured interval. Reusing already-built source state is the explicitly measured premise.',
                'Serial and concurrent conditions have the same ready boundary; serial foreground waits for preparation. Service-only TTFT hides that wait, so common-ready-to-first/finish and makespan are reported independently.',
                'Concurrent preparation starts after the first foreground token. Request-lifetime overlap includes host and queue time; it is not evidence of concurrent CUDA kernels or SM/HBM contention.',
                'Source-vs-destination placement compares foreground locality during source-to-destination preparation. Both GPUs remain reserved; this is not a complete scheduler comparison or evidence that a spare GPU is free.',
                'Copy validity requires successful exact-length generation, consumed prepared prefixes and source-cached continuation parity. Replay cold/cached parity is a separate quality gate; failing replay quality cannot be claimed as semantically equivalent acceleration.',
                'Deterministic token fixtures and fixed independent foreground requests do not establish production agent arrival, tool-duration or reuse distributions.',
                'Preparation-only and foreground-only runs are attribution controls with different work; serial/concurrent is the equal-work comparison. Three paired repetitions show variability, not statistical significance.',
                'Complete GPU disaggregation would need handoff, transfer topology, lost pooling capacity and full application E2E accounting; these two-GPU local measurements identify interference conditions only.'
            ]}


def fmt(value):
    return 'n/a' if value is None else f'{value:.3f}'


def write_report(report, folder):
    folder = Path(folder)
    lines = ['# State preparation and foreground interference', '',
             f'Completed marker: {report["complete"]}; {report["observed_runs"]}/{report["expected_runs"]} conditions recorded; {len(report["audit_errors"])} independent accounting errors.', '',
             'All timings below are seconds. Both GPUs are charged. Negative paired latency/makespan change is better. Rows with invalid mechanism/accounting remain visible but cannot support a valid performance claim.', '',
             '| Endpoint | Load | Rep | Method | Mode | FG success/offered | Prep success/offered | Mechanism | Cached parity | Common first | Common finish | Makespan | GPU-s | Prep s | Lifetime overlap |',
             '|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        lines.append(f'| {r["foreground_target"]} | {r["concurrency"]} | {r["repetition"]} | {r["method"]} | {r["mode"]} | {r["successful_foreground"]}/{r["offered_foreground"]} | {r["successful_preparations"]}/{r["offered_preparations"]} | {r["mechanism_valid"]} | {r["cached_parity_count"]}/{r["validation_count"]} | '+ ' | '.join(fmt(r[m]) for m in ('foreground_ready_to_first_p50_seconds','foreground_ready_to_finish_p50_seconds','measured_makespan_seconds','charged_gpu_seconds','preparation_total_seconds','request_lifetime_overlap_seconds'))+' |')
    lines += ['', '## Paired change (%)', '',
              '| Endpoint | Load | Rep | Method | Common first vs serial | Common finish vs serial | Makespan vs serial | Service TTFT vs FG only | Decode vs FG only | Prep vs prep only | Valid accounting/mechanism | Replay quality |',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for p in report['paired']:
        a=p['concurrent_vs_serial_delta_percent'] or {}; b=p['concurrent_vs_foreground_only_delta_percent'] or {}
        values=[a.get('foreground_ready_to_first_p50_seconds'),a.get('foreground_ready_to_finish_p50_seconds'),a.get('measured_makespan_seconds'),b.get('foreground_ttft_p50_seconds'),b.get('foreground_decode_p50_seconds_per_token'),p['concurrent_vs_preparation_only_delta_percent']]
        lines.append(f'| {p["foreground_target"]} | {p["concurrency"]} | {p["repetition"]} | {p["method"]} | '+' | '.join(fmt(v) for v in values)+f' | {p["serial_and_concurrent_accounting_valid"]}/{p["serial_and_concurrent_mechanism_valid"]} | {p["replay_quality_passed_both"]} |')
    if report['audit_errors']:
        lines += ['', '## Accounting errors', '']+[f'- {e}' for e in report['audit_errors']]
    telemetry=report['telemetry']
    if telemetry['available']:
        lines += ['', '## Whole-campaign telemetry', '',
                  f'{telemetry["samples"]} samples; sampled own-process CPU total {fmt(telemetry["sampled_cpu_seconds"])} CPU-seconds. Observer complete: {telemetry["complete"]}; error: {telemetry["sampler_error"]}.', '',
                  '| GPU | Minimum sampled free MiB | Maximum sampled used MiB | Samples |',
                  '|---|---:|---:|---:|']
        lines += [f'| {gpu} | {fmt(value["minimum_free_mib"])} | {fmt(value["maximum_used_mib"])} | {value["samples"]} |' for gpu,value in telemetry['gpu_memory'].items()]
        lines += ['',telemetry['scope']]
    lines += ['', '## Scope and limits', '']+[f'- {s}' for s in report['limits']]
    related = sorted(p for p in folder.parent.glob('*/RESULTS.md') if p.parent != folder)
    if related:
        lines += ['', '## Other suite reports', '']+[f'- [{p.parent.name}](../{p.parent.name}/RESULTS.md)' for p in related]
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def plot(report, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)/'figures';folder.mkdir(exist_ok=True)
    metrics=[('concurrent_vs_serial_delta_percent','foreground_ready_to_first_p50_seconds','Common-ready to first vs serial'),
             ('concurrent_vs_serial_delta_percent','foreground_ready_to_finish_p50_seconds','Common-ready to finish vs serial'),
             ('concurrent_vs_serial_delta_percent','measured_makespan_seconds','Makespan vs serial (both GPUs charged)'),
             ('concurrent_vs_foreground_only_delta_percent','foreground_decode_p50_seconds_per_token','Decode time/token vs foreground only')]
    for group,metric,title in metrics:
        fig,ax=plt.subplots(figsize=(10,4));labels=[];values=[]
        for p in report['paired']:
            value=(p[group] or {}).get(metric)
            if value is None:continue
            labels.append(f'{p["foreground_target"][:3]} c{p["concurrency"]}\n{p["method"]} r{p["repetition"]}'+('*' if not p['serial_and_concurrent_mechanism_valid'] or not p['serial_and_concurrent_accounting_valid'] else ''))
            values.append(value)
        ax.scatter(range(len(values)),values);ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(range(len(labels)),labels,rotation=60,ha='right',fontsize=7)
        ax.set_ylabel('Paired change (%)');ax.set_title(title)
        fig.text(.01,.01,'* invalid accounting/mechanism; replay quality is separate; lower is better',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));fig.savefig(folder/f'{metric}.png',dpi=160);fig.savefig(folder/f'{metric}.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    report=build_report(args.campaign);write_report(report,args.campaign)
    if not args.no_plots:plot(report,args.campaign)
    print(json.dumps({'complete':report['complete'],'runs':report['observed_runs'],'expected':report['expected_runs'],'audit_errors':report['audit_errors']}))


if __name__=='__main__':main()
