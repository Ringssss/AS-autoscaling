#!/usr/bin/env python3
"""Audit and report constructed agentic CPU-isolation experiments, without GPU access."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

METRICS = {'e2e_p50': 'Agent E2E p50 (s)', 'ttft_p50': 'Request TTFT p50 (s)',
           'decode_s_per_token_p50': 'Decode time/token p50 (s)',
           'tool_total_p50': 'Tool queue + service p50 (s)',
           'context_total_p50': 'Context queue + service p50 (s)'}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = int(k)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (k - lo)


def delta_percent(value, baseline):
    return 100 * (value / baseline - 1) if value is not None and baseline not in (None, 0) else None


def telemetry_metrics(folder, summary):
    records = read_jsonl(folder / 'telemetry.jsonl')
    samples = [r for r in records if r.get('event') == 'sample']
    final = next((r for r in reversed(records) if r.get('event') == 'summary'), None)
    metadata = next((r for r in records if r.get('event') == 'metadata'), {})
    engine = {r['pid'] for field in ('engine_affinity', 'engine_affinity_end') for r in summary.get(field, [])}
    workers = {r['pid'] for r in summary.get('worker_pings', [])}
    own_cpu = defaultdict(float)
    for sample in samples:
        for process in sample['processes']:
            category = 'engine' if process['pid'] in engine else 'workers' if process['pid'] in workers else 'controller_and_helpers'
            own_cpu[category] += process['cpu_seconds_delta']
    before, after = {}, {}
    for field, target in [('engine_affinity', before), ('engine_affinity_end', after)]:
        for proc in summary.get(field, []):
            for thread in proc.get('affinity_exceptions', []):
                target[(proc['pid'], thread['tid'])] = thread
    exceptions = []
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        exceptions.append({'pid': key[0], 'tid': key[1], 'name': (b or a)['name'],
                           'cpu_seconds_delta': max(0, b['cpu_seconds'] - a['cpu_seconds']) if a and b else None,
                           'before_cpu_seconds': a['cpu_seconds'] if a else None,
                           'after_cpu_seconds': b['cpu_seconds'] if b else None})
    gpus = [gpu for sample in samples for gpu in sample.get('gpu', {}).get('devices', [])]
    return {'samples': len(samples), 'complete': final is not None, 'error': final.get('error') if final else None,
            'sampled_cpu_seconds': final['sampled_cpu_seconds'] if final else sum(own_cpu.values()),
            'cpu_seconds_by_category': dict(own_cpu),
            'rss_sum_peak_bytes': max((r['rss_sum_bytes'] for r in samples), default=None),
            'sampling_wall_seconds': sum(r['sampling_wall_seconds'] for r in samples),
            'sampling_wall_max_seconds': max((r['sampling_wall_seconds'] for r in samples), default=None),
            'sampling_wall_sum_divided_by_workflow_seconds': sum(r['sampling_wall_seconds'] for r in samples) / summary['seconds'] if summary.get('seconds') else None,
            'process_discovery': metadata.get('process_discovery', 'legacy_or_unrecorded'),
            'gpu_utilization_sample_mean_percent': statistics.mean([g['utilization_percent'] for g in gpus if g.get('utilization_percent') is not None]) if any(g.get('utilization_percent') is not None for g in gpus) else None,
            'gpu_memory_peak_mib': max((g['memory_used_mib'] for g in gpus if g.get('memory_used_mib') is not None), default=None),
            'affinity_exceptions': exceptions,
            'affinity_exception_observed_cpu_seconds': sum(r['cpu_seconds_delta'] or 0 for r in exceptions),
            'affinity_exception_unmatched_threads': sum(r['cpu_seconds_delta'] is None for r in exceptions),
            'scope': 'Sampler boundaries include start/stop observer work outside benchmark seconds; categories use recorded PID sets; own CPU counters only. CPU includes client replay, workers and sampler overhead. Short-lived subprocesses and exit tails may be missed. GPU sample mean is not time-weighted.'}


def audit_run(folder, summary, args, records):
    errors = []
    ids = [r['agent'] for r in records]
    if sorted(ids) != list(range(args['agents'])):
        errors.append('offered agent IDs missing, duplicated, or unexpected')
    successful = [r for r in records if r.get('success')]
    if len(records) != summary['offered'] or len(successful) != summary['successful']:
        errors.append('summary offered/successful count differs from records')
    for row in records:
        for field in ('policy', 'concurrency', 'repetition'):
            if row[field] != summary[field]:
                errors.append(f'agent {row["agent"]}: {field} mismatch')
        if not row.get('success'):
            continue
        prefix = f'agent {row["agent"]}: '
        first, second, context, tool = [row[key] for key in ('first', 'second', 'context', 'tool')]
        for stage_name, stage in [('first', first), ('second', second)]:
            if stage['output_tokens'] != args['output_tokens'] or len(stage['output_ids']) != args['output_tokens']:
                errors.append(prefix + stage_name + ' output length mismatch')
        if first['input_tokens'] != args['prompt_tokens'] or second['input_tokens'] != len(context['input_ids']):
            errors.append(prefix + 'input length mismatch')
        digest = hashlib.sha256(json.dumps(context['input_ids']).encode()).hexdigest()
        tool_digest = hashlib.sha256(tool['text'].encode()).hexdigest()
        if digest != context['input_sha256']:
            errors.append(prefix + 'context hash invalid')
        if tool_digest != context['tool_sha256'] or tool_digest != tool['metrics']['output_sha256']:
            errors.append(prefix + 'tool output not matched to consumed context hash')
        if not tool['success'] or context['tool_bytes'] != len(tool['text'].encode()):
            errors.append(prefix + 'tool success/bytes mismatch')
        if context['input_ids'][args['prompt_tokens']:args['prompt_tokens'] + args['output_tokens']] != first['output_ids']:
            errors.append(prefix + 'first generated output not preserved in second input')
        if context['input_ids'][0] != 30000 + row['agent']:
            errors.append(prefix + 'unique agent first token mismatch')
    cpus = list(range(args['cpu_start'], args['cpu_start'] + args['cores']))
    if summary['union_cpus'] != cpus or summary['gpu_budget'] != 1:
        errors.append('CPU union or GPU budget mismatch')
    if summary['controller_cpus'] != [64, 65, 66, 67] or set(cpus) & set(summary['controller_cpus']):
        errors.append('controller affinity overlaps pool or differs from fixed controller budget')
    split = 2 if summary['policy'] == 'frontend-reserved2' else len(cpus)//2
    engine_cpus = cpus if summary['policy'] in ('shared', 'precomputed') else cpus[:split]
    tool_cpus = cpus if summary['policy'] in ('shared', 'precomputed') else cpus[split:]
    context_cpus = cpus[split:] if summary['policy'] in ('frontend-isolated', 'frontend-reserved2') else engine_cpus
    if summary['tool_cpus'] != tool_cpus or summary['context_cpus'] != context_cpus:
        errors.append('declared tool/context CPU sets differ from policy contract')
    for field in ('engine_affinity', 'engine_affinity_end'):
        for process in summary.get(field, []):
            exceptions = {str(r['tid']) for r in process.get('affinity_exceptions', [])}
            for tid, affinity in process.get('thread_affinity', {}).items():
                if tid not in exceptions and affinity != engine_cpus:
                    errors.append(f'{field} process {process["pid"]} thread {tid}: unexpected affinity')
    expected_workers = [tool_cpus] * args['tool_workers'] + [context_cpus] * 2
    if len(summary['worker_pings']) != len(expected_workers):
        errors.append('worker count mismatch')
    for worker, expected in zip(summary['worker_pings'], expected_workers):
        if worker['affinity'] != expected or any(a != expected for a in worker.get('thread_affinity', {}).values()):
            errors.append(f'worker {worker["pid"]}: unexpected affinity')
    return errors


def build_report(campaign):
    campaign = Path(campaign)
    contract = json.loads((campaign / 'campaign.json').read_text())
    args = contract['args']
    runs, all_records, errors = [], {}, []
    for folder in sorted(campaign.iterdir()):
        if not folder.is_dir() or not (folder / 'summary.json').exists():
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        records = read_jsonl(folder / 'requests.jsonl')
        key = (summary['concurrency'], summary['repetition'], summary['policy'])
        if key in all_records:
            errors.append(f'duplicate run key {key}')
        all_records[key] = records
        run_errors = audit_run(folder, summary, args, records)
        errors.extend(f'{folder.name}: {message}' for message in run_errors)
        keep = ['policy', 'concurrency', 'repetition', 'offered', 'successful', 'seconds', 'agent_goodput',
                'e2e_p50', 'e2e_p90', 'ttft_p50', 'ttft_p90', 'decode_s_per_token_p50', 'tool_total_p50',
                'context_total_p50', 'union_cpus', 'controller_cpus', 'gpu_budget']
        row = {k: summary[k] for k in keep}
        row.update(folder=folder.name, audit_errors=run_errors, telemetry=telemetry_metrics(folder, summary))
        successes = [r for r in records if r.get('success')]
        row['tool_service_cpu_seconds'] = sum(r['tool']['metrics']['cpu_seconds'] for r in successes) if summary['policy'] != 'precomputed' else 0
        row['context_service_cpu_seconds'] = sum(r['context']['cpu_seconds'] for r in successes)
        row['agent_admission_wait_p50'] = percentile([r['admitted'] - r['offered'] for r in successes], 50)
        row['tool_service_wall_p50'] = percentile([r['tool']['metrics']['wall_seconds'] for r in successes], 50) if summary['policy'] != 'precomputed' else 0
        row['context_service_wall_p50'] = percentile([r['context']['finish'] - r['context']['start'] for r in successes], 50)
        runs.append(row)
    identities = defaultdict(lambda: defaultdict(set))
    within_load = defaultdict(lambda: defaultdict(set))
    second_outputs = defaultdict(set)
    global_tools = set()
    for key, records in all_records.items():
        for row in records:
            if not row.get('success'):
                continue
            # Across repetitions AND policy; also checks both concurrency levels.
            identity = identities[row['agent']]
            identity['first_output'].add(tuple(row['first']['output_ids']))
            identity['second_input'].add(row['context']['input_sha256'])
            identity['tool_output'].add(row['tool']['metrics']['output_sha256'])
            load_identity = within_load[(key[0], row['agent'])]
            load_identity['first_output'].add(tuple(row['first']['output_ids']))
            load_identity['second_input'].add(row['context']['input_sha256'])
            load_identity['tool_output'].add(row['tool']['metrics']['output_sha256'])
            global_tools.add(row['tool']['metrics']['output_sha256'])
            second_outputs[(key[0], key[1], row['agent'])].add(tuple(row['second']['output_ids']))
    identity_differences = [{'agent': agent, 'field': field, 'distinct_values': len(values)}
                            for agent, fields in identities.items() for field, values in fields.items() if len(values) != 1]
    within_load_differences = [{'concurrency': load, 'agent': agent, 'field': field, 'distinct_values': len(values)}
                               for (load, agent), fields in within_load.items() for field, values in fields.items() if len(values) != 1]
    corpus = json.loads((campaign / 'corpus-manifest.json').read_text())
    corpus_bytes = sum(entry['bytes'] for entry in corpus['files'])
    corpus_digest = hashlib.sha256(json.dumps(corpus['files'], sort_keys=True).encode()).hexdigest()
    if corpus_digest != corpus['sha256']:
        errors.append('frozen corpus manifest digest mismatch')
    for key, records in all_records.items():
        for row in records:
            if not row.get('success'):
                continue
            metrics = row['tool']['metrics']
            if (metrics.get('manifest_sha256') != corpus_digest or metrics.get('corpus_bytes') != corpus_bytes
                    or metrics.get('corpus_files') != len(corpus['files']) or metrics.get('passes') != args['tool_passes']
                    or metrics.get('input_bytes') != corpus_bytes * args['tool_passes'] or metrics.get('operation') != 'ast'):
                errors.append(f'{key}, agent {row["agent"]}: tool corpus/operation/pass accounting mismatch')
    pairs = []
    indexed = {(r['concurrency'], r['repetition'], r['policy']): r for r in runs}
    for row in runs:
        if row['policy'] == 'shared':
            continue
        key = (row['concurrency'], row['repetition'])
        baseline = indexed.get((*key, 'shared'))
        if baseline is None:
            continue
        left = {r['agent']: r for r in all_records[(*key, 'shared')] if r.get('success')}
        right = {r['agent']: r for r in all_records[(*key, row['policy'])] if r.get('success')}
        mismatches = [i for i in sorted(set(left) & set(right))
                      if left[i]['first']['output_ids'] != right[i]['first']['output_ids']
                      or left[i]['context']['input_sha256'] != right[i]['context']['input_sha256']
                      or left[i]['tool']['metrics']['output_sha256'] != right[i]['tool']['metrics']['output_sha256']]
        second_mismatches = [i for i in sorted(set(left) & set(right)) if left[i]['second']['output_ids'] != right[i]['second']['output_ids']]
        metrics = list(METRICS) + ['e2e_p90', 'ttft_p90', 'agent_goodput']
        pairs.append({'concurrency': key[0], 'repetition': key[1], 'policy': row['policy'],
                      'equal_work_treatment': row['policy'] != 'precomputed',
                      'identity_mismatch_agent_ids': mismatches,
                      'second_output_mismatch_agent_ids': second_mismatches,
                      'second_output_mismatch_with_identical_input_agent_ids': [i for i in second_mismatches if left[i]['context']['input_sha256'] == right[i]['context']['input_sha256']],
                      'identity_unverified_agent_ids': sorted(set(range(args['agents'])) - (set(left) & set(right))),
                      'baseline_successful': baseline['successful'], 'treatment_successful': row['successful'],
                      'delta_percent': {m: delta_percent(row[m], baseline[m]) for m in metrics},
                      'sampled_cpu_delta_percent': delta_percent(row['telemetry']['sampled_cpu_seconds'], baseline['telemetry']['sampled_cpu_seconds'])})
    expected = {(r['concurrency'], r['repetition'], r['policy']) for r in contract['specifications']}
    missing = sorted(expected - set(indexed))
    if contract['complete'] and missing:
        errors.append('campaign marked complete with missing run summaries')
    limitations = [
        'Constructed repository-AST tool workflow, not a native production Qwen/Kimi agent trace. Tool outputs actually feed second-turn context.',
        'Finite burst with whole-agent concurrency cap; agent_goodput is burst completion rate, not sustainable steady-state SLO goodput.',
        'Shared and isolation variants retain the same configured physical CPU union and one GPU plus four controller cores. Core identity/topology needs the separate topology artifact; numeric CPU IDs alone do not prove physical-core equivalence.',
        'Precomputed removes actual tool work and is attribution-only; its speedup is not an equal-work disaggregation gain.',
        'Local CPU partitioning does not measure remote RPC, network serialization or multi-node placement costs.',
        'Kernel io_uring SQ threads may reject pinning. Recorded before/after exception CPU deltas are included; thread-exit tails or missing endpoints remain unestimated.',
        'CPU telemetry includes request client, workers and sampler overhead; its boundaries slightly exceed timed workflow. Categories are based on recorded engine/worker PID sets; other sampled processes are controller/helpers. Process RSS can double-count shared pages.',
        'One-second GPU samples may miss short phases. GPU utilization is driver device utilization, not SM occupancy. Torch/Nsight evidence must be interpreted separately.',
        'Observer wall time and maximum sample duration must be checked; this is not identical to added workflow delay because GPU subprocess waits release the GIL. Legacy smoke telemetry repeated whole-host process scans and heavily perturbed timing; it is correctness evidence only. Revised sampling traverses per-thread children within the workload tree.',
        'TTFT begins when an inference request is sent; agent E2E also includes admission queue, both generations, tool/context work and handoffs. Decode time/token is a streaming elapsed proxy, not kernel-only latency.',
        'Latency percentiles include successful agents only; all offered/successful counts and incomplete identities remain visible. Three repetitions do not establish significance.',
        'Exact input tokens and fixed output lengths preserve the workload contract but do not imply semantic parity. Second-turn output token identity is audited separately, including divergence despite identical second-turn inputs.',
        'Isolation can reduce engine CPU pooling. Improvements and regressions must be assessed jointly at the same work and total budget; this does not implement closed-loop autoscaling.'
    ]
    return {'campaign': str(campaign.resolve()), 'complete': contract['complete'], 'expected_runs': len(expected),
            'completed_runs': len(runs), 'missing_runs': missing, 'args': args, 'runs': runs, 'paired': pairs,
            'audit': {'errors': errors, 'identity_differences_across_policies_repeats_and_loads': identity_differences,
                      'identity_differences_across_policies_and_repeats_within_load': within_load_differences,
                      'second_output_differing_agent_keys': [{'concurrency': key[0], 'repetition': key[1], 'agent': key[2], 'distinct_outputs': len(values)} for key, values in second_outputs.items() if len(values)>1],
                      'second_output_observed_agent_keys': len(second_outputs),
                      'distinct_tool_hashes': len(global_tools), 'all_observed_tool_outputs_identical': len(global_tools) == 1,
                      'all_observed_agent_identities_identical': not identity_differences and bool(identities)},
            'corpus': {'sha256': corpus_digest, 'bytes': corpus_bytes, 'files': len(corpus['files']),
                       'passes_per_tool_call': args['tool_passes'], 'scope': 'Frozen manifest checksum and every successful tool record checked; runtime validates file bytes/hashes during each call. No post-hoc current-file contents substituted.'},
            'limitations': limitations}


def fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def write_markdown(report, path):
    audit = report['audit']
    lines = ['# Agentic CPU-isolation experiment', '',
             f'Completed {report["completed_runs"]}/{report["expected_runs"]} runs; campaign complete: {report["complete"]}.', '',
             f'CPU pool: {report["args"]["cores"]} configured physical cores plus 4 fixed controller cores; GPU budget: 1.', '',
             f'Tool corpus: {report["corpus"]["files"]} files, {report["corpus"]["bytes"]} bytes, {report["corpus"]["passes_per_tool_call"]} pass(es) per tool call. Every successful call is checked against the frozen manifest and byte counts.', '',
             f'Accounting/affinity audit errors: {len(audit["errors"])}. Cross-policy/repetition/load identity differences: {len(audit["identity_differences_across_policies_repeats_and_loads"])}. Distinct tool outputs: {audit["distinct_tool_hashes"]}.', '',
             f'Within-load cross-policy/repetition identity differences: {len(audit["identity_differences_across_policies_and_repeats_within_load"])}. Paired comparisons with mismatched/unverified identities: {sum(bool(p["identity_mismatch_agent_ids"] or p["identity_unverified_agent_ids"]) for p in report["paired"])}/{len(report["paired"])}. Cross-load differences alone do not invalidate exact matched-work comparisons within each load.', '',
             f'Second-turn output divergence across policies occurs for {len(audit["second_output_differing_agent_keys"])}/{audit["second_output_observed_agent_keys"]} (load, repetition, agent) keys. Identical inputs and fixed output lengths do not establish semantic parity; report.json lists second-output differences for every pair, including those with identical inputs.', '',
             'A negative latency delta is better. Precomputed is an attribution control with less work, never an equal-work win. All paired measurements remain shown, including identity mismatches and failures.', '',
             '| Load | Rep | Policy | Success/offered | E2E p50 s | TTFT p50 s | Decode ms/token | Tool total s | Context total s |',
             '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in report['runs']:
        lines.append(f'| {row["concurrency"]} | {row["repetition"]} | {row["policy"]} | {row["successful"]}/{row["offered"]} | {fmt(row["e2e_p50"])} | {fmt(row["ttft_p50"])} | {fmt(row["decode_s_per_token_p50"] * 1000 if row["decode_s_per_token_p50"] is not None else None)} | {fmt(row["tool_total_p50"])} | {fmt(row["context_total_p50"])} |')
    lines += ['', '## Paired change versus shared (%)', '',
              '| Load | Rep | Policy | E2E p50 | TTFT p50 | Decode | Tool total | Context total | Identity mismatch / missing |',
              '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in report['paired']:
        values = ' | '.join(fmt(row['delta_percent'][m], 1) for m in METRICS)
        lines.append(f'| {row["concurrency"]} | {row["repetition"]} | {row["policy"]} | {values} | {len(row["identity_mismatch_agent_ids"])}/{len(row["identity_unverified_agent_ids"])} |')
    lines += ['', '## Sampled CPU accounting', '',
              '| Load | Rep | Policy | Total CPU s | Engine CPU s | Workers CPU s | Controller/helpers CPU s | Pin-exception observed CPU s | Observer wall s |',
              '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in report['runs']:
        t = row['telemetry']; c = t['cpu_seconds_by_category']
        lines.append(f'| {row["concurrency"]} | {row["repetition"]} | {row["policy"]} | {fmt(t["sampled_cpu_seconds"])} | {fmt(c.get("engine",0))} | {fmt(c.get("workers",0))} | {fmt(c.get("controller_and_helpers",0))} | {fmt(t["affinity_exception_observed_cpu_seconds"])} | {fmt(t["sampling_wall_seconds"])} |')
    if audit['errors']:
        lines += ['', '## Audit errors', ''] + [f'- {e}' for e in audit['errors']]
    if audit['identity_differences_across_policies_repeats_and_loads']:
        lines += ['', '## Identity differences', '', 'The following aggregate includes differences between load levels. Use the per-pair mismatch column and within-load audit to assess matched-work validity; cross-load differences alone do not invalidate within-load pairs.']
        lines += [f'- Agent {r["agent"]}, {r["field"]}: {r["distinct_values"]} distinct values.' for r in audit['identity_differences_across_policies_repeats_and_loads']]
    lines += ['', '## Limits', ''] + [f'- {line}' for line in report['limitations']]
    path.write_text('\n'.join(lines) + '\n')


def plot(report, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(exist_ok=True)
    policies = sorted({r['policy'] for r in report['paired']})
    loads = sorted({r['concurrency'] for r in report['paired']})
    for name, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = []
        position = 0
        for load in loads:
            for policy in policies:
                rows = [r for r in report['paired'] if r['concurrency'] == load and r['policy'] == policy and r['delta_percent'][name] is not None]
                if not rows:
                    continue
                x = [position + (i-(len(rows)-1)/2)*0.09 for i in range(len(rows))]
                ax.scatter(x, [r['delta_percent'][name] for r in rows], marker='x' if policy == 'precomputed' else 'o')
                for xx, row in zip(x, rows):
                    ax.annotate(f'r{row["repetition"]}' + ('*' if row['identity_mismatch_agent_ids'] or row['identity_unverified_agent_ids'] else ''), (xx, row['delta_percent'][name]), xytext=(4, 3), textcoords='offset points', fontsize=7)
                labels.append(f'c{load}\n{policy}')
                position += 1
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel('Paired change vs shared (%) — lower is better')
        ax.set_title(label + '; finite-burst workflow')
        fig.text(.01, .01, '* unmatched/unverified identity; precomputed removes work (attribution only)', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(folder / f'{name}.png', dpi=160)
        fig.savefig(folder / f'{name}.svg')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    report = build_report(args.campaign)
    (args.campaign / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_markdown(report, args.campaign / 'RESULTS.md')
    if not args.no_plots:
        plot(report, args.campaign / 'figures')
    print(json.dumps({'completed_runs': report['completed_runs'], 'audit': report['audit']}))


if __name__ == '__main__':
    main()
