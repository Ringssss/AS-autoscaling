"""Offline audit of exact recorded repartition trials, including incomplete campaigns."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.freetoken_elastic.benchmark_repartition import check_geometry, geometry_for, trial_matrix


def audit_trial(trial, spec, pair, config):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    for key, value in spec.items():
        require(trial.get(key) == value, f'identity mismatch: {key}')
    operations = trial.get('operations', [])
    by_label = {op['label']: op for op in operations}
    require(len(by_label) == len(operations), 'duplicate operation labels')
    require(trial.get('success') is True, f'trial unsuccessful: {trial.get("error")}')
    for op in operations:
        require(op.get('success') is True, f'failed operation: {op["label"]}: {op.get("error")}')
    for label, role, count in [('cold-parent', 'parent', config['parent_output']),
                               ('first-child', 'child', config['child_output']),
                               ('repeated-child', 'child', config['child_output'])]:
        op = by_label.get(label, {})
        prompt = pair[f'{role}_text']
        require((op.get('payload') or {}).get('prompt') == prompt, f'{label}: exact prompt mismatch')
        require(op.get('prompt_sha256') == hashlib.sha256(prompt.encode()).hexdigest(), f'{label}: prompt hash mismatch')
        require((op.get('payload') or {}).get('max_tokens') == count, f'{label}: requested output mismatch')
        require((op.get('usage') or {}).get('prompt_tokens') == pair[f'expected_{role}_tokens'], f'{label}: prompt token count mismatch')
        require((op.get('usage') or {}).get('completion_tokens') == count, f'{label}: output token count mismatch')
        require(op.get('cached_tokens') is not None, f'{label}: missing cache count')
        require(op.get('stream_done') is True, f'{label}: no stream terminal')
    require(config.get('cache_report_enabled') is True, 'cache reporting not asserted')
    require(by_label.get('cold-parent', {}).get('cached_tokens') == 0, 'reset parent not cold')
    initial = geometry_for(SimpleNamespace(**config))
    final = geometry_for(SimpleNamespace(**config), spec['condition'])
    changed = {k: v for k, v in final.items() if v != initial[k]}
    rebuild = by_label.get('measured-rebuild')
    if spec['condition'] == 'no-resize':
        require(rebuild is None, 'no-resize issued measured rebuild')
    else:
        require(rebuild is not None and rebuild.get('payload') == {'mode': 'if_idle', **changed}, 'measured rebuild includes wrong/unchanged pool fields')
    for label, expected in [('initial-status', initial), ('before-status', initial),
                            ('after-first-status', final), ('final-status', final)]:
        try:
            check_geometry(by_label.get(label, {'success': False}), expected)
        except ValueError as exc:
            errors.append(f'{label}: {exc}')
    instances = [(by_label.get(label, {}).get('body') or {}).get('instance_id')
                 for label in ('before-stats', 'after-first-stats', 'final-stats')]
    require(all(instances) and len(set(instances)) == 1, 'instance identity missing or changed')
    child = by_label.get('first-child', {})
    repeat = by_label.get('repeated-child', {})
    metrics = trial.get('metrics', {})
    rb_seconds = rebuild.get('elapsed_seconds') if rebuild else 0.0
    expected_metrics = {'rebuild_seconds': rb_seconds,
                        'first_child_text_seconds': child.get('first_text_seconds'),
                        'first_child_e2e_seconds': child.get('elapsed_seconds'),
                        'first_child_cached_tokens': child.get('cached_tokens'),
                        'repeated_child_cached_tokens': repeat.get('cached_tokens'),
                        'repeated_child_first_text_seconds': repeat.get('first_text_seconds'),
                        'repeated_child_e2e_seconds': repeat.get('elapsed_seconds')}
    if rb_seconds is not None and child.get('first_text_seconds') is not None:
        expected_metrics['rebuild_plus_first_text_seconds'] = rb_seconds + child['first_text_seconds']
    if rb_seconds is not None and child.get('elapsed_seconds') is not None:
        expected_metrics['rebuild_plus_child_e2e_seconds'] = rb_seconds + child['elapsed_seconds']
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        require(expected is not None and observed is not None and abs(observed - expected) < 1e-7,
                f'metric mismatch: {key}')
    for common, pure in [('intervention_to_first_text_seconds', 'rebuild_plus_first_text_seconds'),
                         ('intervention_to_completion_seconds', 'rebuild_plus_child_e2e_seconds')]:
        require(metrics.get(common, -1) >= metrics.get(pure, 0) - 1e-6, f'common clock shorter than components: {common}')
    return {**spec, 'recorded_success': trial.get('success', False), 'audit_pass': not errors,
            'errors': errors, 'metrics': metrics, 'instance_id': instances[0],
            'in_progress': 'finished_unix' not in trial and not trial.get('error'),
            'occupancy_observations': {label: (by_label.get(label, {}).get('body') or {})
                                       for label in ('before-stats', 'after-first-stats', 'final-stats')},
            'initial_geometry': initial, 'final_geometry': final,
            'first_output_sha256': hashlib.sha256(child.get('text', '').encode()).hexdigest(),
            'repeat_output_sha256': hashlib.sha256(repeat.get('text', '').encode()).hexdigest(),
            'first_repeat_text_agreement': child.get('text') == repeat.get('text') if child and repeat else None}


def build_report(root):
    campaign = json.loads((root / 'campaign.json').read_text())
    errors = []
    config = campaign['config']
    manifest = campaign['manifest']
    manifest_path = Path(config['manifest'])
    manifest_verified = False
    if manifest_path.exists():
        manifest_verified = (hashlib.sha256(manifest_path.read_bytes()).hexdigest() == campaign['manifest_sha256']
                             and json.loads(manifest_path.read_text()) == manifest)
        if not manifest_verified:
            errors.append('manifest file hash/content differs from frozen campaign')
    else:
        errors.append('original manifest file unavailable; byte hash cannot be verified')
    expected_matrix = trial_matrix(manifest['pairs'], config['conditions'], config['repetitions'], config['seed'])
    if campaign['matrix'] != expected_matrix:
        errors.append('frozen matrix differs from declared seed/design')
    pairs = {pair['id']: pair for pair in manifest['pairs']}
    rows, missing = [], []
    for spec in campaign['matrix']:
        path = root / f'trial-{spec["index"]:03d}.json'
        if not path.exists():
            missing.append(spec)
            continue
        trial = json.loads(path.read_text())
        rows.append(audit_trial(trial, spec, pairs[spec['pair_id']], config))
    recorded_runs = {r['index']: r for r in campaign.get('runs', [])}
    for row in rows:
        logged = recorded_runs.get(row['index'])
        if logged and logged.get('metrics') != row['metrics']:
            errors.append(f'campaign summary/raw metrics mismatch: {row["index"]}')
    baseline = {(r['pair_id'], r['repetition']): r for r in rows if r['condition'] == 'no-resize'}
    for row in rows:
        base = baseline.get((row['pair_id'], row['repetition']))
        row['first_text_matches_noresize'] = row['first_output_sha256'] == base['first_output_sha256'] if base else None
    instances = {r['instance_id'] for r in rows if r['instance_id']}
    if len(instances) > 1:
        errors.append('engine instance changed across trials')
    complete = campaign.get('complete') is True and not missing and len(recorded_runs) == len(expected_matrix)
    evidence_candidates = [root.parent / 'memory-24.jsonl', root.parent / 'memory-32.jsonl',
                           root.parent / 'ballast-24.json', root.parent / 'ballast-32.json',
                           root.parent / 'baseline-24/launch.json', root.parent / 'baseline-24/engine.log',
                           root.parent / 'baseline-32/launch.json', root.parent / 'baseline-32/engine.log']
    return {'campaign': str(root), 'separate_evidence_files': [str(p.resolve()) for p in evidence_candidates if p.exists()], 'complete': complete, 'planned_trials': len(expected_matrix),
            'present_trials': len(rows), 'missing_trials': missing, 'manifest_hash_verified': manifest_verified,
            'audit_pass': complete and not errors and all(r['audit_pass'] for r in rows),
            'errors': errors, 'trials': rows,
            'scope': 'Diagnostic unequal-final-geometry controls. Text agreement is not token-ID/numerical equivalence.',
            'offload_and_memory_proof': 'Separate environment/provenance artifacts required; cache geometry and HTTP stats alone do not prove peak capacity or expert offload.'}


def markdown(report):
    lines = ['# Repartition baseline audit', '',
             f"Completed campaign: {report['complete']}; present/planned: {report['present_trials']}/{report['planned_trials']}; audit pass: {report['audit_pass']}.", '',
             'All trials are listed, including failures. Controls have different final pools: these are diagnostic comparisons, not equal-budget system winners.', '',
             '| Pair | Rep | Condition | Valid | Child cache | Repeat cache | Rebuild s | First text s | Rebuild + text s | Common wall to text s | Child E2E s | Repeat E2E s | Text vs noresize |',
             '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in report['trials']:
        m = row['metrics']
        def f(key):
            value = m.get(key)
            return '-' if value is None else f'{value:.4f}'
        lines.append(f"| {row['pair_id']} | {row['repetition']} | {row['condition']} | {row['audit_pass']} | {m.get('first_child_cached_tokens')} | {m.get('repeated_child_cached_tokens')} | {f('rebuild_seconds')} | {f('first_child_text_seconds')} | {f('rebuild_plus_first_text_seconds')} | {f('intervention_to_first_text_seconds')} | {f('first_child_e2e_seconds')} | {f('repeated_child_e2e_seconds')} | {row['first_text_matches_noresize']} |")
    lines += ['', 'Repeated child can hit a deeper snapshot than the original parent; its timing is not an exact parent-only warm counterfactual. Common intervention clocks include client journal overhead; sums of component HTTP timings exclude gaps. First text is a decoded chunk, not an instrumented first raw token.', '',
              'Decoded-output agreement must be reported separately: different text can change decode work. Greedy sampling is not proof of cold/cached equivalence. CPU/HBM/host-bank residency and expert miss/fetch evidence are separate artifacts; no desktop-speed or capacity claim follows from this table.', '', '## Audit details', '']
    lines += ['Separate evidence files (links do not certify offload or memory-cap validity):', '']
    lines.extend(f'- [{Path(path).name}]({path})' for path in report.get('separate_evidence_files', []))
    lines.append('')
    lines.extend('- ' + error for error in report['errors'])
    for row in report['trials']:
        lines.extend(f"- Trial {row['index']}: {error}" for error in row['errors'])
    if report['missing_trials']:
        lines.append(f"- Missing trial indices: {[r['index'] for r in report['missing_trials']]}")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.campaign)
    (args.campaign / 'audit-report.json').write_text(json.dumps(report, indent=2))
    (args.campaign / 'RESULTS.md').write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ('complete', 'planned_trials', 'present_trials', 'audit_pass', 'errors')}))
