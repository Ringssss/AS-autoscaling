"""Offline audit and paired accounting for the bounded preservation experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from experiments.freetoken_elastic.report_repartition import audit_trial


def expected_matrix(campaign):
    c = campaign['config']; rng = random.Random(c['seed']); result = []
    for rep in range(c['repetitions']):
        for pair in campaign['manifest']['pairs']:
            for condition in c['conditions']:
                order = ['drop', 'preserve']; rng.shuffle(order)
                for policy in order:
                    result.append(dict(index=len(result), pair_id=pair['id'], condition=condition,
                                       repetition=rep, policy=policy))
    return result


def audit_preservation_trial(trial, spec, pair, config):
    row = audit_trial(trial, spec, pair, config)
    errors = row['errors']
    def require(value, message):
        if not value: errors.append(message)
    operations = {o['label']: o for o in trial.get('operations', [])}
    ids = []
    for label in ['reset-intermediate', 'reset-initial', 'measured-rebuild']:
        op = operations.get(label, {}); policy = op.get('experimental_policy') or {}; hooks = op.get('hook_records') or []
        expected = spec['policy'] if label == 'measured-rebuild' else 'drop'
        require(policy.get('mode') == expected, f'{label}: wrong policy mode')
        require(len(hooks) == 1, f'{label}: missing/ambiguous hook')
        hook = hooks[0] if len(hooks) == 1 else {}
        require(bool(policy.get('id')) and hook.get('id') == policy.get('id'), f'{label}: hook ID mismatch')
        ids.append(policy.get('id'))
        require(hook.get('success') is True and hook.get('mode') == expected, f'{label}: hook mode/failure')
        geometry = {k: v for k, v in (hook.get('geometry') or {}).items() if v is not None}
        requested = {k: v for k, v in (op.get('payload') or {}).items() if k != 'mode'}
        require(geometry == requested, f'{label}: executed hook geometry differs from payload')
        elapsed = hook.get('total_seconds'); begin = hook.get('start'); finish = hook.get('finish')
        require(all(isinstance(v, (float, int)) for v in (elapsed, begin, finish)) and abs(finish-begin-elapsed)<1e-6,
                f'{label}: hook duration inconsistent')
        if elapsed is not None:
            require(op.get('elapsed_seconds', -1) >= elapsed - 1e-5, f'{label}: HTTP shorter than hook')
        if label != 'measured-rebuild':
            require(not hook.get('export') and not hook.get('restore'), f'{label}: reset unexpectedly preserved')
    require(len(set(ids)) == len(ids), 'reused policy IDs within trial')
    op = operations.get('measured-rebuild', {}); hooks = op.get('hook_records') or []
    hook = hooks[0] if len(hooks) == 1 else {}; export = hook.get('export') or {}; restore = hook.get('restore') or {}
    child = operations.get('first-child', {}); boundary = export.get('boundary')
    phase = dict(export_seconds=export.get('export_seconds', 0),
                 original_rebuild_seconds=hook.get('original_rebuild_seconds', 0),
                 restore_seconds=restore.get('restore_seconds', 0), hook_total_seconds=hook.get('total_seconds'))
    if hook.get('total_seconds') is not None:
        phase['hook_other_seconds'] = hook['total_seconds'] - sum(phase[k] for k in ['export_seconds', 'original_rebuild_seconds', 'restore_seconds'])
        require(phase['hook_other_seconds'] >= -1e-6, 'phase durations exceed total hook duration')
    if spec['policy'] == 'preserve':
        candidates = [s['length'] for s in hook.get('snapshots_before', []) if s['length']>0 and s['length'] % 64 == 0]
        require(bool(candidates) and boundary == max(candidates), 'selection not longest existing aligned snapshot')
        require(restore.get('restored') is True and restore.get('boundary') == boundary, 'restore missing/wrong boundary')
        require(child.get('cached_tokens') == boundary, 'restored boundary not exactly consumed by first child')
        require(any(s.get('length') == boundary and s.get('slot') == restore.get('destination_slot') for s in hook.get('snapshots_after', [])), 'restored snapshot absent in published cache')
        unit = ((operations.get('before-status', {}).get('body') or {}).get('geometry') or {}).get('unit_bytes', {})
        if isinstance(boundary, int) and 'kv_per_token' in unit and 'mamba_per_slot' in unit:
            expected_bytes = boundary * (unit['kv_per_token'] + 4) + unit['mamba_per_slot']
            require(export.get('bytes') == restore.get('bytes') == expected_bytes, 'KV + recurrent/conv + int32 token byte mismatch')
            row['byte_accounting'] = dict(boundary=boundary, kv_bytes=boundary*unit['kv_per_token'],
                                          state_bytes=unit['mamba_per_slot'], token_bytes=boundary*4, total_bytes=expected_bytes)
        else: errors.append('insufficient unit-byte metadata')
        require(export.get('bytes', 0)+export.get('host_margin_bytes', 0)<=export.get('host_budget_bytes', -1), 'host export budget violated')
        if config.get('verify_bytes'):
            require(hook.get('bytes_equal') is True and hook.get('export_sha256') == hook.get('restored_sha256') and bool(hook.get('export_sha256')), 'byte verification missing/failed')
    else:
        require(not export and not restore, 'drop unexpectedly exported/restored')
        require(child.get('cached_tokens') == 0, 'drop first child unexpectedly cached')
    row.update(audit_pass=not errors, hook_ids=ids, hook=hook, phase=phase,
               verify_bytes=config.get('verify_bytes', False))
    return row


def build_report(root):
    campaign = json.loads((root/'campaign.json').read_text()); config = campaign['config']; errors=[]
    matrix = expected_matrix(campaign)
    if campaign['matrix'] != matrix: errors.append('matrix differs from declared seed/pair/policy ordering')
    original = Path(config['manifest']); manifest_verified = original.exists() and json.loads(original.read_text()) == campaign['manifest']
    if not manifest_verified: errors.append('manifest missing or changed')
    pairs={p['id']:p for p in campaign['manifest']['pairs']}; rows=[]; missing=[]
    for spec in matrix:
        path=root/f'trial-{spec["index"]:03d}.json'
        if not path.exists(): missing.append(spec); continue
        rows.append(audit_preservation_trial(json.loads(path.read_text()),spec,pairs[spec['pair_id']],config))
    identities=[i for row in rows for i in row['hook_ids'] if i]
    if len(set(identities))!=len(identities): errors.append('hook policy ID reused across trials')
    instances={r['instance_id'] for r in rows if r['instance_id']}
    if len(instances)>1: errors.append('engine instance changed across trials')
    summaries={r['index']:r for r in campaign.get('runs',[])}
    for row in rows:
        summary=summaries.get(row['index'])
        if summary and (summary.get('metrics')!=row['metrics'] or summary.get('hook')!=row['hook']): errors.append(f'raw/summary mismatch trial{row["index"]}')
    paired=[]
    keys={(r['pair_id'],r['condition'],r['repetition']) for r in rows}
    for key in sorted(keys):
        group={r['policy']:r for r in rows if (r['pair_id'],r['condition'],r['repetition'])==key}
        if set(group)!={'drop','preserve'}: continue
        drop,preserve=group['drop'],group['preserve']
        entry=dict(pair_id=key[0],condition=key[1],repetition=key[2],audit_pass=drop['audit_pass'] and preserve['audit_pass'],
                   same_final_geometry=drop['final_geometry']==preserve['final_geometry'],
                   output_text_agreement=drop['first_output_sha256']==preserve['first_output_sha256'])
        if not entry['same_final_geometry']: errors.append(f'paired final geometry mismatch {key}')
        for metric in ['rebuild_plus_first_text_seconds','rebuild_plus_child_e2e_seconds','intervention_to_first_text_seconds','intervention_to_completion_seconds']:
            d=drop['metrics'].get(metric); p=preserve['metrics'].get(metric)
            if d is not None and p is not None:
                entry[metric]=dict(drop=d,preserve=p,delta_seconds=p-d,change_percent=100*(p/d-1))
        pcost=preserve['phase']['export_seconds']+preserve['phase']['restore_seconds']
        total=preserve['metrics'].get('rebuild_plus_child_e2e_seconds')
        entry['export_restore_seconds']=pcost
        entry['export_restore_fraction_preserve_cycle']=pcost/total if total else None
        paired.append(entry)
    summaries_by_group=[]
    for key in sorted({(p['pair_id'],p['condition']) for p in paired}):
        group=[p for p in paired if (p['pair_id'],p['condition'])==key]
        valid=[p for p in group if p['audit_pass'] and p['same_final_geometry']]
        metrics={}
        for name in ['rebuild_plus_first_text_seconds','rebuild_plus_child_e2e_seconds','intervention_to_first_text_seconds','intervention_to_completion_seconds']:
            values=[p[name]['change_percent'] for p in valid if name in p]
            if values:metrics[name]=dict(min_percent=min(values),median_percent=statistics.median(values),max_percent=max(values))
        summaries_by_group.append(dict(pair_id=key[0],condition=key[1],observed_pairs=len(group),valid_pairs=len(valid),metrics=metrics))
    complete=campaign.get('complete') is True and not missing and len(summaries)==len(matrix)
    return dict(campaign=str(root),complete=complete,planned_trials=len(matrix),present_trials=len(rows),missing=missing,
                audit_pass=complete and not errors and all(r['audit_pass'] for r in rows),errors=errors,trials=rows,paired=paired,groups=summaries_by_group,
                manifest_content_verified=manifest_verified,manifest_current_sha256=hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,
                manifest_hash_scope='Campaign embeds original manifest content but did not save byte hash; current hash is provenance, not proof of original byte identity.',
                timing_order_scope='Frozen hook source executes export -> original rebuild -> restore; logs contain durations and whole-hook endpoints, not separate phase start timestamps.',
                strategy_scope='Longest existing 64-aligned checkpoint only; no future child passed to selector. Nonaligned snapshots may also be legal but this prototype excludes them. No threshold strategy implemented or evaluated.')


def render(report):
    lines=['# Matched-geometry preservation results','',f"Complete:{report['complete']}; trials:{report['present_trials']}/{report['planned_trials']}; audit pass:{report['audit_pass']}.",'',
           'Delta is preserve minus drop; negative is faster. Every paired repetition is shown, including invalid rows. Text mismatch is reported separately from transport/state byte correctness.','',
           '| Pair | Condition | Rep | Valid | Drop full cycle s | Preserve full cycle s | Delta % | Common wall delta % | Export+restore s | Cost fraction % | Output text equal |',
           '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for p in report['paired']:
        m=p.get('rebuild_plus_child_e2e_seconds',{}); w=p.get('intervention_to_completion_seconds',{})
        def fmt(v):return '-' if v is None else f'{v:.3f}'
        lines.append(f"| {p['pair_id']} | {p['condition']} | {p['repetition']} | {p['audit_pass']} | {fmt(m.get('drop'))} | {fmt(m.get('preserve'))} | {fmt(m.get('change_percent'))} | {fmt(w.get('change_percent'))} | {fmt(p['export_restore_seconds'])} | {fmt(100*p['export_restore_fraction_preserve_cycle'] if p['export_restore_fraction_preserve_cycle'] is not None else None)} | {p['output_text_agreement']} |")
    lines+=['',report['strategy_scope'],'',report['timing_order_scope'],'',report['manifest_hash_scope'],'',
            'Pure full cycle is rebuild HTTP plus child HTTP. Common wall includes journal/gaps. Formal timings without byte hashing must be distinguished from verified smoke runs. Repeated cold long-child misses are not a warm counterfactual; text chunks are not raw token timestamps. Host/GPU budgets and expert traffic require the separate environment evidence.','', '## Audit issues','']
    lines.extend('- '+e for e in report['errors'])
    for r in report['trials']: lines.extend(f"- Trial{r['index']}: {e}" for e in r['errors'])
    if report['missing']: lines.append(f"- Missing indices:{[s['index'] for s in report['missing']]}")
    return '\n'.join(lines)+'\n'


def plots(report, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid=[r for r in report['trials'] if r['audit_pass']]
    groups=sorted({(r['pair_id'],r['condition'],r['policy']) for r in valid})
    if not groups:return
    data=[]
    for key in groups:
        rs=[r for r in valid if (r['pair_id'],r['condition'],r['policy'])==key]
        parts={name:statistics.median(r['phase'][name] for r in rs) for name in ('export_seconds','original_rebuild_seconds','restore_seconds')}
        parts['other_rebuild']=statistics.median(r['metrics']['rebuild_seconds']-sum(r['phase'][k] for k in ('export_seconds','original_rebuild_seconds','restore_seconds')) for r in rs)
        parts['child']=statistics.median(r['metrics']['first_child_e2e_seconds'] for r in rs)
        data.append(dict(group=key,parts=parts))
    fig,ax=plt.subplots(figsize=(12,5)); bottom=[0.]*len(data)
    for name in data[0]['parts']:
        values=[r['parts'][name] for r in data];ax.bar(range(len(data)),values,bottom=bottom,label=name);bottom=[a+b for a,b in zip(bottom,values)]
    ax.set_xticks(range(len(data)),[' / '.join(r['group']) for r in data],rotation=35,ha='right');ax.set_ylabel('Seconds (component medians)');ax.legend();fig.tight_layout()
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(folder/f'latency-decomposition.{ext}',dpi=180)
    plt.close(fig)
    ps=[p for p in report['paired'] if p['audit_pass']];fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(range(len(ps)),[p['rebuild_plus_child_e2e_seconds']['change_percent'] for p in ps]);ax.axhline(0,color='black',linewidth=.8)
    ax.set_xticks(range(len(ps)),[f"{p['pair_id']}/{p['condition']}/r{p['repetition']}" for p in ps],rotation=35,ha='right');ax.set_ylabel('Full-cycle change % (negative better)');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(folder/f'paired-change.{ext}',dpi=180)
    plt.close(fig);(folder/'figure-data.json').write_text(json.dumps(dict(decomposition=data,paired=ps),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--plots',action='store_true');args=p.parse_args()
    report=build_report(args.campaign);(args.campaign/'audit-report.json').write_text(json.dumps(report,indent=2));(args.campaign/'RESULTS.md').write_text(render(report))
    if args.plots:plots(report,args.campaign)
    print(json.dumps({k:report[k] for k in ('complete','planned_trials','present_trials','audit_pass','errors')}))
