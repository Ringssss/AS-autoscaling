"""Read-only audit of recorded evidence; writes only a derived audit artifact.

Run from any directory: python3 docs/review/audit_evidence.py
No engine requests, dataset imports, or GPU work.
"""
import ast
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = []


def read_json(relative):
    path = ROOT / relative
    SOURCES.append({'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return json.loads(path.read_text())


result = {'audit_date': '2026-09-30', 'scope': 'existing local records; no new serving measurements'}
rows = read_json('results/capacity-vs-hit.json')
capacity = {}
for cond in sorted({r['cond'] for r in rows}):
    selected = [r for r in rows if r['cond'] == cond]
    capacity[cond] = {
        'reps': len(selected),
        'first_event_median_s': statistics.median(r['ttft_s'] for r in selected),
        'free_before_recorded': [r.get('avail_before') for r in selected],
        'free_after': [r['avail_after'] for r in selected],
        'survivors_recorded': [r['survivors'] for r in selected],
    }
capacity['observed_miss_hit_ratio'] = capacity['miss_tight']['first_event_median_s'] / capacity['hit_tight']['first_event_median_s']
capacity['observed_miss_hit_difference_s'] = capacity['miss_tight']['first_event_median_s'] - capacity['hit_tight']['first_event_median_s']
capacity['miss_net_free_slot_decrease'] = 32768 - 4096
capacity['evicted_slots'] = None
capacity['causal_memory_pressure_penalty_s'] = None
result['capacity'] = capacity
sweeps = {}
for line in 'ABCDE':
    rows = read_json(f'experiments/weka_baselines/results/real-20260929/sweep-{line}.json')
    valid = [r for r in rows if r.get('dst_verified_runnable') and r.get('prefill_s') and r.get('transfer_s')]
    summary = []
    for n in sorted({r['n'] for r in valid}):
        selected = [r for r in valid if r['n'] == n]
        p = statistics.median(r['prefill_s'] for r in selected)
        t = statistics.median(r['transfer_s'] for r in selected)
        summary.append({'tokens': n, 'valid_reps': len(selected), 'first_event_median_s': p,
                        'engine_transfer_median_s': t, 'ratio_of_medians': p/t,
                        'state_bytes_median': statistics.median(r['bytes_transferred'] for r in selected)})
    sweeps[line] = {'recorded': len(rows), 'verified': len(valid), 'summary': summary,
                    'invalid': [r for r in rows if r not in valid]}
result['sweeps'] = sweeps
result['sweep_total'] = {'recorded': sum(r['recorded'] for r in sweeps.values()),
                        'verified': sum(r['verified'] for r in sweeps.values())}
result['paper_model_arithmetic'] = {'gamma_over_alpha_tokens': .45 / 2.17e-4,
                                    'kv_seconds_per_token_at_4_36_GiB_s': 20480 / (4.36 * 2**30)}
ceilings = read_json('experiments/weka_baselines/oracle/results/ceilings.json')
result['recorded_ceiling_ratios'] = {k: v / ceilings['total_input_tokens'] for k,v in ceilings['by_engines']['3'].items()}
# Execute only these pure definitions from the existing source; do not load its workload.
source = ROOT / 'experiments/weka_baselines/oracle/ceilings.py'
SOURCES.append({'path': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
names = {'b0_static_perfect', 'b1_mobile_perfect', 'b2_structural'}
tree = ast.parse(source.read_text())
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
namespace = {'hashlib': hashlib, 'BLOCK': 16}
exec(compile(module, str(source), 'exec'), namespace)
# With only one engine, two streams can reuse the same prefix without any migration.
toy = [{'key': 'a', 'ordinal': 0, 'hash_ids': [1, 2]},
       {'key': 'b', 'ordinal': 1, 'hash_ids': [1, 2]}]
result['ceiling_counterexample'] = {
    'engines': 1, 'requests': toy,
    'B0_hit_tokens': namespace['b0_static_perfect'](toy, 1),
    'B1_hit_tokens': namespace['b1_mobile_perfect'](toy, 1),
    'physical_migration_possible': False,
    'interpretation': 'B1-B0 includes cross-stream reuse; it is not an isolated mobility ceiling.'}
result['inputs'] = SOURCES
out = ROOT / 'docs/review/evidence-audit-2026-09-30.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(out)
print(json.dumps({'capacity_ratio': capacity['observed_miss_hit_ratio'],
                  'sweep_total': result['sweep_total'],
                  'counterexample': result['ceiling_counterexample']}, ensure_ascii=False, indent=2))
