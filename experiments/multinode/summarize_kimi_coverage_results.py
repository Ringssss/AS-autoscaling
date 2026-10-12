"""Conditional held-out breadth; not natural migration-event prevalence."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(analysis,design,output):
    data=json.loads(analysis.read_text());d=json.loads(design.read_text());out=[]
    weights={s['stratum']:s['weight'] for s in d['strata']}
    for baseline in ['p2c','dualmap','cacheroute']:
        pairs=[c for c in data['comparisons'] if c['stage']=='coverage' and c['baseline']==baseline and c['treatment']=='p2c+fluidkv']
        if len(pairs)!=12: raise ValueError(f'Expected all 12 held-out pairs for {baseline}; got {len(pairs)}')
        for name in weights: assert sum(c['window'].split('-')[0]==name for c in pairs)==4
        def weighted(predicate):
            return sum(weights[name]*sum(predicate(c) for c in pairs if c['window'].split('-')[0]==name)/4 for name in weights)
        good=lambda c:c['mean_ttft_speedup']>=1.2
        guarded=lambda c:good(c) and c['mean_e2e_speedup']>=1/1.05 and c['treatment_slo']>=c['baseline_slo']-.05 and c['observation_clean']
        request_weight=lambda c:next(s['population_windows'] for s in d['strata'] if s['stratum']==c['window'].split('-')[0])/4
        out.append(dict(baseline=baseline,treatment='p2c+fluidkv',windows=12,
            mean_ttft_speedup_min_median_max=np.percentile([c['mean_ttft_speedup'] for c in pairs],[0,50,100]).tolist(),
            windows_mean_ttft_gain_ge20pct=sum(good(c) for c in pairs),
            windows_mean_ttft_gain_ge2x=sum(c['mean_ttft_speedup']>=2 for c in pairs),
            windows_p95_ttft_gain_ge20pct=sum(c['p95_ttft_speedup']>=1.2 for c in pairs),
            windows_mean_ttft_regression_ge10pct=sum(c['mean_ttft_speedup']<=1/1.1 for c in pairs),
            windows_guarded_gain=sum(guarded(c) for c in pairs),
            windows_with_observation_errors=sum(not c['observation_clean'] for c in pairs),
            stratum_weighted_window_fraction_mean_ttft_gain_ge20pct=weighted(good),
            stratum_weighted_window_fraction_guarded_gain=weighted(guarded),
            paired_requests=sum(c['paired_requests'] for c in pairs),
            requests_ttft_gain_ge20pct=sum(c['requests_ttft_gain_ge20pct'] for c in pairs),
            request_weighted_descriptive_gain_fraction=sum(request_weight(c)*c['requests_ttft_gain_ge20pct'] for c in pairs)/sum(request_weight(c)*c['paired_requests'] for c in pairs),
            scope='Conditional on constructed shared 5760 history, initial source withdrawal and cap64; one noisy run per held-out window. Not natural production migration frequency; no within-window replication CI.'))
    windows=sorted({c['window'] for c in data['comparisons'] if c['stage']=='coverage'})
    both=[]
    for window in windows:
        comparisons={c['baseline']:c for c in data['comparisons'] if c['stage']=='coverage' and c['window']==window and c['treatment']=='p2c+fluidkv'}
        both.append({'window':window,
            'beats_both_routers_ge20pct':all(comparisons[b]['mean_ttft_speedup']>=1.2 for b in ['dualmap','cacheroute']),
            'beats_both_routers_and_p2c_ge20pct':all(comparisons[b]['mean_ttft_speedup']>=1.2 for b in ['dualmap','cacheroute','p2c']),
            'same_router_fluidkv_speedup':comparisons['p2c']['mean_ttft_speedup']})
    result={'per_baseline':out,'joint_window_checks':both,
        'windows_beating_both_routers_ge20pct':sum(w['beats_both_routers_ge20pct'] for w in both),
        'windows_beating_both_routers_and_p2c_ge20pct':sum(w['beats_both_routers_and_p2c_ge20pct'] for w in both)}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--design',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.analysis,a.design,a.output)
