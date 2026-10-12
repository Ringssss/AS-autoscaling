"""Paired routing/preparation comparison retaining failures and token mismatches."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(root):
    groups=defaultdict(dict)
    for path in root.glob('*.json'):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or 'requests' not in data or 'summary' not in data:continue
        s=data['summary']
        key=(s['prefix_tokens'],s['offered_per_engine'],s['regime'],s['repeat'])
        if s['policy'] in groups[key]:raise ValueError('duplicate trial')
        if s['completed']!=len(data['requests']):raise ValueError('request count mismatch')
        groups[key][s['policy']]=data
    pairs=[]
    for key,policies in sorted(groups.items()):
        for base in ('dualmap','cacheroute'):
            augmented=base+'+fluidkv'
            if base not in policies or augmented not in policies:continue
            a,b=policies[base],policies[augmented]
            aa={r['request_id']:r for r in a['requests']};bb={r['request_id']:r for r in b['requests']}
            if len(aa)!=len(a['requests']) or len(bb)!=len(b['requests']):raise ValueError('duplicate request ID')
            matched=0;compared=0;different=[]
            for i in sorted(set(aa)&set(bb)):
                if aa[i]['input_sha256']!=bb[i]['input_sha256']:raise ValueError('policies received different input')
                if aa[i]['success'] and bb[i]['success']:
                    compared+=1
                    if aa[i]['engine']['output_ids']==bb[i]['engine']['output_ids']:matched+=1
                    else:different.append(i)
            x,y=a['summary'],b['summary']
            row={'prefix_tokens':key[0],'offered_per_engine':key[1],'regime':key[2],'repeat':key[3],
                 'base':base,'offered_total':x['offered_total'],'baseline_errors':x['errors'],'augmented_errors':y['errors'],
                 'output_compared':compared,'exact_output_matches':matched,'different_request_ids':different,
                 'ttft_p50_ratio':x['ttft_p50_p95_p99'][0]/y['ttft_p50_p95_p99'][0] if x['ttft_p50_p95_p99'] and y['ttft_p50_p95_p99'] else None,
                 'ttft_p99_ratio':x['ttft_p50_p95_p99'][2]/y['ttft_p50_p95_p99'][2] if x['ttft_p50_p95_p99'] and y['ttft_p50_p95_p99'] else None,
                 'throughput_ratio':y['requests_per_second']/x['requests_per_second'] if x['requests_per_second'] else None,
                 'baseline':x,'augmented':y}
            pairs.append(row)
    grouped=defaultdict(list)
    for r in pairs:grouped[(r['prefix_tokens'],r['offered_per_engine'],r['regime'],r['base'])].append(r)
    summaries=[]
    for key,rows in sorted(grouped.items()):
        row={'prefix_tokens':key[0],'offered_per_engine':key[1],'regime':key[2],'base':key[3],
             'paired_repetitions':len(rows),'output_compared':sum(r['output_compared'] for r in rows),
             'exact_output_matches':sum(r['exact_output_matches'] for r in rows),
             'baseline_errors':sum(r['baseline_errors'] for r in rows),'augmented_errors':sum(r['augmented_errors'] for r in rows)}
        for metric in ('ttft_p50_ratio','ttft_p99_ratio','throughput_ratio'):
            values=[r[metric] for r in rows if r[metric] is not None]
            row[metric+'_median']=float(np.median(values)) if values else None
        for side in ('baseline','augmented'):
            for metric in ('routing_cpu_seconds','preparation_seconds','preparation_exposed_seconds','total_transfer_bytes','slo_attainment'):
                row[side+'_'+metric+'_median']=float(np.median([r[side][metric] for r in rows]))
            row[side+'_ttft_p50_median']=float(np.median([r[side]['ttft_p50_p95_p99'][0] for r in rows if r[side]['ttft_p50_p95_p99']]))
            row[side+'_hit_ratio']=sum(r[side]['actual_cached_tokens'] for r in rows)/sum(r[side]['prompt_tokens'] for r in rows)
        summaries.append(row)
    repeats=[]
    identities=sorted({(k[0],k[1],k[2],policy) for k,policies in groups.items() for policy in policies})
    for length,batch,regime,policy in identities:
        runs=sorted((key[3],policies[policy]) for key,policies in groups.items()
                    if key[:3]==(length,batch,regime) and policy in policies)
        if len(runs)<2:continue
        reference={r['request_id']:r for r in runs[0][1]['requests']}
        for repeat,data in runs[1:]:
            good=[r for r in data['requests'] if r['success'] and r['request_id'] in reference and reference[r['request_id']]['success']]
            if any(r['input_sha256']!=reference[r['request_id']]['input_sha256'] for r in good):raise ValueError('repeat input mismatch')
            repeats.append({'prefix_tokens':length,'offered_per_engine':batch,'regime':regime,'policy':policy,
                            'reference_repeat':runs[0][0],'repeat':repeat,'compared':len(good),
                            'exact_matches':sum(r['engine']['output_ids']==reference[r['request_id']]['engine']['output_ids'] for r in good)})
    return {'scope':'paired common-engine routing algorithms with/without shared-prefix preparation; burst stress workload, not native production trace',
            'statistics':'median paired ratios across repeated runs; individual request p99s are descriptive, not independent replication units',
            'strict_output_equivalence':bool(pairs) and all(r['exact_output_matches']==r['offered_total'] for r in pairs),
            'pairs':pairs,'summaries':summaries,'within_policy_repeat_output_checks':repeats}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    result=summarize(args.directory)
    (args.directory/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summaries'],indent=2))
