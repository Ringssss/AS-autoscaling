"""Audit complete measured windows and paired policy comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


FIELDS=('stage','window','length','withdraw','tool_seconds','rate_scale','output_cap','repeat')
PAIRS=[('p2c','p2c+fluidkv'),('dualmap','dualmap+fluidkv'),('cacheroute','cacheroute+fluidkv'),
       ('dualmap','p2c+fluidkv'),('cacheroute','p2c+fluidkv')]


def execution_balance(record):
    out={}
    for target in sorted({r['target'] for r in record['requests']}):
        rows=[r for r in record['requests'] if r['target']==target and r['success']]
        samples=[s for s in record['telemetry'] if s['engine']==target and 'running' in s]
        out[str(target)]={'requests':len(rows),
            'uncached_input_tokens':sum(r['input_tokens']-r['engine']['meta_info']['cached_tokens'] for r in rows),
            'peak_observed_running_requests':max((s['running'] for s in samples),default=0),
            'peak_observed_waiting_requests':max((s['waiting'] for s in samples),default=0)}
    return out


def summarize(roots,output):
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(dict); runs=[]; errors=[]
    for root in roots:
        for path in sorted(root.glob('[0-9][0-9][0-9]-*.json')):
            data=json.loads(path.read_text())
            if 'summary' not in data: continue
            s=data['summary']; rows=data['requests']; runs.append(s|{'file':str(path)})
            assert len(rows)==s['offered']==len({r['ordinal'] for r in rows}),path
            assert all(r['input_tokens']==r['engine']['meta_info']['prompt_tokens'] and r['requested_output']==r['engine']['meta_info']['completion_tokens'] for r in rows if r['success']),path
            assert all(r['client_queue_seconds']>=-.005 for r in rows),path
            assert all(r['ttft_seconds']<=r['e2e_seconds'] for r in rows if r['success']),path
            assert abs(sum(r['requested_output'] for r in rows)-s['total_requested_output'])==0
            if s['withdraw']: assert all('3.27:' not in r['engine_url'] for r in rows)
            assert s['errors']==sum(not r['success'] for r in rows)
            key=tuple(s[k] for k in FIELDS)
            if s['policy'] in groups[key]: raise ValueError('duplicate paired trial '+str(key))
            groups[key][s['policy']]=data
            errors.extend((str(path),r['error']) for r in rows if not r['success'])
    comparisons=[]
    for key,policies in groups.items():
        for baseline,treatment in PAIRS:
            if baseline not in policies or treatment not in policies: continue
            a,b=policies[baseline],policies[treatment]; x,y=a['summary'],b['summary']
            aa={r['ordinal']:r for r in a['requests']};bb={r['ordinal']:r for r in b['requests']}
            assert aa.keys()==bb.keys()
            assert all((aa[i]['native_index'],aa[i]['input_sha256'],aa[i]['requested_output'],aa[i]['offset'])==(bb[i]['native_index'],bb[i]['input_sha256'],bb[i]['requested_output'],bb[i]['offset']) for i in aa)
            pair=dict(zip(FIELDS,key))|dict(baseline=baseline,treatment=treatment,
                baseline_checkpoint_errors=x['checkpoint_errors'],treatment_checkpoint_errors=y['checkpoint_errors'],
                observation_clean=x['checkpoint_errors']==0 and y['checkpoint_errors']==0,
                mean_ttft_speedup=x['ttft_mean']/y['ttft_mean'],p95_ttft_speedup=x['ttft_quantiles'][1]/y['ttft_quantiles'][1],
                mean_e2e_speedup=x['e2e_mean']/y['e2e_mean'],throughput_speedup=y['throughput']/x['throughput'],
                baseline_slo=x['ttft_slo_10s'],treatment_slo=y['ttft_slo_10s'],
                exact_output_matches=sum(aa[i]['engine']['output_ids']==bb[i]['engine']['output_ids'] for i in aa if aa[i]['success'] and bb[i]['success']),
                paired_requests=len(aa),baseline_mean_ttft=x['ttft_mean'],treatment_mean_ttft=y['ttft_mean'],
                baseline_mean_e2e=x['e2e_mean'],treatment_mean_e2e=y['e2e_mean'])
            paired_good=[i for i in aa if aa[i]['success'] and bb[i]['success']]
            pair['requests_ttft_gain_ge20pct']=sum(aa[i]['ttft_seconds']/bb[i]['ttft_seconds']>=1.2 for i in paired_good)
            pair['requests_ttft_regression_ge10pct']=sum(bb[i]['ttft_seconds']/aa[i]['ttft_seconds']>=1.1 for i in paired_good)
            pair['baseline_full_history_hits']=sum(aa[i]['eligible'] and aa[i]['engine']['meta_info']['cached_tokens']>=x['length'] for i in paired_good)
            pair['treatment_full_history_hits']=sum(bb[i]['eligible'] and bb[i]['engine']['meta_info']['cached_tokens']>=y['length'] for i in paired_good)
            assert x['total_input_tokens']==y['total_input_tokens']
            pair['baseline_uncached_input_tokens']=x['total_input_tokens']-x['observed_cached_tokens']
            pair['treatment_uncached_input_tokens']=y['total_input_tokens']-y['observed_cached_tokens']
            pair['additional_uncached_input_fraction_avoided']=(1-pair['treatment_uncached_input_tokens']/pair['baseline_uncached_input_tokens']) if not (x['errors'] or y['errors']) else None
            pair['baseline_execution_balance']=execution_balance(a)
            pair['treatment_execution_balance']=execution_balance(b)
            for label,mask in [('eligible',lambda r:r['eligible']),('unrelated',lambda r:not r['eligible']),('early',lambda r:r['offset']<5),('late',lambda r:r['offset']>=5)]:
                selected=[i for i in aa if mask(aa[i]) and aa[i]['success'] and bb[i]['success']]
                pair[label+'_requests']=len(selected)
                if selected: pair[label+'_mean_ttft_speedup']=float(np.mean([aa[i]['ttft_seconds'] for i in selected])/np.mean([bb[i]['ttft_seconds'] for i in selected]))
            comparisons.append(pair)
    repeated=defaultdict(list)
    for c in comparisons:
        k=tuple(c[k] for k in FIELDS if k!='repeat')+(c['baseline'],c['treatment'])
        repeated[k].append(c['mean_ttft_speedup'])
    intervals=[]; rng=np.random.default_rng(20260925)
    for k,values in repeated.items():
        if len(values)<5: continue
        v=np.asarray(values); boot=np.median(rng.choice(v,(10000,len(v)),replace=True),axis=1)
        intervals.append(dict(zip([f for f in FIELDS if f!='repeat']+['baseline','treatment'],k))|dict(
            repetitions=len(v),paired_mean_ttft_speedup_median=float(np.median(v)),bootstrap_95=np.percentile(boot,[2.5,97.5]).tolist()))
    result=dict(trials=len(runs),offered=sum(r['offered'] for r in runs),errors=errors,
        checkpoint_errors=sum(r['checkpoint_errors'] for r in runs),runs=runs,comparisons=comparisons,repeated_intervals=intervals)
    (output/'audit-and-comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Kimi trace 回放：逐条件实测表','', '倍率 >1 表示 treatment 更好；均为完整窗口，包含排队与准备暴露时间。screen 限制输出至64 tokens；native 保留原始输出。单次 screening 不附显著性结论。','',
           '|阶段/窗口|历史|源退出|窗口W/速率|对比（基线 → treatment）|均值TTFT|p95 TTFT|均值E2E|吞吐|逐token相同|观测错误(基线/处理)|',
           '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in sorted(comparisons,key=lambda r:tuple(str(r[k]) for k in FIELDS)+(r['baseline'],r['treatment'])):
        lines.append(f"|{c['stage']}/{c['window']}/r{c['repeat']}|{c['length']}|{c['withdraw']}|{c['tool_seconds']}s/{c['rate_scale']}×|{c['baseline']} → {c['treatment']}|{c['mean_ttft_speedup']:.3f}×|{c['p95_ttft_speedup']:.3f}×|{c['mean_e2e_speedup']:.3f}×|{c['throughput_speedup']:.3f}×|{c['exact_output_matches']}/{c['paired_requests']}|{c['baseline_checkpoint_errors']}/{c['treatment_checkpoint_errors']}|")
    (output/'MEASURED_TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('trials','offered','errors','checkpoint_errors','repeated_intervals')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.inputs,a.output)
