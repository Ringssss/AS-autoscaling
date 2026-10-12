"""Full native Kimi field/coverage analysis; no fabricated cache or tool lineage."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from agentshift.workloads.production import parse_kimi_line
from experiments.multinode.audit_kimi import EXPECTED_SHA256

ROOT=Path(__file__).resolve().parent
EXTRA=('e2e_latency','queue_time','prefill_waiting_latency','prefill_launch_latency',
       'api_server_dispatch_finish_ts','inference_time','total_retractions')
EXTRA_PATTERNS={k:re.compile(r"'"+k+r"': ([^,}]+)") for k in EXTRA}
IDENTITY=('conversation_id','routing_key','extra_key','session_params','bootstrap_pair_key')
IDENTITY_PATTERNS={k:re.compile(r'\b'+k+r'=([^,)]+)') for k in IDENTITY}
ASSIGN=re.compile(r'\b(\w+)=')
DICTKEY=re.compile(r"'([a-zA-Z_]\w*)':")
HOST=re.compile(r"bootstrap_host='([^']+)'")
FINISH=re.compile(r"'finish_reason': \{'type': '([^']+)'")


def pct(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return dict(zip(('min','p50','p90','p95','p99','max'),np.percentile(a,[0,50,90,95,99,100]).tolist())) if len(a) else {}


def oracle_budget(ttft, e2e, cohort, valid):
    """Conditional cohort-only ceiling: others/decode fixed, entire TTFT free."""
    valid=np.asarray(valid,dtype=bool)
    chosen=np.asarray(cohort,dtype=bool)&valid
    total_ttft=float(np.sum(ttft[valid]));total_e2e=float(np.sum(e2e[valid]))
    removed=float(np.sum(ttft[chosen]))
    return {'removed_ttft_seconds':removed,'ttft_sum_share':removed/total_ttft if total_ttft else None,
            'cohort_only_zero_ttft_mean_ttft_speedup':total_ttft/(total_ttft-removed) if total_ttft>removed else None,
            'cohort_only_zero_ttft_mean_e2e_speedup':total_e2e/(total_e2e-removed) if total_e2e>removed else None,
            'assumption':'Only this observed cohort changes; its entire TTFT is zero; other requests and decode stay fixed. Not a routing counterfactual, global bound, throughput prediction, or realistic FluidKV estimate.'}


def main(args):
    digest=hashlib.sha256()
    with args.archive.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=EXPECTED_SHA256:raise ValueError('archive hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];extra={k:[] for k in EXTRA};assignment=Counter();dictkeys=Counter()
    identities=Counter();hosts=Counter();reasons=Counter();sources=[]
    old=json.loads((args.normalized.parent/'audit.json').read_text())
    old_sources={s['name']:s for s in old['sources']}
    with tarfile.open(args.archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not re.fullmatch(r'[0-2][0-9]\.redacted\.log',Path(member.name).name):continue
            sha=hashlib.sha256();count=0
            for raw in tar.extractfile(member):
                sha.update(raw);line=raw.decode('utf-8');row=parse_kimi_line(line)
                if row is None:continue
                rows.append(row);count+=1
                assignment.update(set(ASSIGN.findall(line)));dictkeys.update(set(DICTKEY.findall(line)))
                for k,pat in IDENTITY_PATTERNS.items():
                    m=pat.search(line)
                    identities[k+'_present']+=bool(m)
                    identities[k+'_nonnull']+=bool(m and m.group(1).strip() not in ('None',"''",'""'))
                host=HOST.search(line);hosts[host.group(1) if host else '<missing>']+=1
                reason=FINISH.search(line);reasons[reason.group(1) if reason else '<missing>']+=1
                for k,pat in EXTRA_PATTERNS.items():
                    m=pat.search(line)
                    try:value=float(m.group(1)) if m else np.nan
                    except ValueError:value=np.nan
                    extra[k].append(value)
            source={'name':member.name,'rows':count,'sha256':sha.hexdigest()}
            assert source['sha256']==old_sources[member.name]['sha256']
            assert count==old_sources[member.name]['rows']
            sources.append(source)
    assert len(sources)==24 and len(rows)==245555
    # Independently verify every normalized row used by the earlier audit.
    with gzip.open(args.normalized,'rt') as f:
        normalized_count=0
        for i,line in enumerate(f):
            prev=json.loads(line)
            assert {k:prev[k] for k in rows[i]}==rows[i],('normalized mismatch',i)
            normalized_count+=1
    assert normalized_count==len(rows)
    n=len(rows)
    x=np.array([r['input_length'] for r in rows],dtype=np.int64)
    y=np.array([r['output_length'] for r in rows],dtype=np.int64)
    cached=np.array([r['cached_tokens'] if r['cached_tokens'] is not None else -1 for r in rows],dtype=np.int64)
    arrival=np.array([r['arrival_seconds'] for r in rows]);finished=np.array([r['finished_seconds'] for r in rows])
    ttft=np.array([r['ttft_seconds'] for r in rows]);e2e=finished-arrival
    extra={k:np.array(v) for k,v in extra.items()}
    valid_cache=(cached>=0)&(cached<=x)
    valid_latency=np.isfinite(ttft)&np.isfinite(e2e)&(ttft>=0)&(e2e>=0)&(ttft<=e2e+1e-3)
    t0=arrival.min();span=arrival.max()-t0
    ranks=np.array([r['dp_rank'] for r in rows])
    assert len({r['request_id'] for r in rows})==n
    np.savez_compressed(args.output/'native_numeric_fields.npz',input_tokens=x,output_tokens=y,cached_tokens=cached,
                        arrival=arrival,finished=finished,ttft=ttft,e2e=e2e,dp_rank=ranks,**extra)
    cohorts=[]
    for threshold in (8192,16384,32768,65536):
        for kind,mask in [('long_input',x>=threshold),('observed_long_cache',valid_cache&(cached>=threshold))]:
            valid=mask&valid_cache
            cohorts.append({'kind':kind,'threshold':threshold,'requests':int(mask.sum()),
                'fraction_all':float(mask.mean()),'fraction_cache_valid':float(valid.sum()/valid_cache.sum()),
                'input_token_share':float(x[mask].sum()/x.sum()),'output_token_share':float(y[mask].sum()/y.sum()),
                'ttft':pct(ttft[mask&valid_latency]),'e2e':pct(e2e[mask&valid_latency]),
                'cached_tokens_valid':pct(cached[valid]),'cached_fraction_valid':pct(cached[valid]/x[valid]),
                'oracle_budget':oracle_budget(ttft,e2e,mask,valid_latency)})
    bins=((arrival-t0)//60).astype(int);nb=int(bins.max())+1
    counts=np.bincount(bins,minlength=nb)
    minute=[]
    for b in range(nb):
        m=bins==b
        minute.append({'minute_from_first_arrival':b,'requests':int(m.sum()),
            'input_tokens':int(x[m].sum()),'output_tokens':int(y[m].sum()),
            'long_input_32k':int((m&(x>=32768)).sum()),'long_input_64k':int((m&(x>=65536)).sum()),
            'long_cache_32k':int((m&valid_cache&(cached>=32768)).sum()),
            'long_cache_64k':int((m&valid_cache&(cached>=65536)).sum()),
            'ttft_p50':float(np.median(ttft[m&valid_latency])) if (m&valid_latency).any() else None,
            'ttft_p95':float(np.percentile(ttft[m&valid_latency],95)) if (m&valid_latency).any() else None})
    with (args.output/'minute_profile.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=minute[0].keys());writer.writeheader();writer.writerows(minute)
    midpoint=t0+np.arange(nb)*60+30
    live=np.searchsorted(np.sort(arrival),midpoint,side='right')-np.searchsorted(np.sort(finished),midpoint,side='right')
    active_valid=e2e>0
    event_t=np.r_[arrival[active_valid],finished[active_valid]]
    delta=np.r_[np.ones(active_valid.sum(),dtype=int),-np.ones(active_valid.sum(),dtype=int)]
    order=np.lexsort((delta,event_t)) # finishes before arrivals at identical timestamps
    peak=int(np.cumsum(delta[order]).max())
    q=np.quantile(counts,[.25,.75]);loads=[]
    for label,bmask in [('low',counts<=q[0]),('middle',(counts>q[0])&(counts<q[1])),('high',counts>=q[1])]:
        m=bmask[bins]
        loads.append({'label':label,'minutes':int(bmask.sum()),'requests':int(m.sum()),
                      'ttft':pct(ttft[m&valid_latency]),'input_tokens':pct(x[m]),
                      'long_input_32k_fraction':float(np.mean(x[m]>=32768)),
                      'long_cache_32k_fraction':float(np.mean(valid_cache[m]&(cached[m]>=32768)))})
    extra_stats={k:{'finite':int(np.isfinite(v).sum()),'zeros':int((v==0).sum()),'percentiles':pct(v)} for k,v in extra.items()}
    dispatch=extra['api_server_dispatch_finish_ts']-arrival
    report={'scope':'Native logged-request opportunity characterization; NOT FluidKV/DualMap/CacheRoute policy replay.',
        'archive_sha256':digest.hexdigest(),'sources_verified_against_prior_audit':sources,
        'normalized_rows_independently_verified':n,'requests':n,'unique_request_ids':n,
        'input_redacted':sum(r['input_ids_redacted'] for r in rows),'output_redacted':sum(r['output_ids_redacted'] for r in rows),
        'identity_fields':dict(identities),'assignment_field_census':dict(assignment),'dictionary_field_census':dict(dictkeys),
        'bootstrap_host_count':len(hosts),'bootstrap_host_request_counts_sorted':sorted(hosts.values()),
        'finish_reasons':dict(reasons),'dp_rank_counts':dict(Counter(ranks.tolist())),
        'arrival_span_hours':float(span/3600),'logged_arrival_rate':float(n/span),
        'cache_valid_rows':int(valid_cache.sum()),'cache_invalid_rows':int((~valid_cache).sum()),
        'invalid_cache_all_one_input_one_output':bool(np.all((x[~valid_cache]==1)&(y[~valid_cache]==1))),
        'valid_latency_rows':int(valid_latency.sum()),'invalid_latency_rows':int((~valid_latency).sum()),
        'input_tokens':pct(x),'output_tokens':pct(y),'ttft':pct(ttft[valid_latency]),'e2e':pct(e2e[valid_latency]),
        'valid_token_weighted_cache_hit':float(cached[valid_cache].sum()/x[valid_cache].sum()),
        'top_cached_counts_valid':Counter(cached[valid_cache].tolist()).most_common(12),
        'cache_in_5440_to_5888':{'requests':int((valid_cache&(cached>=5440)&(cached<=5888)).sum()),
                               'fraction_all':float(np.mean(valid_cache&(cached>=5440)&(cached<=5888)))},
        'outputs_at_most_64_fraction':float(np.mean(y<=64)),
        'ttft_sum_fraction_of_e2e_sum':float(ttft[valid_latency].sum()/e2e[valid_latency].sum()),
        'cohorts':cohorts,'extra_fields':extra_stats,'observed_dispatch_gap_seconds':pct(dispatch),
        'reported_e2e_vs_timestamp_max_abs_error':float(np.nanmax(abs(extra['e2e_latency']-e2e))),
        'minute_request_counts':pct(counts),'minute_boundaries':'Relative to first logged arrival; last bin can be partial; finish-only cohort is left/right censored.',
        'logged_inflight_peak':peak,'logged_inflight_minute_midpoint':pct(live),
        'inflight_semantics':'Overlapping observed arrival-to-finish intervals, not GPU batch size, utilization, queue length, or all original offered traffic.',
        'arrival_load_cohorts':loads,
        'cache_cohort_semantics':'Observed cached counts at the historical serving path, not private-history identity, source transferability, target residency, or an upper bound on all migration opportunities.',
        'native_policy_effect_identifiable':False,
        'missing_for_policy_replay':['stable session/agent and causal turn IDs','privacy-preserving exact prefix/block identities','tool start/end or dependency readiness','serving-engine/cache residency and membership events','Kimi backend state format and transfer/compute calibration'],
        'inference_time_warning':'Named inference_time is not accepted as elapsed duration; magnitude resembles a monotonic timestamp. Zero queue_time is not proof of zero end-to-end queueing.'}
    (args.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'command.json').write_text(json.dumps({'argv':__import__('sys').argv,'cwd':str(Path.cwd())},indent=2)+'\n')
    plot(report,minute,live,args.output)
    print(json.dumps({k:report[k] for k in ('requests','arrival_span_hours','cache_valid_rows','cache_invalid_rows','valid_latency_rows','logged_inflight_peak','native_policy_effect_identifiable')},indent=2))
    for row in cohorts:print(row['kind'],row['threshold'],row['requests'],round(100*row['fraction_all'],3))


def plot(report,minute,live,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    xx=np.arange(len(minute))/60
    axes[0,0].plot(xx,[r['requests'] for r in minute],lw=.8)
    axes[0,0].set(title='Native logged arrivals',xlabel='Hours since first logged arrival',ylabel='Requests / minute')
    axes[0,1].plot(xx,live,lw=.8,color='tab:green')
    axes[0,1].set(title='Overlapping logged request lifetimes',xlabel='Hours since first logged arrival',ylabel='Observed in-flight requests (not GPU batch)')
    x=np.arange(4);cohorts=report['cohorts']
    for kind,offset,label in [('long_input',-.18,'Input length'),('observed_long_cache',.18,'Observed cached length')]:
        rows=[r for r in cohorts if r['kind']==kind]
        axes[1,0].bar(x+offset,[100*r['fraction_all'] for r in rows],width=.36,label=label)
    axes[1,0].set(title='Long input is not long cached history',xticks=x,xticklabels=['>=8K','>=16K','>=32K','>=64K'],ylabel='% of all logged requests')
    axes[1,0].legend()
    rows=[r for r in cohorts if r['kind']=='observed_long_cache']
    axes[1,1].bar(x,[100*r['oracle_budget']['ttft_sum_share'] for r in rows],color='tab:orange')
    axes[1,1].set(title='Observed long-cache cohort: TTFT sum share\nCoverage measure, not FluidKV speedup',xticks=x,xticklabels=['>=8K','>=16K','>=32K','>=64K'],ylabel='% of summed observed TTFT')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'kimi-opportunity.png',dpi=180);fig.savefig(out/'kimi-opportunity.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=ROOT/'data/kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07.tar.gz')
    p.add_argument('--normalized',type=Path,default=ROOT/'results/kimi-audit/requests.jsonl.gz')
    p.add_argument('--output',type=Path,default=ROOT/'results/kimi-opportunity')
    main(p.parse_args())
