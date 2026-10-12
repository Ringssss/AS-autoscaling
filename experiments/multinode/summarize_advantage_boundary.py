"""Compare paired independent runs; keep waves and precompute controls separate."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/advantage-boundary'

def summarize():
    groups=defaultdict(dict)
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or folder.name.startswith('failed'):continue
        for path in folder.glob('*.json'):
            data=json.loads(path.read_text())
            if not isinstance(data,dict) or 'requests' not in data:continue
            s=data['summary']
            key=(folder.name,s['prefix_tokens'],s['offered_per_engine'],s.get('wave',0),s['repeat'])
            groups[key][s['policy']]=data
    pairs=defaultdict(list)
    for key,policies in groups.items():
        for router in ('dualmap','cacheroute'):
            for comparator in (router,router+'+precompute','sticky'):
                if comparator not in policies or router+'+fluidkv' not in policies:continue
                a,b=policies[comparator],policies[router+'+fluidkv']
                ar={r['request_id']:r for r in a['requests']}
                br={r['request_id']:r for r in b['requests']}
                if ar.keys()!=br.keys():raise ValueError('unpaired request IDs')
                if any(ar[i]['input_sha256']!=br[i]['input_sha256'] for i in ar):raise ValueError('input mismatch')
                x,y=a['summary'],b['summary']
                row={'repeat':key[-1],'errors':x['errors']+y['errors'],
                     'ttft_ratio':x['ttft_p50_p95_p99'][0]/y['ttft_p50_p95_p99'][0],
                     'p95_ratio':x['ttft_p50_p95_p99'][1]/y['ttft_p50_p95_p99'][1],
                     'throughput_ratio':y['requests_per_second']/x['requests_per_second'],
                     'baseline_ttft':x['ttft_p50_p95_p99'][0],'fluidkv_ttft':y['ttft_p50_p95_p99'][0],
                     'baseline_prepare':x['preparation_seconds'],'fluidkv_prepare':y['preparation_seconds'],
                     'fluidkv_bytes':y['total_transfer_bytes'],
                     'output_compared':sum(ar[i]['success'] and br[i]['success'] for i in ar),
                     'exact_outputs':sum(ar[i]['success'] and br[i]['success'] and ar[i]['engine']['output_ids']==br[i]['engine']['output_ids'] for i in ar)}
                label=router+':sticky' if comparator=='sticky' else comparator
                pairs[key[:-1]+(label,)].append(row)
    rows=[]
    for key,values in sorted(pairs.items()):
        row=dict(zip(('campaign','length','concurrency','wave','comparator'),key))
        row['n']=len(values);row['pairs']=values
        row['errors']=sum(v['errors'] for v in values)
        for metric in ('ttft_ratio','p95_ratio','throughput_ratio','baseline_ttft','fluidkv_ttft','baseline_prepare','fluidkv_prepare','fluidkv_bytes'):
            a=np.array([v[metric] for v in values]);row[metric]=float(np.median(a))
            row[metric+'_range']=[float(a.min()),float(a.max())]
            if len(a)>=5:
                rng=np.random.default_rng(20261002)
                boot=np.median(rng.choice(a,(10000,len(a))),axis=1)
                row[metric+'_ci95']=np.percentile(boot,[2.5,97.5]).tolist()
        rows.append(row)
    (OUT/'paired-boundary.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# FluidKV advantage boundary: measured comparisons','',
           'Three nodes, TP2 per node, Qwen3.6; models and communication groups prewarmed. Ratios >1 favor FluidKV. Screening has one repetition; it does not establish stability. Held-out runs use five paired repetitions; only the persistence campaign retains caches across three waves. Source-withdrawal runs have two eligible targets and do not physically retire GPUs. Bootstrap intervals resample runs, not individual requests. Precompute receives the same known prefix and tool window and uses parallel targets. Labels ending in :sticky compare that router plus FluidKV against direct warm-source return.','',
           '| Campaign | Tokens | Concurrency/engine | Wave | Comparator | n | TTFT ratio [95% CI] | p95 ratio | Throughput ratio | Errors |',
           '|---|---:|---:|---:|---|---:|---|---:|---:|---:|']
    for r in rows:
        ci=r.get('ttft_ratio_ci95');interval=f' [{ci[0]:.2f}, {ci[1]:.2f}]' if ci else ''
        lines.append(f"| {r['campaign']} | {r['length']} | {r['concurrency']} | {r['wave']} | {r['comparator']} | {r['n']} | {r['ttft_ratio']:.2f}{interval} | {r['p95_ratio']:.2f} | {r['throughput_ratio']:.2f} | {r['errors']} |")
    lines+=['','Scope: shared-prefix returning bursts; not private agent histories, production tool traces, continuous mixed-load migration, or physical autoscaling. Source state creation is common prior work. TTFT starts at tool return and includes exposed preparation. Long tool windows also let precomputation hide its cost. No universal or exclusive algorithmic advantage follows from this experiment.','']
    (ROOT/'ADVANTAGE_BOUNDARY_RESULTS.md').write_text('\n'.join(lines))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for router,color in [('dualmap','tab:blue'),('cacheroute','tab:orange')]:
            for suffix,style in [('', '-'),('+precompute','--')]:
                selected=sorted((r for r in rows if r['campaign'].startswith('screen-window-') and
                                 r['length']==65536 and r['concurrency']==16 and r['comparator']==router+suffix),
                                key=lambda r:float(r['campaign'].split('-')[-1]))
                axes[0].plot([str(float(r['campaign'].split('-')[-1])) for r in selected],
                             [r['ttft_ratio'] for r in selected],linestyle='none',marker='x' if suffix else 'o',color=color,label=router+suffix)
            selected=sorted((r for r in rows if r['campaign']=='heldout-persistence' and r['comparator']==router),key=lambda r:r['wave'])
            if selected:
                y=np.array([r['ttft_ratio'] for r in selected])
                low=np.array([r.get('ttft_ratio_ci95',[r['ttft_ratio']]*2)[0] for r in selected])
                high=np.array([r.get('ttft_ratio_ci95',[r['ttft_ratio']]*2)[1] for r in selected])
                axes[1].errorbar([r['wave']+1 for r in selected],y,yerr=[y-low,high-y],marker='o',color=color,label=router)
            withdrawn=[r for r in rows if r['campaign']=='heldout-source-withdrawal' and r['comparator']==router]
            if withdrawn:
                r=withdrawn[0];y=r['ttft_ratio'];low,high=r.get('ttft_ratio_ci95',[y,y])
                axes[2].errorbar([0 if router=='dualmap' else 1],[y],yerr=[[y-low],[high-y]],marker='o',capsize=5,color=color)
        for ax in axes:
            ax.axhline(1,color='gray',linewidth=1)
            ax.set_ylabel('Baseline TTFT / FluidKV TTFT')
            ax.grid(alpha=.2)
            handles,labels=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=8)
        axes[0].set(title='64K, concurrency 16\nExploratory single runs',xlabel='Controlled tool window (seconds)')
        axes[1].set(title='64K, concurrency 16, window 0.5s\nCaches retained between waves',xlabel='Consecutive return wave',xticks=[1,2,3])
        axes[2].set(title='64K, concurrency 4\nLogical source withdrawal',xlabel='Two eligible replicas; window 0.5s',xticks=[0,1],xticklabels=['DualMap','CacheRoute'],xlim=(-.5,1.5))
        fig.savefig(OUT/'advantage-boundary.png',dpi=180)
        fig.savefig(OUT/'advantage-boundary.pdf')
        plt.close(fig)
    print(json.dumps({'comparisons':len(rows),'paired_errors':sum(r['errors'] for r in rows)}))

if __name__=='__main__':summarize()
