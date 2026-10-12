"""Per-rank GPU interval accounting with explicit model-layer annotations."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent/'results'
ANALYSIS=ROOT/'analysis'
ANALYSIS.mkdir(exist_ok=True)


def union(intervals):
    total=0;start=end=None
    for a,b in sorted(intervals):
        if start is None:start,end=a,b
        elif a<=end:end=max(end,b)
        else:total+=end-start;start,end=a,b
    return total+(end-start if start is not None else 0)


def category(name):
    if 'nccl' in name.lower() or 'AllReduce' in name:return 'TP collective'
    if '_fwd_grouped_kernel' in name or name in ['_fwd_kernel','_fwd_kernel_stage2','_fwd_kernel_stage1']:return 'Full attention'
    if 'moe' in name.lower() or 'topk' in name.lower():return 'MoE / routing'
    if name.startswith('Cijk_') or 'gemm' in name.lower():return 'GEMM'
    if 'delta_rule' in name or 'causal_conv' in name or 'chunk_' in name or 'recompute_w' in name:return 'Linear attention'
    if 'norm' in name.lower() or 'rms' in name.lower():return 'Norm'
    return 'Other'


def analyze(path):
    data=json.load(gzip.open(path));events=data['traceEvents'];kernels=sorted([e for e in events if e.get('cat')=='kernel' and e.get('dur',0)>0],key=lambda e:e['ts'])
    assert kernels,path
    byname=collections.Counter();counts=collections.Counter();cats=collections.Counter()
    for e in kernels:
        byname[e['name']]+=e['dur'];counts[e['name']]+=1;cats[category(e['name'])]+=e['dur']
    busy=union([(e['ts'],e['ts']+e['dur']) for e in kernels]);span=max(e['ts']+e['dur'] for e in kernels)-kernels[0]['ts']
    steps=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('step[')]
    anchor_name='_moe_sum_reduce_kernel' if 'EXTEND' in path.name else 'triton_per_fused_copy__mul_sum_0'
    anchors=[e for e in kernels if e['name']==anchor_name]
    layer_rows=[]
    for step_idx,step in enumerate(steps):
        # GPU launch is asynchronous; CPU annotation end is not the GPU end.
        # Require exactly 40 known once-per-MoE-layer anchors per full trace
        # forward before partitioning consecutive GPU end anchors.
        if len(anchors)!=40*len(steps):break
        aa=anchors[step_idx*40:(step_idx+1)*40]
        lower=kernels[0]['ts'] if step_idx==0 else anchors[step_idx*40-1]['ts']+anchors[step_idx*40-1]['dur']
        upper=aa[-1]['ts']+aa[-1]['dur']
        within=[e for e in kernels if lower<=e['ts']<upper]
        # Anchor is at MoE end, once per 40-layer forward. Validate counts per
        # annotated step before attaching indices; never infer TP/layer count.
        if len(aa)!=40:continue
        prior=lower
        for layer,anchor in enumerate(aa):
            end=anchor['ts']+anchor['dur'];ks=[e for e in within if prior<=e['ts']<end]
            sums=collections.Counter()
            for e in ks:sums[category(e['name'])]+=e['dur']
            layer_rows.append(dict(step=step_idx,layer=layer,type='full_attention' if layer%4==3 else 'linear_attention',
                start_us=prior,end_us=end,wall_us=end-prior,kernel_sum_us=sum(e['dur'] for e in ks),categories_us=dict(sums)))
            prior=end
    result=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kernel_count=len(kernels),
        kernel_sum_ms=sum(byname.values())/1000,gpu_busy_union_ms=busy/1000,gpu_span_ms=span/1000,
        between_kernel_gap_ms=(span-busy)/1000,categories_ms={k:v/1000 for k,v in cats.items()},
        top_kernels=[dict(name=n,total_ms=t/1000,launches=counts[n]) for n,t in byname.most_common(20)],
        steps=[dict(name=e['name'],start_us=e['ts'],duration_ms=e['dur']/1000) for e in steps],
        anchor_kernel=anchor_name,anchor_count=len(anchors),verified_layer_intervals=len(layer_rows),layers=layer_rows,
        limitations='Graph-off stack-enabled profiler; duration sum != union != unprofiled wall. '
            'ROCtracer duplicate-flow warnings retained. Gaps may include profiler overhead; not proof of removable idle time. '
            'Prefill capture is first five 4096-token chunks, not the entire 116K prefill. '
            'MoE-end anchored layer intervals include preceding residual/collective boundary work.')
    derived=ANALYSIS/path.name.replace('.trace.json.gz','.navigation.json.gz')
    with gzip.open(derived,'wt') as f:json.dump({'traceEvents':[e for e in events if e.get('cat') in ['kernel','user_annotation','gpu_user_annotation'] or e.get('ph')=='M']},f)
    result['derived_navigation_file']=str(derived)
    (ANALYSIS/path.name.replace('.trace.json.gz','.analysis.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    rows=[]
    for path in sorted(ROOT.glob('*/*.trace.json.gz')):
        cached=ANALYSIS/path.name.replace('.trace.json.gz','.analysis.json')
        r=json.loads(cached.read_text()) if cached.exists() else analyze(path)
        if 'anchor_kernel' not in r:r=analyze(path)
        rows.append(r)
        print(path.name,'kernel ms',round(r['kernel_sum_ms'],2),'union',round(r['gpu_busy_union_ms'],2),'span',round(r['gpu_span_ms'],2),'layers',r['verified_layer_intervals'],flush=True)
    (ROOT/'gpu-summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='layers'} for r in rows],indent=2)+'\n')
