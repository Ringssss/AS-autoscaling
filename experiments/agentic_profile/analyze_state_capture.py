"""Offline two-engine state-preparation timeline interpretation."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from experiments.agentic_profile.analyze_engine_traces import clipped, merged, union
from experiments.agentic_profile.analyze_workflow_capture import intersection


def extract(path, anchor, row):
    data = json.load(gzip.open(path)); groups=defaultdict(list);names=Counter();counts=Counter()
    origin = anchor['epoch']*1e6-data['baseTimeNanoseconds']/1000
    for e in data['traceEvents']:
        if e.get('ph')!='X' or not e.get('dur'):
            continue
        a=anchor['monotonic']+(e['ts']-origin)/1e6;b=a+e['dur']/1e6
        intervals=clipped([(a,b)],row['start'],row['finish'])
        if not intervals:continue
        cat=e.get('cat','');name=e['name']
        if cat in ('kernel','gpu_memcpy','gpu_memset'):
            groups[cat].extend(intervals);counts[cat]+=1
            if cat=='kernel':
                names[name]+=union(intervals)
                if any(x in name.lower() for x in ('nccl','sendrecv','copy','index','scatter','gather')):
                    groups['transfer_or_index_named'].extend(intervals)
        elif cat in ('cuda_runtime','cuda_driver'):
            if 'Synchronize' in name or 'WaitEvent' in name:groups['cpu_wait_api'].extend(intervals)
            if 'Launch' in name:groups['cpu_launch_api'].extend(intervals)
    groups['gpu_event_union']=merged(groups['kernel']+groups['gpu_memcpy']+groups['gpu_memset'])
    prep=[(p['start'],p['finish']) for p in row['preparations']]
    decode=[(f['first'],f['finish']) for f in row['foreground']]
    metric={'trace':str(path),'event_counts_in_measured_window':dict(counts),
            'interval_unions_seconds':{k:union(v) for k,v in groups.items()},
            'gpu_event_during_preparation_seconds':union(intersection(groups['gpu_event_union'],prep)),
            'gpu_event_during_foreground_decode_envelope_seconds':union(intersection(groups['gpu_event_union'],decode)),
            'gpu_event_during_prep_and_decode_envelopes_seconds':union(intersection(groups['gpu_event_union'],intersection(prep,decode))),
            'top_kernel_summed_seconds':names.most_common(12),
            'transfer_or_index_named_kernels':[(n,v) for n,v in names.most_common() if any(x in n.lower() for x in ('nccl','sendrecv','copy','index','scatter','gather'))][:15]}
    return metric,groups


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('experiments/agentic_profile/results/20260915/state-profile'));args=parser.parse_args();root=args.root
    anchor=json.loads((root.parent/'state-diagnostic-clock.json').read_text());rows=[json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()]
    results=[];plots=[]
    for row in rows:
        result={'method':row['method'],'index':row['index'],'profiled':row['profiled'],'start':row['start'],'finish':row['finish'],
                'measured_seconds':row['finish']-row['start'],'preparation_wall_union_seconds':union([(p['start'],p['finish']) for p in row['preparations']]),
                'foreground_decode_envelope_union_seconds':union([(f['first'],f['finish']) for f in row['foreground']]),
                'engines':{}}
        groups=[]
        for engine in range(2):
            metric,group=extract(next((root/f'profile-{row["index"]}-engine-{engine}').glob('*.gz')),anchor,row)
            result['engines'][str(engine)]=metric;groups.append(group)
        result['simultaneous_source_destination_gpu_event_seconds']=union(intersection(groups[0]['gpu_event_union'],groups[1]['gpu_event_union']))
        results.append(result);plots.append((row,groups))
    report={'scope':'Two profiler-enabled destination-c2/out64 concurrent diagnostic rows: replay and copy. Excluded from final performance comparisons; no causal latency claims from these timings.',
            'clock_anchor':anchor,'rows':results,
            'caveats':['Per-engine GPU event unions are never added as one elapsed duration.','Kernel names suggest operations but do not attribute each target kernel to foreground vs preparation.','Prep and decode envelopes are host request lifetimes; GPU overlap with them is not proof that both execute simultaneously on one GPU.','CUDA waits overlap useful GPU work; no scheduler-stack or SM utilization evidence.','No source inference in these two rows; source activity during copy can include transfer packaging and communication.']}
    (root/'PROFILE_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plots),1,figsize=(12,7),constrained_layout=True)
    for ax,(row,groups) in zip(axes,plots):
        tracks=[('Source GPU',groups[0]['gpu_event_union']),('Destination GPU',groups[1]['gpu_event_union']),('Destination CUDA wait',groups[1]['cpu_wait_api']),('Preparation host envelope',[(p['start'],p['finish']) for p in row['preparations']]),('Foreground decode envelope',[(f['first'],f['finish']) for f in row['foreground']])]
        for i,(name,spans) in enumerate(tracks):
            ax.broken_barh([((a-row['start'])*1000,(b-a)*1000) for a,b in merged(spans)],(i-.35,.7),facecolors=f'C{i}')
        ax.set_yticks(range(len(tracks)),[n for n,_ in tracks]);ax.set_xlim(0,(row['finish']-row['start'])*1000)
        ax.set_title(row['method']+' diagnostic; GPU events and host envelopes are distinct');ax.set_xlabel('Milliseconds from measured interval start')
    fig.savefig(root/'PROFILE_TIMELINE.png',dpi=160);fig.savefig(root/'PROFILE_TIMELINE.svg');plt.close(fig)
    lines=['# State-preparation GPU profiles','',report['scope'],'','| Method | Measured window (s) | Prep host union (s) | Source GPU union (s) | Destination GPU union (s) | Destination GPU during prep (s) |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['method']} | {r['measured_seconds']:.4f} | {r['preparation_wall_union_seconds']:.4f} | {r['engines']['0']['interval_unions_seconds']['gpu_event_union']:.4f} | {r['engines']['1']['interval_unions_seconds']['gpu_event_union']:.4f} | {r['engines']['1']['gpu_event_during_preparation_seconds']:.4f} |")
    lines +=['','Each preparation is a distinct8192-token checkpoint, four sequential preparations. Replay performs a real cold prefill with one output token on destination; copy performs full state transport. Foreground outputs64tokens. These diagnostic rows have the profiler enabled on both engines and cannot estimate profiler-off performance deltas.','']
    for r in results:
        lines +=[f"## {r['method']} kernel evidence",'']
        for engine,label in [('0','Source'),('1','Destination')]:
            metrics=r['engines'][engine];lines.append(f"{label} top kernels (summed named event durations, not additive elapsed decomposition):")
            lines +=[f"- `{name[:150]}`: {seconds*1000:.3f}ms" for name,seconds in metrics['top_kernel_summed_seconds'][:5]]
            names=metrics['transfer_or_index_named_kernels']
            if names:lines.append('Transfer/index-name candidates include '+', '.join('`'+name[:100]+'`' for name,_ in names[:4])+'.')
            lines.append('')
    lines +=['Replay visibly places model work on the foreground destination while source has no requested work. Copy creates source packaging/communication work and destination communication/installation alongside model execution. The presence of NCCL/index/copy names supports this operation-level distinction; timestamps and names do not establish which individual kernel caused a decode delay.','']
    lines += report['caveats']
    lines +=['','Clock alignment maps Torch epoch-based timestamps through the saved host epoch/monotonic anchor. Capture excludes setup/validation intervals by clipping to each recorded measured start/finish. No cross-GPU summation, device-busy inference, or inference of a race from timestamps.','']
    (root/'PROFILE_ANALYSIS.md').write_text('\n'.join(lines));print(json.dumps([{'method':r['method'],'source_gpu':r['engines']['0']['interval_unions_seconds']['gpu_event_union'],'destination_gpu':r['engines']['1']['interval_unions_seconds']['gpu_event_union'],'prep':r['preparation_wall_union_seconds']} for r in results]))


if __name__=='__main__':main()
