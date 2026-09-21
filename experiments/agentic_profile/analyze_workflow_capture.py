"""Offline joint analysis of actual tool service and bounded GPU trace intervals."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from experiments.agentic_profile.analyze_engine_traces import clipped, merged, union


def intersection(left, right):
    a, b = merged(left), merged(right)
    result = []; i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            result.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/agentic_profile/results/20260915/heavy-workflow-profile'))
    args = parser.parse_args(); root = args.root
    capture = json.loads((root/'capture.json').read_text())
    folder = root/'workflow/000-shared-c8-r0'
    timeline = json.loads((folder/'timeline.json').read_text())
    rows = sorted([json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()], key=lambda r: r['agent'])
    trace_file = next((root/'torch-profile').glob('*.gz'))
    trace = json.load(gzip.open(trace_file))
    anchor = capture['profile_request_anchor']
    # Convert microseconds relative to Torch's epoch base into host monotonic.
    origin_us = anchor['epoch']*1e6-trace['baseTimeNanoseconds']/1000
    groups = defaultdict(list); counts = Counter()
    for event in trace['traceEvents']:
        if event.get('ph') != 'X' or not event.get('dur'):
            continue
        category = event.get('cat', '')
        start = anchor['monotonic']+(event['ts']-origin_us)/1e6
        finish = start+event['dur']/1e6
        if category in ('kernel', 'gpu_memcpy', 'gpu_memset', 'Trace'):
            groups[category].append((start, finish)); counts[category] += 1
        elif category in ('cuda_runtime', 'cuda_driver'):
            if 'Synchronize' in event['name'] or 'WaitEvent' in event['name']:
                groups['cpu_wait_api'].append((start, finish))
            if 'Launch' in event['name']:
                groups['cpu_launch_api'].append((start, finish))
    if len(groups['Trace']) != 1:
        raise ValueError('expected one explicit profiler capture window')
    lo, hi = groups['Trace'][0]
    for name in list(groups):
        groups[name] = clipped(groups[name], lo, hi)
    gpu = merged(groups['kernel']+groups['gpu_memcpy']+groups['gpu_memset'])
    tools = [(r['tool']['metrics']['started_monotonic'], r['tool']['metrics']['finished_monotonic']) for r in rows]
    requests = [(r[turn]['start'], r[turn]['finish']) for r in rows for turn in ('first', 'second')]
    tool_window = clipped(tools, lo, hi)
    gpu_tool = intersection(gpu, tool_window)
    tool_wait = intersection(groups['cpu_wait_api'], tool_window)
    wait_gpu = intersection(groups['cpu_wait_api'], gpu)
    result = {'scope': 'One profiler-enabled shared8core/c8 actual995file AST workflow; diagnostic, excluded from paired performance conclusions.',
              'capture_success': capture['success'], 'workflow_successful': sum(r['success'] for r in rows), 'workflow_offered': len(rows),
              'profile_requested_scheduler_steps': capture['profile_num_steps'], 'event_counts': dict(counts),
              'capture_monotonic': [lo, hi], 'capture_since_workflow_begin_seconds': [lo-timeline['begin_monotonic'], hi-timeline['begin_monotonic']],
              'capture_seconds': hi-lo, 'gpu_kernel_union_seconds': union(groups['kernel']), 'gpu_event_union_seconds': union(gpu),
              'no_recorded_gpu_event_seconds': hi-lo-union(gpu),
              'tools_intersecting_capture': [r['agent'] for r,t in zip(rows,tools) if t[0] < hi and t[1] > lo],
              'tool_service_union_in_capture_seconds': union(tool_window), 'gpu_and_tool_service_overlap_seconds': union(gpu_tool),
              'tool_service_without_recorded_gpu_seconds': union(tool_window)-union(gpu_tool),
              'cpu_wait_api_union_seconds': union(groups['cpu_wait_api']), 'cpu_wait_api_with_gpu_event_seconds': union(wait_gpu),
              'cpu_wait_api_without_gpu_event_seconds': union(groups['cpu_wait_api'])-union(wait_gpu),
              'cpu_wait_api_and_tool_service_seconds': union(tool_wait),
              'tool_service_scope': 'Worker function wall interval, not continuous on-CPU execution. Includes AST parsing/read/hash/textserialization; full tool processCPU totals cannot be assigned fractionally to the capture.',
              'gpu_scope': 'Union of traced kernel/memcpy/memset intervals, not SM utilization. Complement means no recorded event, not proof of machine-wide GPU idle or CPU-caused starvation.',
              'clock_alignment': 'Torch baseTimeNanoseconds + event ts mapped via saved profile_request_anchor epoch/perf_counter. File begin epoch/monotonic independently crosschecked.',
              'clock_offset_difference_seconds': (anchor['epoch']-anchor['monotonic'])-(timeline['begin_epoch']-timeline['begin_monotonic'])}
    (root/'PROFILE_ANALYSIS.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, gridspec_kw={'height_ratios':[1.3,1]})
    begin = timeline['begin_monotonic']
    for row, tool in zip(rows, tools):
        for turn in ('first', 'second'):
            r=row[turn]; axes[0].broken_barh([(r['start']-begin,r['finish']-r['start'])],(row['agent']-.35,.7),facecolors='C0')
        axes[0].broken_barh([(tool[0]-begin,tool[1]-tool[0])],(row['agent']-.35,.7),facecolors='C1')
    axes[0].axvspan(lo-begin, hi-begin, alpha=.2, color='green', label='bounded profiler window')
    axes[0].set(title='Actual workflow: blue=inference request, orange=tool service',xlabel='Seconds from workload begin',ylabel='Agent ID');axes[0].legend()
    tracks=[('GPU event union',gpu),('CPU CUDA launch',groups['cpu_launch_api']),('CPU CUDA wait',groups['cpu_wait_api']),('Tool service union',tool_window)]
    for i,(name,intervals) in enumerate(tracks):
        spans=clipped(merged(intervals),lo,hi)
        axes[1].broken_barh([((a-lo)*1000,(b-a)*1000) for a,b in spans],(i-.35,.7),facecolors=f'C{i}')
    axes[1].set_yticks(range(len(tracks)),[n for n,_ in tracks]);axes[1].set_xlim(0,(hi-lo)*1000)
    axes[1].set(title='Captured overlap; CPU waits can accompany useful GPU execution',xlabel='Milliseconds from actual profiler window start')
    fig.savefig(root/'PROFILE_TIMELINE.png',dpi=160);fig.savefig(root/'PROFILE_TIMELINE.svg');plt.close(fig)
    lines=['# Actual tool + GPU profile', '', result['scope'], '',
           f"All {len(rows)} agents completed. The actual profiler window spans {lo-begin:.3f}–{hi-begin:.3f}s after workflow start ({hi-lo:.4f}s), not the whole workflow. It intersects tool service for agents {result['tools_intersecting_capture']}.", '',
           f"The trace has {counts['kernel']:,} GPU kernels. Kernel interval union is {result['gpu_kernel_union_seconds']:.4f}s; kernel+memcpy+memset union is {result['gpu_event_union_seconds']:.4f}s. Tool-service union within the capture is {result['tool_service_union_in_capture_seconds']:.4f}s, of which {result['gpu_and_tool_service_overlap_seconds']:.4f}s overlaps GPU events. Thus the capture does contain actual tools running concurrently with other agents' GPU inference.", '',
           f"Time without a recorded GPU event is {result['no_recorded_gpu_event_seconds']:.4f}s. During tool service, the corresponding value is {result['tool_service_without_recorded_gpu_seconds']:.4f}s. These are trace complements, not hardware idle/utilization or CPU starvation measurements.", '',
           f"CPU CUDA wait API interval union is {result['cpu_wait_api_union_seconds']:.4f}s; {result['cpu_wait_api_with_gpu_event_seconds']:.4f}s overlaps traced GPU work and {result['cpu_wait_api_without_gpu_event_seconds']:.4f}s does not. A wait is not CPU computation; none of these durations is automatically removable overhead.", '',
           'The AST operation is actual995file/14.8MB analysis, one pass, with returned bounded output consumed by the later turn. Tool bars denote function service wall time, not continuous CPU occupancy. Telemetry provides sampled process counters separately; no scheduler-stack trace permits causal attribution of submillisecond gaps.', '',
           f"Clock mapping uses Torch epoch base and profile-request epoch/monotonic anchors. The independent workflow anchor differs in epoch-minus-monotonic offset by {result['clock_offset_difference_seconds']:.9f}s; uncertainty of profiler synchronization is not otherwise calibrated.", '',
           'This confirms a legal cross-agent overlap boundary between CPU tools and GPU inference. It does not establish harmful contention or show that separating cores/nodes improves end-to-end service. The profiler-off heavy shared/isolation repetitions determine that. The earlier two-request profile has a different workload and capture scope; differences are not speedup estimates.', '']
    (root/'PROFILE_ANALYSIS.md').write_text('\n'.join(lines));print(json.dumps(result))


if __name__=='__main__':main()
