"""Offline engine timeline analysis; interval unions never sum concurrent GPU work."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import statistics


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def union(intervals):
    return sum(b-a for a, b in merged(intervals))


def clipped(intervals, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in intervals if a < hi and b > lo]


def classify(name):
    name = name.lower()
    if any(k in name for k in ('moe', 'expert', 'gatingsoftmax')):
        return 'explicit_moe_routing_experts'
    if any(k in name for k in ('flashattn', 'flash_attn')):
        return 'full_attention'
    if any(k in name for k in ('gated_delta', 'gateddelt', 'recurrent', 'causal_conv', 'chunk_', 'cumsum')):
        return 'linear_attention_named'
    if any(k in name for k in ('gemm', 'matmul')):
        return 'matrix_multiply_unattributed'
    if 'copy' in name:
        return 'copy_kernels'
    return 'other_kernels'


def overhead(path):
    rows = json.loads(path.read_text())
    compact = [{k: row[k] for k in ('mode', 'elapsed', 'epoch', 'monotonic')} for row in rows]
    off = [r['elapsed'] for r in compact if r['mode'] == 'off']
    for r in compact:
        if r['mode'] != 'off':
            r['change_vs_off_mean_percent'] = (r['elapsed']/statistics.mean(off)-1)*100
            r['change_vs_each_off_percent'] = [(r['elapsed']/x-1)*100 for x in off]
    return rows, compact


def extract_torch(path, rows):
    data = json.load(gzip.open(path))
    app = next(r for r in rows if r['mode'] == 'torch')
    origin = app['epoch']*1e6-data['baseTimeNanoseconds']/1000
    groups = defaultdict(list)
    names = defaultdict(Counter)
    counts = Counter()
    for e in data['traceEvents']:
        if e.get('ph') != 'X' or not e.get('dur'):
            continue
        cat = e.get('cat', '')
        interval = ((e['ts']-origin)/1e6, (e['ts']-origin+e['dur'])/1e6)
        if cat == 'kernel':
            groups['gpu_kernels'].append(interval)
            groups[classify(e['name'])].append(interval)
        elif cat in ('gpu_memcpy', 'gpu_memset'):
            groups[cat].append(interval)
        elif cat in ('cpu_op', 'cuda_runtime', 'cuda_driver'):
            groups[cat].append(interval)
            if cat.startswith('cuda'):
                if any(k in e['name'] for k in ('Synchronize', 'StreamWaitEvent')):
                    groups['cpu_cuda_wait_api'].append(interval)
                if 'Launch' in e['name']:
                    groups['cpu_cuda_launch_api'].append(interval)
        else:
            continue
        counts[cat] += 1
        names[cat][e['name']] += e['dur']/1e6
    gpu = groups['gpu_kernels']+groups['gpu_memcpy']+groups['gpu_memset']
    groups['gpu_activity_union'] = merged(gpu)
    details = {'trace': str(path), 'event_counts': counts, 'application_profile_window_seconds': app['elapsed'],
               'clock_alignment': 'epoch = baseTimeNanoseconds/1e9 + trace ts/1e6; subtract paired app epoch. App monotonic = app.monotonic + relative timestamp.',
               'app_epoch': app['epoch'], 'app_monotonic': app['monotonic'],
               'interval_union_seconds_in_application_window': {k: union(clipped(v, 0, app['elapsed'])) for k, v in groups.items()},
               'interval_union_seconds_full_capture': {k: union(v) for k, v in groups.items()},
               'top_inclusive_event_name_seconds': {cat: names[cat].most_common(12) for cat in names},
               'cpu_caveat': 'CPU op/API values are elapsed traced intervals, not process CPU time. CPU ops nest and may wrap runtime calls; never add these totals. Synchronization may wait for useful GPU execution.'}
    return details, groups, app


def extract_nsys(path, rows):
    app = next(r for r in rows if r['mode'] == 'nsys')
    conn = sqlite3.connect(path)
    names = dict(conn.execute('select id,value from StringIds'))
    epoch = conn.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]/1e9
    offset = epoch-app['epoch']
    tables = {x[0] for x in conn.execute("select name from sqlite_master where type='table'")}
    groups = defaultdict(list)
    counts = {}
    source_tables = {'gpu_kernels': 'CUPTI_ACTIVITY_KIND_KERNEL', 'gpu_memcpy': 'CUPTI_ACTIVITY_KIND_MEMCPY', 'gpu_memset': 'CUPTI_ACTIVITY_KIND_MEMSET', 'gpu_graph_envelopes': 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'}
    for category, table in source_tables.items():
        if table not in tables:
            continue
        groups[category] = [(a/1e9+offset, b/1e9+offset) for a, b in conn.execute(f'select start,end from {table}')]
        counts[category] = len(groups[category])
    api_names = Counter()
    for a, b, name_id in conn.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        name = names[name_id]
        api_names[name] += (b-a)/1e9
        category = 'cpu_cuda_wait_api' if 'Synchronize' in name or 'WaitEvent' in name else 'cpu_cuda_launch_api' if 'Launch' in name else 'cpu_other_cuda_api'
        groups[category].append((a/1e9+offset, b/1e9+offset))
    osrt = Counter()
    for a, b, name_id in conn.execute('select start,end,nameId from OSRT_API'):
        osrt[names[name_id]] += (b-a)/1e9
    groups['gpu_activity_envelope_union'] = merged(sum([groups[k] for k in source_tables], []))
    details = {'sqlite': str(path), 'event_counts': counts, 'application_profile_window_seconds': app['elapsed'],
               'app_epoch': app['epoch'], 'app_monotonic': app['monotonic'], 'session_epoch': epoch,
               'clock_alignment': 'Nsight normalized start/end ns + TARGET_INFO_SESSION_START_TIME.utcEpochNs; subtract app epoch; use paired app epoch/monotonic for application alignment.',
               'interval_union_seconds_in_application_window': {k: union(clipped(v, 0, app['elapsed'])) for k, v in groups.items()},
               'interval_union_seconds_full_capture': {k: union(v) for k, v in groups.items()},
               'top_cuda_api_inclusive_seconds': api_names.most_common(12), 'os_runtime_summed_thread_seconds': osrt.most_common(12),
               'graph_caveat': 'Graph-level tracing hides decode child kernels: graph envelopes include internal gaps and are not measured SM busy time. Union envelopes with other GPU events; never add envelopes to overlapping kernels. Kernel-only summaries omit most decode.',
               'os_caveat': 'OS runtime calls include waiting across multiple threads/processes; sums are neither CPU utilization nor application critical path. CPU sampling/thread scheduler tracing disabled; no OS scheduler-stack attribution.'}
    conn.close()
    return details, groups, app


def plot(output, torch_data, nsys_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    tracks = ['gpu_kernels', 'gpu_graph_envelopes', 'gpu_memcpy', 'cpu_cuda_launch_api', 'cpu_cuda_wait_api']
    for ax, (title, data) in zip(axes, [('PyTorch profiler', torch_data), ('Nsight Systems (graph envelopes)', nsys_data)]):
        _, groups, app = data
        used = [name for name in tracks if groups.get(name)]
        for i, name in enumerate(used):
            spans = clipped(merged(groups[name]), 0, app['elapsed'])
            ax.broken_barh([(a*1000, (b-a)*1000) for a, b in spans], (i-.35, .7), facecolors=f'C{i}')
        for request in app['batch']:
            first = request['first']-app['monotonic']
            ax.axvline(first*1000, color='gray', alpha=.5, linestyle=':')
        ax.set_yticks(range(len(used)), [s.replace('_', ' ') for s in used]); ax.set_xlim(0, app['elapsed']*1000)
        ax.set_title(title+'; dotted lines = application first-token arrivals')
        ax.set_xlabel('Milliseconds from paired application epoch/monotonic anchor')
    fig.savefig(output/'profile-timeline.png', dpi=160)
    fig.savefig(output/'profile-timeline.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/agentic_profile/results/20260915'))
    args = parser.parse_args(); root = args.root
    tr, to = overhead(root/'torch-profile/summary.json')
    nr, no = overhead(root/'nsys-profile/summary.json')
    td = extract_torch(next((root/'torch-profile').glob('*.gz')), tr)
    nd = extract_nsys(root/'nsys-engine/engine.sqlite', nr)
    result = {'torch': td[0], 'nsys': nd[0], 'off_on_off': {'torch': to, 'nsys': no},
              'scope': 'Short warmed two-request 2048-input/64-output engine profiles, separate from CPU interference campaign. One off/on/off triplet each; no significance, no production profile representativeness, no complete critical path attribution.'}
    (root/'profile-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(root, td, nd)
    tu = td[0]['interval_union_seconds_in_application_window']; nu = nd[0]['interval_union_seconds_in_application_window']
    lines = ['# Engine profile interpretation', '', result['scope'], '',
             f"PyTorch records {td[0]['event_counts']['kernel']:,} kernel events. Within the {td[2]['elapsed']:.4f}s application window, kernel union is {tu['gpu_kernels']:.4f}s and kernel+memcpy+memset union is {tu['gpu_activity_union']:.4f}s. These measure event residency intervals, not achieved occupancy or SM utilization.", '',
             f"Nsight records {nd[0]['event_counts']['gpu_graph_envelopes']} CUDA graph executions. Within the {nd[2]['elapsed']:.4f}s application window, individually visible kernel union is {nu['gpu_kernels']:.4f}s, graph-envelope union is {nu['gpu_graph_envelopes']:.4f}s, and the combined GPU envelope union is {nu['gpu_activity_envelope_union']:.4f}s. Looking only at kernel summaries would omit most graph decode. Graph envelopes may contain internal idle gaps.", '',
             '| Torch kernel name category | Interval union in app window (ms) |', '| --- | ---: |']
    for name in ('explicit_moe_routing_experts', 'matrix_multiply_unattributed', 'full_attention', 'linear_attention_named', 'copy_kernels', 'other_kernels'):
        lines.append(f'| {name} | {tu.get(name,0)*1000:.3f} |')
    lines += ['', 'Categories use names, not architectural proof. Unattributed GEMMs are not all MoE. Attention, recurrent updates and expert computation remain dependencies within model forward; high kernel time alone does not prove they can be independently disaggregated. Category unions can overlap; do not add them as an exclusive latency decomposition.', '',
              f"Torch CPU CUDA wait-API union is {tu['cpu_cuda_wait_api']*1000:.2f}ms and launch-API union is {tu['cpu_cuda_launch_api']*1000:.2f}ms. They overlap useful GPU execution, and nested CPU operators include child calls. These are not CPU busy-time or removable overhead estimates.", '',
              'OS runtime tracing includes epoll/poll/condition waits across multiple processes/threads. CPU sampling and scheduler tracing were disabled; this capture cannot identify CPU scheduling stacks or prove contention from OS waits.', '',
              '| Capture | Off-before (s) | On (s) | Off-after (s) | On vs mean off |', '| --- | ---: | ---: | ---: | ---: |']
    for name, values in [('Torch', to), ('Nsight', no)]:
        on = next(v for v in values if v['mode'] != 'off')
        lines.append(f"| {name} | {values[0]['elapsed']:.4f} | {on['elapsed']:.4f} | {values[-1]['elapsed']:.4f} | {on['change_vs_off_mean_percent']:+.2f}% |")
    lines += ['', 'One ordered triplet is only a descriptive overhead check. Torch-on lies between its off runs; cache/history, warmup, timing and noise prevent a causal overhead estimate. Final performance comparisons must use profiler-off paired repetitions.', '',
              'Alignment uses Torch baseTimeNanoseconds plus trace timestamps, or Nsight session UTC epoch plus relative nanoseconds, then the simultaneously saved application epoch/monotonic anchor. This provides sub-millisecond-scale visualization but wall-clock sampling/clock-sync uncertainty is not calibrated. Dotted first-token lines are network-observed application timestamps, not exact prefill/decode GPU boundaries.', '',
              'Decision implication: these captures identify substantial GPU model execution and CPU/GPU coordination; they do not reveal an additional independent phase whose separation is already justified. The equal-resource CPU isolation and serial/concurrent preparation experiments must establish contention and net benefit. No NCU run is needed without a concrete kernel-optimization hypothesis.', '']
    (root/'profile-analysis.md').write_text('\n'.join(lines))
    print(root/'profile-analysis.md')


if __name__ == '__main__':
    main()
