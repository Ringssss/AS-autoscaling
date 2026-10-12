"""Read-only interim statistics; unfinished cells never enter comparisons."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).parent/'results'/'campaign-20260928-r3'


def read_rows(path):
    if not path.exists():return [],None
    data=path.read_bytes()
    complete=data[:data.rfind(b'\n')+1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()],hashlib.sha256(complete).hexdigest()


def quantiles(values):
    a=sorted(values)
    return {str(q):a[max(0,math.ceil(len(a)*q/100)-1)] for q in [50,95,99]} if a else {}


def stats(directory):
    rs,sha=read_rows(directory/'requests.jsonl')
    es,_=read_rows(directory/'events.jsonl')
    telemetry,_=read_rows(directory/'telemetry.jsonl')
    spec=json.loads((directory/'spec.json').read_text())
    complete=(directory/'summary.json').exists()
    data=dict(directory=str(directory),spec=spec,complete=complete,record_count=len(rs),
        errors=sum(not r['success'] for r in rs),records_sha256=sha,ordinals=[r['ordinal'] for r in rs],
        cache_reset_count=sum(e['kind']=='initial_cache_reset' for e in es),
        preparation_events=[e for e in es if e['kind'].startswith('preparation')],
        migration_bytes=sum(e.get('bytes',0) for e in es if e['kind']=='preparation'))
    if not complete:
        if (directory/'heartbeat.json').exists():data['heartbeat']=json.loads((directory/'heartbeat.json').read_text())
        return data
    s=json.loads((directory/'summary.json').read_text())
    assert len(rs)==s['expected'] and len({r['ordinal'] for r in rs})==len(rs)
    duration=s['duration_seconds']
    total_in=sum(r['input_tokens'] for r in rs);total_out=sum(r['output_tokens'] for r in rs)
    data.update(duration_seconds=duration,throughput_requests_s=len(rs)/duration,output_tokens_s=total_out/duration,
        total_input_tokens=total_in,total_output_tokens=total_out,
        ttft=quantiles([r['ttft_seconds'] if r['ttft_seconds'] is not None else float('inf') for r in rs]),
        eligible_ttft=quantiles([r['ttft_seconds']-r['dependency_delay_seconds'] for r in rs if r['ttft_seconds'] is not None]),
        e2e=quantiles([r['e2e_seconds'] for r in rs]),
        mean_tpot=quantiles([r['tpot_seconds'] for r in rs if r['tpot_seconds'] is not None]),
        tasks=quantiles(list(s['task_completion_seconds'].values())),
        goodput=s['goodput'],input_token_cache_hit_fraction=sum(r['cached_tokens'] or 0 for r in rs)/total_in,
        cache_hit_requests=sum((r['cached_tokens'] or 0)>0 for r in rs),
        retractions=sum((r['engine_meta'] or {}).get('num_retractions',0) for r in rs),
        requests_with_retractions=sum((r['engine_meta'] or {}).get('num_retractions',0)>0 for r in rs),
        output_length_checks=all(r['engine_meta']['completion_tokens']==r['output_tokens'] for r in rs if r['success']),
        preparation_counts=s['preparation_counts'])
    components=collections.defaultdict(list)
    for r in rs:
        if r['ttft_seconds'] is None:continue
        components['dependency'].append(r['dependency_delay_seconds'])
        components['client_admission'].append(r['admission_delay_seconds'])
        components['routing_and_pre_dispatch'].append(r['dispatch_seconds']-r['admitted_seconds'])
        components['dispatch_to_first_token'].append(r['ttft_seconds']-(r['dispatch_seconds']-r['offered_seconds']))
    total_ttft=sum(r['ttft_seconds'] for r in rs if r['ttft_seconds'] is not None)
    data['ttft_decomposition']={k:dict(mean_seconds=sum(v)/len(v),fraction_of_sum_ttft=sum(v)/total_ttft,quantiles=quantiles(v)) for k,v in components.items()}
    data['phases']=[]
    for phase in range(3):
        group=[r for r in rs if r['phase']==phase]
        data['phases'].append(dict(phase=phase,requests=len(group),
            ttft=quantiles([r['ttft_seconds'] for r in group if r['ttft_seconds'] is not None]),
            eligible_ttft=quantiles([r['ttft_seconds']-r['dependency_delay_seconds'] for r in group if r['ttft_seconds'] is not None]),
            cache_hit_requests=sum((r['cached_tokens'] or 0)>0 for r in group),
            token_hit_fraction=sum(r['cached_tokens'] or 0 for r in group)/sum(r['input_tokens'] for r in group)))
    maxrun=[0]*3;maxwait=[0]*3;max_used=[0]*3;max_active_pinned=[0]*3;hbm=collections.defaultdict(list);errors=0
    cards={'node27':[6,7],'node28':[4,5],'node29':[0,1]}
    for t in telemetry:
        errors+='error' in t
        if 'load' in t:
            i=t['engine']
            for v in t['load']:
                maxrun[i]=max(maxrun[i],v['num_reqs']-v['num_waiting_reqs'])
                maxwait[i]=max(maxwait[i],v['num_waiting_reqs'])
            pool=t.get('pool',{});free=pool.get('available_token_slots');evictable=pool.get('evictable_token_slots')
            if free is not None:
                max_used[i]=max(max_used[i],1048576-free)
                if evictable is not None:max_active_pinned[i]=max(max_active_pinned[i],1048576-free-evictable)
        if 'hbm' in t:
            for c in cards[t['node']]:hbm[t['node']+':'+str(c)].append(int(t['hbm']['card'+str(c)]['vram Total Used Memory (MiB)'])/1024)
    data['telemetry']=dict(errors=errors,peak_running=maxrun,peak_waiting=maxwait,peak_occupied_kv_tokens=max_used,
        peak_non_evictable_kv_tokens=max_active_pinned,hbm_GiB_per_card={k:dict(min=min(v),max=max(v)) for k,v in hbm.items()})
    return data


def main():
    now=datetime.datetime.now().astimezone()
    out=ROOT/'interim'/now.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True,exist_ok=False)
    cells=[stats(d) for d in sorted((ROOT/'runs').iterdir()) if (d/'spec.json').exists()]
    report=dict(as_of=now.isoformat(),complete_cells=sum(c['complete'] for c in cells),planned_cells=160,
        complete_formal_requests=sum(c['record_count'] for c in cells if c['complete']),
        partial_formal_requests=sum(c['record_count'] for c in cells if not c['complete']),cells=cells,
        limitations=['One independent run per cell; no population confidence claim.',
            'Arrival time is a controlled workload clock; dependency/admission delays included.',
            'All p50/p95/p99 here use nearest rank; interpolated percentiles in earlier summaries differ slightly.',
            'Allocated KV pool includes evictable warm cache; full pool does not alone establish active-memory pressure.',
            'Length-valid performance only; semantic/batch-output equivalence not certified.',
            'Incomplete cells omitted from comparisons; experimental process continues.'])
    (out/'snapshot.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# AgentX baseline 阶段统计','',f'统计截点：{now.isoformat()}。实验继续运行。','',
        f"完整完成 {report['complete_cells']}/160 项；完整项 {report['complete_formal_requests']} 请求，未完成项已返回 {report['partial_formal_requests']} 请求。",'',
        '| 已完成方案 | 场景/BS | 请求/失败 | 总时长 s | TTFT p50/p95/p99 s | 请求完成 p50/p95/p99 s | 请求/s | 缓存输入 token 比例 |',
        '|---|---|---:|---:|---|---|---:|---:|']
    for c in cells:
        s=c['spec']
        if not c['complete']:continue
        fmt=lambda x:' / '.join(f'{x[str(p)]:.2f}' for p in [50,95,99])
        lines.append(f"| {s['policy']} | {s['scenario']}/{s['bs']} | {c['record_count']}/{c['errors']} | {c['duration_seconds']:.2f} | {fmt(c['ttft'])} | {fmt(c['e2e'])} | {c['throughput_requests_s']:.4f} | {c['input_token_cache_hit_fraction']:.2%} |")
        lines+=['',f"## {s['policy']} 明细",'',
            f"输出 {c['total_output_tokens']:,} tokens；平均输出吞吐 {c['output_tokens_s']:.2f} tokens/s。",
            f"请求具备前序完成条件之后，到首 token 的 p50/p95/p99：{fmt(c['eligible_ttft'])} 秒；该辅助指标不替代全流量延迟。",
            f"选定子流任务完成时间 p50/p95/p99：{fmt(c['tasks'])} 秒。",
            f"平均 TPOT 的跨请求 p50/p95/p99：{fmt({k:v*1000 for k,v in c['mean_tpot'].items()})} ms/token。",'',
            '| 延迟组成 | 平均秒数 | 占全部 TTFT 总和 |','|---|---:|---:|']
        for k,v in c['ttft_decomposition'].items():lines.append(f"| {k} | {v['mean_seconds']:.2f} | {v['fraction_of_sum_ttft']:.2%} |")
        lines+=['','各组成的 p50 不具有可加性；上表使用均值及总时间份额。','',
            '| 轮次 | 全流量 TTFT p50/p95/p99 s | 可执行后 TTFT p50 s | 命中缓存请求 | 输入 token 命中比例 |','|---|---|---:|---:|---:|']
        for p in c['phases']:lines.append(f"| {p['phase']} | {fmt(p['ttft'])} | {p['eligible_ttft']['50']:.2f} | {p['cache_hit_requests']}/{p['requests']} | {p['token_hit_fraction']:.2%} |")
        lines+=['','| SLO：TTFT / 平均 TPOT | 达标率 | goodput req/s |','|---|---:|---:|']
        for g in c['goodput']:
            if (g['ttft_slo'],g['tpot_slo']) in [(10,.2),(10,.5),(30,.2),(30,.5)]:
                lines.append(f"| {g['ttft_slo']}s / {g['tpot_slo']}s | {g['attainment']:.2%} | {g['requests_per_second']:.5f} |")
        t=c['telemetry']
        lines+=['',f"实际最高 running：{t['peak_running']}；最高引擎等待数：{t['peak_waiting']}。",
            f"KV 最高占用 slots：{t['peak_occupied_kv_tokens']}；其中不可驱逐部分最高：{t['peak_non_evictable_kv_tokens']}。",
            f"记录到 request retraction {c['retractions']} 次，涉及 {c['requests_with_retractions']} 个请求。",
            f"迁移流量 {c['migration_bytes']} bytes；缓存 reset {c['cache_reset_count']} 次（仅轨迹开始）；采集错误 {t['errors']}。",'',
            '| 节点:卡 | HBM 最小/最大 GiB |','|---|---:|']
        for k,v in t['hbm_GiB_per_card'].items():lines.append(f"| {k} | {v['min']:.2f} / {v['max']:.2f} |")
    lines+=['','## 未完成项','']
    for c in cells:
        if not c['complete']:
            lines.append(f"- {c['spec']['policy']} / {c['spec']['scenario']} / BS={c['spec']['bs']}：已返回 {c['record_count']}/{c['spec']['requests']}，失败 {c['errors']}，成功准备 {len([e for e in c['preparation_events'] if e['kind']=='preparation' and not e.get('error')])} 次。不据此计算完整尾延迟或排名。")
    lines+=['','## 解释边界','',
        '当前仅一个完整方案时，不能判断迁移优于 DualMap 或 CacheRoute。',
        '“均衡”描述 key 到达分布，不表示系统处于低负载。384 个请求在约 150 秒内计划到达；较慢本地生成与子流依赖会显著推迟后续轮次。',
        '延迟含依赖、admission、路由和引擎路径，不能全部归因于 KV-cache gravity。',
        'SLO 是预设实验阈值；达标为零表明当前到达/硬件/并发组合未满足该阈值，不能单独说明某算法无价值。',
        '当前是符号前缀、原始长度回放；仅认证长度，未认证输出语义等价。每格一次独立运行，不报告统计显著性。',
        '部分请求结果不与完整结果混排；零请求错误不包含此前保留的启动预检失败。']
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    complete=[c for c in cells if c['complete']]
    if len(complete)==1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        c=complete[0];phases=c['phases'];x=np.arange(3)
        fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        axs[0].bar(x-.18,[p['ttft']['50'] for p in phases],.36,label='From scheduled arrival')
        axs[0].bar(x+.18,[p['eligible_ttft']['50'] for p in phases],.36,label='After predecessor completes')
        axs[0].set(xticks=x,xticklabels=['Round 1','Round 2','Round 3'],ylabel='Median TTFT (s)',title='Dependency delay changes the picture')
        axs[0].legend(fontsize=8)
        axs[1].bar(x,[100*p['token_hit_fraction'] for p in phases],color='#2a9d8f')
        axs[1].set(xticks=x,xticklabels=['Round 1','Round 2','Round 3'],ylabel='Cached input tokens (%)',ylim=(0,100),title='Cache persists across rounds')
        parts=c['ttft_decomposition']
        labels=['Predecessor dependency','Routing / pre-dispatch','Dispatch to first token']
        values=[parts[k]['fraction_of_sum_ttft']*100 for k in ['dependency','routing_and_pre_dispatch','dispatch_to_first_token']]
        axs[2].barh(labels,values,color=['#e9c46a','#457b9d','#e76f51'])
        axs[2].set(xlabel='Share of summed TTFT (%)',xlim=(0,100),title='Where time is spent')
        for i,v in enumerate(values):axs[2].text(v+1,i,f'{v:.1f}%',va='center')
        fig.suptitle('DualMap | balanced arrivals | global concurrency cap 128 | one complete run')
        fig.savefig(out/'FIGURE.png',dpi=180);fig.savefig(out/'FIGURE.pdf');plt.close(fig)
        with (out/'REPORT.zh.md').open('a') as f:f.write('\n![阶段结果](FIGURE.png)\n')
    print(out)
    for c in cells:
        if c['complete']:print(json.dumps({k:v for k,v in c.items() if k not in ['ordinals','spec','preparation_events','goodput']},indent=2))


if __name__=='__main__':main()
