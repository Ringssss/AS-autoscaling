"""Report all fixed six-worker comparisons, including unfavorable cells."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import statistics

def report(root):
    d=json.loads((root/'summary.json').read_text())
    grouped=defaultdict(list)
    for t in d['trials']:
        grouped[t['clients'],t['policy'],t['prep']].append(t)
    requests=[]
    if (root/'requests.jsonl').exists():
        requests=[json.loads(l) for l in (root/'requests.jsonl').open()]
    gaps=[]
    for path in root.glob('telemetry-*.jsonl'):
        previous=None
        for line in path.open():
            try:
                event=json.loads(line)['event'];current=event['counter_id']
            except (ValueError,KeyError):continue
            if previous is not None and current!=previous+1:
                gaps.append(dict(file=path.name,previous=previous,current=current))
            previous=current
    rows=[]
    for (n,policy,prep),trials in sorted(grouped.items()):
        selected=[r for r in requests if (r['clients'],r['policy'],r['prep'])==(n,policy,prep)]
        summary={metric:statistics.median(t['summary'][metric] for t in trials)
            for metric in ('wall','paid_wall','task_p50','task_p90','ttft_p90','tpot_p90','slo_attainment')}
        rows.append(dict(clients=n,policy=policy,prep=prep,repeats=len(trials),
            wall_runs=[t['summary']['wall'] for t in trials],summary=summary,
            preparation_seconds=[t['preparation_seconds'] for t in trials],
            client_router_seconds=[t['client_router_seconds'] for t in trials],
            dispatch_counts=dict(Counter(r['target'] for r in selected)),
            cold_requests=sum(r['cached_tokens']<65536 for r in selected),
            computed_input_tokens=sum(r['input_tokens']-r['cached_tokens'] for r in selected),
            cache_mismatches=sum(len(t['cache_mismatches']) for t in trials),
            peak_decode=[t['peak_decode'] for t in trials]))
    comparisons=[]
    for n in d['levels']:
        required=[('blitz','none'),('dualmap','none'),('blitz','copy'),('blitz','recompute')]
        by_rep=defaultdict(dict)
        for t in d['trials']:
            if t['clients']==n:by_rep[t['repetition']][t['policy'],t['prep']]=t
        complete=[group for group in by_rep.values() if all(k in group for k in required)]
        if not complete:continue
        arms={k:dict(policy=k[0],summary={metric:statistics.median(g[k]['summary'][metric] for g in complete)
            for metric in ('wall','slo_attainment')}) for k in required}
        base=min((arms[p,'none'] for p in ('blitz','dualmap')),key=lambda r:r['summary']['wall'])
        copy=arms['blitz','copy'];recompute=arms['blitz','recompute']
        comparisons.append(dict(clients=n,paired_repeats=len(complete),strongest=base['policy'],
            copy_vs_strongest_pct=100*(1-copy['summary']['wall']/base['summary']['wall']),
            copy_vs_recompute_pct=100*(1-copy['summary']['wall']/recompute['summary']['wall']),
            copy_slo=copy['summary']['slo_attainment'],baseline_slo=base['summary']['slo_attainment']))
    lines=['# 六实例快速验证','',f'状态：{d["status"]}；已完成 {len(d["trials"])} 个 trial。',
        '三节点、每节点两个 TP2 实例，32/96/192 闭环客户端，各两轮，64K 共享前缀。',
        '固定合成冷扩容压力场景。计入复制及提前重算成本；连接初始化单列。',
        '原始 Blitz/DualMap 策略通过统一 SGLang 适配器运行。节点 28 保留了空闲常驻 GPU 上下文。','',
        '|并发|策略|重复|完成 wall 秒（每次）|任务 p90 秒|TTFT p90 秒|TPOT p90 ms|联合 SLO|',
        '|---:|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        s=r['summary']
        lines.append(f'|{r["clients"]}|{r["policy"]}+{r["prep"]}|{r["repeats"]}|'+
            ', '.join(f'{x:.2f}' for x in r['wall_runs'])+
            f'|{s["task_p90"]:.2f}|{s["ttft_p90"]:.2f}|{s["tpot_p90"]*1000:.1f}|{s["slo_attainment"]*100:.1f}%|')
    lines+=['','|并发|更快的纯路由|复制相对纯路由 wall 降幅|复制相对提前重算 wall 降幅|',
        '|---:|---|---:|---:|']
    for c in comparisons:
        lines.append(f'|{c["clients"]}|{c["strongest"]}|{c["copy_vs_strongest_pct"]:+.1f}%|{c["copy_vs_recompute_pct"]:+.1f}%|')
    lines+=['',f'遥测序号缺口：{len(gaps)}。缓存估计拒绝项：{sum(r["cache_mismatches"] for r in rows)}。',
        'SLO 固定为 TTFT <=10s 且 TPOT <=200ms。两次重复报告原始数值，不以两个样本声称统计显著性。',
        '单 worker 峰值不能直接相加为同时运行峰值。客户端路由开销、冷请求数与重算 token 详见 report.json。','']
    if d.get('failure'):lines.append(f'失败原因：`{d["failure"]}`。')
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    (root/'report.json').write_text(json.dumps(dict(status=d['status'],rows=rows,comparisons=comparisons,telemetry_gaps=gaps),indent=2)+'\n')
    print(json.dumps(dict(status=d['status'],trials=len(d['trials']),comparisons=comparisons,telemetry_gaps=len(gaps)),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    report(p.parse_args().root)
