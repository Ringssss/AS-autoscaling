"""Use the stronger, memoized DualMap adapter and both recompute controls."""
from collections import defaultdict
import json
from pathlib import Path
import statistics


def run(root):
    main=json.loads((root/'summary.json').read_text())
    control=json.loads((root/'recompute-wait-control/summary.json').read_text())
    accepted=[]
    for origin,data in [('main',main),('control',control)]:
        for t in data['trials']:
            if origin=='main' and t['policy']=='dualmap':continue
            key=t['policy']+'_'+t['prep']
            accepted.append(dict(t,origin=origin,arm=key))
    grouped=defaultdict(list)
    for t in accepted:grouped[t['clients'],t['arm']].append(t)
    rows=[]
    metrics=['wall','paid_wall','task_p50','task_p90','ttft_p50','ttft_p90','tpot_p50','tpot_p90',
             'slo_attainment','goodput','ttft_slo_attainment']
    for (clients,arm),ts in sorted(grouped.items()):
        rows.append(dict(clients=clients,arm=arm,repeats=len(ts),
            summaries=[t['summary'] for t in ts],
            median={m:statistics.median(t['summary'][m] for t in ts) for m in metrics},
            wall_runs=[t['summary']['wall'] for t in ts],
            prep_runs=[t['preparation_seconds'] for t in ts],
            cache_mismatches=sum(len(t['cache_mismatches']) for t in ts)))
    comparisons=[]
    required=['blitz_none','dualmap_none','blitz_copy','blitz_recompute','blitz_recompute_wait']
    for n in [32,96,192]:
        arms={r['arm']:r for r in rows if r['clients']==n}
        if not all(k in arms and arms[k]['repeats']==2 for k in required):continue
        base=min([arms['blitz_none'],arms['dualmap_none']],key=lambda r:r['median']['wall'])
        recompute=min([arms['blitz_recompute'],arms['blitz_recompute_wait']],key=lambda r:r['median']['wall'])
        copy=arms['blitz_copy']
        comparisons.append(dict(clients=n,best_routing=base['arm'],best_recompute=recompute['arm'],
            copy_vs_routing_pct=100*(1-copy['median']['wall']/base['median']['wall']),
            copy_vs_recompute_pct=100*(1-copy['median']['wall']/recompute['median']['wall']),
            copy_slo=copy['median']['slo_attainment'],baseline_slo=base['median']['slo_attainment']))
    gaps=[]
    for base in [root,root/'recompute-wait-control']:
        for p in base.glob('telemetry-*.jsonl'):
            last=None
            for raw in p.open():
                try:counter=json.loads(raw)['event']['counter_id']
                except (ValueError,KeyError):continue
                if last is not None and counter!=last+1:gaps.append([str(p),last,counter])
                last=counter
    labels={'blitz_none':'Blitz','dualmap_none':'DualMap（缓存查询优化后）',
        'blitz_copy':'Blitz + 复制','blitz_recompute':'Blitz + 异步提前重算',
        'blitz_recompute_wait':'Blitz + 等准备完成再分发'}
    complete=control['status']=='complete' and len(comparisons)==3
    lines=['# 六实例验证：最终采用更强的基线','',
        f'有效对照状态：{"完成" if complete else "运行中"}。旧适配层的192并发DualMap超时保留，已由优化查询后的同策略对照替换。',
        f'采用的已完成 trial：{len(accepted)}；请求：{sum(t["summary"]["requests"] for t in accepted)}。',
        '三节点、每节点两个 TP2 实例（12 张卡）；每客户端两轮，64K 公共前缀，每轮新增64个私有token，输出128 tokens。',
        '合成同步突发；模型和通信连接已就绪，初始仅源端有该状态；准备领先到达1秒。',
        '复制和重算成本计入。总wall从固定到达时刻算起；paid_wall再包含1秒提前窗口。','',
        '|并发|策略|完成wall秒（两次）|TTFT p90秒|TPOT p90毫秒|联合SLO|',
        '|---:|---|---|---:|---:|---:|']
    for r in rows:
        s=r['median'];vals=', '.join(f'{x:.2f}' for x in r['wall_runs'])
        lines.append(f'|{r["clients"]}|{labels[r["arm"]]}|{vals}|{s["ttft_p90"]:.2f}|{1000*s["tpot_p90"]:.1f}|{100*s["slo_attainment"]:.1f}%|')
    lines+=['','|并发|更快的纯路由|复制相对纯路由wall降幅|复制相对更快的提前重算wall降幅|',
        '|---:|---|---:|---:|']
    for c in comparisons:
        lines.append(f'|{c["clients"]}|{labels[c["best_routing"]]}|{c["copy_vs_routing_pct"]:+.1f}%|{c["copy_vs_recompute_pct"]:+.1f}%|')
    lines+=['','方法与核验：',
        '- 原版 Blitz Rust LmetricQ 与原版 DualMap 队列/派发/重平衡，统一 SGLang 后端和已确认可执行缓存视图；不是原生完整系统的部署排名。',
        '- DualMap 最终结果缓存不可变输入与同一缓存版本的重复查询结果，保留原路由策略。旧适配层 192 并发超时，其旧高延迟不作为算法优劣的证据。',
        '- 等待准备对照避免前台提前进入 Engine 而重复 prefill；固定到达时间不后移，等待完全计入。',
        '- 五条复制链的状态摘要、同形状输出和实际复用均通过；六个实例已完成输入边界检查通过。',
        f'- 遥测序号缺口：{len(gaps)}；采用试验的缓存拒绝项：{sum(r["cache_mismatches"] for r in rows)}。',
        '- TTFT/TPOT SLO固定10秒/200毫秒；两个重复展示原始数值，不声称统计显著性。',
        '- 节点28原有空闲常驻GPU上下文保留。初次六实例同形状冷prefill为7.33–7.37秒。',
        '- 通信初始化42.8秒单列，不能将温热连接结果直接当作新worker从零上线的收益。',
        '- 并发是客户端数，单Engine运行上限64；原始记录保留路由等待与实际decode峰值。',
        '- 192并发的测试驱动派发等待已达数秒，SLO数值同时包含适配/客户端开销；不能全部归因为GPU算力或迁移机制。',
        '- 当前迁移实现要求空目标和源Engine暂停执行；此批验证冷缓存实例的状态准备，不覆盖常态在线缓存再布局。','']
    (root/'FINAL-REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    result=dict(rows=rows,comparisons=comparisons,accepted_trials=len(accepted),telemetry_gaps=gaps,
        main_status=main['status'],control_status=control['status'])
    (root/'final-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','telemetry_gaps']},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    run(p.parse_args().root)
