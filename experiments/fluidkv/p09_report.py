"""Concurrency effects, with exploratory and fresh confirmation runs separated."""
import json
from pathlib import Path
from collections import defaultdict
import statistics


def comparisons(path):
    d=json.loads(path.read_text())
    by=defaultdict(dict)
    for t in d.get('trials',[]):
        by[t['case'],t['repetition']][t['policy'],t['prep']]=t
    rows=[]
    for (case,rep),arms in sorted(by.items()):
        if not all((p,'none') in arms for p in ('blitz','dualmap')):continue
        if any(arms[p,'none']['summary']['errors'] for p in ('blitz','dualmap')):continue
        winner=min(('blitz','dualmap'),key=lambda p:arms[p,'none']['summary']['wall'])
        base=arms[winner,'none']['summary']
        for (policy,prep),t in sorted(arms.items()):
            if prep=='none':continue
            q=t['summary']
            if q['errors']:continue
            rows.append(dict(case=case,clients=int(case.rsplit('c',1)[1]),rep=rep,
                policy=policy,prep=prep,baseline=winner,baseline_wall=base['wall'],wall=q['wall'],
                reduction_pct=100*(1-q['wall']/base['wall']),
                slo=q['slo_attainment'],baseline_slo=base['slo_attainment'],
                ttft_p90=q['ttft_p90'],tpot_p90=q['tpot_p90'],
                peak_decode=t['peak_decode'],target_requests=q['target_counts'].get('1',0),
                requests=q['requests'],errors=q['errors'],
                copy_seconds=t['preparation'].get('seconds',0),
                copy_bytes=t['preparation'].get('bytes',0)))
    return d,rows


def report(root):
    lines=['# 并发扫描：寻找状态准备的收益区间','',
        '每个客户端两轮闭环调用，每轮输出固定 128 tokens。客户端数是提供的并发，GPU 实际执行峰值另列；两 worker 的峰值可能发生于不同时刻，不能直接相加。',
        '单前缀为 64K；四前缀为各 32K，复制其中一份，覆盖 25% 客户端。每个扫描内只改变并发。',
        '源端聚集、冷目标、预建连接、1s 准备窗口；固定两台 TP2、每台运行上限 64、262144 token/512 snapshot 容量。',
        '不是原生 AgentX 并发，也不是持续稳态试验。原版策略通过统一 SGLang 后端适配。',
        '路由在冷请求首 token 时获知已验证可读的完整 family prefix，避免等到生成结束才发现自然副本。','']
    result=[]
    for path in sorted(root.glob('*/summary.json')):
        d,rows=comparisons(path)
        lines += [f'## {path.parent.name}：{d.get("status")}，{len(d.get("trials",[]))} trials','']
        if d.get('http_connection_limit',100)==100:
            lines += ['本批客户端连接上限为 100；128-client 档只作受限探索，不能证明 128 个同时在途请求下的效果。','']
        else:
            lines += ['本批已显式解除客户端连接数上限；Engine 运行并发上限仍为每台 64。','']
        if d.get('failure'):lines += [f'失败：`{d["failure"]}`','']
        lines += ['| 客户端 | 策略 | rep | 强基线 wall s | 准备后 wall s | 降幅 | TTFT p90 s | TPOT p90 ms | SLO | 两 worker decode 峰值 |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in rows:
            lines.append(f'| {r["clients"]} | {r["policy"]}+{r["prep"]} | {r["rep"]} | '
                f'{r["baseline_wall"]:.2f} | {r["wall"]:.2f} | {r["reduction_pct"]:+.1f}% | '
                f'{r["ttft_p90"]:.2f} | {1000*r["tpot_p90"]:.1f} | {100*r["slo"]:.1f}% | {r["peak_decode"]} |')
        gaps=[]
        for file in path.parent.glob('telemetry-*.jsonl'):
            prev=None
            for raw in file.open():
                try:cur=json.loads(raw)['event']['counter_id']
                except (ValueError,KeyError):continue
                if prev is not None and cur!=prev+1:gaps.append((file.name,prev,cur))
                prev=cur
        lines += ['',f'进度序号缺口：{len(gaps)}。','']
        result.append(dict(batch=path.parent.name,status=d.get('status'),rows=rows,feedback_gaps=gaps))
    lines += ['## 确认标准','',
        '首轮扫描用于找候选。新随机顺序的两次确认单独报告；同一固定策略两次均改善、降幅中位数至少 5%、联合 SLO 至少 95% 且不低于强基线，才标为值得继续的候选点。',
        '相邻多个候选点才能支持一个区间；若到 128 客户端仍未出现收益回落，不能声称找到峰值。复制与提前重算同时比较，保留所有不利点。','']
    findings=[]
    for rec in result:
        if 'confirm' not in rec['batch']:continue
        by=defaultdict(list)
        for r in rec['rows']:by[r['clients'],r['policy'],r['prep']].append(r)
        for (clients,policy,prep),rs in sorted(by.items()):
            if len(rs)<2:continue
            passed=(min(r['reduction_pct'] for r in rs)>0
                and statistics.median(r['reduction_pct'] for r in rs)>=5
                and min(r['slo'] for r in rs)>=.95
                and all(r['slo']>=r['baseline_slo'] and r['errors']==0 for r in rs))
            findings.append(dict(batch=rec['batch'],clients=clients,policy=policy,prep=prep,
                reductions=[r['reduction_pct'] for r in rs],passed=passed))
            lines.append(f'- {rec["batch"]} / {clients} / {policy}+{prep}：'
                f'{[round(r["reduction_pct"],1) for r in rs]}%，筛选标准 {"通过" if passed else "未通过"}。')
    for name,value in [('REPORT.zh.md','\n'.join(lines)+'\n'),
                       ('report.json',json.dumps(dict(batches=result,findings=findings),indent=2)+'\n')]:
        tmp=(root/name).with_suffix('.tmp')
        tmp.write_text(value);tmp.replace(root/name)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    report(p.parse_args().root)
