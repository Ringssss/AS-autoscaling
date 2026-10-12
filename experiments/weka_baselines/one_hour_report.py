"""Automatic paired report for the bounded campaign, excluding partial cells."""
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).parent/'results'/'one-hour-20260929'


def readlines(path):
    if not path.exists():return []
    raw=path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
    return [json.loads(s) for s in raw.splitlines() if s.strip()]


def percentile(values,p):
    a=sorted(values);return a[max(0,math.ceil(len(a)*p/100)-1)]


def render():
    status=json.loads((ROOT/'status.json').read_text());cells=[]
    for path in sorted((ROOT/'runs').glob('*/summary.json')):
        s=json.loads(path.read_text());rs=readlines(path.parent/'requests.jsonl');es=readlines(path.parent/'events.jsonl')
        assert len(rs)==48 and len({r['ordinal'] for r in rs})==48
        assert all(r['success'] and len(r['output_ids'])==r['output_tokens'] and r['meta']['prompt_tokens']==r['input_tokens'] for r in rs)
        assert sum(e['kind']=='initial_cache_reset' for e in es)==1
        post=[r for r in rs if r['phase']>0]
        duration=max(r['completion_seconds'] for r in post)-min(r['offered_seconds'] for r in post)
        s.update(directory=str(path.parent),post_ttft_p50=percentile([r['ttft_seconds'] for r in post],50),
            post_ttft_p95=percentile([r['ttft_seconds'] for r in post],95),
            post_e2e_p95=percentile([r['e2e_seconds'] for r in post],95),post_duration=duration,
            post_slo_attainment=sum(r['ttft_seconds']<=10 and r['tpot_seconds']<=.2 for r in post)/len(post),
            post_goodput=sum(r['ttft_seconds']<=10 and r['tpot_seconds']<=.2 for r in post)/duration,
            cached_requests=sum(r['cached_tokens']>0 for r in rs),cache_reset_count=1,
            preparation_skip_reasons=dict(collections.Counter(e.get('reason') for e in es if e['kind']=='skip_preparation')))
        for m in s['migrations']:
            m['used_on_first_return']=any(r['phase']==m['phase'] and r['key']==m['key'] and r['target']==m['target'] and r['cached_tokens']>=min(m['tokens'],r['input_tokens']-1) for r in rs)
        max_running=[0]*3;peak_hbm={};peak_occupied=[0]*3;observation_errors=0
        for row in readlines(path.parent/'telemetry.jsonl'):
            observation_errors+='error' in row
            if 'load' in row:
                i=row['engine'];max_running[i]=max([max_running[i]]+[v['num_reqs']-v['num_waiting_reqs'] for v in row['load']])
                free=row.get('pool',{}).get('available_token_slots')
                if free is not None:peak_occupied[i]=max(peak_occupied[i],1048576-free)
            if 'hbm' in row:
                for c in {'node27':[6,7],'node28':[4,5],'node29':[0,1]}[row['node']]:
                    key=row['node']+':'+str(c);n=int(row['hbm']['card'+str(c)]['vram Total Used Memory (MiB)'])/1024
                    peak_hbm[key]=max(peak_hbm.get(key,0),n)
        s['telemetry']=dict(max_running=max_running,peak_occupied_tokens=peak_occupied,peak_hbm_GiB=peak_hbm,errors=observation_errors)
        cells.append(s)
    lookup={(c['policy'],c['repeat']):c for c in cells};pairs=[]
    for base in ['dualmap','cacheroute']:
        for repeat in [0,1]:
            if (base,repeat) not in lookup or (base+'+move',repeat) not in lookup:continue
            a=lookup[base,repeat];b=lookup[base+'+move',repeat]
            ar={r['ordinal']:r for r in readlines(Path(a['directory'])/'requests.jsonl')}
            br={r['ordinal']:r for r in readlines(Path(b['directory'])/'requests.jsonl')}
            assert set(ar)==set(br)
            for i in ar:
                for field in ['input_tokens','output_tokens','token_sha256','key','turn']:assert ar[i][field]==br[i][field]
            pairs.append(dict(base=base,repeat=repeat,
                post_p95_ttft_ratio=a['post_ttft_p95']/b['post_ttft_p95'],
                post_duration_ratio=a['post_duration']/b['post_duration'],
                full_duration_ratio=a['duration_seconds']/b['duration_seconds'],
                slo_attainment_delta=b['post_slo_attainment']-a['post_slo_attainment'],
                equal_outputs=sum(ar[i]['output_sha256']==br[i]['output_sha256'] for i in ar),requests=len(ar),
                migrations=len(b['migrations']),migration_bytes=b['migration_bytes'],
                migrations_used=sum(m['used_on_first_return'] for m in b['migrations'])))
    data=dict(updated=datetime.datetime.now().astimezone().isoformat(),status=status,cells=cells,pairs=pairs,
        scope='Controlled quiet-boundary 2-to-3 warm-model scaleout; original selected AgentX lengths and native hash identities; only complete paired cells used. Not busy-source migration or complete semantic agent execution.')
    (ROOT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# 一小时 AgentX 聚焦验证','',f"更新时间：{data['updated']}",
        f"状态：{status['state']}；完整 {len(cells)}/8 项，完整配对 {len(pairs)}/4。截止 {status['deadline']}。",'',
        f"当前：{status.get('current_policy','—')}，重复 {status.get('current_repeat','—')}，轮次 {status.get('current_phase','—')}，{status.get('current_requests',0)}/48 请求。",'',
        '三个节点六卡，BS=16。16 条按输入长度分层、三轮原始输出合计≤1536 tokens 的 AgentX 子轨迹；不截断所选输入或输出。',
        '第一轮两节点，回合结束后扩至三节点，第三轮继续保留缓存。每轮边界是共同的闭环屏障；1 秒为受控准备窗口。',
        '仅独立方案开头 reset 一次。迁移只用已完成历史和历史路由锚点选择一个合法状态，不读取下一轮输入或结果来挑候选。',
        '当前后端要求源空闲、目标空缓存，第二轮的新节点提供合法机会；第三轮不能清空目标制造机会。', '',
        '| 方案 | 重复 | 全部完成 s | 扩容后 TTFT p50/p95 s | 扩容后全部完成 s | 扩容后 SLO 达标率 | goodput req/s | 迁移次数/实际使用 | 迁移 GiB |',
        '|---|---:|---:|---|---:|---:|---:|---|---:|']
    for c in cells:
        lines.append(f"| {c['policy']} | {c['repeat']} | {c['duration_seconds']:.2f} | {c['post_ttft_p50']:.3f}/{c['post_ttft_p95']:.3f} | {c['post_duration']:.2f} | {c['post_slo_attainment']:.2%} | {c['post_goodput']:.4f} | {len(c['migrations'])}/{sum(m['used_on_first_return'] for m in c['migrations'])} | {c['migration_bytes']/1024**3:.3f} |")
    lines+=['','主指标的“扩容后”合并第二、三轮全部 32 请求，含未迁移请求。goodput 阈值：TTFT≤10s，平均 TPOT≤0.2s。','',
        '| 配对 | 重复 | 扩容后 p95 TTFT 比值 | 扩容后完成时间比值 | 全部完成时间比值 | SLO 达标率变化 | 输出完全一致 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for p in pairs:
        lines.append(f"| {p['base']} / {p['base']}+move | {p['repeat']} | {p['post_p95_ttft_ratio']:.3f}× | {p['post_duration_ratio']:.3f}× | {p['full_duration_ratio']:.3f}× | {100*p['slo_attainment_delta']:+.2f} pp | {p['equal_outputs']}/{p['requests']} |")
    lines+=['','比值为 baseline / 加迁移，>1 表示加迁移该指标更快；SLO 变化为加迁移减 baseline。',
        '原始轨迹、长度与输入哈希在配对内逐条核验。输出可能随 batch 形状变化，保留完整输出，不将长度有效的性能结果冒充语义一致性认证。', '',
        '## 当前判读','']
    if not pairs:lines.append('尚无完整配对，暂不判断优势。')
    for base in ['dualmap','cacheroute']:
        ps=[p for p in pairs if p['base']==base]
        if not ps:continue
        lines.append(f"- {base}：{len(ps)} 个完整配对；p95 TTFT 比值中位数 {statistics.median(p['post_p95_ttft_ratio'] for p in ps):.3f}×，扩容后完成时间比值中位数 {statistics.median(p['post_duration_ratio'] for p in ps):.3f}×；实际使用迁移 {sum(p['migrations_used'] for p in ps)} 次。")
    lines+=['','## 扩容与后续缓存保留分开看','',
        '| 方案/重复 | 扩容轮 p95 TTFT s | 保留轮 p95 TTFT s | 扩容轮/保留轮输入缓存命中比例 |',
        '|---|---:|---:|---|']
    for c in cells:
        a,b=c['phases']['1'],c['phases']['2']
        lines.append(f"| {c['policy']}/{c['repeat']} | {a['ttft']['95']:.3f} | {b['ttft']['95']:.3f} | {a['cache_fraction']:.2%}/{b['cache_fraction']:.2%} |")
    lines+=['','## 内存与实际并发','', '| 方案/重复 | 峰值 running（各节点） | 峰值 KV slots（各节点） | 单卡峰值 HBM GiB 范围 |','|---|---|---|---|']
    for c in cells:
        t=c['telemetry'];v=list(t['peak_hbm_GiB'].values())
        lines.append(f"| {c['policy']}/{c['repeat']} | {t['max_running']} | {t['peak_occupied_tokens']} | {min(v):.2f}–{max(v):.2f} |" if v else f"| {c['policy']}/{c['repeat']} | {t['max_running']} | {t['peak_occupied_tokens']} | 无采样 |")
    lines+=['','## 边界','',
        '- 这是少量轨迹、最多两次重复的条件性验证，不外推到全量 AgentX、持续繁忙源迁出或所有 BS。',
        '- 输入长度覆盖短/中/长上下文，输出长度筛选用于一小时时间预算；不按任何方案的性能筛选。',
        '- 共同轮次屏障与准备窗口是实验控制，不声称真实生产工具时钟；完整任务时间保留屏障和窗口成本。',
        '- 模型和通信已加载，第三节点为逻辑扩容；不包含物理冷启动。',
        '- 未完成项不计入配对，不根据部分返回请求计算尾延迟。到时自动停止并释放本任务容器。']
    if status.get('error'):lines+=['','运行状态说明：'+status['error']]
    (ROOT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        try:
            s=render()
            if not args.watch or s['state'] in ['complete','failed','budget_expired']:break
        except (FileNotFoundError,json.JSONDecodeError):
            if not args.watch:raise
        time.sleep(5)
