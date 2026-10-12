"""Complete-cell, all-request held-out comparison with output checks."""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).parent/'results/compare'


def read(path):
    if not path.exists():return []
    b=path.read_bytes();return [json.loads(s) for s in b[:b.rfind(b'\n')+1].splitlines() if s]


def q(a,p):return sorted(a)[max(0,math.ceil(len(a)*p/100)-1)]


def render():
    status=json.loads((ROOT/'status.json').read_text());cells=[];requests={}
    for p in sorted(ROOT.glob('runs/*/summary.json')):
        s=json.loads(p.read_text());rs=read(p.parent/'requests.jsonl');es=read(p.parent/'events.jsonl')
        assert len(rs)==36 and len({r['ordinal'] for r in rs})==36
        assert sum(e['kind']=='initial_cache_reset' for e in es)==1
        assert all(r['success'] and r['meta']['prompt_tokens']==r['input_tokens'] and len(r['output_ids'])==r['output_tokens'] for r in rs)
        post=[r for r in rs if r['phase']>0];duration=max(r['completion_seconds'] for r in post)-min(r['offered_seconds'] for r in post)
        s.update(post_duration=duration,post_p95_ttft=q([r['ttft_seconds'] for r in post],95),
            post_median_ttft=q([r['ttft_seconds'] for r in post],50),
            post_goodput=sum(r['ttft_seconds']<=10 and r['tpot_seconds']<=.2 for r in post)/duration,
            post_slo_attainment=sum(r['ttft_seconds']<=10 and r['tpot_seconds']<=.2 for r in post)/len(post),
            migration_used=sum(any(r['phase']==m['phase'] and r['key']==m['key'] and r['target']==m['target'] and r['cached_tokens']>=min(m['tokens'],r['input_tokens']-1) for r in rs) for m in s['migrations']),
            prep_skips=dict(collections.Counter(e.get('reason') for e in es if e['kind']=='skip_preparation')),
            request_file_sha256=hashlib.sha256((p.parent/'requests.jsonl').read_bytes()).hexdigest())
        maxrun=[0]*3;peak=[0]*3;hbm={};err=0
        for r in read(p.parent/'telemetry.jsonl'):
            err+='error' in r
            if 'load' in r:
                i=r['engine'];maxrun[i]=max([maxrun[i]]+[l['num_reqs']-l['num_waiting_reqs'] for l in r['load']])
                free=r.get('pool',{}).get('available_token_slots')
                if free is not None:peak[i]=max(peak[i],1048576-free)
            if 'hbm' in r:
                for i in {'node27':[6,7],'node28':[4,5],'node29':[0,1]}[r['node']]:
                    key=r['node']+':'+str(i);hbm[key]=max(hbm.get(key,0),int(r['hbm']['card'+str(i)]['vram Total Used Memory (MiB)'])/1024)
        s['telemetry']=dict(errors=err,max_running=maxrun,peak_token_slots=peak,peak_hbm_GiB=hbm)
        cells.append(s);requests[s['policy']]={r['ordinal']:r for r in rs}
    by={c['policy']:c for c in cells};pairs=[]
    for base in ['dualmap','cacheroute']:
        for suffix in ['+move','+multi']:
            if base not in by or base+suffix not in by:continue
            a,b=by[base],by[base+suffix];ra,rb=requests[base],requests[base+suffix]
            assert set(ra)==set(rb)
            for i in ra:
                for key in ['input_tokens','output_tokens','token_sha256','key','turn']:assert ra[i][key]==rb[i][key]
            pairs.append(dict(base=base,candidate=base+suffix,
                post_duration_ratio=a['post_duration']/b['post_duration'],post_p95_ttft_ratio=a['post_p95_ttft']/b['post_p95_ttft'],
                full_duration_ratio=a['duration_seconds']/b['duration_seconds'],
                slo_delta=b['post_slo_attainment']-a['post_slo_attainment'],
                equal_outputs=sum(ra[i]['output_sha256']==rb[i]['output_sha256'] for i in ra),requests=len(ra)))
    report=dict(updated=datetime.datetime.now().astimezone().isoformat(),status=status,cells=cells,pairs=pairs)
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 非空缓存导入：未见轨迹配对复测','',f"更新时间：{report['updated']}；状态：{status['state']}；完整 {len(cells)}/6 项。",'',
        '同一后端、同一 12 条未见 AgentX 子流、三轮 36 请求，原始长度不截断；原始总输出 6,606 tokens/方案。',
        '初始两节点、随后逻辑扩到三节点，回合之间保留缓存；一次独立运行，不能宣称统计显著。',
        '原版不迁移；+move 为旧空目标/单次准备；+multi 为非空导入及每轮 3GiB/1s 预算下多状态准备。', '',
        '| 方案 | 总完成 s | 扩容后完成 s | 扩容后 p50/p95 TTFT s | 扩容后 SLO 达标率 | goodput req/s | 迁移/首轮使用 | 迁移 GiB |',
        '|---|---:|---:|---|---:|---:|---|---:|']
    for c in cells:lines.append(f"| {c['policy']} | {c['duration_seconds']:.2f} | {c['post_duration']:.2f} | {c['post_median_ttft']:.3f}/{c['post_p95_ttft']:.3f} | {c['post_slo_attainment']:.2%} | {c['post_goodput']:.4f} | {len(c['migrations'])}/{c['migration_used']} | {c['migration_bytes']/1024**3:.3f} |")
    lines+=['','| baseline → candidate | 扩容后完成时间比值 | 扩容后 p95 TTFT 比值 | 全程比值 | SLO变化 | 输出相同 |','|---|---:|---:|---:|---:|---|']
    for p in pairs:lines.append(f"| {p['base']} → {p['candidate']} | {p['post_duration_ratio']:.3f}× | {p['post_p95_ttft_ratio']:.3f}× | {p['full_duration_ratio']:.3f}× | {p['slo_delta']*100:+.2f} pp | {p['equal_outputs']}/{p['requests']} |")
    lines+=['','比值=baseline/candidate，>1 才是该指标更快。主 SLO 为 TTFT≤10s 且平均 TPOT≤0.2s。',
        '输出不同不一定由迁移损坏导致，但仅长度正确不能认证语义等价，所有差异保留。', '',
        '| 方案 | 扩容轮 p95 TTFT s | 后续轮 p95 TTFT s | 两轮输入命中比例 |','|---|---:|---:|---|']
    for c in cells:
        a,b=c['phases']['1'],c['phases']['2'];lines.append(f"| {c['policy']} | {a['ttft']['95']:.3f} | {b['ttft']['95']:.3f} | {a['cache_fraction']:.2%}/{b['cache_fraction']:.2%} |")
    lines+=['','## 占用与实际并发','']
    for c in cells:lines.append(f"- {c['policy']}：{c['telemetry']}；跳过原因：{c['prep_skips']}。")
    lines+=['','## 结论边界','',
        '不读取下一轮实际输入/输出来选迁移。新能力仅支持空闲时导入非空 cache，仍不支持繁忙源后台迁出；源缓存未回收。',
        '短输出分层样本及共同回合屏障限制了外推范围；不要把改善个别 TTFT 或单个 run 的微小差值解释为全局必胜。',
        '所有方案使用同一 opt-in importer 后端；未调用该接口的 baseline 不获得或付出额外复制成本。',
        'profiling 期间没有运行正式配对；原始 profiler trace 不参与吞吐测量。']
    if status.get('error'):lines+=['','运行失败：'+status['error']]
    (ROOT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:
            s=render()
            if not a.watch or s['state'] in ['complete','failed','budget_expired']:break
        except (FileNotFoundError,json.JSONDecodeError):
            if not a.watch:raise
        time.sleep(5)
