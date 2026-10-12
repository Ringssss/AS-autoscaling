"""Finish the concurrency audit from completed, fresh confirmation pairs."""
import argparse
import json
from pathlib import Path
import statistics
import time

from experiments.fluidkv.p09_report import report
from experiments.fluidkv.p09_plot import plot


def finalize(root):
    report(root);plot(root)
    status=json.loads((root/'campaign-status.json').read_text())
    data=json.loads((root/'report.json').read_text())
    lines=['# 高并发扫描结论','',f'运行状态：{status["status"]}。','',
        '本轮为固定长度、两轮闭环突发的控制实验。单前缀 64K，四前缀各 32K；前缀长度和共享度跨两组同时变化，不能把组间差异单独归因于共享度。',
        '预建连接、1s 准备窗口、源端缓存聚集且目标初始冷；不能据此声称持续稳态或原生 AgentX 总体收益。','']
    totals=dict(trials=0,requests=0,errors=0)
    for p in root.glob('*/summary.json'):
        d=json.loads(p.read_text())
        totals['trials']+=len(d['trials'])
        totals['requests']+=sum(t['summary']['requests'] for t in d['trials'])
        totals['errors']+=sum(t['summary']['errors'] for t in d['trials'])
    lines += [f'完成 {totals["trials"]} 个 trial、{totals["requests"]} 个请求，错误 {totals["errors"]}。','']
    if status['status']!='complete':
        lines += ['批次未完整完成，不能据此形成最终收益区间结论。',f'原因：{status.get("failure","见状态记录")}。','']
    else:
        for kind in ('shared','mixed'):
            lines += [f'## {kind} 确认结果','']
            for policy in ('blitz','dualmap'):
                rows=[r for r in data['findings'] if r['batch']==kind+'-confirm' and r['policy']==policy and r['prep']=='copy']
                good=sorted(r['clients'] for r in rows if r['passed'])
                lines.append(f'- {policy}＋复制：达到预先定义门槛的客户端并发为 {good}。')
                for r in rows:
                    lines.append(f'  - {r["clients"]} 客户端：两次相对强基线的 wall 降幅 {[round(x,2) for x in r["reductions"]]}%。')
            d=json.loads((root/(kind+'-confirm')/'summary.json').read_text())
            arms={(t['case'],t['repetition'],t['policy'],t['prep']):t for t in d['trials']}
            by={}
            for (case,rep,policy,prep),t in arms.items():
                if prep!='copy':continue
                rec=arms.get((case,rep,policy,'recompute'))
                if rec:
                    by.setdefault((case,policy),[]).append(100*(1-t['summary']['wall']/rec['summary']['wall']))
            lines += ['', '复制相对同策略提前重算的 wall 降幅（正值表示复制更快）：','']
            for (case,policy),gains in sorted(by.items()):
                lines.append(f'- {case}/{policy}：{[round(x,2) for x in gains]}%。')
            lines += ['']
        lines += ['## 解释边界','',
            '达到门槛仅表示在该受控负载、硬件和 SLO 配置下值得继续。两次重复给出描述性稳定性，不等于总体统计显著性。',
            '若 128 档仍通过，扫描没有确定收益回落的上边界。若复制和重算都改善，应区分提前准备的贡献与传输机制的额外贡献。','']
    (root/'DECISION.zh.md').write_text('\n'.join(lines)+'\n')
    (root/'final-audit.json').write_text(json.dumps(dict(status=status['status'],totals=totals,findings=data['findings']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    if a.watch:
        while json.loads((a.root/'campaign-status.json').read_text())['status'] not in ('complete','failed','stopped_by_user'):
            time.sleep(30)
    finalize(a.root)
