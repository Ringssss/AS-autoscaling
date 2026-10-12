"""Compare each fixed preparation arm against BOTH routing baselines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time


def build(root):
    lines=['# 相对于 Blitz 与 DualMap 的性能门槛','',
        '同一批次、同一 case、同一次重复内，取 Blitz/none 与 DualMap/none 中更快的 wall 作为严格参考。',
        '这只是评价参考，不向在线策略提供事后最优选择。每个准备策略单独报告，不事后挑选最好的准备策略组成方法。',
        '差异为描述性结果；重复不足、目标副本未使用或其他指标恶化时，不宣称方法已胜出。','',
        '| 批次 | Case | 固定准备策略 | 完整配对数 | 相对强基线 wall 降幅中位数 | 目标请求/全部 |',
        '|---|---|---|---:|---:|---:|']
    output=[]
    for p in sorted(root.glob('*/summary.json')):
        data=json.loads(p.read_text())
        by=defaultdict(dict)
        for trial in data.get('trials',[]):
            by[trial['case'],trial['repetition']][trial['policy'],trial['prep']]=trial
        contrasts=defaultdict(list)
        for (case,rep),arms in by.items():
            if not all((policy,'none') in arms for policy in ('blitz','dualmap')):
                continue
            baselines={policy:arms[policy,'none']['summary']['wall'] for policy in ('blitz','dualmap')}
            for (policy,prep),trial in arms.items():
                if prep=='none':continue
                summary=trial['summary']
                contrasts[case,policy,prep].append(dict(repetition=rep,baselines=baselines,
                    wall=summary['wall'],reduction_pct=100*(1-summary['wall']/min(baselines.values())),
                    target_requests=summary['target_counts'].get('1',0),requests=summary['requests'],
                    errors=summary['errors'],ttft_p90=summary['ttft_p90'],tpot_p50=summary['tpot_p50']))
        for (case,policy,prep),pairs in sorted(contrasts.items()):
            delta=statistics.median(r['reduction_pct'] for r in pairs)
            targets=sum(r['target_requests'] for r in pairs)
            requests=sum(r['requests'] for r in pairs)
            lines.append(f'| {p.parent.name} | {case} | {policy}+{prep} | {len(pairs)} | '
                f'{delta:+.2f}% | {targets}/{requests} |')
            output.append(dict(batch=p.parent.name,case=case,policy=policy,prep=prep,
                median_reduction_pct=delta,pairs=pairs))
    lines += ['', '目标请求计数只说明目标是否被使用，不单独证明搬入状态的因果收益。',
        '工程来源：Fluxon 组件作为引用明确的实现基础；新增调度机制及其消融效果才构成 FluidKV 的候选贡献。','']
    for name,content in [('BEST-BASELINES.zh.md','\n'.join(lines)),
                         ('best-baselines.json',json.dumps(output,indent=2)+'\n')]:
        dest=root/name
        tmp=dest.with_suffix(dest.suffix+'.tmp')
        tmp.write_text(content);tmp.replace(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        build(args.root)
        if not args.watch:break
        status=json.loads((args.root/'campaign-status.json').read_text())['status']
        if status in ('complete','failed','stopped_by_user'):break
        time.sleep(30)
