"""Report every completed cell, preparation consumption, and feedback integrity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def report(root):
    lines=['# 状态准备实验实时报告','',
        '原版 routing 算法经统一 SGLang 后端适配；不是原生全栈系统比较。',
        '表格包含已完成 cell。未完成批次不作最终结论，三重复也不代表总体显著性。','']
    all_reports=[]
    for path in sorted(root.glob('*/summary.json')):
        d=json.loads(path.read_text())
        trials=d.get('trials',[])
        lines += [f'## {path.parent.name}：{d.get("status","unknown")}，{len(trials)} trials','']
        if d.get('failure'):
            lines += [f'失败原因：`{d["failure"]}`','']
        by=defaultdict(list)
        requests=defaultdict(list)
        reqpath=path.parent/'requests.jsonl'
        if reqpath.exists():
            for line in reqpath.open():
                try:r=json.loads(line)
                except ValueError:continue
                requests[r['case'],r['policy'],r['prep'],r['repetition']].append(r)
        for r in trials:
            by[r['case'],r['policy'],r['prep']].append(r)
        lines+=['| Case | Router | 准备 | 重复 | Wall 中位数 s | TTFT p90 中位数 s | TPOT p50 中位数 ms | 错误 | 目标请求/全部 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
        cells=[]
        for (case,policy,prep),rs in sorted(by.items()):
            metric=lambda key:statistics.median(r['summary'][key] for r in rs)
            rows=[r for t in rs for r in requests[case,policy,prep,t['repetition']]]
            dest=sum(r['target']==1 for r in rows)
            errors=sum(r['summary']['errors'] for r in rs)
            cell=dict(case=case,policy=policy,prep=prep,repetitions=len(rs),wall=metric('wall'),
                ttft_p90=metric('ttft_p90'),tpot_p50=metric('tpot_p50'),errors=errors,
                destination_requests=dest,requests=len(rows),
                preparations=[dict(seconds=t['preparation'].get('seconds'),
                    bytes=t['preparation'].get('bytes'),
                    prepared_tokens=t['preparation'].get('prepared_tokens'),
                    lateness=t['exposed_preparation']) for t in rs])
            cells.append(cell)
            lines.append(f'| {case} | {policy} | {prep} | {len(rs)} | {cell["wall"]:.3f} | '
                f'{cell["ttft_p90"]:.3f} | {1000*cell["tpot_p50"]:.2f} | {errors} | {dest}/{len(rows)} |')
        lines+=['','目标请求数为 0 时，不能把 wall 波动解释为状态副本被消费产生的收益。',
            '准备完成晚于首次到达，不等于消费者被阻塞：异步重算期间源端可以继续执行。','']
        paired=[]
        for (case,policy,prep),rs in sorted(by.items()):
            if prep=='none':continue
            baseline={r['repetition']:r for r in by.get((case,policy,'none'),[])}
            pairs=[(baseline[r['repetition']],r) for r in rs if r['repetition'] in baseline]
            if pairs:
                deltas=[a['summary']['wall']-b['summary']['wall'] for a,b in pairs]
                fractions=[100*(1-b['summary']['wall']/a['summary']['wall']) for a,b in pairs]
                paired.append(dict(case=case,policy=policy,prep=prep,pairs=len(pairs),
                    wall_saving_s=deltas,wall_reduction_pct=fractions))
                lines.append(f'- {case}/{policy}/{prep} 对 none：{len(pairs)} 配对，'
                    f'wall 降幅 {", ".join(f"{v:.1f}%" for v in fractions)}。')
        gaps=[]
        for telem in path.parent.glob('telemetry-*.jsonl'):
            previous=None
            for raw in telem.open():
                try:ev=json.loads(raw)['event']; cur=ev['counter_id']
                except (ValueError,KeyError):continue
                if previous is not None and cur!=previous+1:
                    gaps.append(dict(file=telem.name,previous=previous,current=cur))
                previous=cur
        lines+=['',f'进度事件序号缺口：{len(gaps)}。','']
        all_reports.append(dict(batch=path.parent.name,status=d.get('status'),cells=cells,
            paired=paired,feedback_gaps=gaps))
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    (root/'report.json').write_text(json.dumps(all_reports,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    report(p.parse_args().root)
