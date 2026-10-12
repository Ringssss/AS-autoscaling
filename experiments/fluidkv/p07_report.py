"""Summarize complete paired cells; never turn an ideal-ready result into a claim."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parents[2]/'results/routing-space-20261010'


def main():
    rows=[]
    reports=[]
    for path in sorted(BASE.glob('*/summary.json')):
        data=json.loads(path.read_text())
        reports.append(dict(run=path.parent.name,trials=len(data['trials']),
                            group_init_seconds=data.get('group_init_seconds')))
        for t in data['trials']:
            rows.append(dict(run=path.parent.name,**t))
    groups=defaultdict(list)
    for row in rows:
        groups[row['run'],row['case'],row['policy'],row['prep']].append(row)
    table=[]
    for key,rs in groups.items():
        med={field:statistics.median(r['summary'][field] for r in rs)
             for field in ('wall','task_p50','task_p90','ttft_p50','ttft_p90',
                           'tpot_p50','goodput','slo_attainment')}
        med.update(run=key[0],case=key[1],policy=key[2],prep=key[3],repeats=len(rs),
                   requests=sum(r['summary']['requests'] for r in rs),
                   errors=sum(r['summary']['errors'] for r in rs),
                   median_prep_seconds=statistics.median(r['preparation_wall'] for r in rs))
        table.append(med)
    pairs=[]
    indexed={(r['run'],r['case'],r['policy'],r['prep'],r['repetition']):r for r in rows}
    for r in rows:
        if r['prep']=='none':
            continue
        baseline=indexed.get((r['run'],r['case'],r['policy'],'none',r['repetition']))
        if baseline:
            pairs.append(dict(run=r['run'],case=r['case'],policy=r['policy'],prep=r['prep'],
                repetition=r['repetition'],
                wall_reduction=1-r['summary']['wall']/baseline['summary']['wall'],
                task_reduction=1-r['summary']['task_p50']/baseline['summary']['task_p50'],
                goodput_increase=r['summary']['goodput']/baseline['summary']['goodput']-1
                    if baseline['summary']['goodput'] else None))
    corrections=[x for r in rows for x in r['corrections']]
    audit=[x for r in rows for x in r['final_cache_audit']]
    result=dict(runs=reports,table=table,paired=pairs,
        total_requests=sum(r['summary']['requests'] for r in rows),
        errors=sum(r['summary']['errors'] for r in rows),
        cache_prediction=dict(count=len(corrections),
            corrections_over_512=sum(abs(r['difference'])>512 for r in corrections),
            max_absolute_difference=max([abs(r['difference']) for r in corrections],default=0),
            final_audit_overprediction=sum(r['predicted']>r['actual_matched_tokens']+512 for r in audit)),
        scope='Same-backend LMetric-score port and profile-based router. Ideal preparation excludes '
              'preparation delay; native fragments use surrogate content and declared output cap. '
              'Three repeats do not establish a population confidence interval.')
    (BASE/'tables.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Routing 与状态准备：实测记录','',
           '本报告只汇总已经完成的 trial；最终去留判断另见 DECISION.zh.md。','',
           '| run | case | router | prep | repeats | wall s | task p50 s | TTFT p90 s | TPOT p50 ms | goodput |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['run']} | {r['case']} | {r['policy']} | {r['prep']} | {r['repeats']} | "
            f"{r['wall']:.3f} | {r['task_p50']:.3f} | {r['ttft_p90']:.3f} | {r['tpot_p50']*1000:.2f} | {r['goodput']:.4f} |")
    lines+=['',f"消费者请求 {result['total_requests']}，错误 {result['errors']}。",'',
            '配对增减、实际 cache 反馈校正与各轮原始数据见 tables.json 及各 run 的 summary.json / requests.jsonl。',
            '',result['scope'],'']
    (BASE/'REPORT.zh.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
