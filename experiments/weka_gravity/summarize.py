"""Generate the measured-results note from complete raw campaigns."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).parent
OUT=ROOT/'results'
links=['node27-node28','node27-node29']
text=['# 本轮三节点实测', '',
      '2026-09-28；node27 卡 6,7，node28 卡 4,5，node29 卡 0,1；每节点一个 TP2 引擎，共六卡。',
      'Qwen3.6-35B-A3B BF16，128K 上下文，262,144 token slots；镜像、参数见 `results/launch.json`。',
      '节点存在其他任务，CPU/网络并非整机独占。node30 显存被占用，未纳入本轮。', '',
      '## 迁移组件成本', '',
      '每格三次配对重复，中位数，单位秒。时间从续写可派发时开始，迁移列包含复制/RPC/所有权提交。',
      '模型、执行形状和通信组事先预热；初始化另列。每次清理的是本次专用引擎缓存。',
      '公共文本 token 重复补足至 96K，输出固定 16 tokens；这是组件校准，不是 Claude 原始内容或全 trace 重放。', '',
      '| 链路 | 前缀 | 温热源 | 冷目标 | 温热目标 | 返回时迁移 | 冷目标/迁移 |',
      '|---|---:|---:|---:|---:|---:|---:|']
allrows=[]
initialization=[]
for link in links:
    summary=json.loads((OUT/link/'summary.json').read_text())
    rows=[json.loads(s) for s in (OUT/link/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==36
    assert len({(r['repeat'],r['prefix_tokens'],r['mode']) for r in rows})==36
    assert all(all(r['checks'].values()) for r in rows)
    allrows+=rows
    for length in [1024,32768,98304]:
        bymode={r['mode']:r['ttft_median_seconds'] for r in summary['summaries'] if r['prefix_tokens']==length}
        modes=['sticky-warm-source','reroute-cold-destination','reroute-warm-destination','migrate-on-return']
        values=[bymode[m] for m in modes]
        text.append(f'| {link} | {length:,} | '+' | '.join(f'{v:.4f}' for v in values)+f' | {values[1]/values[3]:.2f}× |')
    initialization.append(f"{link} 通信组初始化 {summary['group_initialization_seconds']:.3f}s，首次传输预热 {summary['group_first_transfer_warmup_seconds']:.3f}s。")
text+=['']+initialization+['', '72 次正式续写全部通过逐 token 输出、缓存命中、输入/输出长度和所有权提交检查。',
       '1K 是不值得搬运的反例；无负载温热源和已有温热目标也优于额外复制。冷目标比值不是整体系统加速。', '',
       '## 三节点同时返回：三份互不共享的历史', '',
       '三个 32K 历史首 token 不同，源端先分别执行并缓存；三个续写同时返回，各生成 32 tokens。',
       'sticky 全留在源；其余策略每节点一个续写，负载放置完全一致。copy 按源端串行复制到两个目标；',
       'precompute 可在两个目标并行重算。两者均获得同一个受控 1 秒准备窗口，超出部分计入返回延迟。',
       '这个窗口不是从完整工具执行恢复的；实验检验一次回流机制，不是持续集群吞吐或生产流量重放。', '',
       '**此组未通过严格输出一致性门槛。下列时间仅保留为诊断观测，不作为有效加速结论。**', '',
       '| 策略 | 返回后 TTFT 中位数 s | 三请求全部完成 s | 单请求完成中位数 s |',
       '|---|---:|---:|---:|']
burst=json.loads((OUT/'three-node-burst/summary.json').read_text())
raw=[json.loads(s) for s in (OUT/'three-node-burst/requests.jsonl').read_text().splitlines()]
assert len(raw)==12 and len({(r['repetition'],r['mode']) for r in raw})==12
failed_checks=[dict(repetition=r['repetition'],mode=r['mode'],agent=i,checks=c)
               for r in raw for i,c in enumerate(r['checks']) if not all(c.values())]
(OUT/'burst-failed-checks.json').write_text(json.dumps(failed_checks,indent=2)+'\n')
for mode,metrics in burst['modes'].items():
    text.append(f"| {mode} | {metrics['median_ttft_s']:.4f} | {metrics['median_makespan_s']:.4f} | {metrics['median_agent_e2e_s']:.4f} |")
text+=['', f"每策略三次重复，共 36 次正式续写；严格检查失败记录 {len(failed_checks)} 条。",
       '全部请求完成、输入/输出长度和缓存命中符合预期；同一历史在 sticky 三请求 batch 与分布式执行之间输出不同。',
       'copy、cold reroute、precompute 在相同分布式放置下输出一致。没有更换输入或截短输出以绕过失败。',
       f"两对通信组创建共 {burst['group_initialization_s']:.3f}s，另做首次复制预热。",
       '全部完成时间的统计单位是每轮三个请求的 makespan；不能把其倒数称为长期集群吞吐。',
       '没有并发背景流量、持续到达或压力下驱逐；准备时源空闲、目标为空，源历史仍可保留。', '',
       '### 同节点无迁移诊断', '']
diagnosis=json.loads((OUT/'batch-diagnosis/summary.json').read_text())
for mode in ['warm-single','warm-batch3','cold-single']:
    cases=[r for r in diagnosis['cases'] if r['mode']==mode]
    text.append(f"- node27 {mode}：{sum(r['matches_distributed_reference'] for r in cases)}/{len(cases)} 次与分布式参考输出一致。")
text+=['', '诊断未执行迁移，也未改变设备。若串行与 batch 在同节点已产生不同输出，就不能把原差异直接归因于 KV 传坏。',
       '这仍不等同于排除全部数值/质量问题；具体算子和质量容差尚未定位或验证，原始失败门槛保持不变。', '',
       '## 当前实现限制的运行验证', '']
probe=json.loads((OUT/'constraints/results.json').read_text())
copy=probe['successful_copy']
src=copy['source_after']['result'];dst=copy['destination_after']['result']
assert src['full_prefix_available'] and dst['full_prefix_available']
assert src['state_sha256']==dst['state_sha256']==copy['source_before']['result']['state_sha256']
assert probe['nonempty_destination']['rejected']
assert 'destination cache must be empty' in probe['nonempty_destination']['error']
assert probe['busy_source_digest_attempt']['http_status']==409
assert 'idle scheduler' in probe['busy_source_digest_attempt']['result']['message']
text += [f"- 实际复制 {copy['bytes']:,} bytes 后，目标 owner 提交成功，两端状态摘要一致；源前缀仍然完整存在。",
         '- 源端生成进行中，要求 idle 的状态摘要接口返回 HTTP 409；迁移使用同一 idle 检查。',
         '- 目标已有缓存时，真实迁移请求被拒绝；owner 保持在源端。',
         '- 这些检查支持实现边界的判断，不能把一次成功复制等同于源 HBM 已回收。', '',
         '## 如何解释', '',
         'trace 支持长历史、高复用和潜在准备窗口；本轮硬件试验证明部分条件下跨节点状态准备有效。',
         '是否能在真实持续热点上净受益，仍取决于繁忙源端迁出、非空目标导入、源回收、在线预测和带宽预算。',
         '详细判断与下一步验证标准见 [ASSESSMENT.zh.md](ASSESSMENT.zh.md)。']
(ROOT/'MEASUREMENTS.zh.md').write_text('\n'.join(text)+'\n')

# Conservative use of observed component cost: interpolate only in the
# measured domain, require the full previous input boundary and local context.
with (OUT/'requests.csv').open() as f: trace_rows=list(csv.DictReader(f))
points=[]
for length in [1024,32768,98304]:
    points.append((length,statistics.median(r['migration_and_commit_seconds'] for r in allrows
                                          if r['prefix_tokens']==length and r['mode']=='migrate-on-return')))
eligible=[r for r in trace_rows if r['gap'] and float(r['gap'])>=0
          and int(r['previous_lcp'])==int(r['previous_input']) and 32768<=int(r['previous_lcp'])<=98304
          and int(r['input'])+int(r['output'])<=131072]
def interp(n):
    for (a,x),(b,y) in zip(points,points[1:]):
        if a<=n<=b:return x+(y-x)*(n-a)/(b-a)
    raise ValueError(n)
cal=dict(scope='Offline window opportunity for the measured local model and size domain only; no queue/capacity/online predictability assumed.',
         measured_copy_commit_points=points,eligible_pairs=len(eligible),
         fits=sum(interp(int(r['previous_lcp']))<=float(r['gap']) for r in eligible))
(OUT/'calibrated-window-opportunity.json').write_text(json.dumps(cal,indent=2)+'\n')
print(json.dumps({'component_requests':72,'component_checks_passed':True,'burst_requests':36,
                  'burst_checks_passed':burst['all_checks_passed'],'burst_failed_records':len(failed_checks),
                  'calibrated_opportunity':cal},indent=2))
