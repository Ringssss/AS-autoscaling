"""Generate the final report only from the corrected complete campaigns."""
import json
from pathlib import Path

from experiments.multinode.summarize_highbs import summarize


def main():
    base=Path(__file__).resolve().parent
    root=base/'results/highbs'
    final=root/'checkpoint-aware-final'
    runs=json.loads((final/'summary.json').read_text())['runs']
    if len(runs)!=96:raise ValueError('corrected matrix incomplete')
    report=summarize(final)
    (final/'paired_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[]
    for row in report['summaries']:
        if row['regime']!='single-hot':continue
        lines.append(f"| {row['prefix_tokens']//1024}K | {row['offered_per_engine']} | {row['base']} | "
                     f"{row['baseline_ttft_p50_median']:.3f} | {row['augmented_ttft_p50_median']:.3f} | "
                     f"{row['ttft_p50_ratio_median']:.2f}× | {row['throughput_ratio_median']:.2f}× |")
    null=[]
    for row in report['summaries']:
        if row['regime']=='all-hot':
            controls=[p for p in report['pairs'] if p['prefix_tokens']==row['prefix_tokens']
                      and p['offered_per_engine']==row['offered_per_engine'] and p['base']==row['base'] and p['regime']=='all-hot']
            ratios=[p['ttft_p50_ratio'] for p in controls if p['ttft_p50_ratio'] is not None]
            null.append(f"| {row['prefix_tokens']//1024}K | {row['offered_per_engine']} | {row['base']} | "
                        f"{row['ttft_p50_ratio_median']:.2f}× | {min(ratios):.2f}–{max(ratios):.2f}× | {row['throughput_ratio_median']:.2f}× |")
    cold=[];cold_count=0;cold_errors=0
    for length in [32768,65536]:
        directory=root/f'three-node-cold-{length}'
        data=json.loads((directory/'summary.json').read_text())['runs']
        if len(data)!=8 or any(r['total_transfer_bytes'] for r in data):raise ValueError('invalid cold control')
        cold_count+=sum(r['completed'] for r in data);cold_errors+=sum(r['errors'] for r in data)
        summary=summarize(directory)
        (directory/'paired_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        for row in summary['summaries']:
            cold.append(f"| {length//1024}K | {row['offered_per_engine']} | {row['base']} | "
                        f"{row['ttft_p50_ratio_median']:.2f}× | {row['throughput_ratio_median']:.2f}× |")
    integrity=[]
    for node in [29,28]:
        data=json.loads((root/f'state-byte-integrity-node{node}/result.json').read_text())['rows']
        if len(data)!=2:raise ValueError('integrity checks incomplete')
        for row in data:
            integrity.append({'destination':node,'prefix_tokens':row['prefix_tokens'],'checks':row['checks']})
    complete=sum(r['completed'] for r in runs)
    errors=sum(r['errors'] for r in runs)
    obs_errors=sum(sum('error' in o for o in r['checkpoint_observations']) for r in runs)
    matches=sum(p['exact_output_matches'] for p in report['pairs'])
    compared=sum(p['output_compared'] for p in report['pairs'])
    byte_ok=all(all(row['checks'].values()) for row in integrity)
    cr32=next(r for r in report['summaries'] if r['prefix_tokens']==32768 and r['offered_per_engine']==128 and r['base']=='cacheroute' and r['regime']=='single-hot')
    cr64=next(r for r in report['summaries'] if r['prefix_tokens']==65536 and r['offered_per_engine']==128 and r['base']=='cacheroute' and r['regime']=='single-hot')
    slowest=max(runs,key=lambda r:r['duration_seconds'])
    peak_running={str(i):max(r['observed_peak_running_by_engine'].get(str(i),0) for r in runs) for i in range(3)}
    (root/'final-evidence.json').write_text(json.dumps({'primary_requests':complete,'primary_errors':errors,
        'observation_errors':obs_errors,'cold_requests':cold_count,'cold_errors':cold_errors,
        'exact_output_matches':matches,'output_compared':compared,'integrity':integrity,
        'peak_running_by_engine':peak_running},indent=2)+'\n')
    text=f'''# 长请求、大 BS：实际路由与状态准备对比

本轮最终比较使用 **Qwen3.6-35B-A3B BF16、三节点、每节点一个 TP2 引擎（共六卡）**。
Qwen3.8-Flash-Next 已完整下载并通过真实权重参考推理，但当前海光 SGLang
没有其完整模型/状态实现，**未完成 Qwen3.8 的同后端高 BS 路由/迁移对比**。
不能把不同后端的数字混用。新模型详情见 [QWEN38_STATUS.md](QWEN38_STATUS.md)。

最终信息对等矩阵为 32K/64K × 名义每引擎并发 16/128 × 两种缓存状态 ×
四种策略 × 三次重复，共 **{complete:,}** 次请求；请求错误 **{errors}**，
checkpoint 观测错误 **{obs_errors}**。另有 **{cold_count}** 次全冷控制请求，
错误 **{cold_errors}**。这里的完成率不是输出数值等价率。

最直接的结论是：**没有测得对所有 routing 场景都成立的优势**。
在“仅源端热”的最高档，CacheRoute 的32K首token中位数为
{cr32['baseline_ttft_p50_median']:.3f}→{cr32['augmented_ttft_p50_median']:.3f} s，64K为
{cr64['baseline_ttft_p50_median']:.3f}→{cr64['augmented_ttft_p50_median']:.3f} s，
但吞吐比约为{cr32['throughput_ratio_median']:.3f}×与{cr64['throughput_ratio_median']:.3f}×，基本持平。
DualMap 有较大的观测改进，也有极端重平衡波动；对应全热无操作控制也不稳定，
所以不把最大的观测倍率写成可稳定复现的算法优势。
最慢轮次耗时约{slowest['duration_seconds']:.0f} s、发生{slowest['upstream_rebalances']}次队列重平衡，已完整保留。

## 比较方式

- 真正执行固定上游版本 DualMap，以及 CacheRoute 论文 Algorithm 1 的实现。
  各自比较仅路由与同一个 router 加 FluidKV 共享前缀准备，未用“强制冷路由”
  冒充整个算法。DualMap 上游源码 commit 不变。
- 在独立 8K 校准负载上选择 pending-prefill 门槛 32，再固定用于两边；
  候选 1/8/32 的中位批次完成时间为 30.86/12.08/9.77 s。
- 基线与增强版共用真实、只读、可在服务期间查询的 checkpoint 观测。
  目标被普通请求自然预热后，基线也会更新其可执行状态信息。
  不从输入长度猜测 recurrent checkpoint，不通过查询触碰 LRU 或拆树。
- 模型已驻留；“冷”指缺少前缀状态，不是模型/节点刚启动。共享前缀来自
  LongBench 公开报告，后缀是固定长度的确定性合成 token。
  Kimi 的脱敏文件没有原生前缀身份与会话键，故这不是其原生 production replay。
- 状态准备在受控 500 ms 窗口中进行，不是测得的生产工具时长。包含复制
  尾部、路由/队列、网络和推理；初始化、内核预热另行记录。
  这是共享前缀准备消融，不是完整在线多 agent 迁移、fencing 或物理扩缩容。

## 仅源端热、目标初始冷

表中 TTFT 为每次突发内的 p50，再取三次重复的中位数，单位秒。
倍率列是**成对比值的中位数**：TTFT 为基线/增强，吞吐为增强/基线。
倍率大于1有利于增强版。成对中位比值不必等于两个中位延迟的比值；
原始每轮数值和范围均保留，波动较大的点不应据单一倍率作因果结论。

| 前缀 | 名义每引擎并发 | 实际 router | 基线 TTFT s | 加准备 TTFT s | 成对 TTFT 比 | 成对吞吐比 |
|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+f'''

这里的并发是集群提交量除以三个副本：16/128 对应 48/384 个请求。
实际分配不强制均分，路由目标计数和运行/等待队列由引擎实测。
源端亲和造成的排队、目标冷重算及自然预热都在结果内。

## 全热无操作控制

全热时不执行任何复制；两种标签在状态准备上是无操作。这一表反映
时间敏感调度、批次形成、运行噪声和控制条件的变动，**不当作迁移收益**。
大 BS 附近如果无操作也出现较大差异，相应点不能给出可靠的通用加速结论。

| 前缀 | 名义每引擎并发 | router | 成对 TTFT 比中位数 | 三次比值范围 | 成对吞吐比 |
|---|---:|---|---:|---:|---:|
'''+ '\n'.join(null)+f'''

## 完全不共享、无源状态的冷控制

这些请求的首 token 不同，前缀无法互相命中，也没有可搬的源状态。
所有增强版的实际传输量均为0。此类冷负载不能靠状态迁移免除重算。
两次重复仅作为控制，不作精确显著性宣称。它们使用相同引擎预算；
新观察逻辑在 unique-cold 模式明确禁用，因为没有被查询的公共可复用前缀。

| 前缀 | 名义每引擎并发 | router | 成对 TTFT 比 | 成对吞吐比 |
|---|---:|---|---:|---:|
'''+ '\n'.join(cold)+f'''

## BS 上限与正确性

相同配置为每引擎 262144 KV slots、512 recurrent-state slots、请求配置上限128。
服务端回报的**有效上限是102**，热前缀提交256并发的复测达到102运行、
154等待，256个请求均完成。32K独立冷前缀只达到约7个运行请求，继续加到
128主要增加队列，吞吐约0.23请求/s。这是此缓存/状态预算下的容量，
不是整节点八卡 TP8 的绝对硬件上限。容量图见
[capacity-32k.png](results/highbs/capacity-32k.png)。
最终三节点矩阵采样到的各引擎运行峰值为 **{list(peak_running.values())}**。

配对输出共比较 **{compared:,}** 个输入，其中 **{matches:,}** 个64-token
序列逐 token 完全一致。不同批次/路径下存在数值输出差异，甚至原基线
重复也有变化；因此不能把请求成功或平均延迟改进写成严格语义等价证明。
逐请求输出和基线重复控制都在 paired_summary.json 中。

独立迁移完整性检查（两条跨节点链路，各32K/64K）结果：
**{'全部通过' if byte_ok else '存在未通过项，必须查看原始检查'}**。
检查每个TP分片的完整KV、卷积与recurrent-state字节摘要、总传输字节数、
源状态未被修改、短温热续写和所有权提交。摘要采集不计入性能时间。
这支持所测状态包的复制完整性，不将高BS数值差异隐藏掉。

## 能得出的 insight

1. 缓存热度与计算负载是两维。状态准备可以在回流前创造新的温热执行位置，
   放松“留在热点”与“去冷目标”的冲突，但不能保证每种router都更优。
2. 大BS改变瓶颈：批次、decode、recurrent槽与排队可能覆盖冷启动收益；
   一个孤立请求的冷重算倍数不能直接外推为大批吞吐倍数。
3. 缓存观测必须公平。自然预热后的真实checkpoint要同时提供给基线；
   否则会把信息差错算成迁移收益。早期缺少该观测的单热源结果已降为诊断。
4. 已经全热或根本没有源状态时，不应期待复制带来收益。实际部署需要
   把观测、剩余工具窗口、准备成本、队列和容量共同用于选择，而不是总搬。

## 复现与范围

完整命令：[HIGHBS_COMMANDS.md](HIGHBS_COMMANDS.md)。最终主目录为
`results/highbs/checkpoint-aware-final/`，每次请求含输出、缓存命中、时序、
目标及遥测；目录保存实际argv、服务器参数、固定校准及原始数据。
全冷控制在 `three-node-cold-32768/`、`three-node-cold-65536/`。
复制完整性在 `state-byte-integrity-node28/`、`state-byte-integrity-node29/`。

原 `three-node-primary/` 及更早轮次包含缓存信息、默认参数或轮询驱动
敏感性问题，不作为最终单热源优势证据；源码快照、失败日志及无操作
独立进程控制仍保留供审计。最终只读观察版源码在 `checkpoint-aware-source/`。
节点并非整机独占，硬件监测已保存。没有把不同版本、不同后端或TP布局
拼成统一性能结论，也没有声称已完成四节点32卡或Qwen3.8优化后端的比较。
'''
    (base/'HIGHBS_RESULTS.md').write_text(text)
    print('Wrote HIGHBS_RESULTS.md')


if __name__=='__main__':main()
