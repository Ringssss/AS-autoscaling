"""Generate conservative profiler findings from immutable trace analyses."""
import collections
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).parent
OUT=ROOT/'results'
analyses=[json.loads(p.read_text()) for p in sorted((OUT/'analysis').glob('*.analysis.json'))]
records=json.loads((OUT/'records.json').read_text())
config=json.loads((OUT/'text-config.json').read_text())
lines=['# AgentX 瓶颈 profiling 与优化证据','',
       '执行环境：海光 BW，SGLang 0.5.12，PyTorch 2.10.0 / HIP 6.3.26113，Qwen3.6-35B-A3B BF16，TP2。',
       '三个 BBuf skills 已安装；实际采集使用已部署 vendor 的 HTTP profiler API。所有 TP rank 原始 trace 保留。',
       'graph-off、Triton attention/MoE、无 speculative、原始输入 hash 映射和记录输出长度。', '',
       '模型：40 层，30 层 linear attention / 10 层 full attention；hidden=2048，16 Q heads / 2 KV heads，head_dim=256，256 experts、top-k=8。',
       '当前脚本对 hybrid 模型的通用分类仅作导航；实际层型来自 text_config.layer_types。', '',
       '## 采样之外的真实耗时','',
       '| 场景 | 次序 | 输入/输出 tokens | TTFT s | 平均 TPOT ms | 实际缓存 tokens |','|---|---|---|---:|---:|---:|']
for r in records:
    if r['mode']=='profiled':continue
    x=r['result'];m=x['meta']
    lines.append(f"| {r['case']} | {r['mode']} | {m['prompt_tokens']}/{m['completion_tokens']} | {x['ttft']:.3f} | {x['mean_tpot']*1000:.2f} | {m.get('cached_tokens',0)} |")
lines+=['', '冷请求 TTFT 约 16.3s，warm 请求约 0.67s，但两者后续生成都约 132–135ms/token。',
        '这说明可复用前缀能显著减少 prefill，而命中之后的长生成仍形成独立瓶颈。', '',
        '## 各 rank：kernel 总时长、忙碌并集与跨度','',
        '| 文件 | kernels | kernel sum ms | GPU busy union ms | GPU span ms | 可核对层区间数 |','|---|---:|---:|---:|---:|---:|']
for a in analyses:
    lines.append(f"| {Path(a['file']).name} | {a['kernel_count']} | {a['kernel_sum_ms']:.2f} | {a['gpu_busy_union_ms']:.2f} | {a['gpu_span_ms']:.2f} | {a['verified_layer_intervals']} |")
lines+=['', 'sum 可以包含并行重叠，union 才是该 rank 已观测 kernel 的忙碌并集；span 不是未采样服务时间。',
        '两 rank 的 collective 时间不对称且在轻量采集中方向变化，不能解释为固定某张网卡慢。',
        'ROCtracer 报告 duplicate flow start，原始日志保留。结合 profile 对延迟的显著影响，',
        '主机发射/同步开销是下一步候选，但不能用 span−union 直接承诺可消除的耗时。', '',
        '## Kernel/source attribution','',
        '| 观察窗口 | 主要 kernel | 证据 | 已核对源码路径 |','|---|---|---|---|',
        '| cold prefill 前五个 4K chunks | fused_moe_kernel | rank0 约 727ms / 46% kernel 总时长 | layers/moe/moe_runner/triton_utils/fused_moe.py:328 |',
        '| 同上 | _fwd_kernel | rank0 约 258ms / 16% | layers/attention/triton_ops/extend_attention.py:559 |',
        '| 116K 长上下文 decode 的五步 | _fwd_grouped_kernel_stage1 | rank0 约 44ms / 37%；10 full-attention 层/步 | layers/attention/triton_ops/decode_attention.py:430 |',
        '| 同上 | fused_moe_kernel | rank0 约 15ms / 13% | layers/moe/moe_runner/triton_utils/fused_moe.py:328 |',
        '| 各 rank decode | ncclDevKernel_AllReduce | rank0/rank1 等待量差别很大，需结合主机时间线 | distributed/parallel_state.py:818 |', '',
        '以上占比是各自窗口的 kernel duration share，不是端到端份额或可直接叠加的加速比例。',
        '完整三表由 skill 生成，见 `results/cold-prefill-skill-report.md`、`results/cold-decode-skill-report.md`。', '',
        '## Overlap opportunities','',
        '| 候选 | 当前证据 | 下一步验证 / 限制 |','|---|---|---|',
        '| 减少 eager 发射与 CPU 同步 | decode kernel 启动多，busy 与 span 差距大 | 需要无 stack 或设备图 A/B；本轮不将采样空隙当作确定收益 |',
        '| collective 与计算重叠 | rank 等待不对称 | 不可把 NCCL 时长直接当网络传输成本；必须定位 producer/consumer |',
        '| 工具间隙状态准备 | 上轮 0.2s 级复制已隐藏 | 继续压缩复制时间收益有限；应扩大有意义的状态放置覆盖面 |', '',
        '## Fusion candidates','',
        '| 候选 | 证据等级 | 是否本轮实现 |','|---|---|---|',
        '| attention 的 RoPE/reshape/cache write | skill 的模式候选；需验证 vendor dispatch、dtype/layout | 否 |',
        '| MoE activation/reduction fusion | 高 prefill MoE 占比不等于 epilogue 都可消除 | 否 |',
        '| shared-expert 与 routed-expert 路径 | skill 跨版本目录为线索，不能证明本 vendor 支持 | 否 |', '',
        '通用 kernel 优化应同时应用于 DualMap/CacheRoute，不能只给我们的方法使用后宣称算法胜出。', '',
        '## 层级分析','']
for a in analyses:
    if 'cold-critical' not in a['file'] or 'TP-0' not in a['file']:continue
    rows=a['layers'];sums=collections.defaultdict(list)
    for r in rows:
        if r['step']>0 and r['layer']>0:sums[r['type']].append(r['wall_us'])
    lines.append(f"- {Path(a['file']).name}：已核对 {a['anchor_count']} 个 `{a.get('anchor_kernel')}` 锚点。")
    for kind,values in sums.items():lines.append(f"  {kind} 的边界区间平均 {statistics.mean(values)/1000:.3f}ms（排除第一步和 layer0 的跨 forward 头部）。")
lines+=['', 'MoE-end 是导航边界，区间仍包含边界 collective/residual，不能等同于单个 PyTorch module 的严格 self time。',
        'mixed trace 同时有 prefill 和 decode，自动计数不满足 40 个锚点/纯 forward 的区间保持未分类，不硬凑层数。', '',
        '## 已实现的系统改进与门槛','',
        '非空目标导入为 opt-in；保留 scheduler idle、完整快照、同模型/TP、临时容量等约束。',
        '导入调用 vendor radix insert 去重 KV，并释放重复 recurrent slot，不清空其他缓存。',
        '真实跨节点四组检查全部通过：独立前缀、共享扩展、重复导入、失败回滚；状态摘要、slot 数量、旧缓存和输出均核对。',
        '在此基础上加入每轮 3GiB/1s 的多状态准备预算，旧 +move 作为消融。',
        '下一份报告比较未见过的原始长度轨迹；不在本报告提前声称加速。', '',
        '[复测结果](results/compare/REPORT.zh.md) · [运行/复现说明](README.md)']
(ROOT/'PROFILE.zh.md').write_text('\n'.join(lines)+'\n')
print(ROOT/'PROFILE.zh.md')
