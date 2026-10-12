# AgentX 瓶颈 profiling 与优化证据

执行环境：海光 BW，SGLang 0.5.12，PyTorch 2.10.0 / HIP 6.3.26113，Qwen3.6-35B-A3B BF16，TP2。
三个 BBuf skills 已安装；实际采集使用已部署 vendor 的 HTTP profiler API。所有 TP rank 原始 trace 保留。
graph-off、Triton attention/MoE、无 speculative、原始输入 hash 映射和记录输出长度。

模型：40 层，30 层 linear attention / 10 层 full attention；hidden=2048，16 Q heads / 2 KV heads，head_dim=256，256 experts、top-k=8。
当前脚本对 hybrid 模型的通用分类仅作导航；实际层型来自 text_config.layer_types。

## 采样之外的真实耗时

| 场景 | 次序 | 输入/输出 tokens | TTFT s | 平均 TPOT ms | 实际缓存 tokens |
|---|---|---|---:|---:|---:|
| cold-critical | unprofiled-before | 116480/702 | 16.318 | 131.87 | 0 |
| cold-critical | unprofiled-after | 116480/702 | 16.314 | 132.12 | 0 |
| warm-critical | unprofiled-before | 57920/598 | 0.673 | 132.26 | 54848 |
| warm-critical | unprofiled-after | 57920/598 | 0.674 | 135.05 | 54848 |

冷请求 TTFT 约 16.3s，warm 请求约 0.67s，但两者后续生成都约 132–135ms/token。
这说明可复用前缀能显著减少 prefill，而命中之后的长生成仍形成独立瓶颈。

## 各 rank：kernel 总时长、忙碌并集与跨度

| 文件 | kernels | kernel sum ms | GPU busy union ms | GPU span ms | 可核对层区间数 |
|---|---:|---:|---:|---:|---:|
| cold-critical-1790660329.7701972-TP-0-DECODE.trace.json.gz | 5732 | 117.42 | 117.42 | 1417.83 | 200 |
| cold-critical-1790660329.7701972-TP-0-EXTEND.trace.json.gz | 8784 | 1575.02 | 1564.54 | 1677.75 | 200 |
| cold-critical-1790660329.7701972-TP-1-DECODE.trace.json.gz | 5732 | 817.57 | 817.03 | 1417.62 | 200 |
| cold-critical-1790660329.7701972-TP-1-EXTEND.trace.json.gz | 8784 | 1579.56 | 1569.87 | 1676.99 | 200 |
| light-1790661074.0080326-TP-0-DECODE.trace.json.gz | 9121 | 611.91 | 611.87 | 1426.75 | 320 |
| light-1790661074.0080326-TP-0-EXTEND.trace.json.gz | 1421 | 216.72 | 216.69 | 233.39 | 0 |
| light-1790661074.0080326-TP-1-DECODE.trace.json.gz | 9121 | 157.22 | 157.22 | 1426.37 | 320 |
| light-1790661074.0080326-TP-1-EXTEND.trace.json.gz | 1421 | 84.79 | 84.76 | 231.61 | 0 |
| mixed-1790660923.7646763-TP-0-DECODE.trace.json.gz | 8038 | 1070.67 | 1070.09 | 8898.28 | 0 |
| mixed-1790660923.7646763-TP-0-EXTEND.trace.json.gz | 8833 | 1618.21 | 1607.81 | 1713.28 | 200 |
| mixed-1790660923.7646763-TP-1-DECODE.trace.json.gz | 8038 | 222.47 | 222.47 | 8897.95 | 0 |
| mixed-1790660923.7646763-TP-1-EXTEND.trace.json.gz | 8833 | 1537.62 | 1530.45 | 1710.75 | 200 |
| warm-critical-1790660715.0883033-TP-0-DECODE.trace.json.gz | 5732 | 230.57 | 230.55 | 1125.83 | 200 |
| warm-critical-1790660715.0883033-TP-0-EXTEND.trace.json.gz | 1725 | 428.44 | 428.41 | 441.53 | 40 |
| warm-critical-1790660715.0883033-TP-1-DECODE.trace.json.gz | 5732 | 201.60 | 201.55 | 1125.48 | 200 |
| warm-critical-1790660715.0883033-TP-1-EXTEND.trace.json.gz | 1725 | 415.95 | 415.92 | 438.97 | 40 |

sum 可以包含并行重叠，union 才是该 rank 已观测 kernel 的忙碌并集；span 不是未采样服务时间。
两 rank 的 collective 时间不对称且在轻量采集中方向变化，不能解释为固定某张网卡慢。
ROCtracer 报告 duplicate flow start，原始日志保留。结合 profile 对延迟的显著影响，
主机发射/同步开销是下一步候选，但不能用 span−union 直接承诺可消除的耗时。

## Kernel/source attribution

| 观察窗口 | 主要 kernel | 证据 | 已核对源码路径 |
|---|---|---|---|
| cold prefill 前五个 4K chunks | fused_moe_kernel | rank0 约 727ms / 46% kernel 总时长 | layers/moe/moe_runner/triton_utils/fused_moe.py:328 |
| 同上 | _fwd_kernel | rank0 约 258ms / 16% | layers/attention/triton_ops/extend_attention.py:559 |
| 116K 长上下文 decode 的五步 | _fwd_grouped_kernel_stage1 | rank0 约 44ms / 37%；10 full-attention 层/步 | layers/attention/triton_ops/decode_attention.py:430 |
| 同上 | fused_moe_kernel | rank0 约 15ms / 13% | layers/moe/moe_runner/triton_utils/fused_moe.py:328 |
| 各 rank decode | ncclDevKernel_AllReduce | rank0/rank1 等待量差别很大，需结合主机时间线 | distributed/parallel_state.py:818 |

以上占比是各自窗口的 kernel duration share，不是端到端份额或可直接叠加的加速比例。
完整三表由 skill 生成，见 `results/cold-prefill-skill-report.md`、`results/cold-decode-skill-report.md`。

## Overlap opportunities

| 候选 | 当前证据 | 下一步验证 / 限制 |
|---|---|---|
| 减少 eager 发射与 CPU 同步 | decode kernel 启动多，busy 与 span 差距大 | 需要无 stack 或设备图 A/B；本轮不将采样空隙当作确定收益 |
| collective 与计算重叠 | rank 等待不对称 | 不可把 NCCL 时长直接当网络传输成本；必须定位 producer/consumer |
| 工具间隙状态准备 | 上轮 0.2s 级复制已隐藏 | 继续压缩复制时间收益有限；应扩大有意义的状态放置覆盖面 |

## Fusion candidates

| 候选 | 证据等级 | 是否本轮实现 |
|---|---|---|
| attention 的 RoPE/reshape/cache write | skill 的模式候选；需验证 vendor dispatch、dtype/layout | 否 |
| MoE activation/reduction fusion | 高 prefill MoE 占比不等于 epilogue 都可消除 | 否 |
| shared-expert 与 routed-expert 路径 | skill 跨版本目录为线索，不能证明本 vendor 支持 | 否 |

通用 kernel 优化应同时应用于 DualMap/CacheRoute，不能只给我们的方法使用后宣称算法胜出。

## 层级分析

- cold-critical-1790660329.7701972-TP-0-DECODE.trace.json.gz：已核对 200 个 `triton_per_fused_copy__mul_sum_0` 锚点。
  linear_attention 的边界区间平均 5.429ms（排除第一步和 layer0 的跨 forward 头部）。
  full_attention 的边界区间平均 5.126ms（排除第一步和 layer0 的跨 forward 头部）。
- cold-critical-1790660329.7701972-TP-0-EXTEND.trace.json.gz：已核对 200 个 `_moe_sum_reduce_kernel` 锚点。
  linear_attention 的边界区间平均 6.781ms（排除第一步和 layer0 的跨 forward 头部）。
  full_attention 的边界区间平均 11.979ms（排除第一步和 layer0 的跨 forward 头部）。

MoE-end 是导航边界，区间仍包含边界 collective/residual，不能等同于单个 PyTorch module 的严格 self time。
mixed trace 同时有 prefill 和 decode，自动计数不满足 40 个锚点/纯 forward 的区间保持未分类，不硬凑层数。

## 已实现的系统改进与门槛

非空目标导入为 opt-in；保留 scheduler idle、完整快照、同模型/TP、临时容量等约束。
导入调用 vendor radix insert 去重 KV，并释放重复 recurrent slot，不清空其他缓存。
真实跨节点四组检查全部通过：独立前缀、共享扩展、重复导入、失败回滚；状态摘要、slot 数量、旧缓存和输出均核对。
在此基础上加入每轮 3GiB/1s 的多状态准备预算，旧 +move 作为消融。
下一份报告比较未见过的原始长度轨迹；不在本报告提前声称加速。

[复测结果](results/compare/REPORT.zh.md) · [运行/复现说明](README.md)
