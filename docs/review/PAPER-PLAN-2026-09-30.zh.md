# FluidKV：现在开始写的论文蓝图

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

2026-09-30。与 [证据审稿](REVIEW-2026-09-30.zh.md) 配套。本文件中的策略公式、实验目标是建议设计，不是已验证成果。英文草稿只使用本轮确认过的事实，不提供虚构的系统收益。

## 1. 写作定位

建议标题：**FluidKV: Resource-Aware Runnable-Prefix Mobility for Hybrid LLM Serving**。

研究问题：当一个请求可在远端复用混合状态时，如何安全地把该状态变成目标引擎上的可执行前缀，以及何时这种动作比原地服务或目标重算更有利？

核心对象：`runnable prefix = token identity + position/branch boundary + attention KV + recurrent snapshot + compatible layout`。这只是必要语义集合；具体实现不一定需要把它们序列化成同一个对象。

研究范围建议先锁定：同checkpoint、相同TP布局、已完成边界、向新启用/空闲空缓存worker导入。若论文要求常态非空目标迁移，必须把安全插入与reservation做完再扩展范围。

关键指标首先选全流量TTFT-SLO goodput。任务完成时间作为补充，并明确prefill省时可能被decode/依赖关键路径掩盖。不以hit ratio最大化替代服务目标。

## 2. 整篇文章的每节任务

| 章节 | 读者必须带走的结论 | 图/证据 | 写作状态 |
|---|---|---|---|
| Introduction | hybrid state relocation有明确场景和约束，我们解决其中一个具体问题 | 一个移入空worker的因果示意 | 可写；摘要收益句最后填 |
| Background & Motivation | token匹配不够；机制便宜不等于请求快；局部快不等于全局收益 | runnable边界图、实测成本曲线、关键路径反例 | 下附草稿 |
| Design | 判断可运行性→判断资源可行性→比较同终点动作→验证发布 | 协议状态机；三动作决策 | 已有部分与提案明确区分 |
| Implementation | 后端支持什么、不支持什么，如何约束安全边界 | backend capability表、版本manifest | 可写；依实际代码 |
| Evaluation | 正确性成立；同router增加copy有净收益；gate避免何种损失 | correctness表、paired E2E、eligibility、ablation | 部分可写，关键E2E需补 |
| Related Work | 与routing、state transport、hybrid reuse逐维比较 | 具体能力表，不做“一切皆无”总结 | 一手核对已开始 |
| Limitations | idle/empty、单模型、样本、oracle、工作负载外推 | negative regimes | 可写 |

图控制在约6–8张主图，按正文空间调整；17张占位图会让动机、实验、贡献重复讲同一件事。宏大体系结构图不能替代一个实机闭环。

## 3. 建议动机正文（英文，可作为重写底稿）

### 2. Background and Motivation

Prefix caching avoids repeated computation only when the required execution state is available where a request runs. Distributed serving therefore couples request placement with state placement. Cache-aware routing and distributed cache systems already address important parts of this coupling. Our focus is narrower: relocating an executable prefix of a hybrid-attention model, and identifying when this action is both feasible and beneficial.

### 2.1 A matching prefix is not necessarily an executable prefix

A hybrid-attention model maintains both attention KV tensors and recurrent state. Reusing a prefix requires these components to describe the same token history and execution boundary. A token match alone does not establish this condition. In our runtime, recurrent snapshots are available at selected boundaries; the usable migration boundary can therefore be shorter than the request's token-level longest common prefix.

We treat the complete, boundary-consistent state as the unit of relocation. The implementation checks that this state exists at the source and installs its components before publishing the destination prefix. This distinction also changes capacity accounting: token slots and recurrent-snapshot slots are separate resources, and a destination with enough token space may still be unable to admit the complete state. These requirements motivate explicit state inspection and resource reservation rather than inferring readiness from a token match.

[Evidence placement: cite the runtime's boundary checks and separately report output-equivalence tests on the backend where they were run. Do not use the performance sweep's availability flags as a substitute for output correctness.]

### 2.2 Transfer can be cheaper than reconstruction, but its endpoint matters

Our measurements show a substantial difference between reconstruction and the engine's state-transfer phase for long prefixes. On same-node pair A, the median client time to the first stream event for a cold 130,048-token prefix was 28.51 seconds, whereas the median engine-reported transfer phase was 0.992 seconds for 2.73 GB of hybrid state. These measurements establish a promising mechanism-level cost asymmetry. They do not by themselves establish a 28.7-fold reduction in request TTFT: the transfer measurement excludes group initialization and does not include serving the first token at the destination.

The difference is smaller for short prefixes. In the short-prefix sweep, the corresponding measurements at 4,096 tokens were 0.507 and 0.472 seconds. This point is close in absolute cost, but does not locate a crossover: smaller lengths and the complete migration-to-first-token path must also be measured. We therefore use the measurements as cost observations, rather than assuming a universal length threshold or constant transfer cost.

[Figure: retain separate first-event and engine-transfer curves; state their endpoints in the caption. Replace with a common-endpoint comparison once available. Keep topology-specific series separate.]

### 2.3 A profitable transfer is not yet a profitable serving decision

A transfer may save computation for one request without improving the service objective. In a recorded two-arm replay, transferring a 71K-token prefix reduced the affected request's TTFT from 11.497 to 3.944 seconds, while the round's completion time remained nearly unchanged at 125.350 and 125.324 seconds. Another request determined the round's completion time. This example motivates evaluating the affected request, the rest of the workload, and the task critical path separately.

Feasibility also limits the opportunities to act. Our current Hygon transfer path requires idle schedulers and an empty destination cache. Its measured transfer cost therefore applies to an import into a fresh destination, not to arbitrary relocation under cache pressure. Extending the decision to occupied destinations requires accounting for destination reservations, recurrent-state capacity, temporary transfer storage, and any displaced cache state. The source and destination may hold the state simultaneously during transfer, so relocation is not immediately equivalent to freeing capacity.

A separate 32K pilot illustrates why allocator observations must be interpreted carefully. The occupied-hit and occupied-miss conditions began with 32,768 free token slots and had median first-event latencies of 1.67 and 9.54 seconds. Free slots after generation were 32,768 and 4,096, respectively. These conditions used different engines and token sequences, and resident-prefix survival was not measured in the miss condition. We consequently treat the result as a motivation for a paired capacity experiment, not as evidence of an eviction cascade or a causal memory-pressure penalty.

Together, these observations motivate three checks for relocation: whether a complete executable prefix exists, whether the destination can safely admit it, and whether relocation improves the chosen service objective relative to the available alternatives. Prefix length informs the last check, but does not replace the first two.

## 4. 决策模型：定义同一个终点，避免公式制造证据

令请求到达为t=0，所有T均到首个输出token。三个行动：

- `T_stay = Q_source + H_source(b, suffix)`。
- `T_cold = Q_dest + R_dest(full_prompt)`。
- `T_import = max(Q_dest, C_ready) + H_dest(b, suffix)`，**仅在目标排队与状态准备可重叠时**。

H包括从边界b执行剩余prompt并产生首token，不能设为零。C_ready从本次决策开始，包含源等待、transfer资源排队、预留、复制、安装与检查；若依赖不能重叠，使用真实时间线的串行和，不能强行套max。

先过滤infeasible行动，再比较预测完成时间及外部性。如果加入干扰/驱逐损失I、E，需说明是总系统延迟还是单请求延迟的权重，避免把不同单位直接相加。

资源可行性建议写成分量约束：

`delta_KV + decode_reserve <= reservable_KV`

`delta_snapshots <= reservable_snapshots`

`peak_temporary_bytes <= temporary_memory_budget`

既有evictable KV不自动等于可无代价使用的free；源pin与目标reservation必须计入同时驻留。状态slot预留和请求准入分开：拒绝一次migration应回退到合法服务动作，不能无意拒绝客户端请求。

现有 `agentshift/controller/preparation_decisions.py` 已用 `max(ready, queue)+service`，比当前稿件的n-only规则更接近这个模型。复用这部分思路，但capacity_feasible目前来自调用者，不能把这个布尔参数当作完成了实际资源预留。

## 5. 一周内值得形成的“最小论文闭环”——按结果推进，不承诺日历工期

1. 冻结主backend/version和数据manifest，确定fresh-worker场景。
2. 从已有短prefix正确性扩展到论文主length，比较cold/local/full-bundle和失败负控。
3. 选一个长prefix测完整import-to-first-token；拆出init/ready/resume，不先扩长度矩阵。
4. 同一个真实router开启/关闭copy，在同GPU预算、同warmup、同arrival上跑paired E2E。对新worker的启用时间与idle资源成本一致处理。
5. 报告eligible funnel。若比例很低，依据最大损失来源决定实现nonempty import还是收窄主张。
6. 再做cost/capacity gate消融和长度/占用扫描；若全流量收益没有出现，诚实写机制/测量论文，不反过来挑请求制造headline。

并行于实验可以完成动机、协议、限制和related work的写作。结论的动词必须随证据变化：implemented、observed、estimated、proposed不能混用。
