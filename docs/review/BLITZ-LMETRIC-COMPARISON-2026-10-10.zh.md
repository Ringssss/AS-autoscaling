# Blitz Router / LMetric 与 StateShift：常见 AgentX 场景的判断

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

2026-10-10。只读审查 `/home/zhujianian/blitz-router` 的源码与文档，并核对论文；没有启动两系统对跑，因此下述胜负是有依据的预期，不是 measured comparison。

## 1. 结论

**当前版本在常见、稳态、长前缀复用的 AgentX 条件下，没有依据预期会稳定大幅胜过 LMetric。** LMetric 是应保留的强router，不是只均衡请求、忽略KV的弱对照。

StateShift 的空间是：当仅有的warm候选已不能及时服务、其他兼容worker有真实余量时，让已有长状态提前在新目标可执行，减少换位置的冷准备成本。更合理的主比较是 **LMetric vs LMetric + executable-state preparation**，并保留原算法的正常缓存、负载与热点处理能力。

## 2. 版本和论文身份

- Local HEAD：`a005ceb3a6689d30514bd27e45ede0db933d0fea`（2026-05-22）。审查时工作树clean。
- remote：`git@github.com:blitz-serving/blitz-router.git`。
- 本仓库的路由核心策略是 **LMetric：Simple Is Better: Multiplication May Be All You Need for LLM Request Scheduling（OSDI'26）**。
- **BlitzScale（OSDI'25）** 是同项目体系的模型权重快速扩容工作，不能把整套权重传输能力当成本路由仓库已经提供的KV准备API。

官方论文：https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan

分析版全文：https://arxiv.org/html/2603.15202v2

相关BlitzScale：https://www.usenix.org/conference/osdi25/presentation/zhang-dingyan

## 3. 当前代码已经做到什么

`router/src/scheduler/policies/lmetric.rs:18`：

```text
score_i = prefill_tokens(req, observation_i) * (bs_i + 1)
```

`router/src/scheduler/policies/dsl_runtime.rs:143`：

```text
prefill_tokens = queued_uncached_prefill_tokens + input_tokens - cached_prefix_tokens
```

因此它同时考虑当前排队的新prefill工作、候选请求的缓存复用与请求负载。`state.rs:107`的bs是已经分配的active请求计数，包含waiting；不能把它误读成只统计已完成工作或瞬时GPU batch。

`dsl_runtime.rs:545`在决定目标后立即增量登记bs、waiting和prefill tokens。`engine/colocation.rs:457`起根据逐forward-step SSE消费量/生命周期扣减；`:606`与`:654`根据backend evicted/used block IDs更新prefix tree。它不会复现p03/p04只累计完成请求耗时的错误。

另有`least-ttft-q`策略（`policies/least_ttft.rs`）：对每个worker调用模拟器、选择最小预计TTFT，并立即登记admission。模拟器考虑chunked prefill、running/waiting、缓存mirror和逐步观测。它需要正确模型/硬件成本网格与runtime启用；文档中的首轮stopgap网格结果不能视作其最佳性能。

`docs/architecture/README.md:34`明确本repo是routing层；`engine/client.rs:135`的抽象是add request、abort与step反馈，没有export/import executable prefix的trait方法。这个边界支持我们作为额外准备能力接入；不代表所有Blitz后端或其他仓库都没有状态搬运。

论文已经覆盖coding-agent工作负载，并评价TTFT和TPOT，不能写“它只考虑TTFT或未考虑Agent”。其Coder与ToolAgent数据不等于本地AgentX/Weka，不能据论文改善数字直接预测本地效果。

## 4. 为什么多数原生长复用请求可能由它处理得很好

AgentX原生98,827请求中84.58%有≥32K的已完成同stream输入前缀。每conversation观察活动数峰值中位数2、p90=6；并非GPU排队测量。长复用与其他同模型stream活动同时出现17.03%，仍不证明warm worker难以服务。

source可以及时batch时，LMetric会保留复用。仅以“去冷目标重算19s、复制2s”不能证明胜过LMetric，因为LMetric可能根本不会选择那次冷重算。

一个仅说明score的假设例子（不预测时延）：原前缀65,536，新输入64 tokens；source无待处理prefill、已有8个active请求，而target空闲且cold：

```text
source score = 64 * (8 + 1) = 576
target score = 65,600 * (0 + 1) = 65,600
```

它明显偏好source，这是原地命中更划算时的合理行为。有大量新prefill积压时，source score增加，它又可以向其他worker分流。不是always-sticky。

## 5. 哪些事件仍可能提供准备价值

| 场景 | LMetric已有能力 | 额外准备的预期 |
|---|---|---|
| 低/中负载、source warm且可及时服务 | 利用cache优势、合理batch | 增量空间小，额外准备可能浪费 |
| target已有所需副本 | 在多个warm worker间选目标 | 继续用原router，通常无需复制 |
| 一次冷miss后缓存稳定保留 | 重算形成副本，后续复用 | 主要避免首次重算；随重复次数摊薄 |
| warm set服务能力不足、cold兼容worker有余量 | 分流可以卸载，但cold路径付准备成本 | 若有及时准备窗口与容量，可减少换位置损失 |
| 下一轮状态不确定、网络/内存紧张 | 路由无需投机准备整份状态 | 准备容易浪费，应选择性禁用或回退 |
| worker退出或新worker需要接纳长状态 | router可改变目标 | 状态连续性能力可能有价值；这不是常见稳态事件的证据 |

前缀热度并不等于可盈利准备。足够长的历史、source服务约束、target余量、便宜的准备和后续消费要共同成立。

我们此前14.3%的受控结果对比的是balanced，不是LMetric。其64分支长共同前缀超过原生每conversation相同≥32K前缀distinct streams的最大9，不能作为典型AgentX的LMetric胜负证据。16分支提高并发上限后，相对最好routing增量已只有4.1%。

## 6. 当前实现额外限制

`AS-autoscaling/experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py:44`要求transfer时整个scheduler idle；`:221`要求destination root无children。它不能直接作为繁忙稳态worker的准备层。

已有受控runtime复制约0.81s/2.37GB，连接建立31–32s。如果建链、inspect、同步等待或临时容量代价暴露，会抹掉准备收益。GPU buffer传输本身不消除协调成本；Rust router本来也没有在路由热路径上搬几GB状态。

Hybrid的“可执行边界”metadata与预留协议可作为扩展能力。应在相同兼容模型/后端上比较，不能用另一个backend缺少功能或错误cache命中估计来制造优势。

## 7. 一个更具体的共同问题：warm set 与需求

LMetric论文§5.2把prefix class c的请求份额x，与已有warm实例集合M相联系。其潜在失衡条件涉及：

```text
该类请求对每个warm实例的需求，超过其他请求对其余实例的需求。
```

论文提出两阶段detector，必要时过滤warm热点以分流；当前本地`lmetric.rs`仅见乘法body，未在本地搜索到相同detector。评估已发表方法时不能据此删去论文中的保护能力。

这一分析与本项目最有价值的连接是：**router会在当前warm set内或之外选择；准备层可以判断是否值得提前改变warm set。**

不是把“LMetric有失败条件”变成我们的motivation。其论文认为该类极端hotspot在评估数据中罕见；本地AgentX缺实际全局到达与worker归属，也不能验证该条件常见。合理问题是：在需要改变warm set的事件上，保存已有计算能否比到达后重算更便宜、更及时。

## 8. 集成边界和公平比较

1. 保留LMetric的原score、即时admission与真实engine cache反馈。
2. 准备层利用确认的执行状态、阶段提示和成本判断创建目标副本。预留成功或bytes到达都不等于cache ready。
3. 目标能实际消费后，发布backend cache/readiness事实，LMetric自然看到更小的new_tokens并继续选worker。
4. 准备失败回退，但报告已发生的带宽、容量与控制开销。迁移后仍存在的source cache不能算物理释放。

主要比较：**LMetric vs LMetric + StateShift**；根据policy/compiler feature明确所用策略。默认Cargo features并不自动选lmetric，README推荐`--features lmetric-q`；未正确开启simulator的least-ttft-q会退化，不能作为有效基线。

完整比较还要使用相同backend telemetry语义、cache budget、warmup、输入/输出、到达与依赖。最终报告全部请求和多轮任务，不只挑冷目标subset。

本轮结论是保留它作为首要强基线和集成对象。当前证据不支持宣布赢或项目原理无效。
