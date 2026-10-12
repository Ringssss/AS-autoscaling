# StateShift：状态准备与路由的联合设计

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

日期：2026-10-09。本文根据用户明确的比较对象重新定调：**系统主比较是我们的方法与 routing；状态迁移和 GPU-resident transfer 是内部机制。**

## 1. 固定定位

> StateShift: Coordinating State Preparation and Routing for Agentic LLM Serving

本文研究：在长生命周期 Agent serving 中，如何提前准备可执行状态，扩大下一阶段的执行选择，并控制准备过程对当前关键路径的干扰。

核心命题：

> **将已有计算成果及时放到有服务余量的兼容 worker 上，可以把闲置算力变成下一轮真正可用的执行能力。Agent 阶段切换提供准备机会，资源争用和状态依赖决定这些机会如何使用。**

范围是兼容模型与执行布局的 worker；空闲 GPU 不一定已经加载目标模型，异构模型之间不能直接复用 KV。模型权重管理与工具应用状态不是本系统的主体。

## 2. 两个相互扣合的 insight

### 2.1 算力空闲与状态可用是两件事

Agent 的历史状态保存了已经付出的计算。未来计算位置的选择，同时取决于 worker 的服务余量，以及所需状态何时能被正确消费。

状态在 A、B 有余量时，缓存感知 routing 可以留在 A、选择已有副本、从共享存储取回，或在 B 重算。每种选择都付出不同等待和准备代价。主动准备能改变下一轮到来时这些选择的成本。

因此，设计从“到达后选哪个 worker”扩展为“到达前如何改变 worker 的准备条件，再选择执行位置”。Routing 与状态准备是一个决策闭环。

这不是所有 router 的绝对能力边界。Routing 之后的重算会形成新副本，带远端取回或预取的系统也会改变布局。我们的区别必须体现为准备时机、依赖与争用控制共同带来的净收益，而不能靠把 baseline 禁止使用已有功能来建立。

### 2.2 工具窗口是资源有别的窗口

一个 Agent 等待工具时，其 GPU 推理暂停，但 CPU 可能正在执行工具、协调依赖和推进其他 Agent；网络也可能在进行检索或 RPC。**GPU 计算间隙不保证 host 或网络间隙。**

反过来，远端 I/O 工具可能留下大量 CPU 余量。真正的准备机会取决于当前哪些资源有余量，以及哪些资源正在约束工作流的关键路径。

更深的设计原则：

> **准备下一阶段的资源时，限制对当前阶段关键资源的争用。**

GPU-resident 状态移动可以利用已有的设备状态，减少不必要的 host staging；控制元数据和大块状态分离，避免让协调、布局准备与推理调度在同一阻塞路径上串行等待。资源不足时，可保持原地、延后准备或选择其他准备动作。

## 3. 参考论文的准确用法

本地参考：`2608.04458v1.pdf`，14 页，Architectural Implications of Agentic AI Workflows。

论文 §IV-A / Fig.5（印刷页4–5）展示工具阶段的 host 突发需求；§IV-B 描述控制路径、微架构干扰和协调代价。这支持关注准备工作对 Agent 当前关键路径的影响。

论文 §V-A2（印刷页9）使用 host-memory staging 与下一 Agent 的状态预取；§V-B（印刷页10）隔离 scheduler / orchestrator 与 runner 核池。**它支持保护控制路径和利用阶段信息，不支持“CPU 在做调度，因此不能走 host memory”。**

来源：https://arxiv.org/html/2608.04458v1

Host staging 可由 DMA 执行；CPU cores、host DRAM bandwidth、PCIe、NUMA 和控制线程等待是不同代价。核心忙不能直接推出 DMA 慢。Device-buffer collective 也不自动证明物理路径绕过 host memory。

这篇论文对我们的作用是提出资源与阶段的约束。我们的主比较仍是 routing；GPU-resident / host-staged preparation 是内部机制比较，用于解释系统收益及其干扰来源。

## 4. 整体因果链

1. **状态跨请求存活。** 长生命周期 Agent 反复使用历史上下文，已有计算成果有复用价值。
2. **执行需求按阶段变化。** 工具返回、分支释放和集群负载变化，使下一轮最佳执行位置变化。
3. **路由面对不同准备条件。** 有状态的 worker 可能排队；有余量的 worker 可能需要重算、取回或复制。
4. **阶段信息允许提前准备。** 在后续计算就绪前改变状态可用性，但准备需占用有限内存、带宽及运行时资源。
5. **提前准备必须保护当前执行。** GPU 间隙不代表 host、网络或整个源引擎空闲；准备不能无限挤压工具、协调与其他请求。
6. **状态按依赖就绪。** 目标可以在内部依赖满足后推进计算，缓存发布、接管与输出依据正确性协议进行。
7. **联合选择准备与路由。** 将原地、重算、取回和准备后的执行成本放在同一框架内，减少工作流关键路径等待。

## 5. 三个组件

| 挑战 | 组件 | 具体职责 |
|---|---|---|
| 哪些状态准备能真正改善下一轮的执行选择？ | 状态准备与路由联合调度 | 选择兼容目标，比较已有副本、重算、取回、复制；考虑复用、队列、准备时间和容量 |
| 如何准备状态而不堵住当前控制/执行路径？ | 有资源预算的状态准备平面 | 区分 bulk state 与控制元数据；GPU-resident 传输、连接复用、异步推进、源 pin 与目标 reservation、干扰预算 |
| 状态尚未完整准备时，何时可以正确计算与发布？ | 按依赖消费与接管协议 | 描述精确执行边界，按层发布内部 readiness，推进目标计算，协调 TP、缓存可见性及必要的所有权提交 |

GPU→GPU 是第二项的一个实现选择。渐进目标端执行是第三项的一个机制。两者服务于扩大可用执行选择和保护关键路径。

控制工作仍由 CPU 参与。设计目标是减少与数据量成比例的 host 工作和阻塞等待，不能宣称 GPU→GPU 等于没有 CPU 控制开销。当前 JSON token lists、同步 collectives 与调度线程内处理需要在实现上进一步收敛。

## 6. 状态对象由消费者依赖定义

统一抽象是某个明确执行边界上的**可复用计算状态及其依赖**。

- Full attention：需要消费范围内的 KV、位置与模型配置一致性。
- Hybrid：KV 和 recurrent snapshot 必须对应一致的进度；当前状态与历史可复用快照的作用不同。
- 多模态：身份与依赖还包含媒体内容、预处理、位置语义及适用的编码结果。

已完成前缀的复制允许多个 worker 复用，不必变更独占所有权。活跃执行的接管则需要版本与输出所有权协调。不能把副本准备、源端物理释放和执行接管都称作同一件“迁移”。

传输可渐进；每个计算阶段只能读取已正确就绪的依赖。对外可见的缓存、接管与输出仍有清晰发布规则。

## 7. 主比较与机制比较

主比较：相同 GPU / memory budget、到达、输出及工作流下，我们的方法与优秀 routing 策略，包含亲和、负载与缓存感知路由；远端缓存取回、预取或共享状态系统应作为相关强基线保留。

机制比较：固定目标/路由决定，比较准备时机、host staging 与 GPU-resident 数据路径、完整准备与按依赖执行，以及不同状态表示。这些用于归因，不替代主比较。

主要目标是工作流完成时间与 SLO goodput。TTFT、TPOT、token 间隔尾延迟、控制/工具延迟、准备字节、占用与干扰说明收益来自哪里。CPU 争用是从参考论文推导出的设计约束，目前没有被本项目实验验证为 GPU-resident 路径胜出的因果解释。

## 8. 与已有工作的定位

Llumnix 已有源端生成与多阶段复制重叠；SAGA 已有工具期预取；Agora 已有阶段感知预取、角色隔离和布局优化；共享 KV 系统已解耦部分状态存储与计算。不能把 proactive、state first-class 或 GPU-direct 本身写成新颖性。

候选贡献：**把状态的准备条件、当前阶段的资源约束和目标计算依赖统一进路由与准备闭环。** 新颖性最终由该闭环的具体机制及相对强组合基线的能力确定。

## 9. 已有证据对主线的支持

2026-10-09 的27个正式/控制试验共1,440个消费者请求，0个请求错误。在预建连接、64K共享前缀、1s准备窗口的构造fan-out中：

- 引擎上限64、16分支：相对最快不准备策略 sticky，wall 约减少4.1%。
- 引擎上限64、64分支：相对最快不准备策略 balanced，wall 约减少14.3%，request goodput约增加16.6%。
- TPOT基本不变；传输摘要与单请求同形状的64-token输出对照通过。
- 单次复制约0.81s，建链约31–32s。连接建立若暴露在一次事件的关键路径上会抵消收益。

这些结果支持状态准备能改变执行选择的局部动机，并提醒我们关注配置、准备开销和复用范围。它们没有验证CPU争用下GPU-resident路径的优势，也没有验证连续多前缀稳态。

更早的 progressive 结果支持目标端提前消费状态的机制；不能与10月DCU同步路径混作同一个实现。现有“源/目标必须空闲、目标树为空”和同步控制路径，是完整设计需要解决的实现边界。

## 10. 可直接用于正文的主张

中文：

> Agent serving 在推理与工具执行之间反复切换，既有计算状态的驻留位置与后续计算的最佳位置随阶段和负载逐渐分离。缓存感知路由需要在复用、排队与目标准备成本之间选择。本文联合安排状态准备和请求路由，在后续计算就绪前改善目标的执行条件。由于GPU计算间隙不保证host或网络间隙，系统限制准备对当前关键路径的干扰，并依据状态依赖推进目标计算与安全发布。

英文：

> Agentic serving alternates between inference and tool execution, while reusable computation state persists across turns. As demand changes, the worker with available capacity may lack the state needed for the next turn. We coordinate state preparation with request routing to improve execution choices before demand arrives. Preparation is scheduled against the resources available in each workflow phase, and destination computation advances as its state dependencies become ready. This design targets workflow latency and SLO goodput while bounding interference with ongoing orchestration and inference.

贡献按完成后的能力填写：联合准备/路由的决策；隔离批量状态移动与关键控制工作的运行时；支持精确依赖与发布语义的渐进消费协议。现有证据支持的部分与设计目标分开记录，不以换词代替能力实现。

## 11. 2026-10-10 原生 trace 与 routing 组合定位

已核对远端 main / HEAD 与本地一致，Kimi trace 在 Release `kimi-trace-2026-04-07`，本地压缩包hash与Release digest一致。重新解析全部245,555条；字段和分布复算与此前审计一致。该日志没有非空conversation/session/routing身份，也没有token/hash内容；不能识别跨轮状态在哪里或下次请求该去哪里。长input≥32K占12.52%，历史路径cached≥32K占1.77%，两者都不能独立代表可盈利迁移的覆盖率。

AgentX/Weka全量393条conversation、98,827个请求按原始内部时序重新审计：83,590条（84.58%）能复用已完成同stream输入至少32K；其中16,828条（占全量17.03%）到达时还有其他同模型stream活动。跨stream共享≥32K前缀的并行distinct streams最大9，所以64分支合成fan-out不能作为典型原生证据。

在声明的stationary hash亲和、300s TTL、无容量驱逐假设下，2/4/8个兼容worker、3个hash seed产生约5.40%–9.01%的结构候选：source有≥32K已完成input前缀，source观察活动数更高，least-active target缺少该前缀的一部分。其他请求活动不是实测排队，这个比例不是收益率。所有conversation相对零时间独立，不冒充全局集群到达。

原始input+output≤131K且output>0的子集只有46,129条（46.68%）；该子集的结构候选约8.50%–14.14%。大上下文被既有runtime限制排除的事实必须保留。

因此，系统宜定位为**routing可调用的状态准备层**：router持续选择执行位置，准备层为值得改善的目标提供确认的可执行状态与准备成本。已有热目标、source的batch/负载条件已足够、或准备代价高时沿用原策略；对冷目标的选择性复制/取回/重建，改变后续routing可选路径的成本。

主比较使用同一强router原始能力与加准备层的能力，保留原router已有远端取回、预取或复制功能。与另一迁移系统的比较只是相关能力和机制归因。回退仍付出已发生的带宽、内存与协调成本，不承诺零回归。

可用来组织论证的新增能力是：**发现值得改善的执行选择 → 用有限资源准备可执行状态 → 让router在确认就绪后利用这些选择。** 原生结构支持研究这个闭环，但尚不能推出实际净收益事件的频率或幅度。

本轮报告：`results/native-relocation-space-20261010/REPORT.zh.md`，数字、脚本hash和来源记录见同目录的analysis与RUNS.json。本轮只做CPU分析，没有启动新的GPU服务。
