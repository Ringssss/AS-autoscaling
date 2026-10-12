# StateShift：以可执行状态准备扩展 Agent routing

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

2026-10-10。承接 v5 的原生 trace 与实现核验，固定论文主比较为同一强 router 加/不加状态准备能力。本文是研究定位与设计主张；证据范围见第8节。

## 1. 一句话主线

> Agent 的长寿命状态让 routing 能够复用历史计算，也使后续执行受现有状态布局影响。本文通过选择性的可执行状态准备，在请求就绪前改善候选 worker 的条件，让 routing 利用更合适的算力，同时控制准备成本、资源干扰和正确性。

标题候选：

> StateShift: Expanding Routing Choices with Executable State Preparation for Agentic LLM Serving

工作对象是兼容模型/布局上的模型执行状态，首先包括 KV 与 hybrid recurrent snapshots。GPU-resident transfer 是 enabling mechanism；progressive destination continuation 是实现依赖消费的机制。

## 2. Motivation：KV-cache gravity 的正确因果关系

已完成上下文的缓存复用能减少 prefill；routing 因此倾向将相同或相近前缀送到持有状态的 worker。后续调用又在那里形成更多缓存，进一步增强亲和。在前缀热度、会话工作量或阶段活跃度不均时，这种历史路径会把需求集中到少数候选 worker。

这是一种有条件的正反馈：

```text
某worker持有可复用状态
        ↓
亲和routing将相关请求发往该worker
        ↓
后续计算继续在那里形成状态
        ↓
新请求更倾向选择该worker
```

KV-cache gravity 表示复用状态对执行位置选择的吸引，不等于“状态多所以compute一定忙”。工具期间大状态可能驻留而GPU空闲；hot worker 应由排队、服务余量、SLO或容量压力说明。

问题出现于三者相交：**source有值得复用的状态，source的执行条件变差，其他兼容worker有更好的服务条件但缺少相应状态。**

DualMap的motivation支持亲和与均衡的张力。我们的原生trace支持长复用与同模型并发的存在；原生实际worker冷热与净收益事件频率仍需由部署数据支持。

## 3. 更深的问题与 insight

反复讨论“路由还是迁移”“TTFT还是TPOT”“搬多大状态”“什么场景有效”，其实都在问：

> **未来计算的执行位置，应该被已有缓存位置决定到什么程度？当改变这个限制有代价时，系统如何判断值得为哪些执行选择提前付费？**

两个价值必须区分：

- **复用价值：**保存已有状态可以避免多少历史计算。
- **准备到其他位置的边际价值：**这些状态在特定目标、特定时间就绪后，能减少多少等待或改善多少后续执行，以及会给集群增加什么代价。

长prefix、长复用链只能证明第一种价值。第二种还取决于source的负载、已有副本、目标服务余量、准备能否及时、内存与网络争用、以及新增状态的更新需求。

最核心的 insight：

> **把缓存从“复用已有计算的命中对象”进一步视作“决定未来执行选择成本的计算成果”。准备它的时机和位置，会改变后续routing能便宜地执行什么。**

路由在已有条件下选路径；准备层选择性地改善条件。两者联合才能判断一个新warm target是否值得创建。这里的“执行选择”是便宜且及时的选择，冷目标通常也能重算后执行，并非物理不能执行。

这个看法解释所有现有结果：source不拥塞时原地命中已足够；首次冷目标且重算昂贵时单次收益大；routing重算会逐渐形成副本，使准备的后续边际价值降低；一份共同前缀可能服务多条分支、多轮调用，准备事件数不能直接作为收益覆盖率。

## 4. 现有工作的能力与我们的候选问题

| 工作 | 已解决的具体问题 | 与本系统的关系 |
|---|---|---|
| DualMap（ICLR 2026） | 前缀双候选、SLO感知routing、排队请求的热点重平衡和hash-ring scaling | 提供强routing能力；我们改善其候选worker的状态准备条件，不把它描述成忽略负载/缓存 |
| Llumnix | 活跃请求跨实例重调度，源端生成与多阶段KV复制重叠，分配、失败和最终接管协调 | 提供live migration依据；我们聚焦跨请求持久状态的选择性准备与未来routing，不把overlap或状态迁移本身写为新颖性 |
| KVFlow | Agent Step Graph指导固定prompt缓存保留，CPU→GPU预取，按cache loading状态跳过未就绪请求 | 提供未来复用与readiness调度依据；我们的目标是跨worker服务条件与可执行状态准备的联合选择，不把proactive/readiness本身写为新颖性 |
| SAGA | 工作流结构、跨工具边界预取、session affinity、work stealing与任务级调度 | 最接近的组合能力；必须比较联合准备/路由的额外决策和资源/协议能力，不能只用reactive/proactive区分 |
| 共享KV/remote cache系统 | 保存/取回远端状态，部分解耦计算与缓存驻留 | 保留这些能力作为强基线；已能解决的事件不算新增空间 |

从论文设计中可核对的差别是优化对象与协议能力。不能从“joint”或“state first-class”等抽象词推导本项目具有独有贡献。

来源：

- DualMap §3.2–3.3：https://arxiv.org/html/2602.06502v1
- Llumnix §4.2：https://arxiv.org/html/2406.03243v1
- KVFlow §3.1–3.2：https://arxiv.org/html/2507.07400v1
- SAGA §4.3、§5：https://arxiv.org/html/2605.00528v1

KVFlow依赖完整节点ready后dispatch，同时描述HiCache层加载与计算重叠。因此“逐层重叠”也有先例，progressive destination协议的新能力需要在跨worker私有消费、hybrid一致边界与发布语义上具体说明，并由对照支持。

## 5. 问题实例：四个可行选择

Agent s 的已完成状态在 A，下一次调用使用稳定历史前缀并追加新输入。若A排队而B有余量，比较：

1. 在A复用并等待。
2. 在B使用已有副本/远端取回能力。
3. 在B重算所缺前缀；随后形成副本。
4. 提前为B准备可执行前缀，后续由router选择A或B；准备可以复制或采用更便宜的合法重建。

不是每个prefix都做第4项。source已有执行条件更好或目标已经warm时，保留原router路径。共享前缀准备不要求独占所有权转移；活跃执行接管另需ownership和输出协调。

准备比较的简化目标：

\[
\Delta = \text{预期减少的后续关键路径等待/重算}
       - \text{准备暴露等待}
       - \text{对其他请求的干扰}
       - \text{容量占用造成的损失}.
\]

各项应以同一目标/单位计量；容量不是随意与秒直接相减的字节罚项。预测跨多次消费时应反映未来不确定性与状态变化；不把原生未来输入或完成时刻当作线上已知。

副本不是无条件正确选择，迁移也不自动释放源缓存。收益不足时retain/fallback有合理性；准备失败或取消仍要记录已发生开销。

## 6. 三个挑战与三个组件

### C1. 可复用≠值得在目标准备

**组件：状态准备与routing联合调度。**

输入是已确认的可复用边界、候选worker的服务条件、原router分数、现有副本、容量与准备成本，以及可获得的阶段提示。输出是保持现状或创建特定执行选择。重点是评估边际收益，防止热前缀全部复制、负载均衡过度分散状态以及副本oscillation。

### C2. 有准备时间≠有准备资源

**组件：有预算的异步状态准备运行时。**

工具期是该Agent的GPU计算窗口，不保证CPU、源引擎、HBM带宽或网络空闲。运行时需要预留目标KV/recurrent资源并固定源状态，分离轻量协调与bulk state，复用连接，限速和限制并发，失败释放reservation，并让控制路径继续服务。

GPU-resident传输是优先路径之一，CPU staging/remote retrieval仍可有优势。参考2608.04458v1支持控制路径保护，其Agora本身使用host-memory staging，不支持“CPU做调度所以不能走CPU”的结论。

### C3. 字节到达≠状态可执行≠结果可发布

**组件：按依赖消费与发布的执行状态协议。**

对象描述包括稳定前缀、精确执行边界、KV与recurrent依赖、模型/布局身份与版本。目标只消费已就绪依赖；有能力时渐进推进计算，并在TP就绪与正确性规则满足后发布共享cache或接管输出。

Transformer KV与可变recurrent snapshot的语义不同。Hybrid对象既可检验抽象，也必须具体解决边界与一致性，不用toy buffer代替真实模型证据。

这些组件形成一个闭环：**值得改善哪个选择 → 如何有预算地准备 → 何时允许router/计算真正使用该选择。**

## 7. Introduction 的最终逻辑

P1：Agent重复执行推理—工具—推理，历史上下文跨请求积累。缓存避免重算；亲和会让后续计算沿已有状态位置聚集。

P2：阶段释放、前缀热度和会话工作量的skew使部分worker更繁忙。其他兼容worker的余量未必能低成本用于这些请求，因为缺少可复用状态。

P3：优秀routing已经在命中与负载之间选择，迁移系统已支持活跃请求重调度，workflow cache系统已指导保留/预取。我们研究在这些能力之上，如何选择性地改善下一阶段routing的状态条件。

P4：关键洞见是复用价值与目标准备边际价值不同。准备要结合source压力、target服务余量、后续需求及资源窗口。未完成准备还要求消费依赖与发布语义协调。

P5：提出StateShift，作为强router可调用的状态准备层，联合判断目标与准备动作、执行资源有预算的异步准备、按依赖推进目标计算并正确发布。

P6：证据分别展示问题存在的条件、运行时协议能力，以及同router加入准备层后的工作流与SLO收益。原生trace结构用于动机，受控case用于机制，完整真实回放用于系统结论。

可直接使用的核心段落：

> 长生命周期Agent反复复用历史上下文，使缓存亲和与执行位置相互强化。当会话需求和worker负载变化时，持有历史状态的worker可能已不再是下一轮的最佳执行位置。现有routing在已有命中与负载之间选择，而提前准备状态可以改变这些选择的成本。本文研究如何在有限资源下选择性地准备可执行状态，使routing及时利用更合适的worker。我们的设计区分前缀复用价值与目标准备的边际价值，并联合协调准备时机、资源占用与依赖就绪，以降低Agent工作流关键路径上的等待。

英文段落：

> Long-lived agents repeatedly reuse historical context, coupling cache locality with execution placement. As demand and worker load change, the worker holding reusable state may no longer offer the best execution conditions for the next turn. Routing selects among the conditions already available; preparing state can improve those conditions ahead of demand. We study how to selectively prepare executable state under finite resources so that routing can use better workers at the right time. Our design distinguishes the value of retaining computation from the marginal benefit of preparing it elsewhere, and coordinates preparation, resource reservations, and dependency readiness to reduce workflow critical-path delay.

## 8. 已有证据与准确力度

- AgentX全量98,827请求中84.58%有≥32K已完成同stream输入复用，17.03%同时有其他同模型stream活动。这证明结构存在，不证明实际source排队。原生共享≥32K前缀的同时活动distinct streams最多9。
- 在明确的stationary placement敏感性中5.4%–9.0%为长状态+活动差+target缺状态候选，不是可盈利事件频率。不同conversation的绝对启动时间与实际worker归属未知。
- Kimi245,555条匿名日志没有精确prefix/session身份，不能用于原生跨轮routing/准备因果归因。
- 130K隔离case：已预建连接、冷目标下，copy+续算首token中位数2.26s vs目标重算19.35s，支持跨位置复用机制。
- 受控64K fan-out、上限64下16分支的完成窗口比最好routing减少约4.1%，64分支减少约14.3%；TPOT基本不变。64分支不是原生典型形态。
- 单次建链31–32s如暴露于关键路径会抵消本轮服务收益；同步DCU路径要求源/目标空闲且目标空树，不能代表完整准备层。
- 更早progressive实验支持局部目标端提前消费收益，不能与DCU同步路径视作一个已经完整实现的runtime。

出处：`results/native-relocation-space-20261010/REPORT.zh.md`、`results/skew-readiness-20261009/CONCLUSIONS.zh.md`、`experiments/progressive_handoff/E2E_RESULTS.md`。

## 9. 主张和比较固定

主比较：同一strong routing / remote-cache能力，原始路径 vs加入本准备层，相同GPU、内存、到达、输出与可用信息。最终可以评估多个router上的适用性；不能把简化cache_cost视作完成DualMap基线。

核心目标：工作流完成时间、在声明SLO下的goodput。完整TTFT、TPOT、token间隔、重算量、准备字节、峰值双份驻留及邻居/工具/控制路径延迟用于解释得失。

候选贡献围绕具体能力陈述：

1. 前缀复用与目标准备边际价值的分离及联合决策。
2. 让source与target服务继续推进的有预算准备机制。
3. 可复用计算状态的依赖消费与发布协议，覆盖真实hybrid执行边界。

“Gravity”“proactive”“GPU-direct”“state first-class”是组织动机的语言或基础机制，不单独当贡献。三个组件的组合价值要由相对现有强组合能力的结果证明，不声称仅命名新问题就获得新颖性。

## 10. 固定的研究问题

> **在缓存亲和影响执行位置的Agent服务中，系统应为哪些后续routing选择提前准备可执行状态，以及怎样在有限资源下让这些选择及时生效？**

保留“KV-cache gravity → 有条件的worker冷热不均”作为开篇。论文主体回答上述可实现、可归因的问题；收益小的区间用于说明何时应保持原routing。

## 11. LMetric / Blitz Router 对主线的约束

2026-10-10审查了本地blitz-router（HEAD a005ceb3），核心为OSDI'26 LMetric：`(queued_uncached_prefill + request_new_tokens) * (active_requests + 1)`，即时admission并根据引擎step/eviction反馈维护状态。其论文已评价coding agents和TPOT，不应描述为只均衡请求或忽略cache。另有正确成本网格下的simulator least-ttft策略。

在source可及时batch、已有热目标或一次重算即稳定形成副本的常见场景，额外准备空间可能小。当前已测balanced对照的14.3%不能迁移成相对LMetric收益。当前同步DCU路径也尚不能直接介入繁忙、非空稳态worker。

最合适的系统关系是LMetric + StateShift：router优化已有warm set上的选择，准备层只在值得时改变warm set；backend确认可执行就绪后公布事实，原score继续使用真实new_tokens和负载。

详细分析和代码锚点：`BLITZ-LMETRIC-COMPARISON-2026-10-10.zh.md`。这是只读分析和常见AgentX条件下的预期，没有新对跑数据。
