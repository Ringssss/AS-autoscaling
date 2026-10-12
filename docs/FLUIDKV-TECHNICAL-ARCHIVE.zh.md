# FluidKV / AS-autoscaling 技术留档

更新：2026-10-12。本文是近期实验、纠错记录和后续方案的入口。仓库历史名称 AgentShift、研究名称 FluidKV 均指向这里记录的相关工作；不同后端和时期的能力、实验结果分别标注。

## 1. 当前结论

GPU 上的可续算状态复制在已验证的共享长前缀突发场景中有可重复净收益。下一步研究重点是多级状态准备与请求接纳协同；HBM/DRAM/SSD 全链路、完整联合调度器及其收益尚未实现和验证。

核心问题：**状态就绪时间决定空闲 GPU 何时能接手后续计算。准备同时增加当前资源负载、减少未来计算；调度必须表达准备动作和消费者之间的依赖。**

## 2. 2026-10-12 六实例验证

配置：三个节点、每节点两个 TP2 实例，共12张Hygon卡；Qwen3.6-35B-A3B；每实例运行请求上限64。每个客户端两轮，64K共享前缀，每轮增加64个私有token，输出128 tokens。到达前有统一1秒准备窗口；复制或重算的暴露时间计入延迟，通信连接初始化单列。

最终采用30个完整trial、6400个请求。每个条件两次重复，表中为全部客户端完成两轮的时长中位数。Blitz和DualMap原版策略通过同一SGLang后端适配；不是原生完整服务栈的部署排名。

| 并发客户端 | 更快的纯路由 | 复制 | 更快的提前重算 | 复制相对纯路由 | 复制相对提前重算 |
|---:|---:|---:|---:|---:|---:|
|32|44.48秒，Blitz|38.34秒|41.46秒|13.8%|7.5%|
|96|56.26秒，DualMap|45.25秒|49.97秒|19.6%|9.5%|
|192|73.40秒，DualMap|58.35秒|62.48秒|20.5%|6.6%|

32、96并发下，复制组联合SLO达标率均为100%。96并发对应DualMap为67.2%。192并发下复制虽然更快完成，联合SLO达标率约15.9%，低于DualMap的41.7%；驱动派发等待达到数秒，不能把该档位全部延迟归因为GPU或迁移机制。SLO固定为TTFT≤10秒、TPOT≤200毫秒。

### 对照与修正

- 精确记录已完成输入的可续算边界，修正把65600 token错误取整为65536的估计。
- DualMap复测缓存相同输入、相同目录版本的重复查询结果，保持原路由策略；旧适配层192并发超时不作为算法性能结论。
- 增加“等重算完成再分发”的强对照，保持原始到达时间，避免消费者提前进入Engine又重复prefill。
- 五条复制链验证状态摘要、后续输出和实际复用；采用试验请求错误0、缓存拒绝项0、遥测序号缺口0。24项相关单元测试通过，六实例已清理。

完整表格：[最终报告](artifacts/fluidkv-20261012/FINAL-REPORT.zh.md)、[判断与限制](artifacts/fluidkv-20261012/DECISION.zh.md)、[机器可读汇总](artifacts/fluidkv-20261012/final-report.json)、[核验摘要](artifacts/fluidkv-20261012/FINAL-AUDIT.json)。

### 适用范围

这是合成同步突发：模型和连接已就绪，初始只有源端持有共享状态；当前Hygon导入仍要求空目标与源端暂停执行。首次五条通信连接初始化合计42.8秒，不能把后续复用连接的结果当成新worker从零上线的收益。节点28保留了原有空闲GPU上下文；六实例初次同形状冷prefill为7.33–7.37秒。

两次重复显示的收益不是统计显著性声明。此次没有实测多级缓存收益，也没有证明这种高扇出在普通AgentX负载中普遍存在。

## 3. 历史批次索引与证据状态

原始文件、负面结果和当时的解释保留在Release结果包中。下表给出当前阅读口径；历史文档的标题或旧结论不能覆盖这里的状态。

| 日期/批次 | 结果包中的路径 | 当前用途与限制 |
|---|---|---|
|9月28–30日Weka、容量和迁移成本实验|`experiments/weka_baselines/results/`、`experiments/weka_gravity/results/`、`experiments/weka_profile/results/`及根目录JSON|机制探索和审计历史；部分模拟、启发式baseline、计时边界及缺失值解释已被纠正|
|10月1–2日p03/p04/p05、同节点验证|`results/p03-e2e.json`、`p04b-cold-engine.json`、`p04c-same-node-cold.json`、`p05-length-sweep.json`、`decisive-same-node.json`|真实复制/冷启动机制数据；部分同名文件被历史运行覆盖，不把每次启动算独立样本|
|10月1–3日命令追溯|`docs/review/COMMAND-AUDIT-2026-10-01-03.zh.md`|命令、保留产物与统计口径核对；未找到可归属FluidKV的10月3日新增完整评测|
|10月9日就绪与偏斜|`results/skew-readiness-20261009/`|受控偏斜与准备窗口；不能当作原生AgentX热点发生率|
|10月10日路由/准备探索|`results/routing-space-20261010/`|较早观察适配层，作为方法演变记录；性能主结论采用10月12日修正比较|
|10月10日原始trace分析|`results/native-relocation-space-20261010/`|原始结构统计与声明放置假设下的候选空间；不等于真实GPU排队或迁移收益频率|
|10月11日状态准备|`results/state-preparation-20261011/`|保留64个正式trial及停止记录；采用过粗512-token缓存估计，不能作为最新公平对照|
|10月11日并发扫描及v2|`results/concurrency-sweep-20261011/`|包括传输失败、客户端限制与边界估计问题；表面18.4%优势已撤回，见`v2/pause-reason.json`|
|10月12日六实例|`results/six-worker-quick-20261012/`|保留初次门禁失败、旧DualMap超时及补充对照；有效集合由`v2/FINAL-REPORT.zh.md`及`p10_final_report.py`定义|

历史审阅：[9月30日证据审计](review/REVIEW-2026-09-30.zh.md)、[10月1–3日命令核对](review/COMMAND-AUDIT-2026-10-01-03.zh.md)、[Blitz比较](review/BLITZ-LMETRIC-COMPARISON-2026-10-10.zh.md)。历史文件中的相对结果链接在还原结果包后可用。

## 4. 多级状态准备方案（待实现）

HBM之外的命中会把部分计算转成恢复工作。多个请求可能共享一次准备，也可能争用同一SSD、源节点NIC或目标H2D。联合决策的对象是“状态准备动作及其消费者”，候选包括原地等待、跨层加载、GPU复制、重算和加入已在途准备。

| 组件 | 职责 | 当前基础/缺口 |
|---|---|---|
|全局状态与资源视图|真实驻留层、状态版本、合法可续算边界、在途准备、资源积压与容量|已有SnapshotDirectory及执行边界验证；完整多级观测和资源承诺视图待接入|
|恢复计划与代价评估器|为候选worker比较来源、路径、重算/恢复和依赖；估计就绪时间及干扰|已有成本测量和对照框架；动态多资源模型待实现|
|准备—路由—接纳协调器|合并共享准备、选目标与副本数、提交动作、确认就绪后放行、终态回滚|已有协议/运行时基础；完整联合策略未验证|

第一版估计关系：

`预计首token时刻 ≈ max(状态就绪时刻, GPU可接纳时刻, HBM容量就绪时刻) + 剩余prefill耗时`

各时刻应在加入本次承诺后的资源状态下估计。多段恢复可以流水，使用实际依赖的关键路径；重算消耗GPU，恢复占用HBM并可能影响其他状态。相同目标和状态的一次准备只计一份准备工作，每个消费者的后续计算分别计入。

快路径比较少量候选和已有准备；慢路径调整准备队列、副本和热状态驻留。两者共用承诺和终态。GPU-direct是候选数据路径，不能因CPU承担调度就排除DRAM路径。

### Fluxon集成与新颖性边界

已审阅Fluxon固定commit `cc800babb844092362d10369eacb6aa0685f1fde` 的部分协议源码，未部署和完成全仓审计。该版本已有GetStart/GetDone/GetRevoke、真实容量reservation和在途终态。复用这些能力并注明来源；候选评估需无副作用Probe，最终选择才创建真实承诺。详见[Fluxon审阅](review/FLUXON-IMPLICATIONS-2026-10-11.zh.md)。

动态搬运积压和影子记账受到讨论中第三方Dynamo/Fluxon博客的启发；原文无可核实公开URL，本仓库不转载全文，也不把该观察归为原创。

相关边界包括[Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)、[Dynamo分层命中权重](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/configuration-and-tuning)、[Cake的加载/计算协同](https://arxiv.org/abs/2410.03065)、[ObjectCache的共享带宽调度](https://arxiv.org/abs/2605.22850)。多层标签、快慢路径、single-flight及简单`min(加载,重算)`不能单独构成新颖性声明。

### 下一轮只验证三个问题

1. 固定缓存分布，增加真实NIC/H2D争用，最佳动作是否在等待、加载、重算之间反转？
2. 加入已有single-flight和正确等待机制后，共享准备/依赖接纳是否仍优于动态逐请求代价？
3. 同样的数据面、候选与观测下，联合选择worker/来源/准备/接纳是否优于分阶段选择，并改善全部流量的SLO goodput？

先验证HBM、本地DRAM和远端DRAM。只有SSD实际改变瓶颈或计划选择时再引入。若调好分层权重或普通请求合并就消除了差距，应止于工程集成。完整草案：[多级缓存idea留档](review/FLUIDKV-MULTITIER-IDEA-2026-10-12.zh.md)。

## 5. AgentX/Weka trace归档

项目中“AgentX trace”指所使用的公开数据集 **SemiAnalysis `cc-traces-weka-062126`**，不是本项目原创采集。上游为[Hugging Face数据集](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)，固定revision：`23f152f6f0f9399a85901b89a6458def0ef16729`，数据卡许可Apache-2.0。

- 393条conversation；56,798个主stream请求、42,029个子agent请求，共98,827个模型请求。
- 原始`traces.jsonl`为1,847,151,435 bytes，SHA-256：`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`，与上游公开文件一致。
- 内容是hash、长度、相对时间与agent/model元数据，不含原始prompt/response文本字段；全量schema已核对。
- hash ID只在对应conversation范围内解释。`in`是64-token block计数估计，不是真实tokenizer计数；保留上游关于长上下文过计数的说明。
- conversation相对时间不能直接拼成真实全局到达；API重叠不等于GPU排队；completion-to-arrival差值不自动等于工具执行时长。
- 原生同模型、共享≥32K前缀的同时活动stream最大为9，不能把32–192合成并发说成该corpus常态。

归档逐字节保留上游trace、README、统计与分布图，附Apache-2.0许可、来源说明、manifest和校验文件。Weka生成的replay token输入是根据hash构造的合成token，不是恢复了原始语义。

## 6. 下载与还原

[GitHub Release：fluidkv-archive-20261012](https://github.com/Ringssss/AS-autoscaling/releases/tag/fluidkv-archive-20261012)

- `agentx-weka-trace-062126.tar.zst`：公开trace及其许可/来源。
- `fluidkv-results-20260928-20261012.tar.zst`：近期完整结果目录、日志、失败记录、每次运行保留的源码快照及合成replay输入。
- `fluidkv-figures-20260930.tar.zst`：配套历史论文图，不代表新的有效结论。
- 同名`.sha256`与`.manifest.json`：压缩包校验及逐文件清单。[发布资产清单](artifacts/fluidkv-archive-20261012/assets.json)。

```bash
gh release download fluidkv-archive-20261012 \
  --repo Ringssss/AS-autoscaling \
  --pattern '*.tar.zst' --pattern '*.sha256' --pattern '*.manifest.json'
sha256sum -c ./*.sha256
tar --zstd -xf agentx-weka-trace-062126.tar.zst
```

结果包保留原目录结构，建议还原到独立checkout，再运行离线汇总：

```bash
git clone https://github.com/Ringssss/AS-autoscaling.git AS-autoscaling-archive
tar --zstd -xf fluidkv-results-20260928-20261012.tar.zst \
  --strip-components=1 -C AS-autoscaling-archive
tar --zstd -xf fluidkv-figures-20260930.tar.zst \
  --strip-components=1 -C AS-autoscaling-archive
cd AS-autoscaling-archive
python3 -m experiments.fluidkv.p10_final_report results/six-worker-quick-20261012/v2
```

运行历史GPU命令前需核对manifest中的路径、节点、端口、镜像、GPU数量和容器名；这些是实验时的配置，不是自动适配任意环境的部署入口。原始公开trace单独下载，不随`git clone`拉取大文件。

## 7. 对应源码与复现边界

- `experiments/fluidkv/p03_*`至`p10_*`：成本、路由、准备、并发与六实例对照及汇总。
- `experiments/fluidkv/{blitz_policy_bridge.rs,p08_upstream_routers.py,p10_cached_observations.py}`：原版策略适配与精确观测查询。
- `experiments/fluidkv/sota_backend/`：共同后端的forward-progress和checkpoint观测。
- `experiments/weka_baselines/environment/runtime/`：本轮Hygon SGLang runtime overlay；`experiments/hygon_runtime/`保留早期实现和构建记录，二者不能混用。
- `experiments/weka_{baselines,gravity,profile}/`：轨迹构造、历史实验与分析。
- `experiments/fluidkv/analyze_native_relocation_space.py`：原生trace结构分析；`scripts/archive_fluidkv_evidence.py`：打包及逐文件验证。

DualMap源码commit：`24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d`；Blitz源码commit：`a005ceb3a6689d30514bd27e45ede0db933d0fea`。它们是外部依赖，不随此提交伪装为本项目原创代码。每个历史manifest中的源码hash和冻结副本优先于当前脚本；后续维护修复不追溯改变已跑出的结果。

本轮只发布FluidKV相关代码、技术记录、实验产物与公开数据。模型权重、checkpoint产物、私有聊天会话及独立FluidCKPT项目不在此次归档中。
