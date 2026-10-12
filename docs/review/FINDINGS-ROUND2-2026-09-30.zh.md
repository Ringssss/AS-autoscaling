# FluidKV — 实验定稿与本轮实跑结论

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

2026-09-30 第二轮。本文件是本轮**实跑结果**的收口，取代此前的口径冲突。
每一步都注明是 implemented / observed / estimated / proposed。

配套：[论证链与实验矩阵](STORYLINE-v3-2026-09-30.zh.md)、
[接手审计](REVIEW-2026-09-30.zh.md)。

---

## 0. 一句话

> **Agentic 负载的状态局部性使请求放置受缓存位置约束。路由可以转移计算，
> 却让后续请求支付昂贵的状态重建，且从不评估目标有无空位容纳。
> 本轮把"可运行前缀"变成可搬运对象的过程推到了实测阶段，结果是：
> 机制成立、字节无误，但当前实现的导入路径写错了树的形状，
> 使首个消费者拿不到收益。**

---

## 1. 本轮实跑的六个结果

### 1.1 P0-1 容量配对 2×2（已闭合）

设计：单引擎、引擎角色跨 rep 轮换、两臂驻留前缀数**相同**（起始空闲槽一致）、
4 reps/格、请求前后逐对象 `inspect`、读引擎 `num_retractions`。
产物：`results/capacity-2x2.json`、`paper/figs/fig-capacity-2x2.pdf`。

| 条件 | TTFT 中位 (s) | 区间 | 空闲槽（后） | 邻居住在 |
|---|---|---|---|---|
| cold，引擎空 | 3.008 | 3.006–5.590 | 229,376 | — |
| hit，引擎空 | 0.623 | 0.618–0.624 | 229,376 | — |
| **hit，引擎满载** | **0.623** | 0.620–0.628 | 32,768 | **7/7** |
| **miss，引擎满载** | **3.003** | 2.998–3.008 | **4,096** | **6/7** |

**观察（observed）：**
- 两臂起始空闲槽完全相同（32,768），唯一差异是前缀是否已驻留。
- 命中不驱逐任何邻居；未命中**恰好驱逐 1 个**等长驻留前缀
  （28,672 槽，4/4 次掉的是同一个 slot）。
- `num_retractions` 恒为 0 → **该字段是单请求被退回等待队列的次数，
  不是缓存驱逐计数**（源码 `io_struct.py:1125` 注释确认）。

**被推翻（本轮）：** "仅内存压力使 prefill 膨胀 2.4×（3.96s→9.54s）"。
`empty/miss` 与 `full/miss` 在配对设计下无可测差异（3.008 vs 3.003）。
旧数字来自三个不同引擎、不同 token 区域、free 在请求前后错时比较。
**不得再引用。**

### 1.2 P0-2 同终点三动作（本轮最重要的负面结果）

设计：同一引擎、同一 130,048 token 前缀、三条路都测到**首输出事件**。
产物：`results/action-compare-130k.json`、`paper/figs/fig-timeline.pdf`。

| 动作 | TTFT 中位 (s) | 两次实测 |
|---|---|---|
| cold（目标重算） | 19.540 | 19.42 / 19.66 |
| **local（源已驻留）** | **1.402** | 1.11 / 1.69 |
| **import（搬运后服务）** | **19.394** | 19.50 / 19.29 |

**cold / import = 1.01×。搬运在当前实现下没有服务收益**（根因见 E7）。

> ⚠ **本节是全节最重要的历史记录，但结论已被 §1.3 的修复取代。**
> 下表是**修复前**的实测，用于说明缺陷的存在；修复后同终点对比为
> cold 19.258s / import 0.591s，**cold/import ≈ 32.6×**。
> 引用时必须标明是哪一版实现。

### 1.3 根因已定位（五处代码证据 + 实测对照）

隔离复现（`28:31751 → 29:31761`，避开一切并发实验）：

```
转移 0.59s、bytes 2,727,772,160、state COMPLETE（两端一致）
转移后立刻 inspect ： matched=130048, full=True, mamba=True   ← 已发布
在该目标上服务该前缀： 79.40s，cached_tokens=0
```

⚠ **但 79.40s 不能归因于导入。** 同一 130,048 token 长度上，
冷 prefill 耗时随**引擎**分成三档，且每档各自高度可复现：

| 引擎 | 130K 冷 prefill | 该节点的卡占用 |
|---|---|---|
| `node27:31782` | 19.28 / 19.42 / 19.66 s | card4,5 空闲（9 MiB） |
| `node28:31751` | 27.98 / 28.06 s | 部分占用 |
| `node29:31761` | 78.71 / 79.30 / 80.79 s | 八卡全占（合计 377 GB） |

所以 `import130` 里的 79.40s 对应的是**被占满的 node29**，
而 `action-compare` 里 19.39s 的 import 对应的是**独占的 node27**。
两者之差是**引擎/争用差异，不是导入造成的惩罚**。
⇒ **跨引擎的绝对延迟不可直接比较**；论文只能用同一引擎内的配对。

决定性 A/B（`bounds-compare2.log`，同一引擎同一 token）：

| 路径 | snapshot_boundaries | state_sha256 | serve#1 |
|---|---|---|---|
| native 物化 | **32** 条（每 4096 一条） | `dc930ce8…` | — |
| 导入 | **1** 条（`[130048]`） | `dc930ce8…` | `cached_tokens=0` |

**字节搬运完全正确（sha 一致），错的是写入树的形状。**

因果链（经独立代码审计复核，link 1–5 全部 CONFIRMED）：

1. 请求路径把匹配键截掉最后一个 token：
   `max_prefix_len = input_len - 1`（`schedule_batch.py:1028`，
   注释 *"to enable logprob computation"*），**无条件生效**。
2. 导入只写**一个节点**，快照仅挂末端（`_insert_helper` 每次调用只建一个
   `new_node`；目标树为空使 while 循环不执行）。
3. 130047 键是 130048 节点的**严格前缀**：`prefix_len(130047) < len(child.key)(130048)`
   → 触发 `_split_node`。
4. `_split_node` 设 `new_node.mamba_value = None`
   （`mamba_radix_cache.py:1085`，*"mamba cache can not be split"*）；
   原带快照的节点降为 1-token 子节点，循环随即 `break`，**再不会被访问**。
5. → `best_value_len = 0` → `device_indices` 为空 → 全量 prefill。

⭐ **归因必须精确化，不能写成"键短一 token 所以坏了"。**
那个 N−1 截断**普适且本身无害**：native 树同样被截断、同样拆尾部节点。
真正的判据是**快照拓扑**：

- native：32 个快照沿 32 节点链分布 → 尾部被拆后，前面 31 个带快照的祖先
  **仍在匹配路径上** → `best_value_len ≈ 31` → 大量复用。
- import：唯一快照在唯一叶子上 → 尾部一拆，**唯一快照节点被孤立** → `best_value_len = 0`。

准确表述：**"单叶快照 + 强制的 N−1 尾部拆分孤立了唯一快照节点"**。

⚠ **一个竞争解释尚未排除（Rank 1，优先级最高）：**
复用粒度可能由 mamba 分支记账决定，而非节点数。证据：
`serve#2 = 129984 = 2031×64 = n−64`，即 `floor((n−1)/64)×64`
（`mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, page_size) = 64`），
而 **native 温命中是 28672 = n−4096**（`chunked_prefill_size`）。
若该解释成立，"改成写 32 节点链"只会把可达边界从"无"挪到 **n−4096**，
**并不能消除首个消费者惩罚**。
**判别**：导入后服务一个 `n+1` token 的请求（截断后键恰为 n），
看复用落在 `n` 还是 `floor((n−1)/64)×64`。
→ `experiments/fluidkv/discriminate_import.py`

⚠ **另一个待排除项（Rank 2）：** `inspect` 的顶层 `snapshot_boundaries`
是**跨 TP rank 的交集**（`handle()` 里做 `set.intersection`）。
native 侧实测 rank0/rank1 都是 8 条、交集也是 8，所以 native 不受影响；
但**导入侧的 per-rank 值尚未读到**。若任一 rank > 1，则"单节点"是上报假象。


**不是永久失效，而是"首个消费者惩罚"：** 32K 上实测
`serve#1 = 9.47s, cached=0` → `serve#2 = 0.38s, cached=32704`；
引擎侧日志独立确认（`#cached-token` 序列 `0,…,0 → 32704`）。

**仪器有效性对照（同 build、同引擎、同一 run）：** 本地冷服务
`cached_tokens=0`；本地温服务 `cached_tokens=28672`。
所以这不是计数器失效，是**真阴性**。

**修复方向（已设计到可实施，改动规格来自独立代码审计）：**

目标：让导入端按源端的 `snapshot_boundaries` 写**链式节点**，而不是单节点。
改动落在 `hygon_hybrid_control.py` 四处：

| # | 位置 | 改动 |
|---|---|---|
| 0 | 新增同级方法 `boundary_values(tokens)` | 只读遍历，返回每个边界的 `(offset, child.mamba_value)` 张量，而非只有 offset |
| 1 | ~179（源端） | 用 `boundary_values` 枚举全部 N 个边界，`mamba = torch.cat([...])` 得到 K 个槽，并把边界计划发布到 store |
| 2 | ~184（目标端） | 从源端计划读 K，一次 `mamba_pool.alloc(K)`（**已静态核实支持**：`free_slots[:need_size]` 返回 K-tensor，`free` 接受 tensor） |
| 3 | ~204（广播循环） | **无需改动**：`slots = indices if axis==0 else mamba`、`shape[axis] = len(slots)` 已泛化到 K 宽 |
| 4 | ~221（insert） | 改为**按递增前缀长度多次 insert**；每次带 1 元素 mamba 切片，并把 `prev_prefix_len` 设为上一已装边界，使 `_insert_helper` 的释放逻辑不触发 |
| 5 | ~238（finally） | 部分失败时先 `tree_cache.reset()` 拆掉半装链条，再显式释放 `indices` 与全部 K 个 mamba 槽 |

**关键约束（审计发现）：** `insert()` 单独调用**无法**建链——`_insert_helper`
每次只建一个 `new_node`，所以必须循环 N 次。
实现后应验证：导入侧 `snapshot_boundaries` 从 **1 变为 32**，
且 `serve#1` 的 `cached_tokens` 从 **0 变为 ≈126976**（而非仍为 0）。
`state_sha256` 应**保持不变**（末端槽未动）——若变了说明传错了槽。

**修复已实施并验证（2026-09-30，`verify-chained-import.log`）：**

补丁落在 `experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py`：
新增 `boundary_values()`（只读遍历，返回每个边界的 mamba 张量）；
源端枚举全部边界并 `torch.cat` 成 K 槽、同时把边界计划发布到 store；
目标端读计划后一次 `alloc(K)`；
insert 改为**按递增前缀长度循环 N 次**，每次带 1 元素 mamba 切片并设
`prev_prefix_len` 抑制内部释放；部分失败时先 `tree_cache.reset()` 再释放池槽。

验证（32K 前缀，`27:31791 → 28:31791`，跨节点）：

| 断言 | 修复前 | 修复后 | 判定 |
|---|---|---|---|
| A. 导入树边界数 | 1 | **8**（= native 的 8） | ✅ PASS |
| B. 首个消费者 `cached_tokens` | **0** | **28672** | ✅ PASS |
| B'. 首次服务耗时 | 3.262s（全量 prefill） | **0.361s** | 7.4× 改善 |
| C. 跨引擎输出一致 | — | 不一致 | ⚠ 见下 |

**A 与 B 是决定性的**：导入树从单节点变为与 native 同构的链，
首个消费者从"零复用"变为复用 28672（= n−4096，正是 native 的粒度）。

⚠ **C 的失败暂不作为损坏证据**：本 build 已知对 batch 形状敏感、
greedy 输出不保证逐 token 相同，而 C 是**跨引擎**比对（源端末次 serve
vs 目标端首次 serve），batch 形态本不相同。需**同引擎重复服务**的对照
才能判定 C 是否有意义。

⇒ **结论：链式导入修复有效，且效果是决定性的。**

**130K 上的复核（`results/verify-130k.log`）：**

| 断言 | 修复前 | 修复后 | 判定 |
|---|---|---|---|
| A. 导入树边界数 | 1 | **33**（= native 的 33） | ✅ PASS |
| B. 首个消费者 `cached_tokens` | **0** | **129984** | ✅ PASS |

**同终点三动作（130,048 token，跨节点 `27:31791 → 28:31791`）：**

| 动作 | TTFT | 说明 |
|---|---|---|
| cold（目标重算） | 19.258s | — |
| import（搬运后服务） | **0.591s** | 与 local 等价 |
| **cold / import** | **≈32.6×** | 修复前是 1.01× |

（修正：单次测量给出的 0.597s 与 local 的 1.089s 之差是顺序假象，
见下。稳态值 import 0.591s、local 0.594s。）

⚠ **"import 快于 local" 已排除，是顺序假象。**
交换测量顺序后（`results/order-check.log`）：

| 顺序 | 第 1 次服务 | 第 2 次服务 |
|---|---|---|
| import | **1.095s** | **0.591s** |
| local | **1.095s** | **0.594s** |

两者逐格相同，差异**跟着测量顺序走而不是跟着机制走**。
真正的结论是：**import 与 local 等价**（0.59s 对 0.59s），
两者都比 cold（19.258s）快约 **32×**。这正是应得的结果——
搬运的全部价值就是让远端达到本地命中的水平。

**33 条边界的来源已核实：** 列表为 31 个 4096 网格点
（4096…126976）加两个非网格点 **129984** 与 **130048**。
其中 `129984 = n − 64`，正是 `FLA_CHUNK_SIZE` 对齐点。
**引用时应给出 31+2 的构成，不要只写"32 条"。**



### 1.4 P1-1 eligibility 漏斗（离线，真判据）

产物：`results/eligibility-funnel.json`、`paper/figs/fig-eligibility-funnel.pdf`。

| 阶段 | 计数 | 占比 |
|---|---|---|
| 到达 | 1,152 | 100.0% |
| 有可复用前缀 | 799 | 69.4% |
| 高于 L\* | 799 | 69.4% |
| 前缀在别处 | 65 | **5.6%** |
| 容量可行 | 65 | 5.6% |
| 净收益为正 | 65 | 5.6% |

**主损失是 `stays_on_home_engine`（734 条，63.7%）**：sticky 下前缀在本地，
根本不需要搬运。中位节省 5.86s/请求。
⇒ 这个 5.6% 取代旧的"4.8pp"作为**可行动覆盖面**。

### 1.5 group init 的真实成本与摊销

| 项目 | 实测 |
|---|---|
| 裸建链（5 次，同节点） | 16.3 / 21.1 / 21.2 / 21.5 / 31.3 s，5/5 成功 |
| 同一 group 上连续三次迁移 | 0.62 / 0.12 / 0.12 s，均 `dst_verified=True` |
| 重复组名再 init | 被拒（`group name already in use`） |

⇒ 正确成本模型是三段：
`C_ready = init_once(pair) + k · per_transfer + H_dst(残余)`，
`per_transfer ≈ 0.12–0.6s`，`init_once ≈ 16–31s`。
协调器 `_ensure_group` 按 pair 短路，与实测一致。

### 1.6 快照粒度

`32,768` 前缀 → **8** 条边界；`130,048` → **32** 条；每 **4096** 一条，
等于 `chunked-prefill-size`，**不是** `FLA_CHUNK_SIZE`（= 64）。
⇒ 可迁移边界只能落在 4096 的倍数上；token LCP 与可运行边界最多差 4095 token。

---

## 2. 本轮确立 / 推翻 / 未决

**确立：** 容量配对效应（命中不驱逐、未命中驱逐恰好 1 个）；
import 无服务收益及其根因；eligibility 5.6%；init 摊销性质；快照粒度。

**推翻：** "内存压力使 prefill 膨胀 2.4×"；
"路由不看容量"（Preble 已把驱逐重算计入 load cost，CacheRoute 已有 slot 预算）；
"copy 与长度无关"（应为"增长显著慢于重建"，payload 涨 31.75× 时间涨约 2.1×）。

**未决：**
- P0-2 第 3 格仍未补。多次尝试全部败于 group init：
  `DistStoreError: Timed out after 91 seconds waiting for clients. 1/2 clients joined.`
  与 `DistNetworkError: The client socket has timed out after 90000ms`。
  这是**端口被历次失败组泄漏**所致（33000–47000 已占 651 个监听端口），
  不是机制问题。已记录的 2 格（cold 19.54 / local 1.40 / import 19.39）
  方向一致且足以支撑 E7 的结论。
- ⭐ **Net-of-init 是审稿人最可能的一击，且数字不好看。**
  eligibility 漏斗的 5.86s 中位节省是**毛值**，未扣建链成本。
  用实测 `init_once`（10 次样本，中位 **21.5s**，范围 10.9–31.3s）重算：

  | 拓扑 | 每 pair 迁移次数 | init 总开销 | 净收益 | 判定 |
  |---|---|---|---|---|
  | 3 引擎 / 3 pair | 1 | 64.5s | +524.9s | 净正 |
  | 3 引擎 / 3 pair | 3 | 193.5s | +395.9s | 净正 |
  | 6 引擎 / 15 pair | 1 | 322.5s | +266.9s | 净正 |
  | **6 引擎 / 15 pair** | **2** | **645.0s** | **−55.6s** | **净负** |
  | **6 引擎 / 15 pair** | **3** | **967.5s** | **−378.1s** | **净负** |

  **盈亏平衡 = 9.14 次迁移/pair（3 引擎）、1.83 次（6 引擎）。**
  ⇒ 在 6 引擎拓扑上，只要平均每对迁移不到 2 次，**建链成本就吃掉全部节省**。
  这个数必须在正文里给出，且必须说明它取决于"每对复用多少次"，
  而不是取决于单次迁移有多快。
- **`29:31780` 引擎已崩溃**（栈回溯 + `Health check failed. Server couldn't get
  a response from detokenizer`），不再可用。
- P0-3（同 router ± **真 copy** 的 E2E）尚未开始；
  `real_compare.py` 里**没有**任何 `direct_transfer` 路径（`grep -c` = 0），
  真 copy 的模板在 `one_hour.py:188` 的
  `fleet.coordinators[source,2].migrate(agent,'destination')`。
  设计已出（见 workflow 产出），但**前置条件是先修 import**。
- 跨引擎绝对延迟不可比：同一 130K 前缀上 `node27` 冷启动 19.3s、
  `node28` 28.0s、`node29` 78–81s（该节点八卡合计占用 377 GB）。
  **只能同引擎内配对**。
- 已崩引擎与脏引擎需在建组前 flush 或重建容器。

---

## 3. 对论文主张的影响

1. **不能声称端到端加速。** 当前实现的导入路径令首个消费者不复用，
   而"搬运后立刻服务"正是最相关的情形。
2. **可以主张机制与诊断：** 完整边界发现、双向验证协议、
   "字节无误但形状错误"这一具体缺陷及其修复方向，
   以及 5.6% 的可行动覆盖面。
3. **应当把 §2 的动机从"容量压力"改为"容量是绑定资源且未命中驱逐等长前缀"**，
   因为压力膨胀效应未复现。

---

## 4. 图件（`paper/figs/`，字体统一为无衬线、metric 兼容）

| 图 | 文件 | 内容 |
|---|---|---|
| F-capacity-2x2 | `fig-capacity-2x2.pdf` | 四格 TTFT + 容量 + 存活（P0-1）|
| F-funnel | `fig-eligibility-funnel.pdf` | 5.6% 漏斗（P1-1）|
| F-timeline | `fig-timeline.pdf` | stay/cold/import 三终点（P0-2）|
| F-tradeoff | `fig-tradeoff.pdf` | 交换比 vs 长度，四条线 |
| F-capacity-vs-hit | `fig-capacity-vs-hit.pdf` | 旧三条件图，保留作历史 |

生成脚本：`experiments/fluidkv/{plot_capacity_2x2,plot_eligibility_funnel,
plot_action_timeline,plot_exchange_tradeoff}.py`，共用 `figstyle.py`。
