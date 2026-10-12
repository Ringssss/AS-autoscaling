# 2026-10-01—03 FluidKV 命令与产物核对

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

核对日期：2026-10-09。全部时间使用 Asia/Shanghai。仅查阅本地历史、源码和产物；未执行以下 GPU 命令。

主要命令来源：`/home/zhujianian/.claude/projects/-home-zhujianian/4e82260a-996a-424d-9ede-2328ca003740.jsonl` 的 Bash tool_use。Codex 10 月 2 日审查记录在 `/home/zhujianian/.codex/sessions/2026/10/02/rollout-2026-10-02T13-58-47-01a0fb31-9b87-73f2-a9a7-3ed79842ebfc.jsonl`。

历史工具调用表明尝试发起命令；成功与结果另由日志和 JSON 核对。

## 找到的运行

| 时间 | 程序/动作 | 可查产物 |
|---|---|---|
| 10-01 01:39:38 | init_cost_decisive.py | results/init-cost-decisive.json、results/init-decisive.log |
| 10-01 01:53—02:23 | 临时端口、warmup、交错建链、clean init、连接复用探针 | port-range.log、warmup.log、final-init.log、clean-same.log、reuse.log 等；部分临时脚本已删除 |
| 10-01 02:30—12:18 | p03_real_copy_e2e.py 多次启动；8 streams 改成 6 streams | results/p03-e2e.json、results/p03-e2e.log，被多次同名覆盖 |
| 10-01 13:34:28 | launch_fleet.py --tag fk | experiments/fluidkv/results/fleet-fk.json |
| 10-01 13:53:37 | p04_cold_engine.py --streams 8 --hot-frac 0.5 --n-hot 4 --reps 2 | results/p04.log；当前未发现对应完整 p04-cold-engine.json |
| 10-01 14:15、14:21、14:26 | p05_length_sweep.py；16K/32K/64K，每格 3 reps | results/p05-length-sweep.json、results/p05.log，同名覆盖 |
| 10-01 14:44:46 | p04b_cold_engine_prewarmed.py；8 streams、n_hot=2、2 reps | results/p04b-cold-engine.json、results/p04b.log |
| 10-02 02:51:11 | decisive_same_node.py --reps 3 | results/decisive-same-node.json、results/decisive.log |
| 10-02 02:59:04 | p04c_same_node_cold.py；10 streams、n_hot=1、1 rep | results/p04c-same-node-cold.json、results/p04c.log |
| 10-02 10:56:33 | probe_inspect_cost.py | results/inspect-cost.log：安装探针时 RuntimeError: no output，未完成 idle/busy 对照 |
| 10-02 11:02 起 | 清理容器和残留进程 | 会话工具记录；清理不是评测结果 |

## 关键命令（去掉 nohup、等待、轮询和重定向）

解释器：`/home/zhujianian/miniconda3/envs/sglang/bin/python3`，工作目录 `/home/zhujianian/AS-autoscaling`。

```bash
python3 experiments/fluidkv/p03_real_copy_e2e.py --streams 6 --hot-frac 0.5 --arms sticky least_loaded balance_move --out results/p03-e2e.json
python3 experiments/fluidkv/p04b_cold_engine_prewarmed.py --streams 8 --hot-frac 0.5 --n-hot 2 --reps 2 --out results/p04b-cold-engine.json
python3 experiments/fluidkv/p05_length_sweep.py --lengths 16384 32768 65536 --reps 3 --out results/p05-length-sweep.json
python3 experiments/fluidkv/decisive_same_node.py --reps 3 --out results/decisive-same-node.json
python3 experiments/fluidkv/p04c_same_node_cold.py --streams 10 --hot-frac 0.6 --n-hot 1 --reps 1 --out results/p04c-same-node-cold.json
```

## 实际部署版本

`launch_fleet.py` 挂载 `experiments/weka_baselines/environment/runtime/`，不直接挂载 `/home/zhujianian/hygon-runtime-overlay/`。

当前保留的 hybrid 控制源码 SHA256：

```
experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py
dc960414b21f507a5fb8b41fb0c75e78f0d64f659ad127e770a5920c0dedf096

/home/zhujianian/hygon-runtime-overlay/managers/hygon_hybrid_control.py
7ba704a8807c67a24a68d306d7f3be07f74818b4d91099ef8dcb61825ccb1045
```

实际部署副本包含独立控制 CPU group、整条 Mamba 快照链传输和链式导入。两端空闲、目标 root 无 children 的检查仍然存在。文件 hash 说明当前保留版本；历史远端已清理，不能重新读取容器内文件来独立证明每次尝试都用了同一版本。

## 数据与时间归属

- p03 当前完整文件为 6 streams、54 请求；不能把所有历史 p03 启动都当成保留的独立重复样本。
- p04b 当前文件为 8 streams、72 请求；之前总结中的 90 请求不准确。
- p04c 当前文件为 10 streams、90 请求。
- decisive 的 17.43x 口径仅计算迁移后的 serve TTFT；计入 copy_s 后，ratio of medians 为 8.560862，仍不包含 inspect 或建链。
- progressive 的 `real-tools-all-1786344429892748638.json` 有 40 条原始记录，文件名 epoch 对应 2026-08-10 14:47:09 +08:00；相关 git 提交日期为 2026-08-11。中位数 agentshift=0.136502023s、adaptive-progressive=0.122498885s。这是较早 Qwen3-8B TP2 路径的数据，不是 10 月的 DCU hybrid 数据。
- 检索到的 10 月 3 日 Codex 会话和相关 Claude 历史主要属于 FluidCKPT 权重保活、以及 PegaInfer/Kern 讨论；此次未找到可归属 FluidKV 的新增完整评测。

这些区别决定统计单位、可支持的结论和机制归因。
