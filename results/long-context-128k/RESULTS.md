# AgentShift 128K-Class Results

## Scope

本实验验证 AgentShift 是否能迁移接近 128K 的 completed prefix。Qwen3-8B 和
Qwen3-32B 的原生上下文为 32,768 tokens；服务端使用官方 YaRN 4x 配置将上下文
扩展到 131,072。每个样本输入 130,000 tokens，第一轮再生成 4 tokens，因此实际
迁移和目标命中的 prefix 均为 130,004 tokens。

硬件为单机 8x NVIDIA H100 80GB HBM3：

| Model | Parallelism | Source | Destination | KV moved |
| --- | ---: | --- | --- | ---: |
| Qwen3-8B | TP=1 | GPU 0 | GPU 1 | 17.85 GiB |
| Qwen3-32B | TP=4 | GPU 0-3 | GPU 4-7 | 31.74 GiB total, 7.94 GiB/rank |

每个点重复 3 次。表中报告 post-tool latency 的均值。Sticky 保留源端 placement；
Reroute 在目标端重新 prefill；On-return 在工具返回后迁移；AgentShift 在工具阻塞
期间迁移并在目标 ready 后提交 owner handoff。

## Qwen3-8B TP=1

| Tool gap | Sticky | Reroute | On-return | AgentShift | AgentShift vs reroute | AgentShift vs on-return |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 86.42 ms | 11.049 s | 330.08 ms | 308.12 ms | 35.86x | 1.07x |
| 100 ms | 81.94 ms | 11.058 s | 331.38 ms | 245.33 ms | 45.07x | 1.35x |
| 250 ms | 83.31 ms | 11.052 s | 335.29 ms | 83.54 ms | 132.30x | 4.01x |
| 500 ms | 84.73 ms | 11.040 s | 344.04 ms | 84.41 ms | 130.79x | 4.08x |
| 1000 ms | 84.76 ms | 11.037 s | 338.34 ms | 90.88 ms | 121.45x | 3.72x |

- Reroute calibration median: 11.096 s; Sticky calibration median: 81.81 ms.
- Net 130K re-prefill cost: 11.015 s.
- Smoke worker copy: 148.6 ms; protocol migration: 191.9 ms; aggregate
  throughput: 93.05 GiB/s.
- AgentShift and On-return both achieved a 100% full-prefix-hit rate.
- A 250 ms gap reduced exposed migration to approximately zero and returned latency to
  the Sticky floor.

## Qwen3-32B TP=4

| Tool gap | Sticky | Reroute | On-return | AgentShift | AgentShift vs reroute | AgentShift vs on-return |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 157.18 ms | 11.849 s | 377.32 ms | 354.79 ms | 33.40x | 1.06x |
| 100 ms | 150.96 ms | 11.859 s | 384.23 ms | 290.93 ms | 40.76x | 1.32x |
| 250 ms | 151.40 ms | 11.837 s | 373.06 ms | 156.69 ms | 75.54x | 2.38x |
| 500 ms | 156.78 ms | 11.862 s | 369.50 ms | 153.30 ms | 77.38x | 2.41x |
| 1000 ms | 156.50 ms | 11.864 s | 383.32 ms | 153.15 ms | 77.47x | 2.50x |

- Reroute calibration median: 11.888 s; Sticky calibration median: 159.91 ms.
- Net 130K re-prefill cost: 11.728 s.
- Smoke worker copy: 132.9 ms; protocol migration: 183.7 ms; aggregate
  throughput: 172.81 GiB/s across four TP ranks.
- The smoke ended in `SOURCE_RELEASED`, advanced owner epoch from 1 to 2, and
  installed all 130,004 tokens at the destination.
- Every AgentShift and On-return sample achieved a 100% full-prefix-hit rate.
- At gaps of 250 ms or longer, AgentShift was within measurement noise of Sticky.

## Interpretation

The 128K result validates that AgentShift mobility is not limited to 32K prefixes.
Longer context increases the value of mobility because stateless rerouting pays an
approximately 11-12 second historical prefill in these configurations. It also increases
the state footprint: one suspended 128K agent occupies 17.85 GiB for 8B TP=1 or
31.74 GiB total for 32B TP=4. AgentShift therefore needs admission and concurrency
control; longer prefixes are not unconditionally better.

The key result is conditional: when future placement must move to another engine,
AgentShift preserves the full prefix hit and, with a 250 ms tool gap, makes the next
turn execute at Sticky-like latency. Sticky itself remains the latency floor but does not
change placement or relieve the source engine.

The zero-gap proactive samples contain an observed 16-18 ms asyncio scheduling interval.
They should be interpreted as a no-overlap sanity check, not evidence of an intentional
tool gap.

## Artifacts

- `blocked-window-1785136832475469481.json`: Qwen3-8B TP=1 raw records.
- `blocked-window-1785138109761090702.json`: Qwen3-32B TP=4 raw records.
- `qwen3-8b-tp1-128k-summary.{json,csv,md}`: generated 8B summaries.
- `qwen3-32b-tp4-128k-summary.{json,csv,md}`: generated 32B summaries.
- `smoke-32b-128k-1785137060479500117.db`: durable 32B smoke protocol state.
