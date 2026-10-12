> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Kimi K2.5真实trace：FluidKV机会与可验证边界

2026-09-23完成全量检查。**这份trace显示了长输入和高到达率时段的优化机会，但不足以识别FluidKV相对DualMap/CacheRoute的真实收益。此前受控接管实验的6–10倍TTFT改善，不能作为这份trace的回放结果。** 本轮没有用合成会话、前缀身份或工具时间冒充原生Kimi策略回放，也没有新增GPU性能测量。

## 数据与检查

原始release包含24个小时文件、245,555条Finish记录。归档SHA-256为
`e7d8fd50757208220b9bc5bbd111a9d491b94a89a6c42390ab345aab9a0c3513`。
本轮重新扫描原始文件，核对24个成员文件哈希，并逐条核对先前规范化的245,555行。额外检查了身份字段、原始prompt_tokens、等待时间和bootstrap元数据。9项相关测试通过；8个群体的条件预算已从保存数组独立复算。

这是按完成日志收集的请求群体，到达跨度约24.97小时。它不保证覆盖原服务所有未完成、失败或窗口外完成的请求，不能据此计算原服务的完整成功率。Finish reason中记录stop 236,574条、length 8,980条、abort 1条，均保留。

## 长输入存在，但不等于已有长历史可搬

下表百分比统一以全部245,555条记录为分母，K=1024 tokens。

| 阈值 | 输入长度达到阈值 | 占全部请求 | 观测缓存命中达到阈值 | 占全部请求 |
|---|---:|---:|---:|---:|
| 8K | 121,729 | 49.573% | 14,301 | 5.824% |
| 16K | 71,888 | 29.276% | 8,599 | 3.502% |
| 32K | 30,736 | 12.517% | 4,357 | 1.774% |
| 64K | 10,549 | 4.296% | 1,435 | 0.584% |

有199,471条、即**81.23%**的全部请求，记录的缓存命中落在5,440–5,888 tokens。即使只看≥64K输入，缓存命中中位数仍为**5,760 tokens**；该群体缓存命中比例中位数约7.00%。≥32K输入群体的这一比例中位数约12.21%。

长度相同不能证明前缀相同，更不能用cached_tokens当作会话键或prefix hash。以上缓存列只是历史服务路径实际记录的命中数，不证明它是私有agent历史、完整可执行状态或另一目标可接收的状态包。

**0.584%不是FluidKV迁移机会发生率，也不是其全局上限。** 未命中的长输入可能全是新内容，也可能在其他副本已有完整状态；这份trace无法区分。反过来，已命中64K也不表示必须迁移，或有冷目标和可隐藏的工具等待。

8,455条记录出现cached_tokens大于输入长度，且输入/输出均为1。原始meta_info.prompt_tokens也全部为1，不能用另一个长度字段修复；未将它们称为健康检查，未截断缓存计数。这些记录保留在到达/长度/延迟统计中，从合理缓存计数群体排除。有效缓存计数群体为237,100条；若用这个分母，≥64K缓存命中占约0.605%，≥32K约1.838%。有效群体token加权缓存命中率约40.71%。

## 长请求和高负载时段确实值得研究

≥32K输入占约**46.09%输入token**和**33.59%累计TTFT**；≥64K输入占约23.18%输入token和16.12%累计TTFT。这说明长请求不是数量上的主体，却占据相当大的可见请求成本。累计TTFT是请求等待时间之和，不是GPU计算时间或墙钟时间。

按每分钟观测到达数分组：上四分位时段的TTFT中位数约**2.158秒**，中间时段约**0.660秒**。输入长度中位数也从约6,952变为10,041，故不能把差异全部归因于排队或routing。最高一分钟记录389个请求；观测请求生命周期的重叠峰值为266，既不是实际GPU batch size，也不证明存在空闲目标。

原trace的TTFT中位数约1.123秒，E2E中位数约9.266秒，输出长度中位数**350 tokens**；约**82.47%**请求输出超过64 tokens。此前的64-token受控突发实验不能直接代表这份输出分布。TTFT之和约占E2E之和23.80%，该比例也不能当作GPU耗时分解或稳态吞吐上限。

## 为什么无法识别真实策略收益

全量检查发现：

- input/output token IDs全部被脱敏；无block hash或可验证前缀身份。
- conversation_id、routing_key、extra_key、session_params、bootstrap_pair_key均无非空值。
- custom_labels、external_trace_header也全部为空，未找到可替代的稳定会话/agent身份或父子turn关系。
- 没有真实tool开始/结束或下一turn依赖就绪信息；请求间隔不能当作tool时间。
- queue_time全部为0，prefill_waiting_latency和prefill_launch_latency全部为空。这不能证明没有排队，只能说明这些字段不足以归因源/目标等待。
- API dispatch时间相对接收时间的差值中位数约0.144秒、p95约1.533秒，不能未经定义直接当作GPU队列时间。
- inference_time数值约为94万–220万，不被当作耗时；其量级接近单调时钟时间戳，需要运行时定义才能解释。
- 观察到16种bootstrap地址、8个DP rank。bootstrap地址不等于状态owner或完整模型副本ID；地址出现/消失也不证明扩缩容，无法恢复实际可路由集合或各引擎缓存分布。

例如，一个80K输入、命中5.8K的请求，可以是首次提交的新文档；也可以是已有私有历史被发到冷目标，原状态仍在另一个副本。两者在这份日志中的字段可以相同，FluidKV的价值却完全不同。没有身份、位置和依赖时间，无法判定是哪一种，更不能恢复两种router的原生前缀键与CacheRoute的业务键频率。

## 能说什么，不能说什么

**可支持的判断：** 真实流量包含昂贵的长输入和明显时段变化，研究避免丢失已有状态有现实动机。应该重点找“长历史已经存在，但接管/重分配让可用执行位置变冷”的事件，而不只是筛选长输入或高TTFT。

**尚不能支持：** FluidKV在这份原始Kimi流量上相对两种routing的TTFT、吞吐、SLO或GPU成本改善了多少；该接管条件在真实流量中的发生率；把6–10倍接管收益外推到全天。也不能据此断言FluidKV无效，未观测到的跨副本状态机会同样可能存在。

旧的[Kimi/LooGLE/ToolBench回放](../hybrid_state_mobility/TRACE_REPLAY.md)有128个构造agent和442个事件，加入了替代文本、构造turn关系及其他实验的tool时长。它明确属于trace-shaped replay，可检验这些假设下的机制，不能作为原生Kimi会话/工具轨迹或策略收益证据。本轮没有复用其构造关系来计算上表。

## 群体覆盖敏感性，非性能预测

JSON中附带一个有严格限定的数学预算：只改变指定历史群体、把该群体全部TTFT清零、其他请求与decode耗时完全不变。

- 观测缓存≥32K群体占累计TTFT约3.937%；在上述极乐观且固定其他工作的假设下，平均TTFT倍率为1.041×、平均E2E为1.0095×。
- 观测缓存≥64K群体占累计TTFT约1.577%；对应1.016×和1.0038×。

这仅说明“只挑这一小群体”的历史延迟覆盖率。它不是FluidKV预测，不是包括共享队列影响的全局上限，不是吞吐预测，也不能用于插入新的缩容事件后、历史延迟已经改变的反事实。其他副本中不可见的状态机会没有被这两个群体覆盖。

## 真正验证所需的最小补充

不必公开原始对话文本，但至少需要：

1. 稳定匿名agent/session ID、request/parent-turn ID，以及一致的前缀block hashes；hash须跨请求/副本可比较。
2. 工具/依赖开始、结束或就绪时间，与请求时间可对齐；不能仅用两次请求的到达间隔替代。
3. 实际执行引擎和状态owner、可路由成员变化、初始缓存/驱逐或足够的状态驻留信息；bootstrap地址不能直接替代这些字段。
4. 模型状态布局、容量和实际后端的计算/传输成本。Qwen3.6上的状态字节与时延不能直接套到Kimi K2.5。

补齐后才能固定真实到达、真实前缀关系和成员事件，公平比较DualMap、CacheRoute、各自加FluidKV以及FluidKV配简单路由。若仍使用替代模型或合成内容，应继续明确标注，不能冒充Kimi同后端的生产性能结果。

## 复现与产物

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/analyze_kimi_opportunity.py
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/inspect_kimi_metadata.py
/home/zhujianian/miniconda3/envs/sglang/bin/python -m pytest -q \
  tests/test_kimi_opportunity.py tests/test_production_traces.py
```

主脚本拒绝覆盖已有输出；复测可使用`--output`另选目录，补充脚本的OUT需对应修改。原始trace和已有结果均未更改。

[完整分析JSON](results/kimi-opportunity/analysis.json) · [补充字段检查](results/kimi-opportunity/metadata-crosscheck.json) · [分钟级CSV](results/kimi-opportunity/minute_profile.csv) · [验证记录](results/kimi-opportunity/validation.json) · [图PNG](results/kimi-opportunity/kimi-opportunity.png) · [图PDF](results/kimi-opportunity/kimi-opportunity.pdf)
