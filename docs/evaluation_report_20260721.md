# AgentShift Consolidated Evaluation (updated 2026-07-24)

## Executive Judgment

AgentShift 已经实现并闭合了三个核心组件：durable continuation and
ownership、completed-prefix mobility，以及 gap-aware semantic handoff。当前
Qwen3-8B TP=1 的关键实验表明：当下一轮必须迁往另一个 engine 时，AgentShift
能够保持完整 prefix hit，把 32K 历史的 post-tool latency 从 stateless
reroute 的 1260.5 ms 降到 52.4 ms，并把相同 direct-transfer mechanism 的
on-return latency 从 126.1 ms 降到 52.4 ms。

这已经是 paper-worthy system prototype，但还不能表述为“全面超过官方
SOTA”。文献命名的 Agentix/Autellix、Continuum、TokenCake 和 Symphony
baseline 是同一 SGLang testbed 中的 mechanism-equivalent implementation，
不是官方 artifact reproduction。当前最稳妥的论文结论是：

> Among evaluated designs that relocate the next LLM turn while preserving a
> full prefix hit and single-owner execution, AgentShift achieves the lowest
> post-tool latency by moving the completed prefix directly between GPUs and
> overlapping that movement with the blocked interval.

## Implemented System

### 1. Agent Continuation and Ownership

- SQLite WAL 中的 committed step、owner engine 和 monotonic epoch。
- inference step claim、stale-owner rejection 和 duplicate-step rejection。
- global tool mailbox、pending future 和 output stream metadata。
- managed-effect PREPARED/SUBMITTED/COMPLETED/UNKNOWN lifecycle。
- destination ACK 前保留 source shadow；post-commit failure 时递增 epoch
  恢复。
- controller restart reconciliation：PREPARING/COPYING/DEST_READY 保持 source
  authority 并清理或等待；COMMITTED 保持 destination authority 与 source
  shadow。

### 2. Completed-Prefix Mobility

- pin source RadixCache prefix，reserve destination KV slots。
- persistent NCCL groups 和 rank-to-rank MHA K/V transfer。
- async transfer worker、独立 CUDA stream 和 completion polling。
- 所有 layer/rank 完成后才安装完整 token key。
- destination readiness 后 ownership CAS，first-token ACK 后释放 source。
- TP-wide aggregation，禁止 rank 0 success 掩盖落后或失败 rank。

### 3. Gap-Aware Semantic Handoff

- 只迁移 suspended/blocked agent，不迁移 active decode request。
- tool gap 中启动 transfer，在 next turn runnable 前完成 owner handoff。
- capacity、queue、KV size、estimated gap 和 transfer interference admission。
- on-return、oracle、TTL、routing、CPU-tier 和 shared-tier baseline 使用同一
  workload 与 engine configuration。

### Lifecycle Fix Found During Evaluation

中断 arrival-probe smoke 后发现，SGLang `/flush_cache` 重置 RadixCache 时会
遗留 AgentShift pinned-prefix metadata，导致后续 release 引用旧树节点。现在
flush 在活动 transfer/tier operation 存在时拒绝 reset；只有后台操作终止后，
才先清理 AgentShift registry，再重置 RadixCache。真实 endpoint 验证中，flush
后的 retryable release 返回 0 tokens，未再出现 cross-tree reference。

## Baseline Capability Boundary

| Design | Move next turn | Full prefix hit | Gap overlap | Durable owner transfer |
| --- | ---: | ---: | ---: | ---: |
| Sticky | No | Yes | N/A | No |
| Agentix/Autellix-style routing | Optional | Only if source | No | No |
| Continuum-style TTL | No | Before expiry | No | No |
| TokenCake-Source | No | Yes | Yes | No |
| Stateless reroute | Yes | No | N/A | Routing only |
| On-return migration | Yes | Yes | No | Yes |
| Symphony-style shared prefetch | Yes | Yes | Yes | No |
| Shared prefetch + CAS | Yes | Yes | Yes | Yes |
| AgentShift | Yes | Yes | Yes | Yes |
| Oracle | Yes | Yes | Ideal overlap | Yes |

Sticky、Agentix/Autellix、TTL 和 TokenCake-Source 可以达到很低的下一轮
latency，是因为它们没有解除 source placement。它们是 locality/residency
baseline，不是满足 AgentShift constrained objective 的直接竞争者。

## Headline Results

### RQ1: Locality After Placement Change

Qwen3-8B TP=1，32K prefix，500 ms blocked gap，五次重复：

| Strategy | Post-tool mean | Run-level max | Full hit | Placement/owner moved |
| --- | ---: | ---: | ---: | ---: |
| Sticky | 54.4 ms | 58.1 ms | 100% | No |
| Reroute | 1260.5 ms | 1266.2 ms | 0% | Placement only |
| On-return | 126.1 ms | 130.2 ms | 100% | Yes |
| AgentShift | **52.4 ms** | **53.3 ms** | 100% | Yes |
| Oracle | 54.5 ms | 56.5 ms | 100% | Yes |

AgentShift is 24.04x faster than reroute and 2.41x faster than on-return. Its
mean is 3.6% below Sticky because of run noise; the defensible interpretation is
that AgentShift matches the sticky latency floor while changing placement.

Across the three-repeat 4K/16K/32K sweep at the same 500 ms gap, AgentShift is
3.08x/10.61x/23.36x faster than reroute and 2.00x/2.16x/2.29x faster than
on-return. The increasing reroute gap supports the claim that mobility becomes
more valuable as an agent accumulates context.

Artifacts:

- `results/blocked-window-1784566204404099436.json` (five-run headline)
- `results/blocked-window-1784565792789187724.json` (90-point matrix)

### RQ2: Gap Overlap and Alternative State Paths

At 32K, the direct transfer moves 4.50 GiB per migrated agent. With a 100 ms
gap, AgentShift reaches 53.4 ms post-tool latency versus 129.5 ms for on-return
and 1272.7 ms for reroute. At a 500 ms gap it reaches 53.9 ms, close to Sticky
at 53.6 ms.

The mechanism-equivalent source-residency methods also become fast when the
gap is long: at 32K/500 ms, TTL is 55.1 ms and TokenCake-Source is 54.2 ms.
However, both leave future execution on source. In contrast, the local
shared-memory two-hop implementations relocate KV but take 4868.9 ms
(Symphony-style, no owner protocol) and 5241.0 ms (shared prefetch + CAS) in
this testbed. These values diagnose the cost of the current CPU/shared-tier
path; they are not claims about official Symphony.

### RQ3: Correlated-Return Hotspot

Eight agents, 32K prefix, simultaneous return, 500 ms gap, 32 output tokens,
five repeats:

| Strategy | Makespan | Relocated | Owner moved | Full hit | Re-prefilled history |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sticky | 783.7 ms | 0% | 0% | 100% | 0 |
| TokenCake-Source | 2196.4 ms | 0% | 0% | 100% | 0 |
| Reroute | 5579.8 ms | 50% | 0% | 50% | 131,088 tokens |
| On-return | 1250.4 ms | 50% | 50% | 100% | 0 |
| AgentShift | **730.9 ms** | 50% | 50% | 100% | 0 |

AgentShift is 7.63x faster than reroute, 1.71x faster than on-return, and 1.07x
faster than Sticky. The key result is not merely the last 7%: AgentShift moves
half the future execution and owner authority while retaining a full hit for
every agent.

Artifact: `results/hotspot-1784566696908767902.json`.

### RQ4: Continuum-Style TTL Under Capacity Pressure

TTL is calibrated from measured coding-tool durations rather than fixed for one
test point. The resulting TTL is 491.9 ms with 88.9% held-out coverage (9
held-out samples). Eight 16K agents compete with twelve 16K pressure requests:

| Gap | Strategy | Agent makespan | Post-tool mean | Full hit | Owner moved |
| ---: | --- | ---: | ---: | ---: | ---: |
| 400 ms | TTL | 5594.7 ms | 5594.3 ms | 100% | 0% |
| 400 ms | AgentShift | 5636.6 ms | 3238.3 ms | 100% | 50% |
| 500 ms | TTL | 8048.6 ms | 6631.1 ms | 33% | 0% |
| 500 ms | AgentShift | 5545.6 ms | 3129.7 ms | 100% | 50% |

Before expiry, TTL protects locality but does not redistribute computation.
After expiry, AgentShift is 1.45x faster in makespan and 2.12x faster in mean
post-tool latency while preserving all prefixes. This supports the design-space
claim that retain/evict and relocate are different actions.

Artifact: `results/ttl-pressure-1784562956744594834.json`.

### RQ5: Real Coding-Tool Windows

For a 32K prefix, three-run means are:

| Tool | Tool mean | Reroute | On-return | AgentShift | Full hide |
| --- | ---: | ---: | ---: | ---: | ---: |
| `git status --short` | 8.9 ms | 1269.0 ms | 131.5 ms | 113.2 ms | No |
| state-store tests | 403.4 ms | 1258.2 ms | 125.4 ms | 54.5 ms | Yes |
| control-plane tests | 488.2 ms | 1259.2 ms | 124.8 ms | 54.7 ms | Yes |

The short command exposes most of the migration, while the two real test tools
hide it completely. This is direct evidence for a gap-aware admission policy,
not evidence that every tool call is long enough.

Artifact: `results/real-tools-all-1784548799599095961.json`.

### RQ5b: Multi-Turn Coding Replay and Migration Scheduling

Eight agents execute three tool turns while their prefixes grow from 16K to
24K and 32K. Each turn invokes a real subprocess command; four agents relocate
after the first tool submission. Three-run means are:

| Strategy | Makespan | Post-tool mean | Post-tool p95 | Full hit | Historical re-prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sticky | 7.169 s | 1.866 s | 3.377 s | 100% | 0 |
| Reroute | 6.264 s | 1.298 s | 3.372 s | 83.3% | 65,548 tokens |
| On-return | 5.318 s | 0.967 s | **1.708 s** | 100% | 0 |
| **AgentShift** | **5.002 s** | **0.940 s** | 1.711 s | 100% | 0 |

AgentShift is 1.43x faster than Sticky, 1.25x faster than reroute, and 1.06x
faster than on-return in makespan. It moves 50% of agent owners while preserving
all reusable history. The small mean gain and equal p95 relative to on-return
show that proactive overlap helps this replay but does not dominate every tool
turn.

The migration-policy trace contains six blocked agents with 4K--32K prefixes
and decorrelated 60--500 ms gaps. Prefixes are pinned when each turn completes;
every policy therefore receives the same six full prefixes. Three-run means are:

| Policy | Completed in gap | Total exposed migration | Full hit |
| --- | ---: | ---: | ---: |
| FIFO | 50.0% | 405.1 ms | 100% |
| Shortest KV | 50.0% | 388.0 ms | 100% |
| Earliest return | 33.3% | 182.3 ms | 100% |
| **AgentShift admissible-first** | **83.3%** | 374.7 ms | 100% |

AgentShift completes five of six handoffs before their deadlines by deferring a
predicted-late migration. That objective improves coverage but gives the
deferred agent a long tail; earliest-return has lower total exposed time while
completing fewer handoffs in-gap. The calibrated exhaustive schedule is a
model-optimal reference, not a measured clairvoyant upper bound.

Artifacts:

- `results/coding-agent-replay-1784829317183895026.json`
- `results/migration-policy-1784829520064490654.json`

### RQ6: Foreground Interference

One 32K migration overlaps eight foreground streaming decodes. A one-token
arrival probe is submitted 5 ms after migration starts. Five-run means:

| Mode | Throughput | Arrival TTFT | TPOT p95 | Max token gap |
| --- | ---: | ---: | ---: | ---: |
| No migration | 397.7 tok/s | 53.9 ms | 20.55 ms | 41.62 ms |
| Sync transfer | 397.0 tok/s | 78.0 ms | 20.42 ms | 66.17 ms |
| Async transfer | 397.9 tok/s | 64.5 ms | 20.57 ms | 53.38 ms |

Async transfer keeps throughput and steady TPOT effectively unchanged. It
reduces the excess arrival-TTFT cost over baseline from 24.1 ms to 10.6 ms and
the excess maximum token gap from 24.6 ms to 11.8 ms. Copy/compute interference
is therefore bounded but non-zero.

Artifacts:

- `results/interference-1784566871298243631.json` (five-run headline)
- `results/interference-1784564046361063009.json` (4K/16K/32K, concurrency
  1/4/8 matrix)

### RQ7: Correctness and Recovery

The real-engine fault matrix passes 8/8 cases:

- post-commit destination failure with source-shadow recovery;
- destination failure after source release with safe cold reconstruction;
- lost ACK and idempotent release retry;
- controller restart at DEST_READY before CAS;
- tool result racing ownership CAS;
- duplicate managed effect against a real external SQLite table;
- no-owner-fencing ablation;
- flush lifecycle cleanup after a pinned prefix.

At 16K, source-shadow recovery takes 48.1 ms with a full 16,388-token hit.
After the shadow is released, correctness still holds but cold recovery takes
523.2 ms. The shadow therefore improves this recovery point by 10.89x. The
pre-commit restart case first copies all 16,388 tokens, injects failure at
`DEST_READY`, and then verifies that the source remains owner with a full hit.
The original no-fencing case models two accepting executors. A later real-engine
microbenchmark replaces that modeled result with actual Qwen3-32B TP=4
execution. Under injected duplicate delivery, router-only prefetch sends one
`/generate` request to each engine; both hit the full 16,388-token prefix and
generate 128 tokens. One copy is discarded, wasting 50% of decode tokens and
25.95 GPU-seconds per logical step. With epoch and step fencing, all five stale
source attempts are rejected before `/generate`, while the destination executes
once. Mean stale rejection is 62.8 us.

Artifacts:

- `results/fault-matrix-1784829560507240459.json`;
- `results/fencing-microbench/qwen32b-tp4-16k-128t/fencing-microbench-1786008706120873629.json`;
- `results/fencing-microbench/qwen32b-tp4-control-overhead/fencing-microbench-1786008850759397964.json`.

## Model and TP Generality

Previously collected runs use the same 32-token tool observation:

| Configuration, 32K | Sticky | Reroute | AgentShift | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B TP=2 | 75.4 ms | 704.3 ms | 83.2 ms | 8.46x |
| Qwen3-32B TP=4 | 118.7 ms | 1637.3 ms | 120.5 ms | 13.58x |

The Qwen3-32B TP=4 destination hits all 32,772 completed-prefix tokens and adds
1.8 ms over Sticky while changing placement. Full performance matrices remain
TP=1; TP=2/4 currently establish compatibility and headline generality.

Artifacts:

- `results/e2e-1784537113268610412.json`
- `results/e2e-1784539658778872653.json`

## Validation Status

- AgentShift tests: 39 passed.
- SGLang AgentShift prefix/TP tests: 14 passed.
- Current SGLang base: `034dd39189ba1ace1308d3c8a58df275ef301a21`.
- Environment and exact launch flags:
  `results/evaluation-environment-20260721.md`.

## Claim-Evidence Map

| Claim | Evidence | Status |
| --- | --- | --- |
| AgentShift preserves locality after relocation | 32K destination full hit; 52.4 ms vs Sticky 54.4 ms | Supported |
| Gap overlap is a distinct contribution | 52.4 ms vs same-transport on-return 126.1 ms | Supported |
| AgentShift relieves correlated-return hotspots | 50% owner relocation; 730.9 ms vs reroute 5579.8 ms | Supported |
| Async copy has bounded foreground impact | 5-run arrival/TPOT/throughput experiment | Supported for one migration |
| Durable owner transfer prevents double advancement | 8-case fault matrix and real Qwen3-32B duplicate-delivery ablation | Supported within evaluated fault model |
| Source shadow improves recovery, not correctness | 48.1 ms warm vs 523.2 ms cold recovery | Supported |
| AgentShift outperforms official Symphony/TokenCake/Continuum | No official artifact comparison | Not supported |
| Multi-turn coding replay preserves mobility benefits | 8 agents, 3 tool turns, 16K--32K prefixes | Supported for controlled subprocess tools |
| Placement policy improves in-gap coverage | 83.3% versus 33.3--50.0% simple policies | Supported for one serialized six-agent trace |

## Remaining Submission Work

1. Run complete SWE-agent and BFCL/function-calling replays; the current coding
   replay uses real subprocess tools but a controlled conversation skeleton.
2. Evaluate concurrent migrations 1/2/4/8 and enforce a measured bandwidth/
   interference budget. The current NCCL group serializes untagged transfers.
3. Perform independent-process crash-stop injection by killing a destination
   SGLang process; the current matrix injects controller failure at real
   protocol boundaries but keeps both engine processes alive.
4. Revalidate headline points with production CUDA Graph settings and report
   confidence intervals. The current headline uses five repeats; the full
   90-point matrix uses three.
5. Add cross-node results if the paper claims transport generality. Same-model,
   same-revision, same-TP engines should remain the stated compatibility scope.

The current evidence is sufficient to write the system design and preliminary
evaluation. The remaining experiments should strengthen workload generality,
policy quality, and failure realism rather than add new top-level components.
