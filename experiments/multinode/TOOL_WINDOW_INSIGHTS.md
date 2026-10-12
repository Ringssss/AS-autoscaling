> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Tool-window experiment: what FluidKV can change

The completed primary measurements are in [TOOL_WINDOW_RESULTS.md](TOOL_WINDOW_RESULTS.md).

## Mechanism and scope

The intended mechanism is to move a suspended agent's executable state while
the agent waits for a tool. Its next turn can run on another replica with a
warm prefix, instead of choosing between waiting at the source and recomputing
on a cold target. This is an opportunity to reduce imbalance, not a guarantee
that all migration decisions improve load balance.

In this Hygon prototype, state includes full-attention KV, convolution buffers
and recurrent state. Prefix length alone is not evidence of executable reuse.
The destination must have a matching snapshot and become ready before durable
ownership advances. Successful copying does not free the source model or
immediately reclaim the source's cache; that requires a separate lifecycle.

The current experiment compares three mechanisms using a real subprocess
repository search, a controlled minimum tool window, and identical foreground
inference offered to the source. It does not execute DualMap or CacheRoute
routing decisions. The prefix boundary is materialized by a one-token generation,
not an autonomous model-emitted tool-call sequence. Foreground inference uses
normal continuous batching; there is no artificial serial queue added to the
sticky baseline.

The experiment only offers one foreground request. It measures interference
under that load, not saturation, fleet load variance, p99 or throughput capacity.
No production tool durations or causal session links are recovered from the
redacted Kimi dataset.

## Core insights

1. **Compute load and cache locality are separate dimensions.** A replica can
   be lightly loaded yet cold for the returning agent, or heavily loaded and
   warm. Routing reduces load by changing the execution destination, but the
   best destination may lack executable state. FluidKV can alter that state
   placement before dispatch, creating a warm option on a lightly loaded replica.
   Cache-aware routers already exploit warm choices when they exist; copying
   has no inherent advantage over directly selecting an equally idle warm target.

2. **A tool wait is preparation time with a deadline.** Once the completed
   prefix is immutable, its state can move without predicting the tool output.
   Only the tool-result suffix must wait. Copying during the wait can reduce
   exposed latency; copying after return cannot exploit that interval. Short
   waits leave a copy tail and can make staying at the source faster.

3. **Routing moves work toward existing state; migration can move state toward
   future work.** This relaxes the locality/load tradeoff under drift, return
   bursts or planned retirement. It does not eliminate the tradeoff: copying
   consumes network, device bandwidth, cache space and transient duplicated
   state. A planner must decide when creating an additional warm destination
   is worth those costs.

4. **The decision should minimize total exposed cost, including externalities.**
   For intuition, let G be the remaining tool wait, C be preparation time
   including readiness/ownership, R be additional cold prefill, and Qs/Qd be
   source/destination wait or interference. With a common warm continuation
   cost P, idealized post-tool paths are:

   - stay: Qs + P;
   - cold reroute: Qd + R + P;
   - tool-window copy: max(0, C - G) + Qd + P.

   These are scheduling heuristics, not an additive decomposition of measured
   engine time: batching and overlapping queues make components interact.
   Include copy-induced delay to foreground requests and capacity pressure
   when choosing a policy. An uninitialized communication group must also be
   charged or prepared before the request; it is not free.

5. **FluidKV is naturally complementary to a router.** Keep the router's good
   warm-affinity decisions; selectively materialize state when a cold target's
   queue advantage and reuse value exceed migration cost. A fair ablation is
   the same routing/placement policy with and without state preparation, plus
   the original DualMap and CacheRoute baselines. Beating a forced cold-target
   control does not establish superiority over either complete routing algorithm.

## Four-node readiness

The launcher now supports node30, and its identical vendor image has been
downloaded and checked. The device check still reports approximately 53 GB
occupied on every 64 GiB card. The allocation check rejected launching there;
the four-node endpoint gate also rejected the campaign before starting trials.
No existing workload was stopped.

At least two available cards on node30 are needed for one additional TP2 engine.
Four nodes with one engine each means **eight cards**, not the full 32-card
fleet. `run_four_node_tool_window.py` is a prepared sequential source-to-three-
targets experiment; it must not be described as concurrent four-node balancing.
The full-fleet requirements in `PROTOCOL.md` still apply.

## Measurement safeguards

- Same source/target, text, tool result, offered foreground request, model,
  cache sizes and scheduler settings for all modes.
- Prewarm serial and concurrent execution shapes and the lazy communication
  group. Record startup costs separately and do not apply steady-state numbers
  to cold expansion.
- One client monotonic clock records actual tool return and continuation first
  token. The tool-return timestamp does not wait for migration completion.
- Keep exposed copy time, foreground request completion and owner commit, not
  just the migrating request's TTFT.
- Verify all output tokens against both sticky and cold baselines, actual
  prefix cache hits, lengths, transferred bytes and durable ownership.
- Report short-window/short-prefix losses. Few component repeats are not
  evidence of statistically significant fleet-wide capacity gains.
- The first campaign lacked concurrent-shape prewarming and showed a large
  first sticky sample. It is retained as exploratory evidence; the second,
  explicitly prewarmed campaign is the primary result.

Raw campaigns are `results/tool-window-node27-node29/` (exploratory) and
`results/tool-window-prewarmed-node27-node29/` (primary). The main log, raw
request timelines, full generation metadata, SQLite ownership database,
server settings and paired summary are retained. Node30 allocation/image
checks are under `results/node30-*` and `results/four-node-readiness.log`.
