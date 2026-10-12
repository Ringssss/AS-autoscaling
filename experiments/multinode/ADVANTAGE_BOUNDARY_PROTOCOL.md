> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# FluidKV advantage boundary campaign

Allocation checked 2026-09-23: node30 has all eight cards occupied. Use existing
node27/28/29 task engines, one TP2 Qwen3.6 engine per node (six cards total).
Allocation evidence: `results/advantage-boundary/allocation.json`.

Compare unchanged DualMap and paper-implemented CacheRoute, each with no
preparation, FluidKV state copying, or parallel target precomputation. All
receive the same immutable prefix and controlled tool-return deadline.
Precomputation is a stronger diagnostic control, not a claim that the original
routing algorithms already implement speculative prefill.

Screen 8K/64K x offered per-engine concurrency 4/16 x tool windows 0/.5/5s,
one repetition, randomized policy order. These observations locate candidate
boundaries only. Independently predeclare 64K/16/.5s with seed 20261001,
five repetitions and three consecutive return waves per policy. Reset caches
only before wave zero, retain them thereafter, and observe real checkpoints.
All policy waves use identical inputs, so later-wave results describe repeated
shared-prefix requests (including repeated suffixes), not advancing private
agent histories. Zero transfers on later warm waves are expected.

Large stable TTFT advantage requires a paired median ratio >=1.2 and a
run-bootstrap 95% interval lower bound >1, assessed separately for both routers.
Retained waves must also support a claim of persistence. Report p95, throughput,
errors, exact output matches, copy bytes and preparation time. Five runs offer
limited precision; screening and held-out data are not pooled for inference.
TTFT includes preparation that extends past tool return. Add the identical
tool duration when interpreting latency from tool start. State creation before
the tool interval is common prior work. GPU/network externalities under
continuous background load are not measured by this idle-engine prototype.

Current state import requires an empty destination and idle engines. Therefore
this campaign cannot establish the most promising private-history, mixed-cache,
continuous-load regime. That claim requires runtime changes and separate state
integrity / concurrency / failure-recovery validation. No simulated numbers are
substituted for measurements.

Run sequentially from repository root:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/run_advantage_boundary.py
/home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/summarize_advantage_boundary.py
```

The full argv for every job is saved in `results/advantage-boundary/campaign.json`.
The first initialization attempt failed before trials with a TP collective
result mismatch; its directory and source log are retained. Only the three
experiment containers were restarted before retry, with full kernel warmup.

The second initialization attempt completed one transfer but failed on the
second pair with `EADDRINUSE` on port 33110. Ports 33100+ were inside the host
32768–60999 ephemeral range. Retry uses checked ports 25100+ outside that
range. The earlier hypothesis of incomplete scheduler drain was not confirmed
as the cause; the common pre-trial drain guard remains a setup safeguard.

## Adaptive followups (separate held-out data)

Zero-window screening identified 64K / concurrency 4 as a strong candidate.
`selected-validation.json` fixes five new paired repetitions with a new seed.
It additionally includes sticky warm-source routing, to check whether the
source actually needs offloading. This is a diagnostic lower-cost alternative,
not a substitute for DualMap or CacheRoute in the requested comparisons.

Measured 64K precomputation is around 7.3 seconds, so the exploratory ladder is
extended to a 10-second window (`long-window-control.json`) to bracket full
hiding. That extension has one repetition and supports only a mechanism check.
`run_advantage_followups.py` waits for the main campaign to drain before running
these two jobs. It never runs inference concurrently with the main campaign.

The predeclared 1.2x threshold means a practically meaningful 20% improvement;
use >=2x for the stronger colloquial description “much faster.” Statistical
significance of a few percent does not meet the substantive-gain criterion.

## Source membership withdrawal

`source-withdrawal.json` predeclares 64K / four requests per **eligible** engine /
0.5-second tool window / five repetitions, with six routing/preparation modes.
Three physical TP2 engines participate in setup and state movement. At return,
the source is removed from the eligible URL list for every policy; the two
remaining engines receive eight offered requests total. DualMap is constructed
for two replicas and CacheRoute replans for the same two-replica membership
using the fixed prior rates and cost calibration. Neither baseline may route
back to the withdrawn source. Actual destination URLs are retained and audited.

This is logical source withdrawal, not measured physical shutdown, freed GPU
memory, or continuous scale-in/scale-out. Every repetition is a fresh transition.
It isolates the case where warm-source return is unavailable by membership,
without pretending a forced single-target request is the routing algorithm.
The stronger advance-prefill control receives the same prefix and deadline.

Complete reproduction sequence (existing output directories must be archived
before any new campaign; scripts deliberately refuse to overwrite trials):

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/run_advantage_boundary.py
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/run_advantage_followups.py
/home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/audit_advantage_boundary.py
/home/zhujianian/miniconda3/envs/sglang/bin/python experiments/multinode/summarize_advantage_boundary.py
```
