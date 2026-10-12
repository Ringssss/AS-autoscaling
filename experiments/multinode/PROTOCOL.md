> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Four-node comparison protocol

Target: four Hygon BW nodes, eight 64 GiB devices per node, Qwen3.6-35B-A3B
BF16, TP=2: **16 independently routable engines on 32 devices**. A three-node,
six-device component pilot is a separate result and must not be labeled a
32-device policy evaluation.

## Current evidence gates

1. All four nodes are reachable; all have eight devices and four active 400 Gb/s
   InfiniBand ports. Other users' workloads occupy node30's entire device set
   and part of node28. Do not stop them or silently share those devices in a
   supposedly dedicated comparison. Capture the allocation before each run.
2. The supplied Kimi release is genuine production serving telemetry, but its
   245,555 rows redact both token arrays and provide no conversation, routing,
   session, or extra key. All 8,455 `cached_tokens > input_length` records have
   one input and one output token. Preserve and flag these records; do not
   silently clamp their cache counts or assert they are health checks without
   an explicit label. Arrival times span approximately 24.97 hours because
   finish-log cohorts need not coincide with a 24-hour arrival window.
   A finish-only log does not establish coverage of all original offered or
   failed requests; replay success rates must not be described as the original
   service's success rates.
3. This release supports timing/length replay, not native prefix-identity,
   business-key, parent-child, tool-gap, or counterfactual cache-hit replay.
   Request IDs are unique and cannot substitute for repeated business keys.
   Reconstructing keys from cached-token counts would bias the comparison.
4. The current Hygon migration prototype requires idle engines and an empty
   destination cache. It is suitable for isolated state-copy correctness and
   cost measurements. A mixed-load fleet comparison requires admission,
   eviction, import into nonempty caches, transfer lifecycle/cleanup,
   TP-aware port allocation, and host-aware bidirectional groups. It must not
   clear the destination cache for one policy to make its migration work.

## Baselines

- **DualMap:** the existing adapter, unmodified upstream commit
  `24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d`, tree
  `cbe7f87e8707923d59a5acfc42f2d776bd5696ad`. The clean original source is
  available at `/home/zhujianian/DualMap`; its adapter tests run against that
  checkout. This is the project's common-engine adaptation, not a claim of
  reproducing every component of the upstream production system.
- **CacheRoute:** paper Algorithm 1, arXiv:2608.19677v1. `cacheroute.py`
  implements top-rate warm-set admission, load-based replication, distinct
  LPT placement, stable within-interval affinity and cold-tail power-of-two
  choices. This is a paper-faithful implementation, not author-provided code.
  Its equal-slot assumption must be disclosed for heterogeneous prefix sizes.
- **Controls:** sticky affinity, cache-blind P2C, warm-destination rerouting,
  and on-return migration. Cold rerouting is a component control, not a
  replacement for either cache-aware routing baseline.

Use the same engine version, attention kernels, graph settings, model revision,
KV/recurrent capacities, concurrency, batching, sampling, networking, and
request data for every policy. Calibrate service cost and CacheRoute's `qcap`
before the evaluated interval. Rate estimates come only from preceding data.
Hyperparameters may be tuned on training windows with equal tuning budgets;
do not select the winner's best window after looking at test results.

## Routing tax and latency accounting

For each eligible request record, on one monotonic client clock:

- arrival, dependency readiness, router decision start/end, dispatch, first
  output token, and final output token;
- chosen engine, previous owner, observed executable prefix at source and
  destination, actual `cached_tokens`, prompt and completion lengths;
- state bytes, copy start/end, readiness, ownership commit and source cleanup;
- timeout, generation/copy/admission errors, and correctness checks.

Report arrival-to-first-token and arrival-to-completion, including router CPU,
router queue, engine queue, network, and prefill/decode. Do not double-count a
component already inside client TTFT. Dependency delay must not disappear by
changing the time origin. Always include failed/timed-out requests in offered
load and SLO denominators.

**A→B does not imply a zero hit.** B may contain a global template, an earlier
copy, or another exact shared prefix. For this hybrid model, a KV-only match
without the matching recurrent snapshot is not executable reuse. Attribute
lost reusable work to the difference in real executable cache availability;
do not count every noncached token as routing tax. Measure recompute cost on
the engine because it is nonlinear in prefix/suffix size and batch load.

Count exposed transfer/installation time, destination allocation/eviction,
source and destination interference, transient duplicated state, and router
overhead for AgentShift. Hide transfer behind a tool interval only when that
interval is genuinely observed or explicitly labeled as an experimental
controlled delay. A completely prepared destination is not evidence that a
production tool gap hid the migration.

## Workload and statistical design

Keep three results separate:

1. Native telemetry audit and timing/length-only production replay.
2. Real public-content, known-prefix component experiments (LongBench here),
   with all substituted content/lineage identified as such.
3. Production cache-aware policy replay, requiring privacy-preserving exact
   prefix/block identities plus stable opaque session/business keys, causal
   turn links and observed tool boundaries if overlap is claimed.

For the full fleet, use disjoint training/evaluation windows covering low,
median, peak, and burst loads; preserve causal dependencies when known. An
arrival-rate sweep must use identical streams and scaling for all policies.
Report every context/output truncation, exclusion, and failure. A 131,072-token
context fits all supplied observed input+output lengths; a 4,096-token context
would retain only about 6.73%, and 65,536 retains about 95.48%.

Include stable membership, hot-key skew, destination-already-warm, scale-in,
scale-out, and plan-replacement churn. Physical GPU retirement must actually
release serving resources; a router membership change with a live source model
is only logical retirement. Report the longest-lived source shadow and its
memory/resource time.

Use paired seeds and policy order randomization, at least five independent
full runs for headline comparisons, explicit warmup exclusion, and a fixed
offered-load ladder. Report TTFT/E2E p50/p95/p99, SLO goodput, total throughput,
failures, token-weighted actual cache hit, per-engine load skew, copy bytes,
GPU memory high-water marks, and routing/transfer overhead. Confidence
intervals should resample independent runs or trace blocks, not treat correlated
requests as independent trials. Mark capacity estimates as left/right censored
when the tested ladder does not bracket the SLO knee.

Component pilots use few repeats and report median/range only. They establish
the cost of a known cache-location change, not p99, capacity, or superiority
over DualMap/CacheRoute on the unrecoverable original production key stream.
