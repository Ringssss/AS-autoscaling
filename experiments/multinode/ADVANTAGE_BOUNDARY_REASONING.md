> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# What would constitute a distinctive FluidKV advantage?

The experiment tests adding state preparation to the same DualMap / CacheRoute
implementation. A win in that comparison does not establish that every good
router must lose: when the warm source is idle, sticky return may be cheaper.
Likewise, with sufficient tool slack, advance prefill can create warm targets.
The added controls distinguish these explanations.

## Conditional sufficient condition, not a measured decomposition

Let H be warm continuation latency, Qs/Qd source/target queue delays, R cold
recomputation cost, C complete transfer and installation time, and W remaining
tool slack. Let I >= 0 summarize migration-induced extra delay. Assume these
quantities are stable and additive for this simplified comparison; real batch
formation and queue overlap can violate that assumption.

The best stay-or-cold-route choice takes:

    Troute = H + min(Qs, Qd + R).
    Tfluid = H + Qd + max(C - W, 0) + I.

Write B = H + Qd, M = min(Qs - Qd, R), and X = max(C-W,0)+I.
Then a k-fold TTFT gain is possible exactly when, in this simplified model:

    M >= (k - 1) * B + k * X.

Both the queue advantage and recomputation cost must be large enough. A cheap
idle warm source defeats the premise. State transfer being faster than prefill
alone is not sufficient to beat a strong router.

Against a stronger baseline that can also precompute during the same window,
replace R in M by max(R-W, 0). This exposes the distinctive interval:

    C <= W < R,

with a sufficiently congested warm source, an available cold target, and small
interference. W >= R removes the latency distinction from advance prefill;
C > W leaves a transfer tail, but can still win if that tail is small enough.
Long history is helpful only if the measured compute-to-transfer cost ratio is
favorable for the actual architecture, hardware and contention level.

## Repeated wins versus steady-state wins

Independent repetitions that reset caches establish repeatability of the cold
transition. They do not establish steady-state persistence. The retained-wave
experiment directly tests whether ordinary requests warm targets and remove
the initial advantage. It uses identical shared prefixes and suffixes, not
advancing private agent histories.

Sustained gains require a continuing supply of eligible state-location
mismatches: for example private evolving histories, cache eviction, or changing
placement. Transfer demand must remain below usable transport capacity, target
compute below its capacity, and transient duplicated state within memory.
Migration must continue to beat the best available warm route and advance
prefill after accounting for interference. These are workload and resource
conditions, not a guarantee that arbitrarily higher BS improves the method.

The present Hygon import path requires idle engines and an empty destination.
It cannot yet test the central private-history mixed-load claim. Extending it
requires nonempty-cache admission, overlap-safe transfer, eviction/lifecycle,
and integrity/recovery validation. Clearing destinations between trials cannot
stand in for that implementation.

## Logical source withdrawal

When membership makes the source unavailable, set Qs = infinity in the model.
The warm-source alternative disappears for both algorithms, so the comparison
reduces to cold recomputation versus exposed transfer cost on eligible targets.
`heldout-source-withdrawal` tests this with genuine two-replica DualMap and
CacheRoute decisions, identical offered requests, and a common membership
change. Five reset trials assess repeated transition events, not perpetual
speedup after those targets have become warm. Source GPUs remain allocated.

TTFT convergence with long-window advance prefill does not prove equal resource
efficiency: prefill still consumes compute that could serve other work. Measuring
that potential system benefit requires background throughput / interference
accounting that this isolated campaign does not provide.
