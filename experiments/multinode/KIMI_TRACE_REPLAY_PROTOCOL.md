> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Kimi trace-driven FluidKV campaign (2026-09-23)

## Question and scope

Measure whether state movement improves complete trace-driven handoff windows,
including natural cache warming and interference from unrelated requests.
Compare P2C+FluidKV directly against standalone DualMap and CacheRoute, alongside
matched-router ablations. FluidKV standalone means state preparation plus P2C;
FluidKV does not itself replace request dispatch.

node30 was checked at launch and all eight cards have about 53 GiB allocated.
Use existing dedicated task engines on node27/28/29, TP2 each (six cards), with
identical eligible engine counts for all policies. Snapshot allocation again
before the main run. Source withdrawal is logical; no physical GPU release.

## Trace and constructed state

Source: verified complete Kimi release, SHA256
`e7d8fd50757208220b9bc5bbd111a9d491b94a89a6c42390ab345aab9a0c3513`.
Select three 10-second arrival-count windows at p10/p50/p90 among complete bins
after the first 600 seconds, using numpy RNG seed 20260924 for tied bins.
Selection uses no output, latency, cached-token or policy-result criterion.
The selected bins are 7166, 1840, 314, containing 14, 24, 57 requests.
All selected requests are retained, including anomalous original cache counts;
original cache counts are metadata, never used to invent identities.

Actual Qwen token content comes from the existing public LongBench corpus,
repeated when necessary to meet native input length. For controlled history L,
all requests with input length > L share exactly that L-token prefix; a unique
marker immediately after it prevents extra accidental sharing. Shorter requests
have a unique first token and no source state. This is an explicitly constructed,
optimistic shared-history envelope, not recovered Kimi conversations or private
agent histories. Each policy gets identical tokens and offered times.

Source history is prefilled as common prior work outside replay timing. At time
-W state preparation may start; first-bin arrival time is 0. Window arrivals
retain native offsets unless a rate-scale experiment explicitly divides them.
FluidKV copies to both destinations. The implementation must keep engines idle
while copying; dispatch waits for preparation, but arrivals remain on their
original clocks, so all exposed preparation/admission delay is charged.
The baseline has no such preparation work. Baselines and FluidKV can inspect
real checkpoint availability; router code is unchanged.

There is one initial handoff per independent window. Caches are reset only at
common trial setup, never between requests in a window. No claim of repeated
private-history migration, busy-source migration, or import into nonempty caches.
A source-available control keeps the source eligible in every policy.

## Predeclared screening grid

Six policies: P2C, P2C+FluidKV, DualMap, DualMap+FluidKV, CacheRoute,
CacheRoute+FluidKV. Each cell is one exploratory trial, not an independent
reproducibility claim. Randomize policy/cell order with seed 20260924.

- Three selected windows × histories 5760 / 16384 / 65536 × source withdrawn,
  W=.5 seconds, native arrival rate: 54 trials.
- Three windows × history 5760 × source eligible, W=.5: 18 trials.
- High-arrival window, history 5760, withdrawn source, W=.5, rate scales .25/4:
  12 trials. Native-rate counterpart already in grid.
- High-arrival window, history 16384, withdrawn source, W=0/10: 12 trials.

Total: 96 screening trials. Screen output length is min(native output,64), with
per-request and per-window truncation counts reported. This is not full native
compute load. Native input length is never truncated. Every request is allowed
to finish after the arrival window; include this drain in completion throughput.

Use common current prefill calibration and warm batch-16, 64-output throughput
for CacheRoute qcap. Shared-key rate uses preceding 60 seconds only. If the
training rate exceeds fleet capacity, cap replication count at fleet size by
clipping only the rate passed to plan construction; retain actual training rate
and all offered requests. Fixed plan within each short window. Native-output
confirmation uses batch16 with the median original output length in the first
600 seconds, strictly preceding all evaluated windows; it does not reuse a 64-output capacity estimate.
This correction was fixed before any native-output run. It remains a simple
capacity calibration, not a per-request compute model.

## Follow-up and claims

After screening, run original-output-length confirmation, explicitly separate
from capped screening. Choose follow-up cells before running them and save their
job file. Repeated validation uses five independent replays and paired policy
order randomization; never count requests in the same run as independent trials.
An advance-prefill control, if added, gets the same known prefix/window.

Meaningful gain: mean or p95 TTFT speedup >=1.2 with no material completion/SLO
regression; strong gain >=2. Report metrics separately, not an ambiguous combined
winner. For repeated claims show run-paired bootstrap 95% intervals. Include
all failed requests in offered/SLO counts and preserve failed setup attempts.

Report entire-window metrics and eligible/noneligible subgroups; exact generated
output matches, observed cache hits, copies, preparation, router time, offered
and completed lengths, and telemetry. Output differences preclude a lossless
claim. Report latency relative to arrival and preparation time separately.

Coverage means (a) request coverage in measured windows and (b) grid breadth under
explicit scenario assumptions. Three windows (95/245555 requests) cannot estimate
whole-production opportunity frequency. Grid points are not probability-weighted
production events; hypothetical prefix identity and membership are not measured
production facts. Do not infer population confidence intervals from these windows.

### Additional predeclared continuity control

Extend the selected high window from 10 to 60 seconds without selecting on results
(bin 314 start retained), yielding 317 requests. Six policies, history 5760,
withdrawn source, W=.5, native arrival timing, output cap64, one exploratory run
per policy. Job/window files: `results/kimi-trace-long-jobs.json` and
`results/kimi-trace-long-windows.json`. No cache reset inside these 60 seconds.

### Original-output confirmation fixed before screening results

`results/kimi-trace-native-jobs.json` specifies 18 trials: six policies × the same
three selected windows, history5760, source withdrawn, W=.5, original input AND
output lengths, original arrival spacing. These are separate from short-output
screening. Generated text still uses substituted LongBench content; preserving
lengths does not restore Kimi model semantics. No output-length-based filtering.

The .25/4 rate cells change the evaluated-window arrival rate only; the preceding
60-second rate estimate remains native. They are arrival-rate shocks, not steady
rescaled training-and-test traces. Report them separately from native-rate cells.

### Independent primary validation

Fixed while initial screening is still running (not chosen as the best cell):
high-arrival window, L=5760, source withdrawn, W=.5, native rate, cap64. Five
new repetition seeds (100–104), four policies P2C/P2C+FluidKV/DualMap/CacheRoute,
20 trials. Chosen for the frequently observed ~5.7K cache scale and loaded
serving regime. This validates the standalone comparison and P2C ablation;
it does not turn native-output single runs into replicated results.

### Calibration correction before the retained formal matrix

The initial capacity probe (5760-token warm prefix plus a one-token suffix)
underrepresented mixed-input prefill work. Its 17 completed trials / 526 requests
are retained under `kimi-trace-screen-warm-calibration-pilot`; the next trial was
intentionally interrupted. None is pooled into the final formal matrix.

Superseding the capacity rules above: use the first 16 arrivals in trace-relative
[300,600) seconds, before all evaluation windows. Preserve their native input
lengths, construct the same 5760 history, and run a one-engine concurrent batch.
Screening/validation/long-window capacity uses output cap64; native confirmation
uses those calibration requests' original output lengths. qcap is the measured
batch completion rate including drain, a planning parameter rather than a claim
of maximum steady-state capacity. Store all calibration requests and results.
Rerun the entire 96-cell-policy matrix; no selective retention of earlier wins.

### Original-output long-history control

Before the long-history screening pairs finish, add four native-output trials
at low arrival / L=65536 / source exit / W=.5, policies P2C/P2C+FluidKV/DualMap/
CacheRoute. This mechanism control retains the same 14 original requests and
outputs; it tests whether a long-history effect survives full decode. It is
exploratory, separate from the short-history primary validation. Native job file
now has 22 trials (18 primary-window confirmations plus these four).

### Held-out conditional coverage sample

To measure breadth beyond three mechanism windows, sample four nonempty 10-second
windows without replacement from each arrival-count tercile, seed20260926, after
the first 600 seconds. Exclude the original windows and 60-second extension.
Population: 8,902 candidate windows; strata 3010/3250/2642. Twelve windows contain
296 requests. File `kimi-trace-coverage-design.json` stores population weights,
thresholds and selected bins; script `prepare_kimi_coverage_windows.py` reads no
policy results. Four policies P2C/P2C+FluidKV/DualMap/CacheRoute, L5760, source
withdrawn, W=.5, original arrival/input lengths, output cap64: 48 trials.

This estimates descriptive breadth conditional on the imposed shared-history
handoff and output cap, not the production probability of such handoffs. Single
runs per held-out window have measurement noise; use the independently repeated
primary cell for repeatability, and do not present exact population prevalence
or a confidence interval that ignores within-window run uncertainty.

Held-out breadth reports mean-TTFT and p95 thresholds separately. An additional
predeclared guarded-gain count requires >=1.2 mean-TTFT speedup, <=5% mean-E2E
regression, and <=5 percentage-point loss in TTFT<=10s attainment. These guards
are fixed before held-out execution. Stratum weights describe conditional-window
breadth; individual request gains and regressions are also retained.

### Saturated planning boundary fix and continuation

After 27 complete formal trials (772 requests), the next source-available
high-arrival case failed during plan setup, before evaluated requests were
submitted. With qcap=0.7307621427183835, floating-point `(qcap*3)/qcap` is slightly
above 3, so ceil incorrectly requested four replicas. The harness now clips a
saturated rate one representable step below fleet capacity; the upstream router
is unchanged. A regression test reproduces this exact boundary. Retain the
failure file. Continue the remaining 69 jobs in original order under
`kimi-trace-screen-resume`, reusing the exact original calibration. Combine the
27+69 complete trials for the formal matrix; no previously completed result is
replaced. All subsequent phases use the boundary correction.

### Concurrent checkpoint observation correction

The high / L16384 / withdrawn / W0 / DualMap+FluidKV screening case completed all
57 requests but logged one inspect HTTP500. Retained node28 server logs show
`communicator.py: queueing_call: assert self._result_event is None`, caused by
overlapping same-engine observation calls. The running screen retains its original
observer behavior and every error. All later phases coalesce observations to at
most one in-flight call per engine, matching the control interface's constraint;
generation and transfer code are unchanged. A concurrency regression test brings
the relevant suite to 23 passing tests. A two-policy DualMap/DM+FluidKV recheck of
this exact cell, with original calibration, is appended before validation.
The original observation-degraded pair is flagged, not silently discarded or
replaced by its retry. Raw server evidence: `kimi-trace-observer-error-node28.log`.

The ongoing uncoalesced screening subsequently recorded the same observation
failure in additional conditions. Recheck selection is therefore generalized:
after screening, collect every condition with a checkpoint error and rerun all
policy pairs involving the affected policy, with the fixed observer and original
calibration. Select on the error flag only, not on measured speedup. Save the
resulting exact job list before rechecks start. Keep screening and rechecks as
separate stages. Expected total trial/request counts are updated from this list.

Telemetry semantics: `/get_load` is the legacy projection of `/v1/loads`.
`num_tokens` includes waiting-request token demand; it is not GPU memory usage.
For this recorded backend, `num_tokens - num_pending_tokens` equals its reported
`num_used_tokens`. Running-request counts are interface observations, not a GPU
utilization measurement. Do not infer memory pressure from total demand alone.
