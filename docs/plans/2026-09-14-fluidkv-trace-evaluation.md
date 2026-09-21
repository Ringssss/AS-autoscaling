# FluidKV Trace Evaluation

## Authorized scope

Use the supplied Kimi K2.5 Finish logs and FlowPrefill's embedded Qwen-Bailian
A trace to start the evaluation. Retain the existing AgentShift implementation
name until a separate rename. Do not modify original traces or old raw results.

## Evidence and workload contracts

- Kimi: original arrivals, completions, lengths, cache-hit counts, and DP rank.
  Input/output IDs are redacted and conversation IDs are absent. These fields
  cannot identify branch lineage or router prefix keys. Use for characterization
  and explicitly labelled length/timing workload studies; do not invent native
  session chains or call a synthetic reuse graph an original Kimi replay.
- Qwen-Bailian A: original timestamps, parent IDs, input/output lengths, and
  anonymized 16-token block hashes. Join on chat_id, not row number. Use exact
  input lengths; the final block may be partial. Arrival differences are not
  observed tool execution times. Types such as image/search have no media assets
  and their token surrogate is not multimodal execution.
- Token surrogates deterministically map each block hash to 16 vocabulary IDs.
  They preserve block equality and original lengths, not semantic content or
  original output quality. Do not concatenate generated output into the next
  recorded input: the trace does not establish that equivalence.

## Initial execution

1. Audit both complete inputs with SHA256 provenance. Verify deterministic
   parsing and prefix relationships on small fixtures before processing all rows.
2. Build a Qwen parent/child pilot from exact, block-aligned parent prefixes.
   Report eligible coverage and selection rules. Compare sticky, cold reroute,
   same-transport on-return handoff, and prepared handoff. Use actual generation
   and migration timings, streaming TTFT, output equivalence and cached tokens.
   Prepared handoff measures destination readiness, not an observed tool gap.
3. Integrate the pinned upstream DualMap decision code with the same SGLang
   request/telemetry harness. Retain dual hashing, SLO-aware dispatch, waiting
   queue rebalancing and prefix-key selection. Pilot first; do not claim faithful
   NPU/vLLM/DRAM reproduction from an SGLang port.

## Main evaluation after the pilot

| Experiment | Controls | Measurements |
| --- | --- | --- |
| Continuous serving | Same inputs, prefix warmup, model/backend and accelerator budget; low through overloaded arrival rates | TTFT/ITL/E2E, successful SLO goodput, failures and admission delay |
| Fixed membership changes | Same model-ready scale schedule for all policies | Arrival-to-recovery latency, cold misses, reconstruction, actual drain time |
| Closed-loop elasticity | Same SLO and demand; permit each controller to select capacity | GPU-seconds through physical release, goodput, recovery and controller overhead |
| Materialization ablation | Same placement/transport/handoff; full-copy vs legal partial-copy/replay plans | Bytes, replay GPU time, readiness, memory reservations, output validity |
| Scheduling ablation | Same materializer/handoff; independent placement vs joint planning | Drain completion, late preparations, SLO violations, rejected/wasted migrations |
| Handoff fault matrix | Duplicates, early returns, transfer/install/commit/cleanup failures | Single committed advancement, valid state, recovery latency, leaked allocations |
| Architecture coverage | Full attention, Qwen3.5 hybrid, a distinct recurrent architecture, SWA beyond its window | Correctness and cost changes attributable to component semantics |
| Concurrency and cross-node | Shared link contention, foreground load and topology | Transfer queue time, throughput, foreground tail penalty and scaling limits |

Primary baselines: sticky/cache affinity, least-loaded routing, upstream DualMap
on the common backend, on-return full-state migration, proactive full-state
migration with the same handoff, and full FluidKV. Add DualMap plus a documented
simple proactive full-state preparation extension to isolate routing from mobility.
Keep native Mooncake P/D as a separate architecture comparison; use the same
transport for claims about planning and scheduling.

## DualMap audit

- Repository: https://github.com/ASISys/DualMap
- Examined commit: 24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d
- Paper: https://arxiv.org/abs/2602.06502
- Sections 3.2/3.3 implement SLO-aware routing and queued-request rebalancing.
  The inspected rebalance code moves queue entries, not completed GPU state.
- Section 3.4 and Appendix A.2.3 cover consistent-hash membership changes and
  4-to-8-to-4 elasticity. Do not claim DualMap lacks autoscaling.
- The README describes custom vLLM DRAM KV management and NPU measurements.
  Public cache classes track block identities/capacity. Verify actual backend
  caching independently rather than treating these counters as physical hits.
- Calibrate TTFT estimation on our model/hardware; retain SLO and tuning parity.
  Include immutable commit, local adaptations, cache tier budgets and any missing
  native capability in each result's provenance.

## Measurement requirements

Use streaming receipt times for TTFT; distinguish request eligibility, harness
queueing, router queueing, migration preparation and engine service. Count all
failed/timed-out requests in offered load and SLO attainment. Record both
successful requests per second and maximum sustainable offered rate at a stated
SLO attainment target; these are different metrics.

Preserve independent session arrivals. For causal serving, release a child only
after its parent completes and its recorded arrival is reached, and record any
dependency delay. Do not give online policies future arrivals or exact tool gaps.
Oracle duration knowledge belongs only in a labelled upper-bound experiment.

Use held-out tuning windows and multiple chronological windows, paired seeds,
randomized policy execution order and at least three independent repetitions
for the initial campaign. Obtain enough requests for tail quantiles and report
confidence intervals; a 32-pair pilot is not a p99 serving result. Report
filtering, duration compression, output caps and selected-token coverage.

Validation here is inference-oriented: deterministic parser/planner tests,
backend sanity, and small real-engine end-to-end checks apply. Gradient and
training-overfit checks do not apply.

## Existing evidence requiring correction before main figures

- The hour replay's load time series averages all eight engines after scale-in,
  unlike the four-engine aggregate. Use a consistent membership-aware metric.
- Its heterogeneous policy includes 80 reconstructions (41,751 input tokens),
  omitted from the recomputation counter. Preserve the old raw artifact and
  publish corrected accounting separately.
- The Mooncake pilot achieved 32/32 actual parent hits; one 1088-token parent
  was incorrectly checked against a constant 2048-token expectation.
- Existing latency gains on Qwen3 and migration-only timings on Qwen3.5 do not
  establish FluidKV/DualMap cluster goodput or GPU cost improvement.

## Qwen3.5 execution updates

The same-model pilot now covers independently audited A, Coder and Thinking
inputs. All policies share a snapshot-aware cache estimate. This is necessary
because arbitrary KV-block matches do not establish runnable hybrid state.
Actual cache hits and estimation errors remain measured separately.

Chronological prefixes with no parent links are control experiments, not
evidence about next-turn reuse. Exact-parent cohorts retain original lengths
and select only verified block-aligned parent relationships; their selection
and any explicit time compression must be reported. Compare retirement of a
specified state owner against retirement of an unused instance.

The initial Coder capacity-pressure run revealed unmatched NCCL work after
destination reservation rejection. The runtime and coordinator now support
PREPARED on both engines before starting communication. A three-repeat live
regression fills the destination pool, rejects migration, cleans reservations,
and reuses the same communication group successfully. General post-launch
network failure recovery remains outside this fix.

Deterministic inference fixes one sampled ToolBench cold/cached discrepancy but
does not fix the full selected-pair gate (22/32 cold agreement). Both migration
paths still match sticky in 32/32 cases. Keep this distinction in the evidence.

The new drain harness is explicitly a planner prototype with a simple
destination heuristic and logical membership changes. It does not yet
implement the proposed full joint scheduler, exact engine cache directory,
physical GPU release, or complete fault-tolerant execution ingress.
