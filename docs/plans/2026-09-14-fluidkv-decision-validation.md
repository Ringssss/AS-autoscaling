# FluidKV Decision Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task.

**Goal:** Determine whether execution-state mobility adds useful latency, SLO-goodput, or physically measured GPU-cost benefits beyond strong routing and simple migration.

**Hypothesis:** Mobility is useful when required placement changes destroy valuable locality, preparation fits the available time/resources, and its cost is smaller than recomputation or continued engine occupancy. The null result is an acceptable outcome.

**Authorization:** The user approved the proposed go/no-go experiment sequence and requested execution on 2026-09-14. Continue implementation, dedicated-engine validation, and bounded campaigns without another planning checkpoint.

**Validation scope:** Deterministic workload/decision/accounting tests; L0 common backend, actual memory/transfer calibration, and warmed inference; L3 tiny real-engine comparisons and retirement lifecycle. Training gradients and overfitting (L1/L2) do not apply. Cached/cold numerical disagreement remains an explicit unresolved correctness limitation, independently of preservation of the source cached path. Never silently pass it.

**Architecture:** Preserve the existing pilots and raw inputs. Add a separate continuous-workload harness with common routing, measured preparation, decision logs, and explicit timing boundaries. Use CPU characterization and optimistic bounds to determine whether increasingly expensive experiments are warranted. Runtime experiments share three dedicated Qwen3.5 H100 TP1 engines; cross-node claims require actual cross-node evidence.

## Shared Scaffold

- Existing runtime: `sglang-hybrid-state`, async PREPARED protocol, and the three localhost engines 31480–31482.
- Existing routing: pinned DualMap commit and `experiments/fluidkv/dualmap_adapter.py`.
- Existing workload: original Qwen A/Coder/Thinking anonymous block traces; Kimi Finish logs for length/timing characterization only.
- Existing correctness/failure controls: `validate_reservation_failure.py`, `benchmark_native_pairs.py`, and retained raw results.
- New campaign output: `experiments/fluidkv/results/decision-validation-20260914/`. Preserve original results and record current source snapshots.

## Subtask 1: Workload opportunity and representative windows

**Implementation:** Add `experiments/fluidkv/analyze_decision_workloads.py` with deterministic fixture tests. Analyze every Qwen split: chronological load, parent coverage, exact legal snapshot coverage, prefix reuse distances, and background retained by fixed windows. Analyze Kimi observed lengths, timing, and cached-token counts without inventing lineage.

**Selection:** Predeclare windows by time, not by favorable policy outcomes. Inspect 60/120/300-second windows with preceding warmup; choose affordable continuous measurement windows before running comparative policies. Preserve all requests in each selected interval except documented context-limit exclusions. Persist all selected IDs and input hashes. Label any rate scaling. Do not call arrival differences tool gaps.

**Validation:** Fixture joins use chat IDs; partial last blocks and missing warmup parents are counted correctly; no request-cap truncation hides load. Save complete population and selected-window audits.

## Subtask 2: Measured preparation and decision controls

**Implementation:** Add calibrated length/state transfer measurements and a minimal decision model. Compare cold service, warm service, full copy, and reconstruction. Include fixed overhead, actual capacity rejection, shared-resource interference, and explicit no-prepare decisions. The cost controller must not receive future child arrivals or identities. Offline future-informed controls are labelled as such.

**Validation:** Warmed paired probes, real cached tokens, source/destination continuation agreement, failure cleanup; deterministic decision tests cover no-benefit, missed deadline, capacity exhaustion, and time spent waiting.

## Subtask 3: Common continuous-serving comparisons

**Implementation:** Add a separate continuous harness and resumable runner. Policies: DualMap, DualMap plus simple full copy, and DualMap plus cost-aware preparation. Retain a no-retirement control and an explicitly optimistic analysis. Use the same trace, backend, membership schedule, SLO, and resource limits. Preparation cannot silently block causal eligibility or disappear from E2E/GPU accounting.

**Protocol:** Start with tiny L3 controls. Calibrate load on a disjoint window, then freeze parameters. Compare multiple held-out time windows and at least three paired repetitions with randomized policy order. Use original output lengths; use explicit arrival-rate scaling only when required to span underload through overload. Fixed-time engine retirement is a mechanism control, not evidence of an autoscaling policy. Retain unsuccessful/unfinished requests in offered-load and SLO denominators.

**Metrics:** Arrival/eligibility/dispatch/first-token/finish timestamps; actual/estimated hits; parent-only and population metrics; ITL when observable; preparation bytes/input/output/wall time; unused/late preparations; workload offered rate and completion rate separately; SLO sensitivity; queueing and calibration error; per-run uncertainty, never request-level pseudo-replication.

## Subtask 4: Physical retirement and cost

**Implementation:** On dedicated engines only, stop new work, drain in-flight requests and source-dependent preparation, terminate the identified owned process tree, and observe actual GPU process/memory disappearance. Record shutdown delay and model reload readiness. Charge model-resident standby GPUs as occupied. Compare the same demand/SLO with direct drop/recompute and simple-copy draining. Do not equate cache ownership count with an obligation to keep a reconstructible source alive.

**Validation:** No killed foreground request; no restart counted as instantaneous; monitor process identity; all raw lifecycle evidence retained. Only claim GPU savings if the full observation interval and SLO support it.

## Subtask 5: Independent review and final decision

Review variable controls, oracle leakage, tail sample sizes, parity, cold/cached behavior, capacity failures, and GPU accounting. Compile figures/tables and reproducible commands. Distinguish observed results, estimated bounds, and missing evidence.

**Decision rules:** (1) negligible optimistic headroom on the examined target workload: do not build a larger optimizer merely to seek gains; (2) useful gain captured by simple preparation: retain the simple mechanism and reject an unsupported joint-controller contribution; (3) reproducible headroom lost to contention/timing despite simple baselines: proceed to a bounded joint optimizer. A gain must survive warmed paired repeats and include failures and overhead. No universal percentage is treated as a conference acceptance threshold.

## Execution log

- 22:40 +08:00: Three dedicated H100 engines healthy and idle; same model/backend/context/token and recurrent-slot budgets verified. Existing harness still lacks physical release and continuous mobility control. Original files and prior raw results retained.
- L0 correction: enable CUDA graphs at batch sizes 1/2/4/8/16 on all engines, retaining deterministic inference, FA3/Triton, FP32 recurrent state and original capacity limits. Record eager controls separately. Single-engine sampled decode rose from 15.60 to 111.61 tokens/s; this is a backend correction shared by all policies.
- Admission correction: async destination preparation now pins its matching prefix before reclaiming ordinary evictable KV/recurrent cache. Actual free-capacity checks still reject private reservation exhaustion. CPU failure/pin/cleanup tests and three-repeat live evictable-cache/private-exhaustion regressions passed.
- Storage correction: the workspace is NFS4 and durable SQLite transitions there caused large non-RPC delays. Retain NFS measurements; use explicitly labelled local ext4 SQLite FULL/WAL for the new comparisons, archive committed databases after observation, and keep all actual preparation costs in timing.
- Correctness: on graph-enabled production settings the original native32 cold/cached gate remains 22/32; both migration variants match sticky 32/32. Synthetic calibration agreement does not override this limitation.
- L3: a constructed four-request control passed all three policies; full copy and the cost controller recovered the 8k child cache hit, while DualMap rerouted cold. It is a mechanism fixture, not trace-performance evidence. Physical release of its actual owner completed gracefully in 4.335 seconds, with no foreground failures; both surviving engines then served probes. Restart took 42.453 seconds. Routing continues during shutdown.
- Information contract: online policies see child inputs only after causal eligibility. The immutable prompt snapshot may be observed after the current parent's first token; this does not grant future child inputs or migrate the evolving live decode state.
- Frozen load calibration: Coder start 600, 15 seconds measurement plus 15 seconds warmup, full original lengths/background, arrival scales 0.125/0.25/0.5; only DualMap with the same fixed 3-to-2 event. This is disjoint from held-out starts 3000/5400. Main coverage uses 60-second measurement and 60-second warmup (five eligible children in each Coder held-out window); 30-second selections remain engineering screens.
- Frozen controller costs: measured local production curves; busy-copy multiplier 1.6, foreground-interference estimate 0.006 seconds, minimum predicted gain 0.02 seconds, shared SLO 1 second with the prespecified sensitivity thresholds also reported. Exact warm-prefix measurements supply the service floor. This is an observed-demand cost controller, not predictive speculation or MILP.

## Authorized checkpoint-coverage repair (23:43 +08:00)

The user explicitly requested implementation after the identity ambiguity finding. Preserve the main comparison objective, but repair the workload/backend interface before spending on a broad head-to-head campaign.

The initial full-parent filter is not an upper bound on original reuse. A has 18,797 unaligned parents matching every complete parent block; Coder has 15,418 and Thinking 1,085 such ambiguous tails. Their missing final fragment cannot be inferred from anonymous hashes. Never change surrogate tokens to assert unobserved tail equality.

1. Validate the existing SGLang `extra_buffer` strategy on one dedicated engine, first controlling overlap scheduling and then checking its normal supported setting. Test unaligned prompt lengths, actual valid cached boundaries, source-active availability, and existing transfer compatibility. Record both cold/cached and migrated/sticky comparisons, failures, memory and timing. Retain/restorable launch identities.
2. Add a read-only engine prefix inspection endpoint if needed to report valid hybrid snapshot lengths and actual free/evictable resources separately. Inspection must not fabricate snapshots, mutate ownership, pin/evict state, or alter LRU. Both baseline routing and mobility receive the same current metadata. Actual reservations remain authoritative against races.
3. Extend the shared adapter/materializer to use validated engine checkpoint boundaries, including checkpoints already produced at chunked-prefill boundaries in `no_buffer`. Replay the unchanged child suffix from its confirmed boundary. Distinguish this existing backend capability from a new FluidKV contribution.
4. Run deterministic prefix-observation/accounting tests, real unaligned-pair tests and continuous-window smoke; then rerun common-backend routing/full-copy/cost-aware comparisons with all original background and length information retained. Keep strict confirmed-identity and optimistic tail assumptions separate; only the former is the main replay.

The 128k token pool passed controlled memory/admission tests (lowest sampled free memory 3.87 GiB at source and 4.19 GiB at destination); new snapshot strategy memory must still be checked. Larger pool and checkpoint strategy results are separate from the original 65k/no-buffer measurements.

### Checkpoint implementation validation and bounded head-to-head

- The read-only inspector walks complete existing radix edges without splitting, LRU updates, pins or COW; reports only nodes with full KV and recurrent snapshot. Every policy receives the same inspection directory, and automatic prompt-end assumptions are disabled. Context/vocabulary validation and independent actual-RadixKey tests pass.
- `extra_buffer` revealed the expected checkpoints at 64/960/4032/8128/9984/16320 in 24/24 constructed cases; no_buffer already retained 8192 for the two longest prompts. Three copied intermediate checkpoints matched source-cached output and target hits. Cold/cached equality remains failed (18/24 in the full no-overlap matrix), independently of migration preservation.
- All three engines now share extra_buffer, overlap enabled, graphs, 131072 KV slots, and 64 configured recurrent slots. SGLang's recurrent-buffer budget reduces effective concurrent serving to 12; report the effective limit and do not call a 16-request queue a 16-active test.
- The constructed unaligned four-request live harness has no failed requests or inspection failures and reports actual 8128-token child hits for the two mobility variants versus zero after DualMap's required cold reroute. This is a mechanism fixture, not a representative benefit claim.
- Final bounded native comparison uses the two already declared held-out Coder starts 3000 and 5400, 30-second measurement and 30-second warmup windows, all input/output lengths and eligible background retained, arrival scale 0.25, three paired repetitions and randomized policy order. The 30-second subsets were audited before any policy outcomes (25/26 retained parents; 25/25 identity-supported hypothetical floor64 boundaries). Scope is a continuous screening experiment; it does not replace the longer 60/120/300-second population audits or establish peak sustainable goodput.
- Use the same specified owner retirement and actual release for all three policies. Include a same-window no-retirement DualMap reference separately with all three resident engines charged. Record restart time separately, and do not claim full-cycle autoscaling savings unless a common observation and readiness horizon supports them.
- Inspecting snapshot availability and post-completion metadata refresh have explicit event timings; inspection failures clear stale estimates. Metadata refresh cannot postpone causal child eligibility, and source shutdown waits for in-flight inspections and cleaned transfers while routing continues.
- Add the independently reviewed fourth `selective-copy` ablation before native comparative outcomes: same proactive trigger, destination, queue order, transport and no dispatch waiting as `full-copy`; copy only if the measured one-reuse cold-minus-warm proxy exceeds copy, interference and 20 ms margin. The calibration's 64-token suffix is a declared proxy, not a future child prediction. Keep reactive `cost-aware` as a reference; its negative result cannot reject proactive planning because it skips unobserved demand before physical release.
- The SGLang generation path reserves two context tokens (`max_req_len=context-1`, max output `max_req_len-input-1`). Use input+requested output <=32766 to avoid silent output shortening. No A/Coder/Thinking source records equal32767/32768, so this clarification changes no selected native requests. Retain the failed stress-boundary attempts separately.

## Frozen campaign completion — 2026-09-15

Completed24/24 checkpoint-native-paired-v2 runs, including physical release and restart records. Independent artifact audit passed all24;254 CPU tests passed. The bounded campaign is complete; broader no-retirement/long-window/multi-node evaluation remains unperformed under the user instruction to conclude now. See experiments/fluidkv/results/decision-validation-20260914/FINAL_DECISION.md. Conditional latency improvements do not establish stable SLO/goodput/GPU-cost superiority. Retain validated state mechanisms, defer large MILP implementation and broad performance claims. Correctness equivalence and continuous-memory limits remain explicit.
