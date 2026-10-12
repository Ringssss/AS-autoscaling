# AgentX profiling and bounded nonempty-cache preparation

This run applies the installed BBuf `llm-torch-profiler-analysis`,
`llm-pipeline-analysis`, and benchmark fairness guidance. Installation hashes,
source contracts, exact runtime launch/config and native request provenance are
under `results/`. No cross-framework serving comparison is implied: the
comparators remain pinned DualMap and the existing CacheRoute adaptation on
the same SGLang/Hygon model backend.

## Profiling

`capture.py` measures two native-length, symbol-token requests before and after
separate torch captures: 116,480/702 cold and 57,920/598 cached. Each trace keeps
both TP ranks. Five profiled forwards are requested separately for EXTEND and
DECODE. The long prefill capture only covers its first five 4K chunks, not all
116K tokens. A controlled 4-request mixed case is separately labeled.
`light_capture.py` disables stack and shape capture to check profiler effects.

`analyze.py` retains full original captures and emits navigation-only derivatives,
per-rank kernel sum/union/span, categories and source-verified MoE-end anchor
intervals. Both eager stack captures and light traces remain diagnostic; formal
performance uses unprofiled serving. ROCtracer duplicate-flow warnings are
retained; a gap between visible kernels is not automatically removable CPU time.

The skill's generic pipeline labels an interval starting at a layer's MoE-end
anchor. Therefore its `layer 2` interval contains the next layer's attention;
do not call it the original model's layer 2. Our JSON starts intervals at the
previous MoE-end, verifies 40 anchors per forward and labels the corresponding
hybrid layer type. Layer0 includes inter-forward boundary work; intervals are
navigation/accounting guides, not perfectly isolated module execution.

## Scoped implementation change

`build_nonempty.py` produces a task-local opt-in overlay. With
`AGENTSHIFT_NONEMPTY_IMPORT=1`, the destination can have existing radix state.
The importer still reserves a full private copy and requires both engines idle.
Vendor `MambaRadixCache.insert` frees duplicate KV slots; the caller frees a
redundant recurrent-state slot if a complete target snapshot already exists.
Existing cache is never flushed to make import succeed, no forced eviction is
introduced, and source state remains resident. Capacity and busy-source limits
are retained. This is not asynchronous background migration.

`validate_nonempty.py` checks live disjoint import, shared extension, duplicate
import and missing-source rollback. It verifies complete-state hashes, old
cache preservation, KV/snapshot accounting, ownership and exact generated output.
The direct group's failure state after the rollback case is intentionally not
reused. The serving comparison uses separate group ports and names.

## Held-out comparison

`prepare_compare.py` freezes 12 previously unused native substreams, four in
each maximum-input bin (<32K, 32–64K, 64–96K), first three requests, at most 900
total original output tokens per stream. Original lengths are never truncated.
This is a short-output stratified test, not full-corpus performance. All
selection and hashes are saved before results exist.

Generated `comparison_trial.py` and `harness.patch` show the exact change from
the prior three-round, logical-scaleout harness. All six cases use the same
runtime, same full-length requests, same cold initial state, and retain cache
across rounds. One completed trajectory per case; no significance claim.

- `dualmap`, `cacheroute`: original routing, no state copy.
- `+move`: previous empty-target-only, single-candidate preparation.
- `+multi`: same historical router-anchor eligibility, but may prepare multiple
  observed complete states and import into nonempty target caches. Stop starting
  work after the common one-second preparation window and enforce 3 GiB per round.
  Any overrun of a started transfer is charged. Skip already-warm states and
  reject insufficient temporary capacity. Do not peek at the next input/output.

The quiet round boundaries give the idle-only prototype a legal migration window.
This does not answer sustained busy-source migration. Third-round cache is
retained even if it prevents or makes preparation unnecessary. Compare all
requests, including those without transferable prefixes, and retain output
differences caused by batching. Length-valid latency is not semantic equivalence.

`compare.py` is bounded to 35 minutes and releases all three task containers at
completion, timeout or failure. The previous long campaign is not restarted.

## Entry points

Run from the repository root with the `sglang` conda environment and `PYTHONPATH=.`:

```bash
python experiments/weka_profile/build_nonempty.py
python experiments/weka_profile/validate_nonempty.py
python experiments/weka_profile/capture.py
python experiments/weka_profile/light_capture.py
python experiments/weka_profile/analyze.py
python experiments/weka_profile/prepare_compare.py
python -m pytest -q experiments/weka_profile/test_analysis.py
python experiments/weka_profile/compare.py
```

Launch records specify required dedicated endpoints and mounts. Do not run
these scripts against other users' serving endpoints. The capture and comparison
must not overlap on the same devices; `continue_campaign.py` enforces that order.
