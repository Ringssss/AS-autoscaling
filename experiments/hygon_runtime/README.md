# Hygon Qwen3.6 bring-up

This opt-in runtime adapts the Hygon SGLang 0.5.12 vendor image for a local
Qwen3.6-35B-A3B BF16 checkpoint, which declares `Qwen3_5MoeForConditionalGeneration`.
Each engine uses two 64 GiB Hygon BW (gfx936) devices.

The 256- and 1024-token migration and coordinator checks passed locally.
See [RESULTS.md](RESULTS.md) for evidence and the validated scope.

The initial synchronous migration path transfers actual full-attention KV,
convolution state, and FP32 recurrent state using RCCL (PyTorch's `nccl` backend).
It runs in the scheduler while idle and publishes the destination prefix only
after all local TP shards complete copying. Source and destination exchange
readiness before reporting completion to the existing durable coordinator.

Supported scope: two engines, identical checkpoint and TP layout, DP=1, PP=1,
page size 1, standard KV layout, extra-buffer Mamba caching, no speculative
decoding, and an empty destination radix cache. The checkpoint config and index
and the actual tensor layouts are checked; weight content is not rehashed, so
both engines must mount the same immutable checkpoint. This is a correctness
bring-up path. It blocks scheduling during transfer and does not implement
asynchronous overlap, active-request migration, multi-engine group allocation,
or the original KV-only endpoints. A failed transfer group must be replaced.
Use fresh dedicated engines for each validation campaign.

## Build and launch

Run from the repository root. Extract `srt` from the exact image used in
`engines.sh` (an existing unmodified vendor container can be used):

```bash
mkdir -p /home/zhujianian/hygon-runtime-source
docker cp zhujianian-qwen36-check:/usr/local/lib/python3.10/dist-packages/sglang/srt /home/zhujianian/hygon-runtime-source/
python3 experiments/hygon_runtime/build_overlay.py \
  --source /home/zhujianian/hygon-runtime-source/srt \
  --output /home/zhujianian/hygon-runtime-overlay
bash experiments/hygon_runtime/engines.sh start
```

The overlay preserves the vendor implementation and modifies only scheduler
dispatch, message dataclasses, one communicator, and HTTP routing. The builder
checks unique source anchors, parses generated Python, and records original
and patched file SHA-256 values. It also installs `hybrid_control.py`.

Default devices: source 2,3; destination 4,5. Set `SOURCE_DEVICES`,
`DESTINATION_DEVICES`, `MODEL_PATH`, and `HYGON_OVERLAY` before first creation to
override. Existing containers retain their original settings. Both HTTP servers
bind to loopback, at ports 31400 and 31401. The launcher uses host networking
for inter-engine rendezvous. A TP=2 group reserves four ports starting at the
requested base port. Use distinct base ports for separate validations.

Vendor compatibility settings inherited from the local inference investigation:
Triton attention, `SGLANG_KV_LAYOUT_HCU_FA=false`, BF16 Triton MoE with HCU
token alignment, and disabled custom all-reduce. The original single-engine
investigation also used `AMD_SERIALIZE_KERNEL=3`.
The dual-engine launcher removes `AMD_SERIALIZE_KERNEL` and disables RCCL P2P
and shared-memory transports (`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`). The
initial configuration blocked at the first cross-engine broadcast. These are
compatibility settings, not tuned performance settings. Triton compilation
caches persist under `/home/zhujianian/.cache/hygon-agentshift`.

## Validate

Wait for both `/health` endpoints to return 200. These commands flush the
dedicated engine caches and record real generated token IDs and cache hits:

```bash
export PYTHONPATH=.
PYTHON=/home/zhujianian/miniconda3/envs/sglang/bin/python
$PYTHON experiments/hybrid_state_mobility/validate_direct_two_engine.py \
  --source http://127.0.0.1:31400 --destination http://127.0.0.1:31401 \
  --prefix-tokens 256 --continuation-tokens 16 --rendezvous-port 29820 \
  --output-dir experiments/hygon_runtime/results/direct-256
$PYTHON experiments/hybrid_state_mobility/validate_hybrid_coordinator.py \
  --source http://127.0.0.1:31400 --destination http://127.0.0.1:31401 \
  --prefix-tokens 256 --continuation-tokens 16 --base-port 29840 \
  --output-dir experiments/hygon_runtime/results/coordinator-256
```

Inspect a cached boundary with `POST /agentshift/hybrid_state/inspect` and
`{"token_ids": [...]}`. Unknown operations return 404. Rejected operations
return HTTP 409 and an explicit error. Control results include per-TP-rank
details; transfer byte counts are summed across TP ranks.

The later high-concurrency observer walks existing nodes without splitting,
changing LRU order, locking or allocating. Metadata-only inspection can run
between scheduler operations while requests are active. It returns actual
`snapshot_boundaries`, `query_tokens`, and free/evictable token and snapshot
capacities; TP aggregation intersects boundaries and takes minimum capacities.
Add `return_digest: true` for an idle-only complete-state integrity check. The
aggregate digest hashes the ordered per-rank digest/byte records. Transfer and
digest operations retain their idle-engine requirements.

For a single command that waits for readiness and saves both validators' output:

```bash
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/hygon_runtime/run_smoke.py --prefix-tokens 256 --base-port 29900 \
  --output experiments/hygon_runtime/results/smoke-256
```

```bash
bash experiments/hygon_runtime/engines.sh status
bash experiments/hygon_runtime/engines.sh stop
```

Historical NVIDIA results elsewhere in this repository do not validate this
Hygon runtime. See the local `results/` artifacts for completed measurements.
