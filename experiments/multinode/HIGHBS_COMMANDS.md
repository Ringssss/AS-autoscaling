> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Long-request / high-concurrency campaign commands

Working directory: `/home/zhujianian/AS-autoscaling`. Public model download:
no credentials required or written to these commands. Prior diagnostic runs
and failed attempts remain in `results/highbs/` and are not primary evidence.

## Model download

Pinned Qwen revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Initial metadata (read architecture/size before allocating storage):

```bash
PATH=/home/zhujianian/miniconda3/envs/download-tools/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com bash /home/zhujianian/hfd.sh \
  Qwen/Qwen3.8-Flash-Next --tool aria2c -x 16 \
  --include '*.json' '*.md' '*.jinja' \
  --local-dir /home/zhujianian/Qwen3.8-Flash-Next
```

The full 360 GB checkpoint was placed on `/data/node29/zhujianian`, which has
enough storage. A resumed three-worker download assigns disjoint shards to
node27/node28/node29, each using the user's `hfd.sh`, 16 connections per file
and 8 parallel files. All concrete arguments, including the shard lists, are
saved in `results/highbs/download-commands.json`:

```bash
python3 experiments/multinode/download_qwen38.py
PATH=/home/zhujianian/miniconda3/envs/download-tools/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com bash /home/zhujianian/hfd.sh \
  Qwen/Qwen3.8-Flash-Next --tool aria2c -x 16 -j 4 \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --include '*.txt' '*.json' '*.md' '*.jinja' '.gitattributes' LICENSE \
  --local-dir /data/node29/zhujianian/Qwen3.8-Flash-Next

curl -L -fsS \
  'https://hf-mirror.com/api/models/Qwen/Qwen3.8-Flash-Next/revision/de4b8e4d43b917e7706784d8bb445c9af86a3540?blobs=true' \
  -o experiments/multinode/results/highbs/qwen38-file-metadata.json
ssh node29 python3 /home/zhujianian/AS-autoscaling/experiments/multinode/verify_model_download.py \
  --model /data/node29/zhujianian/Qwen3.8-Flash-Next \
  --metadata /home/zhujianian/AS-autoscaling/experiments/multinode/results/highbs/qwen38-file-metadata.json \
  --output /home/zhujianian/AS-autoscaling/experiments/multinode/results/highbs/qwen38-hash-verification.json
```

The completed home-directory model path is a symlink to the full data-disk
checkpoint; the original metadata-only directory was preserved separately.
The first single-host `-j 4` and `-j 16` downloads were checkpointed/resumed,
not treated as separate checkpoints. Their logs are retained.

## Qwen3.8 reference inference (separate backend)

The vendor SGLang image does not recognize `qwen4_exp`. An isolated Transformers
source checkout/archive at commit `0bc252863a4e5c0e709893664cc57369ebcd7353` and
new tokenizer/hub dependencies provide the reference model implementation.
Source and dependency paths are `/home/zhujianian/qwen38-transformers-source`
and `/home/zhujianian/qwen38-reference-deps`.

The exact complete `docker run` argument vector is recorded in
`results/highbs/qwen38-reference-command.json`. The container uses physical
cards 2–7 on node29, the vendor Torch, a read-only checkpoint, and:

```bash
python -m pip uninstall -y torchaudio
mkdir -p /var/log/hylog
source /opt/dtk/env.sh
PYTHONPATH=/source/src:/deps python /run.py
```

`/run.py` is a read-only mount of `run_qwen38_reference.py`. The incompatible
audio package is removed only inside this isolated container. Default grouped
GEMM failed on Hygon; the successful retry used explicit upstream eager experts
and the upstream recurrent GDN scan for prefill. A separate CPU numerical check
against chunked GDN is retained. To rerun the successful script mounted in the
existing stopped reference container:

```bash
ssh node29 docker start -a zhujianian-qwen38-reference
```

This is device-map layer placement with the large N-gram table on CPU, **not
TP=8**, not a SGLang serving benchmark, and not a FluidKV migration result.
Record actual used devices from `qwen38-reference.json`, not just the six-card
visibility mask. Do not mix this latency with the Qwen3.6 routing measurements.

## Qwen3.6 equal-engine configuration

Only this task's idle pilot containers were reconfigured; unrelated workloads
and the earlier local 31360/31400/31401 servers were preserved. Each engine
uses the same vendor image and runtime overlay with TP2, 131072 context,
262144 KV slots, 512 Mamba slots, requested maximum 128 running requests,
static memory fraction 0.90, BF16, Triton attention/MoE, CUDA graphs off.

```bash
python3 experiments/multinode/launch_pilot.py --node node27 --max-running 128 --mamba-slots 512
python3 experiments/multinode/launch_pilot.py --node node28 --max-running 128 --mamba-slots 512
python3 experiments/multinode/launch_pilot.py --node node29 --max-running 128 --mamba-slots 512
```

These are creation commands: existing containers must be idle and explicitly
stopped/removed before recreating them. `results/highbs/runtime-manifest.json`
contains actual container commands, image IDs, devices, mounts, network
environment and overlay hashes. All three use `shca_0`/`IBext_v8` for this run.
Node30 remains occupied and was omitted as requested.

## Capacity probes

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/probe_batch_capacity.py \
  --output experiments/multinode/results/highbs/capacity-32k
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/probe_batch_capacity.py --regimes shared-hot \
  --batches 128 256 \
  --output experiments/multinode/results/highbs/capacity-32k-fresh-connections
```

The second probe fixes stale keepalive-connection failures in the first probe;
the original four disconnected requests remain recorded. Peak running batch
and queue come from engine telemetry. Submitted concurrency is not actual
batch size, and these fixed cache/state budgets are not a universal physical
maximum for the node or a TP8 capacity claim.

## Calibration and retained preliminary campaign

Before the primary test, tune only DualMap's existing pending-prefill parameter
on a separate 8192-token workload. Each candidate uses two repetitions of
single-hot/all-hot, 32 offered requests per engine, and no FluidKV. Selection
minimizes median batch completion time subject to no failed requests:

```bash
for pending in 1 8 32; do
  PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
    experiments/multinode/benchmark_highbs.py --lengths 8192 --batches 32 \
    --regimes single-hot all-hot --policies dualmap --repetitions 2 \
    --output-tokens 64 --pending-prefills "$pending" \
    --output "experiments/multinode/results/highbs/tune-prefill-$pending"
done
```

`results/highbs/prefill-tuning.json` records the selection: **32**. The measured
median completion times were about 30.86 s (1), 12.08 s (8), and 9.77 s (32).
The selected value is fixed for both baseline and augmented DualMap on the
held-out 32K/64K test, with no routing-formula modification. The following
earlier sweep used heuristic hybrid metadata; retain it for diagnosis, not as
final single-hot gain evidence, because naturally warmed checkpoints were not
always discovered by the router:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_highbs.py \
  --lengths 32768 65536 --batches 16 64 128 \
  --regimes single-hot all-hot --repetitions 3 --output-tokens 64 \
  --pending-prefills 32 --base-port 31300 \
  --output experiments/multinode/results/highbs/three-node-primary
```

Default endpoints are `12.11.3.27:31600`, `12.11.3.28:31600`, and
`12.11.3.29:31600`. Policies are `dualmap`, `dualmap+fluidkv`, `cacheroute`,
`cacheroute+fluidkv`. Per-run `command.json` saves the exact argv and cwd.

This is a burst stress workload with public-text prefixes and deterministic
synthetic continuation suffixes. CacheRoute uses an explicitly synthetic
equal-rate prior and measured single-engine cost calibration; those keys/rates
are not recovered from Kimi. The shared-prefix preparation ablation retains
the same router and creates warm target state during a controlled 500 ms
window. It does not claim a general production mixed-cache FluidKV scheduler.

Unique-cold controls, where no transferable source state exists:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_highbs.py --lengths 32768 --batches 16 \
  --regimes unique-cold --repetitions 2 --output-tokens 64 --pending-prefills 32 --base-port 31200 \
  --output experiments/multinode/results/highbs/three-node-cold-32768
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_highbs.py --lengths 65536 --batches 8 \
  --regimes unique-cold --repetitions 2 --output-tokens 64 --pending-prefills 32 --base-port 31220 \
  --output experiments/multinode/results/highbs/three-node-cold-65536
```

Run these sequentially; they share engines and flush caches between trials.
The cold probe already reaches the fixed KV-capacity plateau; sending hundreds
more cold requests increases queue length, not the resident decode batch.

## Final shared-observation campaign

The updated Hygon overlay provides non-mutating metadata inspection while an
engine is serving. It returns actual checkpoint boundaries and conservative
TP-wide capacities. Baseline and augmented routing share this observer;
per-engine requests are coalesced, and freshly materialized checkpoints update
the same SnapshotDirectory. No prompt-end checkpoint is inferred from request
completion. Inspection errors invalidate estimates rather than fabricate hits.

The final high-load comparison uses 32K/64K, nominal per-engine concurrency
16/128 (48/384 cluster requests), both single-hot/all-hot, and three repetitions:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_highbs.py \
  --lengths 32768 65536 --batches 16 128 --regimes single-hot all-hot \
  --repetitions 3 --output-tokens 64 --pending-prefills 32 \
  --observe-checkpoints \
  --calibration-file experiments/multinode/results/highbs/three-node-primary/calibration.json \
  --base-port 32320 \
  --output experiments/multinode/results/highbs/checkpoint-aware-final
```

`observation-pilot/` is the successful actual-engine observation test preceding
this run. `three-node-primary/` and earlier sweeps are retained with their
metadata limitations. The unique-cold controls have no reusable source state
and no repeated prefix identity, so the new observer is deliberately disabled
for that regime and does not change that comparison's path.

For independent integrity checks after the performance workload drains:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/validate_state_bytes.py \
  --destination http://12.11.3.29:31600 --base-port 31820 \
  --output experiments/multinode/results/highbs/state-byte-integrity-node29
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/validate_state_bytes.py \
  --destination http://12.11.3.28:31600 --base-port 31840 \
  --output experiments/multinode/results/highbs/state-byte-integrity-node28
```

Digest inspection remains idle-only and is outside performance timing. It
checks every TP shard's complete KV/conv/recurrent bytes, not just output text.

## Summaries and validation

```bash
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/summarize_highbs.py experiments/multinode/results/highbs/checkpoint-aware-final
FLUIDKV_DUALMAP_SOURCE=/home/zhujianian/DualMap \
  /home/zhujianian/miniconda3/envs/sglang/bin/python -m pytest -q
```

The summary preserves exact-output match rates and errors; an HTTP success is
not a claim of bitwise equivalence under different BF16 batch shapes. Initial
polling-driver and metadata-index diagnostic runs are kept separately and must
not be used to inflate DualMap routing tax. The primary driver reacts to
arrival/first-token/completion events rather than repeatedly rebalancing an
unchanged queue. Original upstream source remains pinned and unmodified.
`three-node-routing/` is the earlier upstream-default-parameter sensitivity
run, not the tuned primary campaign; its `scope.json` records the interruption.

The historical JSON field `routing_cpu_seconds` is elapsed **wall time** inside
enqueue/dispatch functions measured with `perf_counter`, not an exclusive CPU
profiler counter. It excludes some completion/cache-accounting callbacks; the
client latency and full batch duration still include the request-path effects.
Do not use that field to claim a complete CPU utilization decomposition.
"Cold" here means missing executable prefix state on a model-resident engine,
not a machine/model startup. All controllers and first communication groups
are initialized before measured return bursts, with initialization recorded.
