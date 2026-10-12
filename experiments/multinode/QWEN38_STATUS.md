> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Qwen3.8-Flash-Next on this Hygon host

The full checkpoint is downloaded and real-weight reference inference has
succeeded. The Qwen3.8 SGLang/FluidKV high-concurrency comparison is **not
validated** by that reference run.

## Checkpoint

- Repository: `Qwen/Qwen3.8-Flash-Next`.
- Revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Full local path: `/data/node29/zhujianian/Qwen3.8-Flash-Next`.
- Convenient symlink: `/home/zhujianian/Qwen3.8-Flash-Next`.
- 131 safetensors shards; 359,999,963,128 tensor bytes.
- All 144 repository files, 360,023,351,514 bytes, verified against the pinned
  repository's LFS SHA-256 or Git blob hashes. The structural check also
  verifies shard headers, file lengths, tensor names, and the complete index.
- Download used the existing `hfd.sh` with disjoint shard workers on three
  nodes. The public repository did not require credentials.

Evidence: `results/highbs/qwen38-download-validation.json`,
`qwen38-hash-verification.json`, `download-commands.json`, per-worker logs.

## Actual inference result

On node29, the isolated vendor-image container loaded the full trained model
through Transformers source commit
`0bc252863a4e5c0e709893664cc57369ebcd7353` and isolated updated hub/tokenizer
dependencies. The six-card visibility mask is physical 2–7. Accelerate's
device map actually used logical devices 1–5 for the GPU layers; logical 0
remained unused. This is layer placement, **not TP=6 or TP=8**. The large
N-gram embedding can remain on CPU through upstream's no-placement mechanism.

Measured maximum allocated memory per visible GPU was approximately
`[0, 50.44, 53.18, 53.18, 53.19, 25.38] GiB`. Model load took 249.18 s.

| Prompt | Actual answer | Generated IDs |
|---|---|---|
| What is 1 + 1? Answer with just the number. | `2` | `[17, 248046]` |
| What is the capital of France? Answer briefly. | `Paris` | `[57590, 248046]` |

Generation wall times were 17.04 s and 3.10 s, respectively. They include
first-use effects and are **smoke observations, not throughput benchmarks**.
The run used upstream eager experts and the upstream recurrent GDN scan for
prefill after the vendor grouped-GEMM path failed. These are reference
implementations, not the optimized SGLang kernels.

The GDN recurrent/chunked formulations were independently checked on CPU for
sequence lengths 8, 32, 73; maximum output/state absolute error was below
4e-8 for those test tensors. The broken vendor TRSM prefill path was avoided.
The incompatible audio package was removed only inside the isolated container.
Neither the running serving images nor global host packages were replaced.

Evidence: `qwen38-reference.json`, `qwen38-reference-eager-run.log`,
`qwen38-reference-run.log` (retained failed grouped-GEMM attempt),
`qwen38-gdn-reference-validation.log`, `qwen38-reference-command.json`.

## Why the routing benchmark still uses Qwen3.6

The installed Hygon SGLang 0.5.12 image's Transformers 5.6 config loader rejects
`model_type=qwen4_exp`; the installed model registry also predates Qwen4Exp.
The registry's newer `sf_b075-20260903` image was also downloaded for a
CPU-only package inspection; it did not expose a Qwen4Exp model implementation.
Other installed custom SGLang/vLLM model packages were checked as well. These
inspections did not replace any active serving image. Candidate manifests and
inspection logs are retained under `results/highbs/vendor-*`.
Upstream's Qwen3.8 cookbook describes newer model support and separate NVIDIA
and AMD gfx950 builds. Those are not verified Hygon gfx936/DTK replacements.

This checkpoint introduces QSA, N-gram/PLE state and a different recurrent
layout. Merely recognizing the config or executing HF generation does not
establish a correct paged-cache serving implementation or a complete FluidKV
state bundle. Qwen3.6's KV/conv/recurrent byte formula cannot be reused.
Required follow-up is a compatible optimized serving runtime plus discovery,
transfer and continuation checks covering the new state kinds, followed by
the same-engine router comparison. No Qwen3.8 high-BS advantage is claimed.

See [HIGHBS_COMMANDS.md](HIGHBS_COMMANDS.md) for complete download, environment,
runtime and experiment commands. The stopped reference container is retained
as `zhujianian-qwen38-reference` on node29 for reproducibility.
