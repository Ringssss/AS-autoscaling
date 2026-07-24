# AgentShift Evaluation Environment

## Hardware

- Host GPUs: 8 x NVIDIA H100 80GB HBM3
- GPUs used by the TP=1 evaluation: GPU0 and GPU1
- GPU0/GPU1 topology: NV18
- NVIDIA driver: 580.65.06

## Software

- Python: 3.10.20
- PyTorch: 2.10.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- NCCL: 2.27.5
- SGLang package: 0.5.12.post2.dev500+g034dd3918
- SGLang base commit: 034dd39189ba1ace1308d3c8a58df275ef301a21
- AgentShift SGLang worktree: `/home/zhujianian/agentshift-sglang`
- AgentShift controller worktree: `/home/zhujianian/agentshift`

## Model

- Weights: `/mnt/models/Qwen3-8B`
- Dtype: BF16
- Layers: 36
- Attention heads / KV heads: 32 / 8
- Head dimension: 128
- Maximum context: 40,960 tokens

## Server Configuration

Both engines use the same flags. GPU visibility and port differ.

```text
python -m sglang.launch_server
  --model-path /mnt/models/Qwen3-8B
  --host 127.0.0.1
  --port 31000|31001
  --mem-fraction-static 0.82
  --max-total-tokens 270000
  --page-size 1
  --chunked-prefill-size 4096
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
```

The server reports FA3 attention, FlashInfer sampling, 270,000 KV tokens per
engine, and a 40,960-token context limit.

## Comparison Discipline

- Every mechanism uses the same model weights, dtype, SGLang build, endpoints,
  prompt tokens, output lengths, and GPU pair.
- Scenario order is randomized per repeat where the runner supports multiple
  strategies.
- Transfer and CPU-tier allocators are warmed before measured samples.
- Failed and skipped runs are retained or reported separately; interrupted
  smoke samples are excluded from aggregate result files.
- Agentix/Autellix-style routing, Continuum-style TTL, TokenCake-style tiering,
  and Symphony-style shared prefetch are mechanism-equivalent implementations
  in this SGLang testbed, not official artifact reproductions.
- CUDA Graph is disabled in this testbed. Relative strategy comparisons are
  aligned, but absolute serving latency should be revalidated with production
  graph settings before submission.
