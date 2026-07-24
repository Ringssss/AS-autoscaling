# AgentShift Reproduction Guide

This guide reconstructs the evaluated single-node setup. Commands assume Linux,
two or more NVIDIA GPUs, CUDA, NCCL, and Python 3.10 or newer. The reported
environment used eight H100 80 GB GPUs, PyTorch 2.10.0+cu128, CUDA 12.8, and
NCCL 2.27.5.

## 1. Fetch AgentShift and SGLang

```bash
git clone https://github.com/Ringssss/AS-autoscaling.git
cd AS-autoscaling
export AGENTSHIFT_ROOT="$PWD"

git clone https://github.com/sgl-project/sglang.git ../agentshift-sglang
cd ../agentshift-sglang
git checkout 034dd39189ba1ace1308d3c8a58df275ef301a21
git apply "$AGENTSHIFT_ROOT/patches/agentshift-sglang.patch"
export SGLANG_ROOT="$PWD"
```

The patch contains eight files: HTTP and tokenizer control endpoints, scheduler
integration, TP result aggregation, strict allocator accounting, the completed
prefix manager, and its 14 unit tests. Verify the patch before installing:

```bash
git diff --check
git status --short
```

## 2. Install

Use an environment that already satisfies the upstream SGLang GPU dependencies.
Then install both worktrees in editable mode:

```bash
cd "$SGLANG_ROOT"
python -m pip install -e python

cd "$AGENTSHIFT_ROOT"
python -m pip install -e '.[test]'
```

The evaluation uses raw token IDs and Qwen3 MHA models. Set the local model path:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

## 3. Start Two Engines

Run each command in a separate terminal. Both engines must use the same model,
revision, dtype, TP degree, page size, and KV layout.

Terminal A:

```bash
export SGLANG_ROOT=/path/to/agentshift-sglang
export MODEL_PATH=/path/to/Qwen3-8B
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$SGLANG_ROOT/python" \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 1 \
  --mem-fraction-static 0.82 \
  --max-total-tokens 270000 \
  --page-size 1 \
  --chunked-prefill-size 4096 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --log-level warning
```

Terminal B uses GPU 1 and port 31001:

```bash
export SGLANG_ROOT=/path/to/agentshift-sglang
export MODEL_PATH=/path/to/Qwen3-8B
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$SGLANG_ROOT/python" \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 31001 \
  --tp-size 1 \
  --mem-fraction-static 0.82 \
  --max-total-tokens 270000 \
  --page-size 1 \
  --chunked-prefill-size 4096 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --log-level warning
```

Wait until both `/health` endpoints return HTTP 200.

## 4. Validate

Control plane:

```bash
cd "$AGENTSHIFT_ROOT"
pytest -q
```

SGLang prefix and TP aggregation:

```bash
cd "$SGLANG_ROOT"
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_agent_prefix_cache.py
```

Real 32K handoff:

```bash
cd "$AGENTSHIFT_ROOT"
PYTHONPATH=. python scripts/smoke_e2e.py \
  --source http://127.0.0.1:31000 \
  --destination http://127.0.0.1:31001 \
  --context-length 32768 \
  --agent-id smoke-32k \
  --transfer-port 29600
```

## 5. Reproduce Core Experiments

The runners randomize scenario order where supported and write timestamped JSON
under `results/`. Do not compare runs that use different model, KV capacity,
CUDA Graph, or TP settings.

Primary locality, gap, and literature-inspired baseline matrix:

```bash
PYTHONPATH=. python scripts/benchmark_blocked_window.py \
  --prefix-lengths 1024 4096 16384 32768 \
  --gap-ms 0 10 25 50 100 250 500 1000 \
  --repeats 3
```

Eight-agent, three-turn coding replay:

```bash
PYTHONPATH=. python scripts/benchmark_coding_agent_replay.py \
  --agents 8 \
  --turns 3 \
  --initial-prefix 16384 \
  --turn-growth 8192 \
  --max-prefix 32768 \
  --relocation-fraction 0.5 \
  --repeats 3 \
  --python "$(command -v python)"
```

Heterogeneous migration scheduling:

```bash
PYTHONPATH=. python scripts/benchmark_migration_policy.py
```

Return hotspots, TTL pressure, warm elasticity, and foreground interference:

```bash
PYTHONPATH=. python scripts/benchmark_hotspot.py \
  --prefix-lengths 32768 \
  --agent-counts 8 \
  --return-patterns simultaneous \
  --repeats 5

PYTHONPATH=. python scripts/benchmark_ttl_pressure.py
PYTHONPATH=. python scripts/benchmark_elasticity.py
PYTHONPATH=. python scripts/benchmark_interference.py \
  --prefix-lengths 32768 \
  --foreground-concurrencies 8 \
  --repeats 5
```

Fault campaign:

```bash
PYTHONPATH=. python scripts/validate_fault_matrix.py
```

CPU-only controller scaling does not require SGLang engines:

```bash
PYTHONPATH=. python scripts/benchmark_control_plane.py
```

## 6. Regenerate Figures and Paper

Figures use Arial and are generated from the curated JSON paths configured in
the script. Arial's license does not permit redistributing the TTF files in this
repository. Place licensed `Arial.TTF`, `Arialbd.TTF`, `Ariali.TTF`, and
`Arialbi.TTF` files in `paper_rewriting_output/fonts/`, or pass a directory
containing those files with `--font-dir`:

```bash
python scripts/generate_paper_figures.py \
  --font-dir paper_rewriting_output/fonts
```

The USENIX source is `paper_rewriting_output/final_paper/main.tex`. The checked
PDF was built with Tectonic 0.16.9:

```bash
cd paper_rewriting_output/final_paper
tectonic main.tex --keep-logs --keep-intermediates
cp main.pdf paper.pdf
```

## 7. Evaluated Scope

- Single node and direct NCCL GPU transfer.
- Qwen3-8B/32B MHA; TP=1, 2, and 4.
- Identical model revision and TP layout at source and destination.
- Completed turns only; no active decode migration.
- CUDA Graph disabled consistently across compared strategies.
- Agentix/Autellix, Continuum, TokenCake, and Symphony names refer to
  mechanism-equivalent local implementations, not official artifacts.

Cross-node RDMA, independent-machine crash-stop faults, heterogeneous TP, MLA,
Mamba, and full SWE-agent/BFCL integration remain outside this artifact.
