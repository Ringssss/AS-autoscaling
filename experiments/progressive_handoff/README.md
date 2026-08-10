# Progressive Handoff Oracle

This replay asks whether AgentShift can privately consume immutable KV layers
at the destination before the complete prefix is installed in RadixCache.

It uses the measured Qwen3-8B, 32K-prefix, short `git-status` result and models
the handoff as a two-stage flow shop:

```text
copy KV layer group i -> publish private readiness -> compute layer group i
```

The destination still publishes no output and installs no shared prefix until
the full TP-wide migration commits. Therefore this oracle changes scheduling,
not ownership or externally visible semantics.

Run from the AgentShift repository root:

```bash
python3 experiments/progressive_handoff/oracle.py
```

The four profiles sweep concurrent copy/compute slowdown and TP straggler
overhead. Results are an opportunity bound, not a measured implementation:
the source data contains aggregate transfer and continuation times, so the
replay assumes uniform per-layer costs.

## Initial result

The measured case has a 51.399 ms sticky continuation, a 61.815 ms copy tail
after the tool returns, and 113.238 ms current AgentShift post-tool latency.
The inferred 70.740 ms transfer has completed about 4.54 of 36 layers when the
tool returns.

With a 10 us publication cost per group:

| Profile | Concurrent slowdown | TP straggler | 1-layer groups | 8-layer groups |
|---|---:|---:|---:|---:|
| Ideal | 0% | 0% | 63.603 ms (-43.8%) | 71.129 ms (-37.2%) |
| Mild | 10% | 5% | 72.137 ms (-36.3%) | 78.526 ms (-30.7%) |
| Conservative | 25% | 10% | 83.170 ms (-26.6%) | 88.147 ms (-22.2%) |
| Pessimistic | 50% | 20% | 102.737 ms (-9.3%) | 105.559 ms (-6.8%) |

The conservative profile passes a 20% opportunity threshold for every tested
group size. This justifies a prototype, but does not establish an end-to-end
speedup: real layer costs are nonuniform, GPU copy/compute interference must be
measured, and SGLang currently has no private partially installed prefix view.

The research claim must also be stronger than layerwise transfer. The candidate
protocol is prepare-execute-commit: exact layer state may be consumed privately
at the destination, while public RadixCache visibility, ownership transfer,
token output, and managed effects remain fenced until the full handoff commits.

## Opt-in prototype

The local `agentshift-sglang` and `agentshift` trees now contain an opt-in
implementation of that protocol. The original handoff remains the default.

```text
destination copy stream
  copy KV group i -> record CUDA event i -> private readiness
                                              |
destination compute stream                    v
  wait event i -> existing forward_split_prefill(group i)
                                              |
                                              v
full copy + TP barrier -> RadixCache install -> controller ownership commit
                       -> return generated output
```

The prototype reuses SGLang's existing Qwen3/Qwen3-MoE split-prefill path. It
does not add or replace an attention, GEMM, or KV-copy kernel. New request and
transfer fields default to disabled, so an unmodified AgentShift experiment
continues to use the full-transfer barrier.

The direct comparison keeps the original `agentshift` path unchanged. The
`adaptive-progressive` scenario selects progressive execution only when the
measured transfer estimate exceeds the predicted tool duration:

```bash
PYTHONPATH=. python scripts/benchmark_real_tools_all.py \
  --source http://127.0.0.1:31000 \
  --destination http://127.0.0.1:31001 \
  --scenarios agentshift adaptive-progressive \
  --prefix-lengths 32768 \
  --progressive-layer-group-size 12 \
  --tp-size 2
```

Both engines must run the local `agentshift-sglang` tree. The initial prototype
requires PP=1 and isolates the progressive continuation in its own prefill
batch. Its uncached continuation suffix must fit without SGLang chunked
prefill. These restrictions do not affect the baseline path.

Transfer status reports `published_layers` (the TP-local event has been
recorded), `ready_layers` (the event has completed), `layer_group_size`, and
`total_layers`. TP aggregation takes the least advanced rank for readiness.

## Measured end-to-end result

On two local TP=2 Qwen3-8B engines, a 32K prefix, and the short `git-status`
tool, 20 paired runs reduced median post-tool latency from 136.502 ms to
122.499 ms (10.26% by ratio of medians). The mean paired improvement was
10.37% with a 95% bootstrap confidence interval of [8.63%, 12.09%], and the
progressive path won all 20 pairs. All 40 requests had a complete prefix hit.

The same adaptive policy selected the unchanged AgentShift path for two longer
tools whose blocked intervals already hid the complete migration. Their
post-tool latency remained within run-to-run noise instead of paying the
10--20 ms split-prefill overhead observed when progressive execution was forced.
This makes the result conditional: progressive handoff helps when tool output
becomes available before transfer completion, not after the transfer is fully
hidden.

`scripts/validate_progressive_e2e.py` additionally replays the same 32K prompt
through the baseline and progressive paths. The first-turn outputs, eight-token
continuations, and full-prefix-hit checks match. Raw measurements and the full
methodology are recorded in `E2E_RESULTS.md`.
