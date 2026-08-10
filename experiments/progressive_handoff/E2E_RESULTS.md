# Progressive Handoff E2E Results

## Configuration

- Model: `/mnt/models/Qwen3-8B-sglang-tp2` (36 layers)
- Engines: two local SGLang engines, TP=2 each, CUDA Graph disabled
- Prefix: 32,768 input tokens plus four first-turn output tokens
- Continuation: 771 tool-result tokens and one measured output token
- Progressive granularity: 12 layers per group
- Metric: wall time from tool completion until the next output is available
- Baseline: unchanged `agentshift` scenario
- Increment: opt-in `adaptive-progressive` scenario

## Short-Tool Paired Result

Raw artifact:
`results/progressive-e2e-32k-adaptive-git-20/real-tools-all-1786344429892748638.json`

| Metric | AgentShift | Adaptive progressive |
|---|---:|---:|
| Runs | 20 | 20 |
| Median post-tool latency | 136.502 ms | 122.499 ms |
| Mean post-tool latency | 138.576 ms | 124.114 ms |
| Full prefix hits | 20/20 | 20/20 |

The ratio-of-medians reduction is 10.26%. Per-pair reductions have a 10.999%
median and 10.372% mean; the bootstrapped 95% confidence interval for the mean
is [8.632%, 12.092%]. Progressive execution won 20/20 pairs.

## Adaptive Guard

Raw artifact:
`results/progressive-e2e-32k-adaptive/real-tools-all-1786344348577708292.json`

The policy selected progressive execution for all five short-tool trials and
the original AgentShift path for all ten long-tool trials. The two long-tool
paired mean changes were -0.59% and +2.89%, with only five pairs each; these are
treated as noise rather than a performance claim. The important result is that
the forced-progressive 10--20 ms regression disappears when no transfer tail
remains to overlap.

## Correctness

Run:

```bash
PYTHONPATH=. python scripts/validate_progressive_e2e.py \
  --prefix-length 32768 \
  --layer-group-size 12 \
  --tp-size 2
```

For an identical prompt, baseline and progressive execution produced identical
four-token first turns and identical eight-token continuations. Both reported
a complete 32,772-token prefix hit. The performance runs also reported 70/70
complete prefix hits across the 20-pair short-tool run and the five-pair
three-tool adaptive run.

## Interpretation

The earlier conservative oracle predicted 22.2--26.6%; the measured gain is
about 10%, so the oracle overestimated overlap by roughly twofold. The method
has a real but bounded value: it removes part of the exposed transfer tail when
continuation tokens become available early enough. It does not improve a handoff
whose transfer is already fully hidden by the tool-blocked interval.
