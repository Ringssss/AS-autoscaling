# AgentX three-node baseline campaign

The running matrix, scope, cache rules and metrics are specified in
[PROTOCOL.zh.md](PROTOCOL.zh.md).

- [Live progress](results/campaign-20260928-r3/LIVE.md)
- [Machine-readable status](results/campaign-20260928-r3/status.json)
- [Completed cells](results/campaign-20260928-r3/completed.json)
- [Launch configuration](results/launch.json)
- [Native cohort manifest](data/manifest.json)

The 160-cell, original-output-length matrix runs in the background on node27.
Its log is `results/campaign-r3.log`. BS=1 in particular can require days;
starting the campaign does not mean the matrix is already complete. No result
ranking is claimed before complete cells and failure/quality audits exist.

The fixed cohort has 128 substreams with nine original requests each. Every
scenario uses 384 requests over three persistent-cache phases. Only an
independent policy trajectory starts from reset caches. The same model-resident
fleet is used sequentially so baselines do not contaminate one another.

Reproduction components, from the repository root:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_baselines/prepare.py
python3 experiments/weka_baselines/build_runtime.py
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  -m pytest -q experiments/weka_baselines/test_protocol.py
PYTHONPATH=. python3 experiments/weka_baselines/launch.py
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_baselines/campaign.py
```

Use new experiment/container identifiers when reproducing; launch/output
directories intentionally refuse overwriting existing runs. The matrix driver
stops only its three dedicated containers at completion or failure. Do not
launch another runner against the current fleet while this campaign is active.

The first two initialization failures remain in their own attempt directories.
The third uses checked ports below the host ephemeral-port range and a task-local
control overlay with synchronized TP initialization error handling. Six directed
transfer/output checks and 48 capped transport/router smoke requests passed
before the full-length matrix. Smoke records are not included in main results.

Output-dependent numerical differences across batch shapes were observed in
the earlier investigation. This campaign retains output tokens/hashes; its
online goodput check certifies lengths, not semantic equivalence. There is one
independent run per matrix cell, with separate phase summaries. Kernel shape
warmup is limited to the declared calibration/preflight; first-use compilation
and randomized order need to be considered before interpreting small differences.
