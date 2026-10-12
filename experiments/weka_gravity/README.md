# Weka trace applicability assessment

Read [ASSESSMENT.zh.md](ASSESSMENT.zh.md) for the full-trace findings and the
implementation assessment, and [MEASUREMENTS.zh.md](MEASUREMENTS.zh.md) for the
new three-node measurements. Generated data and raw records live in `results/`.

All commands run from the repository root. Python with numpy, matplotlib,
aiohttp and pytest is `/home/zhujianian/miniconda3/envs/sglang/bin/python`.

## Full-trace audit

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  -m pytest -q experiments/weka_gravity/test_audit.py
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/audit.py
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/plot.py
```

The source is the downloaded 1,847,151,435-byte JSONL. `requests.csv` preserves
conversation ID, source line, outer/inner request location, model, stream,
timing, lengths, prefix reuse and previous-stream overlap. It contains no
recovered token IDs. `audit.json` records the input SHA-256 and assumptions.

The event audit only makes completed inputs available. Cache namespaces are
conversation-local and model-specific. TTLs assume unlimited capacity and
expiry since last completed use. Gap opportunities are offline observations,
not online tool-duration predictions, load imbalance, or speedup estimates.

## Live measurements

`launch.py` starts task-specific `zhujianian-weka-gravity` containers only after
checking free cards. It reuses the existing image/overlay and mounts the shared
immutable Qwen checkpoint. Node27/28/29 use two cards each, port 31720. The
scripts flush these dedicated engines and must not target shared serving ports.

```bash
PYTHONPATH=. python3 experiments/weka_gravity/launch.py
```

Wait for the engines to become healthy. The component token file was generated
by repeating `experiments/multinode/data/longbench_tokens.npz` twice and retaining
98,336 tokens. Its provenance is in `results/component-provenance.json`.

Run component measurements sequentially because they share a source engine:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_routing_tax.py \
  --source http://12.11.3.27:31720 --destination http://12.11.3.28:31720 \
  --master-address 12.11.3.27 --base-port 32400 \
  --token-file experiments/weka_gravity/results/component_tokens.npz \
  --prefix-tokens 1024 32768 98304 --repetitions 3 \
  --output experiments/weka_gravity/results/node27-node28

PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_routing_tax.py \
  --source http://12.11.3.27:31720 --destination http://12.11.3.29:31720 \
  --master-address 12.11.3.27 --base-port 32420 \
  --token-file experiments/weka_gravity/results/component_tokens.npz \
  --prefix-tokens 1024 32768 98304 --repetitions 3 \
  --output experiments/weka_gravity/results/node27-node29

PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/benchmark_burst.py
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/diagnose_batch.py
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/probe_constraints.py
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/weka_gravity/summarize.py
```

Output directories for live campaigns must not already exist; archive prior
runs rather than overwrite them. The constraint probe deliberately poisons its
own final transfer group with a nonempty-destination rejection and is run last.

These experiments substitute public-text tokens and control preparation time.
They do not reproduce Claude inference, real tool execution, absolute fleet
arrivals, or all 98,827 requests on GPU. Each paired cell has three repetitions;
there is no claim about production p99 or full-fleet goodput.

The recorded burst campaign exits nonzero because strict cross-batch output
equivalence fails. Its raw outputs and failure remain part of the assessment.
`diagnose_batch.py` examines the same history serially and in a batch on node27,
without any state transfer; it does not waive the original correctness gate.

After collecting logs, stop only the three containers created for this task:

```bash
ssh node27 docker stop zhujianian-weka-gravity
ssh node28 docker stop zhujianian-weka-gravity
ssh node29 docker stop zhujianian-weka-gravity
```
