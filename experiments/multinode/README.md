> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# Hygon multi-node investigation

Native Kimi K2.5 trace assessment: [KIMI_FLUIDKV_ASSESSMENT.md](KIMI_FLUIDKV_ASSESSMENT.md).
It audits all 245,555 raw Finish records and distinguishes observed opportunity
coverage from unidentifiable routing/migration counterfactuals.

Latest long-request/high-concurrency study: [HIGHBS_RESULTS.md](HIGHBS_RESULTS.md),
[complete commands](HIGHBS_COMMANDS.md), and [Qwen3.8 runtime status](QWEN38_STATUS.md).
Use the final shared-checkpoint-observation results rather than preliminary
heuristic-metadata runs when assessing routing/preparation gains.

Requested target: four nodes × eight cards, 16 TP2 serving engines. The current
measured component pilot uses **three nodes × two cards**, with the same local
Qwen3.6-35B-A3B BF16 checkpoint and serving image. Read
[PROTOCOL.md](PROTOCOL.md) before interpreting any performance result.

[RESULTS.md](RESULTS.md) contains the measured results, retained failures,
trace audit, initialization costs, and outstanding full-fleet requirements.

[TOOL_WINDOW_RESULTS.md](TOOL_WINDOW_RESULTS.md) adds the controlled tool-window
and foreground-load experiment, its limitations, and the fourth-node readiness
status. [TOOL_WINDOW_INSIGHTS.md](TOOL_WINDOW_INSIGHTS.md) explains the mechanism.

## Inputs and provenance

- [Kimi production trace release](https://github.com/Ringssss/AS-autoscaling/releases/tag/kimi-trace-2026-04-07):
  58,121,430-byte archive, SHA-256
  `e7d8fd50757208220b9bc5bbd111a9d491b94a89a6c42390ab345aab9a0c3513`.
  Local archive and release metadata are in `data/`.
- `audit_kimi.py` streams all 24 logs without extracting tar paths and records
  source-file hashes and line numbers. Normalized rows preserve source order;
  sort by observed arrival time before a timing replay.
- Public government-report text from `THUDM/LongBench`, downloaded with the
  user's `hfd.sh`, fills the isolated component tests. The first six report
  contexts are concatenated and tokenized with the local Qwen tokenizer.
  `data/longbench_tokens.provenance.json` records source rows and archive hash.
  These are not recovered Kimi conversations or original token IDs.
- [CacheRoute paper](https://arxiv.org/abs/2608.19677), downloaded as PDF with
  extracted text in `data/`. `cacheroute.py` implements Algorithm 1; it is not
  the authors' original code. The existing DualMap adapter uses the clean
  pinned upstream checkout `/home/zhujianian/DualMap`.

The complete audit is `results/kimi-audit/audit.json`. All 245,555 records have
redacted token IDs and no usable conversation/routing/session key. The 8,455
records with cache counts exceeding input length are flagged, not repaired.
Consequently the release cannot by itself establish a native production
cache-aware policy ranking. See the protocol for valid substituted-workload
claims and required additional identity fields.

```bash
PATH=/home/zhujianian/miniconda3/envs/download-tools/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com bash /home/zhujianian/hfd.sh THUDM/LongBench \
  --dataset --include README.md data.zip \
  --local-dir /home/zhujianian/AS-autoscaling/experiments/multinode/data/LongBench -x 16
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/prepare_longbench.py \
  --archive experiments/multinode/data/LongBench/data.zip \
  --model /home/zhujianian/Qwen3.6-35B-A3B \
  --output experiments/multinode/data/longbench_tokens.npz
```

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/audit_kimi.py \
  --archive experiments/multinode/data/kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07.tar.gz \
  --output experiments/multinode/results/new-audit
```

## Runtime

`launch_pilot.py` launches a task-specific container only on verified free
devices. It mounts the existing Hygon runtime overlay and the same read-only
model checkpoint. It does not replace or stop an existing container.

| Node | Cards | Endpoint |
|---|---|---|
| node27 | 6,7 | `http://12.11.3.27:31600` |
| node28 | 4,5 | `http://12.11.3.28:31600` |
| node29 | 0,1 | `http://12.11.3.29:31600` |

The fourth node is occupied by an existing eight-card vLLM workload. Other
node28 cards also have pre-existing workloads. `fleet_plan.json` describes the
requested **not-yet-launched** 32-card layout; the current migration prototype
is not a ready full-fleet scheduler.

The pilot uses 131,072 context, 262,144 KV slots, 128 Mamba slots, 16 maximum
running requests, 4096-token chunked prefill, Triton attention/MoE, no CUDA
graphs, and no custom all-reduce. Intrahost TP uses normal P2P. Cross-node
transfers explicitly load `NCCL_NET_PLUGIN=shca` and `NCCL_NET=IBext_v8` with
`/dev/infiniband` exposed. The launcher now pins `NCCL_IB_HCA=shca_0` following
a timeout with automatic HCA selection on the second target. The original
node27→node28 campaign used automatic selection; retain those configurations
as separate runs. The image's baked topology file is unset for actual
hardware discovery. Merely using an `ib0` IP address does not prove RDMA;
retain the RCCL `NET/IBext_v8` channel logs as evidence.

The Hygon overlay is built using `experiments/hygon_runtime/build_overlay.py`.
This pilot retains its synchronous/idle/empty-destination limitations.

```bash
python3 experiments/multinode/launch_pilot.py --node node27
python3 experiments/multinode/launch_pilot.py --node node28
python3 experiments/multinode/launch_pilot.py --node node29
```

These are launch commands, not restart commands. For the already-running
pilot, inspect with `ssh node28 docker logs --tail 30 zhujianian-as-multinode-pilot`.
Stop only this task's container with
`ssh node28 docker stop zhujianian-as-multinode-pilot` (repeat for other pilot nodes).

## Validation and component measurement

### Controlled tool-window / hot-owner experiment

`benchmark_tool_window.py` executes a real repository-search subprocess and
returns its actual output to the continuation. A **controlled minimum delay**
of 50 or 500 ms creates the tested tool window; neither is asserted to be a
production tool-duration measurement. An identical 64-token foreground
generation is offered to the source 200 ms after tool start under every mode.
The scheduler uses its normal continuous batching; the harness does not impose
an artificial single-request queue on the sticky baseline.

It compares staying on the warm source, routing to a cold idle target, and
copying during the tool window. This probes mechanisms, not the decisions of
DualMap/CacheRoute. A generated one-token warmup materializes the input snapshot;
this is not a full model-generated tool-call trajectory. All comparisons use
the same parent text and tool-output suffix. Baseline caches are reset equally
between isolated trials, consistent with the prototype's empty-target limit.

Both serial and concurrent batch shapes and the first RCCL collective are
prewarmed. Initialization time is reported separately. The same client clock
records actual tool return, transfer commit, continuation dispatch/first/final
token, and foreground dispatch/first/final token. A short tool window does not
wait for the copy before recording tool return. The summary retains exposed
copy time and foreground latency, not just the migrating request's benefit.

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_tool_window.py --base-port 30600 \
  --output experiments/multinode/results/new-tool-window
/home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/summarize_tool_window.py \
  experiments/multinode/results/new-tool-window
```

### Fourth-node readiness

`launch_pilot.py --node node30` is prepared, and the same image was downloaded
and verified on that node. Its current cards remain occupied; the launcher
refuses to share them. When two cards become available (default 0,1), it can
start the fourth TP2 engine. This is **four nodes/eight cards**, not 32 cards.

`run_four_node_tool_window.py` checks all four endpoints before beginning and
then measures source node27 to nodes28/29/30 **sequentially**. That campaign
would establish four-node reachability and pairwise tool-window behavior, not
a concurrent four-node load-balancing/scaling result. Its readiness check
currently fails at node30 and does not silently substitute a third node.

```bash
python3 experiments/multinode/launch_pilot.py --node node30
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/run_four_node_tool_window.py --base-port 30500 \
  --output experiments/multinode/results/four-node-tool-window
```

The existing validators now accept `--master-address` for cross-node groups:

```bash
PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/hybrid_state_mobility/validate_direct_two_engine.py \
  --source http://12.11.3.27:31600 --destination http://12.11.3.28:31600 \
  --master-address 12.11.3.27 --prefix-tokens 1024 --rendezvous-port 30300 \
  --output-dir experiments/multinode/results/new-smoke

PYTHONPATH=. /home/zhujianian/miniconda3/envs/sglang/bin/python \
  experiments/multinode/benchmark_routing_tax.py \
  --source http://12.11.3.27:31600 --destination http://12.11.3.28:31600 \
  --master-address 12.11.3.27 --base-port 30320 \
  --token-file experiments/multinode/data/longbench_tokens.npz \
  --prefix-tokens 1024 8192 32768 --repetitions 3 \
  --output experiments/multinode/results/new-routing-tax
```

Use unused rendezvous ranges (four ports for TP2); current groups persist until
engine restart. Run measurements sequentially when they share an engine.

The component harness randomizes case order, warms every prefill/continuation
shape and the lazy RCCL communicator, retains per-request raw measurements and
ownership databases, and requires exact output/cache/length/ownership checks.
It compares sticky warm source, cold destination, already-warm destination,
and synchronous migration on return. All measured TTFTs include the complete
client-observed path. No tool time is invented or subtracted.

Cold/warm destination controls have the same target hardware. Their TTFT
difference measures the cost of losing that particular executable prefix;
their ratio is **not** a DualMap or CacheRoute speedup. Actual policy outcomes
depend on how frequently those policies choose cold or warm destinations and
on queueing, interference, eviction and migration admission under load.

```bash
FLUIDKV_DUALMAP_SOURCE=/home/zhujianian/DualMap \
  /home/zhujianian/miniconda3/envs/sglang/bin/python -m pytest -q
```
