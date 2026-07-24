# AgentShift PoC: Frozen Scope and Implementation

## Claim

The PoC migrates a completed-turn prefix KV, durable continuation, and execution
ownership between identical SGLang instances while the agent is blocked on a tool.
The next turn runs on the destination without re-prefilling the historical prefix.

This is not active-request migration. There is no decode request during the cutover.

## Repositories and Environment

- Control plane: `/home/zhujianian/agentshift`
- SGLang worktree: `/home/zhujianian/agentshift-sglang`
- SGLang branch: `agentshift-poc`
- Base commit: `034dd39189ba1ace1308d3c8a58df275ef301a21`
- Runtime environment: `/home/zhujianian/miniconda3/envs/sglang-bench`
- Models: `/mnt/models/Qwen3-8B` and `/mnt/models/Qwen3-32B`

The original `/home/zhujianian/sglang` worktree was left untouched because it has
unrelated uncommitted experiments.

## Component 1: Durable Agent Runtime

`SQLiteStateStore` uses WAL and full synchronous commits for:

- continuation: committed step, token IDs, tool future, workspace reference, stream offset;
- ownership: engine and monotonically increasing epoch;
- migration state: PREPARING, COPYING, DEST_READY, COMMITTED, SOURCE_RELEASED,
  RECOVERED, ABORTED;
- tool mailbox: idempotent result delivery by `(agent, step, future)`;
- managed effects: PREPARED, SUBMITTED, COMPLETED, UNKNOWN;
- inference step claims: one durable claim for each `(agent, step)`.

`ManagedAgentExecutor` is the serving ingress. It rejects stale epochs and duplicate
steps before inference, then commits the returned token continuation atomically.

## Component 2: SGLang Completed-Prefix Mobility

`AgentPrefixManager` adds four lifecycle operations over existing RadixCache APIs:

1. Pin a page-aligned resident token key with `inc_lock_ref`.
2. Reserve destination KV slots, evicting only evictable cache when necessary.
3. Copy MHA K/V rows and install the destination token key with `insert`.
4. Unpin and reclaim source cache only after destination acknowledgement.

The SGLang HTTP control API exposes:

- `POST /agentshift/prefix/pin`
- `POST /agentshift/transfer_group/init`
- `POST /agentshift/prefix/transfer`
- `POST /agentshift/prefix/transfer/status`
- `POST /agentshift/prefix/transfer/cleanup`
- `POST /agentshift/prefix/release`
- `POST /agentshift/prefix/rebind`

## Component 3: Gap-Aware Semantic Handoff

Each corresponding source/destination TP rank creates a persistent two-rank custom
NCCL process group on a distinct port. A bounded persistent worker copies one layer
at a time on an independent CUDA stream. Scheduler RPCs reserve and enqueue work,
then return immediately; the controller polls all TP ranks and only accepts
`COMPLETE` after the destination Radix node is installed.

Every control operation is also aggregated across the engine's TP CPU group.
Status advances at the least advanced rank, failures on any rank propagate to
the controller, and physical bytes are summed across rank shards. This prevents
rank zero from exposing a destination prefix before all TP ranks install it.

After either a successful handoff or a failed transfer, the controller invokes
`/agentshift/prefix/transfer/cleanup` on both engines. Cleanup removes terminal
transfer records and stale source index tensors. On a failed destination it also
returns reservations that never became visible in RadixCache; on success the
installed KV remains owned by RadixCache and is released through the normal
prefix lifecycle.

`PlacementPolicy` admits a migration only when the destination has enough KV slots
and scores lower queue depth. `MigrationCoordinator` overlaps copy with the tool
window, then performs destination-ready, SQLite ownership CAS, mailbox rebinding,
and delayed source release. The synchronous path remains available only as an
interference baseline.

## Protocol Invariants

- Durable progress is authoritative; KV is reconstructable acceleration state.
- Ownership changes only by CAS from `source@epoch` to `destination@epoch+1`.
- Destination KV is not visible until every layer has arrived and the full token key
  matches the installed RadixCache node.
- A source copy remains pinned until commit and destination acknowledgement.
- Destination slots reserved by an in-flight async transfer remain visible to the
  strict SGLang pool invariant checker.
- Terminal transfer cleanup never frees successfully installed RadixCache rows and
  does reclaim failed destination reservations.
- A failed pre-commit transfer leaves ownership at the source and marks the migration
  ABORTED. Best-effort destination cleanup runs before returning the error.
- A post-commit failure may recover to the source shadow only by advancing durable
  ownership to `epoch+2`; the failed destination's `epoch+1` is fenced.
- Source release is retryable after lost acknowledgements.
- Managed effects in SUBMITTED or UNKNOWN are never automatically replayed.

## Supported Configuration

- same model and model revision;
- same TP degree and corresponding rank layout;
- MHA KV pool with RadixCache;
- page-aligned completed turns;
- single-node NCCL transport;
- TP=1, TP=2, and TP=4 tested on H100;
- Qwen3-8B and Qwen3-32B tested as compatible MHA targets.

## Explicit Non-Goals

- active decode migration;
- Python stack, browser, shell, or sandbox process migration;
- MLA, Mamba, hybrid SWA, or speculative draft KV;
- heterogeneous TP degree;
- cross-node transport;
- cold model loading or BlitzScale integration;
- arbitrary third-party exactly-once effects;
- distributed consensus or multi-controller failover.

## Reproduction

Start two TP=1 servers on separate GPUs using the worktree and `sglang-bench` env,
then run:

```bash
PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_e2e.py \
  --context-lengths 1024 4096 16384 32768 \
  --repeats 3 \
  --burst-context-length 2048 \
  --burst-concurrency 16 \
  --tool-result-tokens 1024 \
  --burst-output-tokens 128 \
  --burst-repeats 3
```

Correctness and trace calibration:

```bash
PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/validate_correctness.py --context-length 4096 --output-tokens 32

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/analyze_traces.py \
  --benchmark results/e2e-1784487320239905011.json
```

Async interference, blocked windows, real tools, burst, and recovery:

```bash
PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_interference.py

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_blocked_window.py

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_real_tools.py

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_burst_matrix.py

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/validate_recovery.py
```

The Qwen3-32B TP=4 headline configuration uses two terminals:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/home/zhujianian/agentshift-sglang/python \
  /home/zhujianian/miniconda3/envs/sglang-bench/bin/python \
  -m sglang.launch_server \
  --model-path /mnt/models/Qwen3-32B --host 127.0.0.1 --port 31200 \
  --tp-size 4 --mem-fraction-static 0.55 --max-total-tokens 45000 \
  --page-size 1 --chunked-prefill-size 4096 \
  --disable-cuda-graph --disable-piecewise-cuda-graph

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH=/home/zhujianian/agentshift-sglang/python \
  /home/zhujianian/miniconda3/envs/sglang-bench/bin/python \
  -m sglang.launch_server \
  --model-path /mnt/models/Qwen3-32B --host 127.0.0.1 --port 31201 \
  --tp-size 4 --mem-fraction-static 0.55 --max-total-tokens 45000 \
  --page-size 1 --chunked-prefill-size 4096 \
  --disable-cuda-graph --disable-piecewise-cuda-graph
```

Then run the three-policy locality sweep or the proactive/on-return gap sweep:

```bash
PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_e2e.py \
  --source http://127.0.0.1:31200 \
  --destination http://127.0.0.1:31201 \
  --context-lengths 4096 16384 32768 --repeats 3 \
  --transfer-port 29900 --tp-size 4

PYTHONPATH=/home/zhujianian/agentshift \
  /home/zhujianian/miniconda3/envs/crossstage/bin/python \
  scripts/benchmark_blocked_window.py \
  --source http://127.0.0.1:31200 \
  --destination http://127.0.0.1:31201 \
  --scenarios reroute on-return proactive \
  --prefix-lengths 32768 --gap-ms 0 25 50 100 250 --repeats 3 \
  --transfer-port 29920 --tp-size 4
```
