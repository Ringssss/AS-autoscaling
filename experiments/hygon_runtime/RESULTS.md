# Hygon local validation — 2026-09-22

Qwen3.6-35B-A3B BF16 now passes the project's synchronous hybrid-state
migration and durable ownership coordinator checks on the local Hygon machine.

- Source: HCU 2,3, TP=2, `http://127.0.0.1:31400`.
- Destination: HCU 4,5, TP=2, `http://127.0.0.1:31401`.
- Hardware: Hygon BW / gfx936, 65,520 MiB VRAM per device.
- Image ID: `sha256:ee3b9807d9e90cdc3e6ab9c1058f0b78dabcafa6a0fc7661d9c5af69ce1f9588`.
- Transport: RCCL network/socket, P2P and SHM disabled. No
  `AMD_SERIALIZE_KERNEL` override in the validated dual-engine configuration.
- Context capacity: 4096; KV capacity: 8192 tokens; 32 Mamba slots;
  Triton attention and BF16 MoE; CUDA graphs and custom all-reduce disabled.

| Prefix tokens | Actual state bytes (sum of both TP ranks) | Direct validator transfer wall time | Coordinator transfer wall time | Checks |
|---|---:|---:|---:|---|
| 256 | 69,632,000 | 0.572 s | 0.474 s | 7/7 + 5/5 passed |
| 1024 | 85,360,640 | 0.569 s | 0.555 s | 7/7 + 5/5 passed |

All generated 16-token continuations matched the source engine exactly. The
separate direct validator also matched cold inference exactly. Source and
destination each reported the complete migrated prefix cached, and the cold
baseline reported zero cached tokens. These tests use synthetic token-ID
prompts, not a real workload dataset.

The coordinator committed migration state `COMMITTED`, owner `destination`,
and epoch 8 from source epoch 7. Its existing validation script uses a temporary
SQLite database and records the checked state in JSON; that database is not a
persistent running controller service.

State includes full-attention KV, convolution buffers, and FP32 recurrent
state. The measured byte totals match `prefix_tokens * 20480 + 64389120`.

Negative live checks passed: invalid IDs return 409, unsupported operations
return 404, both engines reject migration into a nonempty destination cache,
and the existing 1024-token prefixes remain available after rejection.

Repository tests: **338 passed, 50 skipped**. Skipped external integrations
are not validated by this run. A regression test now prevents the controller
from committing ownership when an engine explicitly reports failure even if
its token and byte counts match.

These are correctness smoke measurements, one run per size and validator,
not performance benchmarks or a full autoscaling campaign. The implementation
is synchronous and requires idle engines with the same checkpoint and TP
layout and an empty destination cache. Async overlap, active decode mobility,
original KV-only endpoints, and multi-engine scaling remain outside this path.

The initial default transport configuration stalled at cross-engine broadcast.
Removing serialized kernel debugging and disabling P2P/SHM produced the
successful results above; this run does not isolate which setting caused the
stall. A separate two-device collective probe with fine-grained PCIe memory
and loopback networking also passed, but those settings were not added to the
validated model engines.

Artifacts (local, intentionally ignored by Git):

- `results/direct-256/direct_two_engine_result.json`
- `results/coordinator-256/hybrid_coordinator_result.json`
- `results/smoke-1024/direct/direct_two_engine_result.json`
- `results/smoke-1024/coordinator/hybrid_coordinator_result.json`
- `results/negative_checks.json`
- `results/source-final.log`, `results/destination-final.log`
- `results/runtime-versions.json`, `results/overlay-manifest.json`

Both test engines are left running. The pre-existing inference server on
31360 (HCU 0,1) was preserved. The temporary probe on HCU 6,7 has exited.
See [README.md](README.md) for launch, validation, and stop commands. Restart
the two test containers before reusing an already allocated rendezvous port
range; communicator groups live until engine shutdown.
