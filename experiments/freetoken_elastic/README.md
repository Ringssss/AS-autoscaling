# State-preserving memory repartition experiments

Completed evidence and limitations: [2026-09-15 results](results/20260915/RESULTS.md).

The experiments use real BF16 Qwen3.5 MoE offload on H100 under separately allocated24/32GiB-class available-VRAM envelopes. These are not RTX performance measurements. The isolated environment and exact dependency hashes are recorded under `environment/` (ignored build artifacts), while the tested FreeToken source changes are retained in `patches/`.

Scripts:

- `hold_vram.py`: separate capacity-only ballast, charged separately from model memory.
- `launch_baseline.py`, `stop_baseline.py`: owned idle engine lifecycle.
- `client.py`, `build_manifest.py`: exact text/usage/cache observation and frozen source-text fixtures.
- `benchmark_repartition.py`: no-resize/expert-only/KV/recurrent destructive baseline.
- `benchmark_preservation.py`: matched-final-geometry drop/preserve comparison.
- `bootstrap/`: explicit opt-in serialized research policy control; not a production multi-tenant API.
- `report_repartition.py`, `report_preservation.py`, `plot_overview.py`: raw artifact audits and plots.
- `watch_memory.py`, `watch_pcie.py`: external observation with explicit sampling limits.

Start only on dedicated available hardware and match the recorded engine's actual auto-resolved geometry. Preserve or stop any foreground work before a cache rebuild. Do not infer legal recurrent checkpoints from arbitrary KV matches. The full suite and GPU gates passed for the recorded successful cases; generated-text/quality equivalence, general failure recovery and other model pool families remain unestablished.
