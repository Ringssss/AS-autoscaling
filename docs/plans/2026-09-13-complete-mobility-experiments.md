# Complete Mobility Experiment Matrix

**Goal:** Move the hybrid-state mobility prototype from correctness milestones to a complete elastic agentic-serving evaluation.

**Hypothesis:** Branch-valid heterogeneous state mobility reduces drain time and preserves continuation SLOs better than sticky, token-only, and KV-only policies, provided transfers are overlapped with tool waits and scheduled with dual-resource capacity.

**Validation scope:**

- L0: transfer latency, group reuse, concurrent-serving interference, GPU memory and pool accounting.
- L1: state descriptor, branch nesting, owner epoch, fault accounting, and policy-decision correctness.
- L3: Kimi-timing/LooGLE-content/ToolBench-boundary replay on real engines under scale-in/out and return bursts.
- Training L2 remains not applicable.

## Experiment Matrix

1. **Non-blocking direct transfer**
   - Add start/poll/cleanup semantics for hybrid bundle transfer.
   - Source completed-prefix reads run outside scheduler forwarding.
   - Destination reserves token/snapshot slots before public visibility.
   - Install only after complete bundle and descriptor checks.

2. **Concurrent interference**
   - Background decode/prefill on source and destination.
   - Compare no-transfer, synchronous transfer, and non-blocking transfer.
   - Measure foreground TTFT/TBT/E2E and transfer completion.

3. **Trace-window scale**
   - Run 32 real replay agents, not only four.
   - Include source/destination continuation and exact parent-prefix hits.
   - Measure migration utility under real trace-shaped tool gaps.

4. **Global policy sensitivity**
   - Sweep bandwidth, tool-gap scale, token capacity, snapshot capacity, scale-in time, and deadline.
   - Compare sticky, token-only, KV-only invalid-state control, and heterogeneous-aware policy.

5. **Real three-engine elasticity**
   - Engines on GPU0/GPU1/GPU3.
   - Generate warm agents from the replay manifest.
   - Inject scale-out and scale-in.
   - Evacuate owners from retiring engine during tool waits.
   - Measure drain-to-idle, migration bytes, continuation TTFT, and owner distribution.

6. **Branch and fault matrix**
   - Branch fanout, stale branch, tool early return, destination failure, partial install, and source/destination mismatch.
   - Success requires source owner preservation or deterministic recovery.

## Current State

Already complete:

- two-engine correctness;
- full bundle direct transfer;
- Mamba ablation;
- durable owner commit;
- trace-shaped workload generation;
- initial global policy simulation.

Not yet complete:

- non-blocking concurrent transfer;
- full SLO elasticity replay;
- crash and branch matrix.
