# Fencing Microbenchmark

This experiment distinguishes prefix availability from execution authority. A
Qwen3-32B TP=4 source and destination both hold the same completed prefix. The
benchmark then injects the same logical next step at both engines. Router-only
requests bypass the managed ingress; the fenced path checks the owner epoch and
claims the step before issuing `/generate`.

## Duplicate-delivery result

The main run uses a 16,388-token completed prefix, 128 output tokens, and five
repetitions. Every executed turn reports a full prefix hit. Each handoff moves
4,296,015,872 bytes.

| Mechanism | Accepted executors | Generate RPCs | Generated tokens | Wasted tokens | Wasted GPU-s |
|---|---:|---:|---:|---:|---:|
| Router-only | 2.0 | 2.0 | 256 | 128 (50%) | 25.95 |
| Epoch + step fencing | 1.0 | 1.0 | 128 | 0 | 0 |

Router-only execution sends one real request to each four-GPU engine. The stale
source consumes 6.49 engine-seconds per logical step. With fencing, all five
stale source requests are rejected before a source `/generate` RPC; mean stale
rejection latency is 62.8 us.

Raw result:
`qwen32b-tp4-16k-128t/fencing-microbench-1786008706120873629.json`.

## Normal-path control cost

A separate ten-repetition Qwen3-32B TP=4 run uses a 4,100-token completed prefix
and 16 output tokens to time the managed ingress directly. The pre-GPU step
claim averages 0.257 ms. The post-GPU durable continuation commit averages
1.550 ms. Their combined mean is 1.807 ms, with 3.788 ms p99 in this small run.
Failure-free end-to-end means are within decode-time noise: 0.885 s for direct
router execution and 0.868 s for managed execution. The negative difference is
not interpreted as a speedup.

Raw result:
`qwen32b-tp4-control-overhead/fencing-microbench-1786008850759397964.json`.

## Scope

- Duplicate delivery is injected to model stale routing, lost-ACK retry, or
  controller recovery. The experiment does not estimate their production rate.
- Temperature-zero executions produce identical output digests. The measured
  consequence is duplicate compute and potential duplicate downstream
  submission, not observed continuation divergence.
- Fencing does not save the prefix-transfer bytes already spent before commit.
  It prevents a stale replica from converting delivery ambiguity into another
  GPU execution.
- A linearizable centralized sequencer would provide an alternative, equivalent
  single-winner fence; owner epochs are not claimed as the only implementation.
