> 归档说明（2026-09-23）：已按要求清理原始实验产物；文字报告及最终统计保留。旧的原始数据／图表链接可能失效，详见 [文字归档](archive/2026-09-23-text-records/README.md)。

# FluidKV advantage boundary: measured comparisons

Three nodes, TP2 per node, Qwen3.6; models and communication groups prewarmed. Ratios >1 favor FluidKV. Screening has one repetition; it does not establish stability. Held-out runs use five paired repetitions; only the persistence campaign retains caches across three waves. Source-withdrawal runs have two eligible targets and do not physically retire GPUs. Bootstrap intervals resample runs, not individual requests. Precompute receives the same known prefix and tool window and uses parallel targets. Labels ending in :sticky compare that router plus FluidKV against direct warm-source return.

| Campaign | Tokens | Concurrency/engine | Wave | Comparator | n | TTFT ratio [95% CI] | p95 ratio | Throughput ratio | Errors |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| heldout-low-concurrency | 65536 | 4 | 0 | cacheroute | 5 | 7.32 [7.16, 7.51] | 6.57 | 1.81 | 0 |
| heldout-low-concurrency | 65536 | 4 | 0 | cacheroute+precompute | 5 | 7.73 [7.58, 7.97] | 7.00 | 1.92 | 0 |
| heldout-low-concurrency | 65536 | 4 | 0 | cacheroute:sticky | 5 | 0.82 [0.81, 0.84] | 0.95 | 1.00 | 0 |
| heldout-low-concurrency | 65536 | 4 | 0 | dualmap | 5 | 5.46 [5.37, 5.54] | 5.04 | 1.80 | 0 |
| heldout-low-concurrency | 65536 | 4 | 0 | dualmap+precompute | 5 | 6.02 [5.92, 6.09] | 5.40 | 1.88 | 0 |
| heldout-low-concurrency | 65536 | 4 | 0 | dualmap:sticky | 5 | 0.61 [0.59, 0.62] | 0.68 | 0.96 | 0 |
| heldout-persistence | 65536 | 16 | 0 | cacheroute | 5 | 5.32 [5.28, 5.43] | 4.17 | 1.50 | 0 |
| heldout-persistence | 65536 | 16 | 0 | cacheroute+precompute | 5 | 5.85 [5.78, 5.95] | 4.79 | 1.65 | 0 |
| heldout-persistence | 65536 | 16 | 0 | dualmap | 5 | 1.48 [1.41, 1.49] | 2.53 | 1.43 | 0 |
| heldout-persistence | 65536 | 16 | 0 | dualmap+precompute | 5 | 4.75 [4.70, 4.77] | 3.29 | 1.60 | 0 |
| heldout-persistence | 65536 | 16 | 1 | cacheroute | 5 | 1.01 [0.90, 1.04] | 0.95 | 0.97 | 0 |
| heldout-persistence | 65536 | 16 | 1 | cacheroute+precompute | 5 | 1.01 [0.91, 1.07] | 0.97 | 0.97 | 0 |
| heldout-persistence | 65536 | 16 | 1 | dualmap | 5 | 1.03 [1.01, 1.06] | 0.99 | 1.00 | 0 |
| heldout-persistence | 65536 | 16 | 1 | dualmap+precompute | 5 | 1.04 [1.01, 1.08] | 1.01 | 1.01 | 0 |
| heldout-persistence | 65536 | 16 | 2 | cacheroute | 5 | 1.03 [1.00, 1.05] | 0.99 | 0.97 | 0 |
| heldout-persistence | 65536 | 16 | 2 | cacheroute+precompute | 5 | 1.02 [0.99, 1.05] | 1.02 | 1.00 | 0 |
| heldout-persistence | 65536 | 16 | 2 | dualmap | 5 | 1.03 [1.02, 1.04] | 1.00 | 1.00 | 0 |
| heldout-persistence | 65536 | 16 | 2 | dualmap+precompute | 5 | 1.04 [1.03, 1.06] | 1.01 | 1.00 | 0 |
| heldout-source-withdrawal | 65536 | 4 | 0 | cacheroute | 5 | 9.61 [9.55, 10.43] | 8.67 | 1.72 | 0 |
| heldout-source-withdrawal | 65536 | 4 | 0 | cacheroute+precompute | 5 | 9.78 [9.70, 10.57] | 8.91 | 1.74 | 0 |
| heldout-source-withdrawal | 65536 | 4 | 0 | dualmap | 5 | 6.89 [6.16, 7.05] | 7.08 | 1.72 | 0 |
| heldout-source-withdrawal | 65536 | 4 | 0 | dualmap+precompute | 5 | 7.02 [6.33, 7.20] | 6.95 | 1.71 | 0 |
| screen-window-0 | 8192 | 4 | 0 | cacheroute | 1 | 1.49 | 1.46 | 1.01 | 0 |
| screen-window-0 | 8192 | 4 | 0 | cacheroute+precompute | 1 | 2.09 | 2.06 | 1.10 | 0 |
| screen-window-0 | 8192 | 4 | 0 | dualmap | 1 | 1.41 | 1.41 | 1.02 | 0 |
| screen-window-0 | 8192 | 4 | 0 | dualmap+precompute | 1 | 1.95 | 1.94 | 1.07 | 0 |
| screen-window-0 | 8192 | 16 | 0 | cacheroute | 1 | 1.41 | 1.38 | 0.99 | 0 |
| screen-window-0 | 8192 | 16 | 0 | cacheroute+precompute | 1 | 1.99 | 1.98 | 1.06 | 0 |
| screen-window-0 | 8192 | 16 | 0 | dualmap | 1 | 1.26 | 1.03 | 1.04 | 0 |
| screen-window-0 | 8192 | 16 | 0 | dualmap+precompute | 1 | 1.85 | 1.67 | 1.07 | 0 |
| screen-window-0 | 65536 | 4 | 0 | cacheroute | 1 | 5.96 | 5.28 | 1.57 | 0 |
| screen-window-0 | 65536 | 4 | 0 | cacheroute+precompute | 1 | 6.47 | 5.80 | 1.68 | 0 |
| screen-window-0 | 65536 | 4 | 0 | dualmap | 1 | 4.29 | 4.08 | 1.57 | 0 |
| screen-window-0 | 65536 | 4 | 0 | dualmap+precompute | 1 | 4.92 | 4.40 | 1.64 | 0 |
| screen-window-0 | 65536 | 16 | 0 | cacheroute | 1 | 3.83 | 3.16 | 1.44 | 0 |
| screen-window-0 | 65536 | 16 | 0 | cacheroute+precompute | 1 | 4.45 | 3.86 | 1.62 | 0 |
| screen-window-0 | 65536 | 16 | 0 | dualmap | 1 | 1.13 | 2.20 | 1.38 | 0 |
| screen-window-0 | 65536 | 16 | 0 | dualmap+precompute | 1 | 3.85 | 2.98 | 1.58 | 0 |
| screen-window-0.5 | 8192 | 4 | 0 | cacheroute | 1 | 1.81 | 1.73 | 1.02 | 0 |
| screen-window-0.5 | 8192 | 4 | 0 | cacheroute+precompute | 1 | 1.49 | 1.45 | 1.04 | 0 |
| screen-window-0.5 | 8192 | 4 | 0 | dualmap | 1 | 1.71 | 1.71 | 1.04 | 0 |
| screen-window-0.5 | 8192 | 4 | 0 | dualmap+precompute | 1 | 1.47 | 1.47 | 1.03 | 0 |
| screen-window-0.5 | 8192 | 16 | 0 | cacheroute | 1 | 1.69 | 1.64 | 1.01 | 0 |
| screen-window-0.5 | 8192 | 16 | 0 | cacheroute+precompute | 1 | 1.45 | 1.42 | 1.02 | 0 |
| screen-window-0.5 | 8192 | 16 | 0 | dualmap | 1 | 1.46 | 1.17 | 1.06 | 0 |
| screen-window-0.5 | 8192 | 16 | 0 | dualmap+precompute | 1 | 1.38 | 1.30 | 1.03 | 0 |
| screen-window-0.5 | 65536 | 4 | 0 | cacheroute | 1 | 10.27 | 8.43 | 1.67 | 0 |
| screen-window-0.5 | 65536 | 4 | 0 | cacheroute+precompute | 1 | 10.32 | 8.67 | 1.73 | 0 |
| screen-window-0.5 | 65536 | 4 | 0 | dualmap | 1 | 6.24 | 5.62 | 1.66 | 0 |
| screen-window-0.5 | 65536 | 4 | 0 | dualmap+precompute | 1 | 6.67 | 5.74 | 1.70 | 0 |
| screen-window-0.5 | 65536 | 16 | 0 | cacheroute | 1 | 5.45 | 4.32 | 1.50 | 0 |
| screen-window-0.5 | 65536 | 16 | 0 | cacheroute+precompute | 1 | 6.00 | 4.97 | 1.62 | 0 |
| screen-window-0.5 | 65536 | 16 | 0 | dualmap | 1 | 1.39 | 2.60 | 1.42 | 0 |
| screen-window-0.5 | 65536 | 16 | 0 | dualmap+precompute | 1 | 4.57 | 3.32 | 1.59 | 0 |
| screen-window-10 | 65536 | 4 | 0 | cacheroute | 1 | 13.36 | 11.08 | 1.93 | 0 |
| screen-window-10 | 65536 | 4 | 0 | cacheroute+precompute | 1 | 0.98 | 0.98 | 1.00 | 0 |
| screen-window-10 | 65536 | 4 | 0 | cacheroute:sticky | 1 | 1.49 | 1.62 | 1.07 | 0 |
| screen-window-10 | 65536 | 4 | 0 | dualmap | 1 | 8.16 | 7.18 | 1.92 | 0 |
| screen-window-10 | 65536 | 4 | 0 | dualmap+precompute | 1 | 1.00 | 1.01 | 1.01 | 0 |
| screen-window-10 | 65536 | 4 | 0 | dualmap:sticky | 1 | 0.90 | 1.00 | 1.03 | 0 |
| screen-window-10 | 65536 | 16 | 0 | cacheroute | 1 | 7.30 | 5.44 | 1.75 | 0 |
| screen-window-10 | 65536 | 16 | 0 | cacheroute+precompute | 1 | 1.01 | 1.01 | 1.00 | 0 |
| screen-window-10 | 65536 | 16 | 0 | cacheroute:sticky | 1 | 2.21 | 2.23 | 1.23 | 0 |
| screen-window-10 | 65536 | 16 | 0 | dualmap | 1 | 1.42 | 3.48 | 1.66 | 0 |
| screen-window-10 | 65536 | 16 | 0 | dualmap+precompute | 1 | 1.02 | 1.03 | 1.01 | 0 |
| screen-window-10 | 65536 | 16 | 0 | dualmap:sticky | 1 | 1.66 | 1.42 | 1.14 | 0 |
| screen-window-5 | 8192 | 4 | 0 | cacheroute | 1 | 1.79 | 1.79 | 1.03 | 0 |
| screen-window-5 | 8192 | 4 | 0 | cacheroute+precompute | 1 | 1.01 | 1.06 | 1.01 | 0 |
| screen-window-5 | 8192 | 4 | 0 | dualmap | 1 | 1.70 | 1.72 | 1.07 | 0 |
| screen-window-5 | 8192 | 4 | 0 | dualmap+precompute | 1 | 1.02 | 1.02 | 1.00 | 0 |
| screen-window-5 | 8192 | 16 | 0 | cacheroute | 1 | 1.66 | 1.64 | 1.01 | 0 |
| screen-window-5 | 8192 | 16 | 0 | cacheroute+precompute | 1 | 0.99 | 0.98 | 0.98 | 0 |
| screen-window-5 | 8192 | 16 | 0 | dualmap | 1 | 1.45 | 1.17 | 1.06 | 0 |
| screen-window-5 | 8192 | 16 | 0 | dualmap+precompute | 1 | 1.03 | 1.02 | 1.01 | 0 |
| screen-window-5 | 65536 | 4 | 0 | cacheroute | 1 | 10.10 | 8.36 | 1.69 | 0 |
| screen-window-5 | 65536 | 4 | 0 | cacheroute+precompute | 1 | 4.06 | 3.53 | 1.25 | 0 |
| screen-window-5 | 65536 | 4 | 0 | dualmap | 1 | 6.29 | 5.67 | 1.68 | 0 |
| screen-window-5 | 65536 | 4 | 0 | dualmap+precompute | 1 | 2.92 | 2.61 | 1.23 | 0 |
| screen-window-5 | 65536 | 16 | 0 | cacheroute | 1 | 5.46 | 4.03 | 1.48 | 0 |
| screen-window-5 | 65536 | 16 | 0 | cacheroute+precompute | 1 | 2.69 | 2.17 | 1.19 | 0 |
| screen-window-5 | 65536 | 16 | 0 | dualmap | 1 | 1.07 | 2.08 | 1.36 | 0 |
| screen-window-5 | 65536 | 16 | 0 | dualmap+precompute | 1 | 1.65 | 1.42 | 1.12 | 0 |

Scope: shared-prefix returning bursts; not private agent histories, production tool traces, continuous mixed-load migration, or physical autoscaling. Source state creation is common prior work. TTFT starts at tool return and includes exposed preparation. Long tool windows also let precomputation hide its cost. No universal or exclusive algorithmic advantage follows from this experiment.
