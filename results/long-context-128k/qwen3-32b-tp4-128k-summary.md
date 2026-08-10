# Blocked-Window Baseline Summary

All named literature baselines are mechanism-equivalent implementations in the same SGLang testbed, not official reproductions.

Source artifact: `results/long-context-128k/blocked-window-1785138109761090702.json`

| Prefix | Gap | Strategy | Mean post-tool | p95 | Full hit | vs reroute | vs on-return | vs sticky |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 130000 | 0 ms | sticky | 157.18 ms | 163.74 ms | 100% | 75.38x | 2.40x | 1.00x |
| 130000 | 0 ms | reroute | 11848.96 ms | 11854.58 ms | 0% | 1.00x | 0.03x | 75.38x |
| 130000 | 0 ms | on-return | 377.32 ms | 384.93 ms | 100% | 31.40x | 1.00x | 2.40x |
| 130000 | 0 ms | agentshift | 354.79 ms | 360.40 ms | 100% | 33.40x | 1.06x | 2.26x |
| 130000 | 100 ms | sticky | 150.96 ms | 151.98 ms | 100% | 78.56x | 2.55x | 1.00x |
| 130000 | 100 ms | reroute | 11858.79 ms | 11885.23 ms | 0% | 1.00x | 0.03x | 78.56x |
| 130000 | 100 ms | on-return | 384.23 ms | 390.26 ms | 100% | 30.86x | 1.00x | 2.55x |
| 130000 | 100 ms | agentshift | 290.93 ms | 303.26 ms | 100% | 40.76x | 1.32x | 1.93x |
| 130000 | 250 ms | sticky | 151.40 ms | 151.98 ms | 100% | 78.18x | 2.46x | 1.00x |
| 130000 | 250 ms | reroute | 11836.65 ms | 11846.43 ms | 0% | 1.00x | 0.03x | 78.18x |
| 130000 | 250 ms | on-return | 373.06 ms | 377.27 ms | 100% | 31.73x | 1.00x | 2.46x |
| 130000 | 250 ms | agentshift | 156.69 ms | 164.96 ms | 100% | 75.54x | 2.38x | 1.03x |
| 130000 | 500 ms | sticky | 156.78 ms | 166.16 ms | 100% | 75.66x | 2.36x | 1.00x |
| 130000 | 500 ms | reroute | 11861.88 ms | 11911.98 ms | 0% | 1.00x | 0.03x | 75.66x |
| 130000 | 500 ms | on-return | 369.50 ms | 371.63 ms | 100% | 32.10x | 1.00x | 2.36x |
| 130000 | 500 ms | agentshift | 153.30 ms | 156.51 ms | 100% | 77.38x | 2.41x | 0.98x |
| 130000 | 1000 ms | sticky | 156.50 ms | 160.81 ms | 100% | 75.81x | 2.45x | 1.00x |
| 130000 | 1000 ms | reroute | 11864.44 ms | 11867.03 ms | 0% | 1.00x | 0.03x | 75.81x |
| 130000 | 1000 ms | on-return | 383.32 ms | 391.27 ms | 100% | 30.95x | 1.00x | 2.45x |
| 130000 | 1000 ms | agentshift | 153.15 ms | 154.15 ms | 100% | 77.47x | 2.50x | 0.98x |
