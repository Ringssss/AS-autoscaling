# Blocked-Window Baseline Summary

All named literature baselines are mechanism-equivalent implementations in the same SGLang testbed, not official reproductions.

Source artifact: `results/long-context-128k/blocked-window-1785136832475469481.json`

| Prefix | Gap | Strategy | Mean post-tool | p95 | Full hit | vs reroute | vs on-return | vs sticky |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 130000 | 0 ms | sticky | 86.42 ms | 92.48 ms | 100% | 127.85x | 3.82x | 1.00x |
| 130000 | 0 ms | reroute | 11049.44 ms | 11107.33 ms | 0% | 1.00x | 0.03x | 127.85x |
| 130000 | 0 ms | on-return | 330.08 ms | 341.05 ms | 100% | 33.48x | 1.00x | 3.82x |
| 130000 | 0 ms | agentshift | 308.12 ms | 309.82 ms | 100% | 35.86x | 1.07x | 3.57x |
| 130000 | 100 ms | sticky | 81.94 ms | 82.30 ms | 100% | 134.95x | 4.04x | 1.00x |
| 130000 | 100 ms | reroute | 11057.91 ms | 11074.51 ms | 0% | 1.00x | 0.03x | 134.95x |
| 130000 | 100 ms | on-return | 331.38 ms | 336.28 ms | 100% | 33.37x | 1.00x | 4.04x |
| 130000 | 100 ms | agentshift | 245.33 ms | 263.05 ms | 100% | 45.07x | 1.35x | 2.99x |
| 130000 | 250 ms | sticky | 83.31 ms | 83.75 ms | 100% | 132.67x | 4.02x | 1.00x |
| 130000 | 250 ms | reroute | 11052.27 ms | 11078.11 ms | 0% | 1.00x | 0.03x | 132.67x |
| 130000 | 250 ms | on-return | 335.29 ms | 349.61 ms | 100% | 32.96x | 1.00x | 4.02x |
| 130000 | 250 ms | agentshift | 83.54 ms | 87.11 ms | 100% | 132.30x | 4.01x | 1.00x |
| 130000 | 500 ms | sticky | 84.73 ms | 85.25 ms | 100% | 130.29x | 4.06x | 1.00x |
| 130000 | 500 ms | reroute | 11039.62 ms | 11078.61 ms | 0% | 1.00x | 0.03x | 130.29x |
| 130000 | 500 ms | on-return | 344.04 ms | 354.52 ms | 100% | 32.09x | 1.00x | 4.06x |
| 130000 | 500 ms | agentshift | 84.41 ms | 87.35 ms | 100% | 130.79x | 4.08x | 1.00x |
| 130000 | 1000 ms | sticky | 84.76 ms | 86.93 ms | 100% | 130.22x | 3.99x | 1.00x |
| 130000 | 1000 ms | reroute | 11037.41 ms | 11054.73 ms | 0% | 1.00x | 0.03x | 130.22x |
| 130000 | 1000 ms | on-return | 338.34 ms | 359.92 ms | 100% | 32.62x | 1.00x | 3.99x |
| 130000 | 1000 ms | agentshift | 90.88 ms | 103.35 ms | 100% | 121.45x | 3.72x | 1.07x |
