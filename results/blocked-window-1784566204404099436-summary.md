# Blocked-Window Baseline Summary

All named literature baselines are mechanism-equivalent implementations in the same SGLang testbed, not official reproductions.

Source artifact: `results/blocked-window-1784566204404099436.json`

| Prefix | Gap | Strategy | Mean post-tool | p95 | Full hit | vs reroute | vs on-return | vs sticky |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 500 ms | sticky | 54.38 ms | 58.10 ms | 100% | 23.18x | 2.32x | 1.00x |
| 32768 | 500 ms | reroute | 1260.48 ms | 1266.22 ms | 0% | 1.00x | 0.10x | 23.18x |
| 32768 | 500 ms | on-return | 126.13 ms | 130.21 ms | 100% | 9.99x | 1.00x | 2.32x |
| 32768 | 500 ms | agentshift | 52.44 ms | 53.34 ms | 100% | 24.04x | 2.41x | 0.96x |
| 32768 | 500 ms | oracle | 54.52 ms | 56.47 ms | 100% | 23.12x | 2.31x | 1.00x |
