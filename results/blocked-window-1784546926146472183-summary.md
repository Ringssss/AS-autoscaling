# Blocked-Window Baseline Summary

All named literature baselines are mechanism-equivalent implementations in the same SGLang testbed, not official reproductions.

Source artifact: `results/blocked-window-1784546926146472183.json`

| Prefix | Gap | Strategy | Mean post-tool | p95 | Full hit | vs reroute | vs on-return | vs sticky |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 100 ms | sticky | 42.52 ms | 42.52 ms | 100% | 3.16x | 2.09x | 1.00x |
| 4096 | 100 ms | reroute | 134.20 ms | 134.20 ms | 0% | 1.00x | 0.66x | 3.16x |
| 4096 | 100 ms | agentix | 42.55 ms | 42.55 ms | 100% | 3.15x | 2.09x | 1.00x |
| 4096 | 100 ms | ttl | 42.98 ms | 42.98 ms | 100% | 3.12x | 2.07x | 1.01x |
| 4096 | 100 ms | tokencake-source | 42.57 ms | 42.57 ms | 100% | 3.15x | 2.09x | 1.00x |
| 4096 | 100 ms | tokencake-remote | 618.29 ms | 618.29 ms | 100% | 0.22x | 0.14x | 14.54x |
| 4096 | 100 ms | symphony | 610.76 ms | 610.76 ms | 100% | 0.22x | 0.15x | 14.36x |
| 4096 | 100 ms | on-return | 89.07 ms | 89.07 ms | 100% | 1.51x | 1.00x | 2.09x |
| 4096 | 100 ms | agentshift | 44.81 ms | 44.81 ms | 100% | 2.99x | 1.99x | 1.05x |
| 4096 | 100 ms | oracle | 50.12 ms | 50.12 ms | 100% | 2.68x | 1.78x | 1.18x |
