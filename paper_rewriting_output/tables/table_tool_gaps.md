# Tool-Wait Characteristics

| Configuration | Tool class | Representative operation | Blocked turns | Wait p50 / p90 (ms) | Prefix p50 / p90 (K tokens) | Covered by p95 preparation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Coding replay (TP=1) | Shell | git status --short | 18 | 5.0 / 11.9 | 24.0 / 32.0 | 0% |
|  | Targeted test | pytest state store | 18 | 426.8 / 434.7 | 24.0 / 32.0 | 100% |
|  | Targeted test | pytest migration protocol | 18 | 504.1 / 510.3 | 24.0 / 32.0 | 100% |
|  | Build/test | pytest full suite | 18 | 623.4 / 628.4 | 24.0 / 32.0 | 100% |
|  | **Overall** | All operations | 72 | 437.2 / 624.1 | 24.0 / 32.0 | 75% |
| Representative operations (TP=2) | Web search | OpenAlex query | 24 | 1649.8 / 2614.5 | 16.0 / 32.0 | 100% |
|  | Page fetch | HTTP fetch + HTML parse | 24 | 709.2 / 2328.4 | 16.0 / 32.0 | 100% |
|  | External API | Open-Meteo request | 24 | 1078.4 / 1184.4 | 16.0 / 32.0 | 100% |
|  | PDF parsing | Parse up to 12 pages | 24 | 1074.9 / 1518.6 | 16.0 / 32.0 | 100% |
|  | Python execution | AST/JSON/hash/sort | 24 | 270.3 / 364.0 | 16.0 / 32.0 | 100% |
|  | **Overall** | All operations | 120 | 1025.1 / 2045.8 | 16.0 / 32.0 | 100% |

Coverage compares each observed interval with the p95 measured Qwen3-8B completed-prefix preparation time for the nearest 4K, 16K, or 32K prefix bucket under the matching TP configuration.

Scope: controlled labeled operations, not a production-frequency distribution. Azure/Kimi synthetic gaps and unlabeled FlowPrefill inter-turn proxies are excluded.
