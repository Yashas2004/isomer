# Equal-time comparison: test_id

- main run (neural): `runs/E2-test_id`, commit `497f19ff91d3183cdc93ab623806dce5180a1e1f`, 2 workers, device `cuda`, max budget 1000
- time run (baselines): `runs/E2time-test_id`, commit `497f19ff91d3183cdc93ab623806dce5180a1e1f`, 4 workers, max budget 10000, max nodes 1500000
- hardware (config.json): `Windows-10-10.0.26300-SP0`, 16 logical CPUs, Python 3.10.11; the GPU model is not recorded in config.json
- **Times are machine-dependent and measured under different worker counts** (neural: 2 workers sharing one GPU; baselines: 4 CPU workers).
- Neural seconds include, for the first problem in each worker process, loading the model and starting CUDA (the searcher's clock starts before the load): at most 2 neural rows are inflated, against neural; the logs do not say which.
- agreement check: OK: 300 problems x 4 budgets (30, 100, 300, 1000) x 2 baselines agree (cell; proof, proof_len, expansions when solved)
- within T: solved = solved and seconds <= T; unsolved = solved after T, or unsolved and still running at T; **censored** = unsolved but stopped (expansion budget / node cap) before T, excluded from n.

## Solve rate within T

Rate with 95% Wilson interval over uncensored problems.

| method | T | n | solved | rate [95% CI] | censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural-w1` | 1 s | 300 | 293 | 0.977 [0.953, 0.989] | 0 |
| `neural-w1` | 3 s | 300 | 296 | 0.987 [0.966, 0.995] | 0 |
| `neural-w1` | 10 s | 300 | 299 | 0.997 [0.981, 0.999] | 0 |
| `neural-w1` | 30 s | 300 | 300 | 1.000 [0.987, 1.000] | 0 |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 279 | 0.930 [0.895, 0.954] | 0 |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 288 | 0.960 [0.931, 0.977] | 0 |
| `bestfirst-symbol_bag-w1` | 10 s | 300 | 294 | 0.980 [0.957, 0.991] | 0 |
| `bestfirst-symbol_bag-w1` | 30 s | 296 | 296 | 1.000 [0.987, 1.000] | 4 |
| `bfs-fwd` | 1 s | 300 | 276 | 0.920 [0.884, 0.946] | 0 |
| `bfs-fwd` | 3 s | 300 | 285 | 0.950 [0.919, 0.969] | 0 |
| `bfs-fwd` | 10 s | 300 | 294 | 0.980 [0.957, 0.991] | 0 |
| `bfs-fwd` | 30 s | 296 | 296 | 1.000 [0.987, 1.000] | 4 |

## Paired comparison: `neural-w1` vs baselines

Problems uncensored for both methods. a = solved within T by `neural-w1` only; b = by the baseline only. Exact two-sided McNemar; Holm-Bonferroni across all rows. * = adjusted p < 0.05.

| vs | T | pairs | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 18 | 4 | 0.00434 | 0.0304 | * |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 11 | 3 | 0.0574 | 0.287 |  |
| `bestfirst-symbol_bag-w1` | 10 s | 300 | 6 | 1 | 0.125 | 0.5 |  |
| `bestfirst-symbol_bag-w1` | 30 s | 296 | 0 | 0 | 1 | 1 |  |
| `bfs-fwd` | 1 s | 300 | 20 | 3 | 0.000488 | 0.00391 | * |
| `bfs-fwd` | 3 s | 300 | 13 | 2 | 0.00739 | 0.0443 | * |
| `bfs-fwd` | 10 s | 300 | 6 | 1 | 0.125 | 0.5 |  |
| `bfs-fwd` | 30 s | 296 | 0 | 0 | 1 | 1 |  |

## Censoring points

Smallest seconds among a method's unsolved rows: beyond it, its curve is incomplete.

| method | censoring point (s) |
| --- | ---: |
| `neural-w1` | none (all solved) |
| `bestfirst-symbol_bag-w1` | 11.42 |
| `bfs-fwd` | 10.98 |
