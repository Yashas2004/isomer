# Equal-time comparison: test_deep

- main run (neural): `runs/E2-test_deep`, commit `497f19ff91d3183cdc93ab623806dce5180a1e1f`, 2 workers, device `cuda`, max budget 1000
- time run (baselines): `runs/E2time-test_deep`, commit `497f19ff91d3183cdc93ab623806dce5180a1e1f`, 4 workers, max budget 10000, max nodes 1500000
- hardware (config.json): `Windows-10-10.0.26300-SP0`, 16 logical CPUs, Python 3.10.11; the GPU model is not recorded in config.json
- **Times are machine-dependent and measured under different worker counts** (neural: 2 workers sharing one GPU; baselines: 4 CPU workers).
- Neural seconds include, for the first problem in each worker process, loading the model and starting CUDA (the searcher's clock starts before the load): at most 2 neural rows are inflated, against neural; the logs do not say which.
- agreement check: OK: 100 problems x 4 budgets (30, 100, 300, 1000) x 2 baselines agree (cell; proof, proof_len, expansions when solved)
- within T: solved = solved and seconds <= T; unsolved = solved after T, or unsolved and still running at T; **censored** = unsolved but stopped (expansion budget / node cap) before T, excluded from n.

## Solve rate within T

Rate with 95% Wilson interval over uncensored problems.

| method | T | n | solved | rate [95% CI] | censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural-w1` | 1 s | 100 | 0 | 0.000 [0.000, 0.037] | 0 |
| `neural-w1` | 3 s | 100 | 1 | 0.010 [0.002, 0.054] | 0 |
| `neural-w1` | 10 s | 100 | 1 | 0.010 [0.002, 0.054] | 0 |
| `neural-w1` | 30 s | 72 | 2 | 0.028 [0.008, 0.096] | 28 |
| `bestfirst-symbol_bag-w1` | 1 s | 100 | 0 | 0.000 [0.000, 0.037] | 0 |
| `bestfirst-symbol_bag-w1` | 3 s | 100 | 0 | 0.000 [0.000, 0.037] | 0 |
| `bestfirst-symbol_bag-w1` | 10 s | 99 | 0 | 0.000 [0.000, 0.037] | 1 |
| `bestfirst-symbol_bag-w1` | 30 s | 41 | 0 | 0.000 [0.000, 0.086] | 59 |
| `bfs-fwd` | 1 s | 100 | 0 | 0.000 [0.000, 0.037] | 0 |
| `bfs-fwd` | 3 s | 100 | 0 | 0.000 [0.000, 0.037] | 0 |
| `bfs-fwd` | 10 s | 46 | 0 | 0.000 [0.000, 0.077] | 54 |
| `bfs-fwd` | 30 s | 0 | 0 | – | 100 |

## Paired comparison: `neural-w1` vs baselines

Problems uncensored for both methods. a = solved within T by `neural-w1` only; b = by the baseline only. Exact two-sided McNemar; Holm-Bonferroni across all rows. * = adjusted p < 0.05.

| vs | T | pairs | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-symbol_bag-w1` | 1 s | 100 | 0 | 0 | 1 | 1 |  |
| `bestfirst-symbol_bag-w1` | 3 s | 100 | 1 | 0 | 1 | 1 |  |
| `bestfirst-symbol_bag-w1` | 10 s | 99 | 1 | 0 | 1 | 1 |  |
| `bestfirst-symbol_bag-w1` | 30 s | 37 | 0 | 0 | 1 | 1 |  |
| `bfs-fwd` | 1 s | 100 | 0 | 0 | 1 | 1 |  |
| `bfs-fwd` | 3 s | 100 | 1 | 0 | 1 | 1 |  |
| `bfs-fwd` | 10 s | 46 | 1 | 0 | 1 | 1 |  |
| `bfs-fwd` | 30 s | 0 | 0 | 0 | 1 | 1 |  |

## Censoring points

Smallest seconds among a method's unsolved rows: beyond it, its curve is incomplete.

| method | censoring point (s) |
| --- | ---: |
| `neural-w1` | 15.45 |
| `bestfirst-symbol_bag-w1` | 8.82 |
| `bfs-fwd` | 5.14 |
