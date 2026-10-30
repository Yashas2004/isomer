# Equal-time comparison: test_depth

- main run (neural): `runs/E2-test_depth`, commit `ccc97a542f1204e4951cb5a19468f9341ac9a7df`, 2 workers, device `cuda`, max budget 1000
- time run (baselines): `runs/E2time-test_depth`, commit `ccc97a542f1204e4951cb5a19468f9341ac9a7df`, 4 workers, max budget 10000, max nodes 1500000
- hardware (config.json): `Windows-10-10.0.26300-SP0`, 16 logical CPUs, Python 3.10.11; the GPU model is not recorded in config.json
- **Times are machine-dependent and measured under different worker counts** (neural: 2 workers sharing one GPU; baselines: 4 CPU workers).
- Neural seconds include, for the first problem in each worker process, loading the model and starting CUDA (the searcher's clock starts before the load): at most 2 neural rows are inflated, against neural; the logs do not say which.
- agreement check: OK: 300 problems x 4 budgets (30, 100, 300, 1000) x 2 baselines agree (cell; proof, proof_len, expansions when solved)
- within T: solved = solved and seconds <= T; unsolved = solved after T, or unsolved and still running at T; **censored** = unsolved but stopped (expansion budget / node cap) before T, excluded from n.

## Solve rate within T

Rate with 95% Wilson interval over uncensored problems.

| method | T | n | solved | rate [95% CI] | censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural-w1` | 1 s | 300 | 151 | 0.503 [0.447, 0.560] | 0 |
| `neural-w1` | 3 s | 300 | 182 | 0.607 [0.550, 0.660] | 0 |
| `neural-w1` | 10 s | 300 | 213 | 0.710 [0.656, 0.758] | 0 |
| `neural-w1` | 30 s | 238 | 223 | 0.937 [0.899, 0.961] | 62 |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 54 | 0.180 [0.141, 0.227] | 0 |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 79 | 0.263 [0.217, 0.316] | 0 |
| `bestfirst-symbol_bag-w1` | 10 s | 294 | 104 | 0.354 [0.301, 0.410] | 6 |
| `bestfirst-symbol_bag-w1` | 30 s | 221 | 122 | 0.552 [0.486, 0.616] | 79 |
| `bfs-fwd` | 1 s | 300 | 38 | 0.127 [0.094, 0.169] | 0 |
| `bfs-fwd` | 3 s | 300 | 57 | 0.190 [0.150, 0.238] | 0 |
| `bfs-fwd` | 10 s | 103 | 68 | 0.660 [0.564, 0.744] | 197 |
| `bfs-fwd` | 30 s | 68 | 68 | 1.000 [0.947, 1.000] | 232 |

## Paired comparison: `neural-w1` vs baselines

Problems uncensored for both methods. a = solved within T by `neural-w1` only; b = by the baseline only. Exact two-sided McNemar; Holm-Bonferroni across all rows. * = adjusted p < 0.05.

| vs | T | pairs | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 103 | 6 | 6.62e-24 | 2.65e-23 | * |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 107 | 4 | 4.79e-27 | 2.39e-26 | * |
| `bestfirst-symbol_bag-w1` | 10 s | 294 | 104 | 1 | 5.23e-30 | 3.14e-29 | * |
| `bestfirst-symbol_bag-w1` | 30 s | 183 | 51 | 0 | 8.88e-16 | 2.66e-15 | * |
| `bfs-fwd` | 1 s | 300 | 113 | 0 | 1.93e-34 | 1.35e-33 | * |
| `bfs-fwd` | 3 s | 300 | 125 | 0 | 4.7e-38 | 3.76e-37 | * |
| `bfs-fwd` | 10 s | 103 | 22 | 0 | 4.77e-07 | 9.54e-07 | * |
| `bfs-fwd` | 30 s | 68 | 0 | 0 | 1 | 1 |  |

## Censoring points

Smallest seconds among a method's unsolved rows: beyond it, its curve is incomplete.

| method | censoring point (s) |
| --- | ---: |
| `neural-w1` | 13.50 |
| `bestfirst-symbol_bag-w1` | 6.74 |
| `bfs-fwd` | 3.81 |
