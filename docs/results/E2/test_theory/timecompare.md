# Equal-time comparison: test_theory

- main run (neural): `runs/E2-test_theory`, commit `ccc97a542f1204e4951cb5a19468f9341ac9a7df`, 2 workers, device `cuda`, max budget 1000
- time run (baselines): `runs/E2time-test_theory`, commit `ccc97a542f1204e4951cb5a19468f9341ac9a7df`, 4 workers, max budget 10000, max nodes 1500000
- hardware (config.json): `Windows-10-10.0.26300-SP0`, 16 logical CPUs, Python 3.10.11; the GPU model is not recorded in config.json
- **Times are machine-dependent and measured under different worker counts** (neural: 2 workers sharing one GPU; baselines: 4 CPU workers).
- Neural seconds include, for the first problem in each worker process, loading the model and starting CUDA (the searcher's clock starts before the load): at most 2 neural rows are inflated, against neural; the logs do not say which.
- agreement check: OK: 300 problems x 4 budgets (30, 100, 300, 1000) x 2 baselines agree (cell; proof, proof_len, expansions when solved)
- within T: solved = solved and seconds <= T; unsolved = solved after T, or unsolved and still running at T; **censored** = unsolved but stopped (expansion budget / node cap) before T, excluded from n.

## Solve rate within T

Rate with 95% Wilson interval over uncensored problems.

| method | T | n | solved | rate [95% CI] | censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| `neural-w1` | 1 s | 300 | 106 | 0.353 [0.301, 0.409] | 0 |
| `neural-w1` | 3 s | 300 | 133 | 0.443 [0.388, 0.500] | 0 |
| `neural-w1` | 10 s | 298 | 159 | 0.534 [0.477, 0.589] | 2 |
| `neural-w1` | 30 s | 224 | 167 | 0.746 [0.685, 0.798] | 76 |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 164 | 0.547 [0.490, 0.602] | 0 |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 178 | 0.593 [0.537, 0.647] | 0 |
| `bestfirst-symbol_bag-w1` | 10 s | 296 | 201 | 0.679 [0.624, 0.730] | 4 |
| `bestfirst-symbol_bag-w1` | 30 s | 248 | 205 | 0.827 [0.775, 0.869] | 52 |
| `bfs-fwd` | 1 s | 300 | 146 | 0.487 [0.431, 0.543] | 0 |
| `bfs-fwd` | 3 s | 300 | 160 | 0.533 [0.477, 0.589] | 0 |
| `bfs-fwd` | 10 s | 254 | 172 | 0.677 [0.617, 0.732] | 46 |
| `bfs-fwd` | 30 s | 210 | 177 | 0.843 [0.788, 0.886] | 90 |

## Paired comparison: `neural-w1` vs baselines

Problems uncensored for both methods. a = solved within T by `neural-w1` only; b = by the baseline only. Exact two-sided McNemar; Holm-Bonferroni across all rows. * = adjusted p < 0.05.

| vs | T | pairs | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-symbol_bag-w1` | 1 s | 300 | 6 | 64 | 2.44e-13 | 1.95e-12 | * |
| `bestfirst-symbol_bag-w1` | 3 s | 300 | 8 | 53 | 2.99e-09 | 1.79e-08 | * |
| `bestfirst-symbol_bag-w1` | 10 s | 294 | 4 | 47 | 2.42e-10 | 1.69e-09 | * |
| `bestfirst-symbol_bag-w1` | 30 s | 213 | 1 | 10 | 0.0117 | 0.0234 | * |
| `bfs-fwd` | 1 s | 300 | 7 | 47 | 2.29e-08 | 1.15e-07 | * |
| `bfs-fwd` | 3 s | 300 | 8 | 35 | 4.19e-05 | 0.000168 | * |
| `bfs-fwd` | 10 s | 254 | 8 | 24 | 0.007 | 0.021 | * |
| `bfs-fwd` | 30 s | 193 | 1 | 7 | 0.0703 | 0.0703 |  |

## Censoring points

Smallest seconds among a method's unsolved rows: beyond it, its curve is incomplete.

| method | censoring point (s) |
| --- | ---: |
| `neural-w1` | 6.69 |
| `bestfirst-symbol_bag-w1` | 8.97 |
| `bfs-fwd` | 4.77 |
