# Results: test_theory

- split: `test_theory`, problems: 300
- methods: `bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd`, `neural-w1@runs/h-v1/best.pt#bf16-fast`
- budgets (node expansions): 30, 100, 300, 1000
- dataset format: `isobench/2`
- git commit: `497f19ff91d3183cdc93ab623806dce5180a1e1f`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 30 | 83/300 | 0.277 [0.229, 0.330] | 5 | 124 | 1.024 |
| `bestfirst-symbol_bag-w1` | 100 | 129/300 | 0.430 [0.375, 0.487] | 11 | 327 | 1.036 |
| `bestfirst-symbol_bag-w1` | 300 | 148/300 | 0.493 [0.437, 0.550] | 14 | 554 | 1.056 |
| `bestfirst-symbol_bag-w1` | 1000 | 166/300 | 0.553 [0.497, 0.609] | 29 | 695 | 1.070 |
| `bfs-bidir` | 30 | 151/300 | **0.503 [0.447, 0.560]** | 5 | 95 | 1.000 |
| `bfs-bidir` | 100 | 186/300 | **0.620 [0.564, 0.673]** | 6 | 154 | 1.000 |
| `bfs-bidir` | 300 | 218/300 | **0.727 [0.674, 0.774]** | 10 | 205 | 1.000 |
| `bfs-bidir` | 1000 | 241/300 | **0.803 [0.755, 0.844]** | 13 | 285 | 1.000 |
| `bfs-fwd` | 30 | 63/300 | 0.210 [0.168, 0.260] | 6 | 115 | 1.000 |
| `bfs-fwd` | 100 | 93/300 | 0.310 [0.260, 0.364] | 16 | 251 | 1.000 |
| `bfs-fwd` | 300 | 119/300 | 0.397 [0.343, 0.453] | 25 | 431 | 1.000 |
| `bfs-fwd` | 1000 | 139/300 | 0.463 [0.408, 0.520] | 43 | 638 | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 94/300 | 0.313 [0.263, 0.368] | 6 | 110 | 1.014 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 124/300 | 0.413 [0.359, 0.470] | 11 | 210 | 1.013 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 147/300 | 0.490 [0.434, 0.546] | 14 | 373 | 1.017 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 169/300 | 0.563 [0.507, 0.618] | 22 | 642 | 1.015 |

## Fair optimality gap

Mean proof_len / dist over the **122** problems with known dist solved by every method at budget 1000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-symbol_bag-w1` | 1.086 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1.018 |

## Paired comparison vs `bestfirst-symbol_bag-w1`

a = solved by the method but not `bestfirst-symbol_bag-w1`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bfs-bidir` | 30 | 70 | 2 | 1.11e-18 | 1.11e-17 | * |
| `bfs-bidir` | 100 | 60 | 3 | 9.05e-15 | 8.14e-14 | * |
| `bfs-bidir` | 300 | 70 | 0 | 1.69e-21 | 1.86e-20 | * |
| `bfs-bidir` | 1000 | 75 | 0 | 5.29e-23 | 6.35e-22 | * |
| `bfs-fwd` | 30 | 9 | 29 | 0.00166 | 0.00829 | * |
| `bfs-fwd` | 100 | 10 | 46 | 1.25e-06 | 9.96e-06 | * |
| `bfs-fwd` | 300 | 13 | 42 | 0.000114 | 0.000795 | * |
| `bfs-fwd` | 1000 | 13 | 40 | 0.000269 | 0.00161 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 26 | 15 | 0.117 | 0.469 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 22 | 27 | 0.568 | 1 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 20 | 21 | 1 | 1 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 21 | 18 | 0.749 | 1 |  |

## Per theory (budget 1000)

| method | lattice | ring |
| --- | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 71/152 (0.467) | 95/148 (0.642) |
| `bfs-bidir` | 117/152 (0.770) | 124/148 (0.838) |
| `bfs-fwd` | 64/152 (0.421) | 75/148 (0.507) |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 82/152 (0.539) | 87/148 (0.588) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
