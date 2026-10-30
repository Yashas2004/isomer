# Results: test_theory

- split: `test_theory`, problems: 600
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/0`
- git commit: `985d4dd5368141ffe1c58b2c47515c5f5a26c099`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 251/600 | 0.418 [0.380, 0.458] | 8 | 202 | 1.039 |
| `bestfirst-size_diff-w1` | 300 | 278/600 | 0.463 [0.424, 0.503] | 10 | 275 | 1.043 |
| `bestfirst-size_diff-w1` | 1000 | 330/600 | 0.550 [0.510, 0.589] | 20 | 490 | 1.073 |
| `bestfirst-size_diff-w1` | 3000 | 362/600 | 0.603 [0.564, 0.642] | 28 | 769 | 1.093 |
| `bestfirst-size_diff-w3` | 100 | 198/600 | 0.330 [0.294, 0.369] | 5 | 154 | 1.046 |
| `bestfirst-size_diff-w3` | 300 | 247/600 | 0.412 [0.373, 0.451] | 9 | 282 | 1.051 |
| `bestfirst-size_diff-w3` | 1000 | 289/600 | 0.482 [0.442, 0.522] | 18 | 484 | 1.078 |
| `bestfirst-size_diff-w3` | 3000 | 321/600 | 0.535 [0.495, 0.575] | 29 | 904 | 1.100 |
| `bestfirst-symbol_bag-w1` | 100 | 302/600 | 0.503 [0.463, 0.543] | 7 | 162 | 1.048 |
| `bestfirst-symbol_bag-w1` | 300 | 338/600 | 0.563 [0.523, 0.602] | 9 | 224 | 1.052 |
| `bestfirst-symbol_bag-w1` | 1000 | 388/600 | 0.647 [0.608, 0.684] | 12 | 392 | 1.068 |
| `bestfirst-symbol_bag-w1` | 3000 | 438/600 | 0.730 [0.693, 0.764] | 20 | 568 | 1.086 |
| `bestfirst-symbol_bag-w3` | 100 | 273/600 | 0.455 [0.416, 0.495] | 5 | 143 | 1.063 |
| `bestfirst-symbol_bag-w3` | 300 | 323/600 | 0.538 [0.498, 0.578] | 8 | 223 | 1.082 |
| `bestfirst-symbol_bag-w3` | 1000 | 352/600 | 0.587 [0.547, 0.625] | 10 | 326 | 1.098 |
| `bestfirst-symbol_bag-w3` | 3000 | 385/600 | 0.642 [0.603, 0.679] | 13 | 487 | 1.112 |
| `bfs-bidir` | 100 | 413/600 | **0.688 [0.650, 0.724]** | 4 | 95 | 1.000 |
| `bfs-bidir` | 300 | 477/600 | **0.795 [0.761, 0.825]** | 5 | 123 | 1.000 |
| `bfs-bidir` | 1000 | 514/600 | **0.857 [0.826, 0.882]** | 5 | 150 | 1.000 |
| `bfs-bidir` | 3000 | 544/600 | **0.907 [0.881, 0.927]** | 7 | 185 | 1.000 |
| `bfs-fwd` | 100 | 247/600 | 0.412 [0.373, 0.451] | 7 | 144 | 1.018 |
| `bfs-fwd` | 300 | 280/600 | 0.467 [0.427, 0.507] | 11 | 197 | 1.020 |
| `bfs-fwd` | 1000 | 313/600 | 0.522 [0.482, 0.561] | 17 | 271 | 1.021 |
| `bfs-fwd` | 3000 | 355/600 | 0.592 [0.552, 0.630] | 30 | 404 | 1.031 |

## Fair optimality gap

Mean proof_len / dist over the **292** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.071 |
| `bestfirst-size_diff-w3` | 1.102 |
| `bestfirst-symbol_bag-w1` | 1.053 |
| `bestfirst-symbol_bag-w3` | 1.086 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.022 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 37 | 33 | 0.72 | 1 |  |
| `bestfirst-size_diff-w1` | 300 | 32 | 34 | 0.902 | 1 |  |
| `bestfirst-size_diff-w1` | 1000 | 41 | 24 | 0.0464 | 0.185 |  |
| `bestfirst-size_diff-w1` | 3000 | 41 | 34 | 0.489 | 1 |  |
| `bestfirst-size_diff-w3` | 100 | 25 | 74 | 8.51e-07 | 1.02e-05 | * |
| `bestfirst-size_diff-w3` | 300 | 28 | 61 | 0.00061 | 0.00488 | * |
| `bestfirst-size_diff-w3` | 1000 | 33 | 57 | 0.0149 | 0.0892 |  |
| `bestfirst-size_diff-w3` | 3000 | 28 | 62 | 0.000438 | 0.00394 | * |
| `bestfirst-symbol_bag-w1` | 100 | 75 | 20 | 1.16e-08 | 1.51e-07 | * |
| `bestfirst-symbol_bag-w1` | 300 | 77 | 19 | 1.86e-09 | 2.61e-08 | * |
| `bestfirst-symbol_bag-w1` | 1000 | 87 | 12 | 3.37e-15 | 5.05e-14 | * |
| `bestfirst-symbol_bag-w1` | 3000 | 96 | 13 | 8.3e-17 | 1.33e-15 | * |
| `bestfirst-symbol_bag-w3` | 100 | 69 | 43 | 0.0178 | 0.0892 |  |
| `bestfirst-symbol_bag-w3` | 300 | 73 | 30 | 2.72e-05 | 0.000299 | * |
| `bestfirst-symbol_bag-w3` | 1000 | 69 | 30 | 0.000111 | 0.00111 | * |
| `bestfirst-symbol_bag-w3` | 3000 | 71 | 41 | 0.00589 | 0.0412 | * |
| `bfs-bidir` | 100 | 166 | 0 | 2.14e-50 | 3.63e-49 | * |
| `bfs-bidir` | 300 | 197 | 0 | 9.96e-60 | 1.89e-58 | * |
| `bfs-bidir` | 1000 | 201 | 0 | 6.22e-61 | 1.24e-59 | * |
| `bfs-bidir` | 3000 | 189 | 0 | 2.55e-57 | 4.59e-56 | * |

## Per theory (budget 3000)

| method | lattice | ring |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 174/300 (0.580) | 188/300 (0.627) |
| `bestfirst-size_diff-w3` | 151/300 (0.503) | 170/300 (0.567) |
| `bestfirst-symbol_bag-w1` | 206/300 (0.687) | 232/300 (0.773) |
| `bestfirst-symbol_bag-w3` | 165/300 (0.550) | 220/300 (0.733) |
| `bfs-bidir` | 270/300 (0.900) | 274/300 (0.913) |
| `bfs-fwd` | 163/300 (0.543) | 192/300 (0.640) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
