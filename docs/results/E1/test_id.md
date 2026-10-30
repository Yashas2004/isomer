# Results: test_id

- split: `test_id`, problems: 236
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/1`
- git commit: `d6394351ffb69ac1f234885dc9b56953d52237ab`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 143/236 | 0.606 [0.542, 0.666] | 8 | 111 | 1.063 |
| `bestfirst-size_diff-w1` | 300 | 170/236 | 0.720 [0.660, 0.774] | 12 | 175 | 1.100 |
| `bestfirst-size_diff-w1` | 1000 | 195/236 | 0.826 [0.773, 0.869] | 19 | 267 | 1.121 |
| `bestfirst-size_diff-w1` | 3000 | 214/236 | 0.907 [0.863, 0.938] | 24 | 324 | 1.153 |
| `bestfirst-size_diff-w3` | 100 | 116/236 | 0.492 [0.428, 0.555] | 6 | 88 | 1.080 |
| `bestfirst-size_diff-w3` | 300 | 141/236 | 0.597 [0.534, 0.658] | 11 | 159 | 1.144 |
| `bestfirst-size_diff-w3` | 1000 | 166/236 | 0.703 [0.642, 0.758] | 20 | 271 | 1.206 |
| `bestfirst-size_diff-w3` | 3000 | 191/236 | 0.809 [0.754, 0.854] | 36 | 487 | 1.256 |
| `bestfirst-symbol_bag-w1` | 100 | 177/236 | 0.750 [0.691, 0.801] | 6 | 85 | 1.086 |
| `bestfirst-symbol_bag-w1` | 300 | 201/236 | 0.852 [0.801, 0.891] | 7 | 110 | 1.114 |
| `bestfirst-symbol_bag-w1` | 1000 | 216/236 | 0.915 [0.873, 0.944] | 8 | 128 | 1.130 |
| `bestfirst-symbol_bag-w1` | 3000 | 225/236 | 0.953 [0.918, 0.974] | 11 | 138 | 1.147 |
| `bestfirst-symbol_bag-w3` | 100 | 163/236 | 0.691 [0.629, 0.746] | 5 | 73 | 1.123 |
| `bestfirst-symbol_bag-w3` | 300 | 186/236 | 0.788 [0.732, 0.835] | 6 | 99 | 1.183 |
| `bestfirst-symbol_bag-w3` | 1000 | 202/236 | 0.856 [0.805, 0.895] | 8 | 120 | 1.216 |
| `bestfirst-symbol_bag-w3` | 3000 | 211/236 | 0.894 [0.848, 0.927] | 9 | 135 | 1.231 |
| `bfs-bidir` | 100 | 236/236 | **1.000 [0.984, 1.000]** | 4 | 48 | 1.000 |
| `bfs-bidir` | 300 | 236/236 | **1.000 [0.984, 1.000]** | 4 | 48 | 1.000 |
| `bfs-bidir` | 1000 | 236/236 | **1.000 [0.984, 1.000]** | 4 | 48 | 1.000 |
| `bfs-bidir` | 3000 | 236/236 | **1.000 [0.984, 1.000]** | 4 | 48 | 1.000 |
| `bfs-fwd` | 100 | 163/236 | 0.691 [0.629, 0.746] | 9 | 100 | 1.000 |
| `bfs-fwd` | 300 | 186/236 | 0.788 [0.732, 0.835] | 13 | 148 | 1.000 |
| `bfs-fwd` | 1000 | 209/236 | 0.886 [0.839, 0.920] | 16 | 205 | 1.000 |
| `bfs-fwd` | 3000 | 228/236 | 0.966 [0.935, 0.983] | 20 | 284 | 1.000 |

## Fair optimality gap

Mean proof_len / dist over the **190** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.160 |
| `bestfirst-size_diff-w3` | 1.257 |
| `bestfirst-symbol_bag-w1` | 1.146 |
| `bestfirst-symbol_bag-w3` | 1.216 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 14 | 34 | 0.00552 | 0.0607 |  |
| `bestfirst-size_diff-w1` | 300 | 15 | 31 | 0.0259 | 0.233 |  |
| `bestfirst-size_diff-w1` | 1000 | 12 | 26 | 0.0336 | 0.268 |  |
| `bestfirst-size_diff-w1` | 3000 | 4 | 18 | 0.00434 | 0.0521 |  |
| `bestfirst-size_diff-w3` | 100 | 9 | 56 | 2.05e-09 | 3.48e-08 | * |
| `bestfirst-size_diff-w3` | 300 | 10 | 55 | 1.18e-08 | 1.76e-07 | * |
| `bestfirst-size_diff-w3` | 1000 | 7 | 50 | 4.24e-09 | 6.78e-08 | * |
| `bestfirst-size_diff-w3` | 3000 | 1 | 38 | 1.46e-10 | 2.62e-09 | * |
| `bestfirst-symbol_bag-w1` | 100 | 33 | 19 | 0.0704 | 0.422 |  |
| `bestfirst-symbol_bag-w1` | 300 | 30 | 15 | 0.0357 | 0.268 |  |
| `bestfirst-symbol_bag-w1` | 1000 | 19 | 12 | 0.281 | 1 |  |
| `bestfirst-symbol_bag-w1` | 3000 | 5 | 8 | 0.581 | 1 |  |
| `bestfirst-symbol_bag-w3` | 100 | 25 | 25 | 1 | 1 |  |
| `bestfirst-symbol_bag-w3` | 300 | 23 | 23 | 1 | 1 |  |
| `bestfirst-symbol_bag-w3` | 1000 | 15 | 22 | 0.324 | 1 |  |
| `bestfirst-symbol_bag-w3` | 3000 | 3 | 20 | 0.000488 | 0.00635 | * |
| `bfs-bidir` | 100 | 73 | 0 | 2.12e-22 | 4.24e-21 | * |
| `bfs-bidir` | 300 | 50 | 0 | 1.78e-15 | 3.38e-14 | * |
| `bfs-bidir` | 1000 | 27 | 0 | 1.49e-08 | 2.09e-07 | * |
| `bfs-bidir` | 3000 | 8 | 0 | 0.00781 | 0.0781 |  |

## Per theory (budget 3000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 102/115 (0.887) | 112/121 (0.926) |
| `bestfirst-size_diff-w3` | 85/115 (0.739) | 106/121 (0.876) |
| `bestfirst-symbol_bag-w1` | 109/115 (0.948) | 116/121 (0.959) |
| `bestfirst-symbol_bag-w3` | 98/115 (0.852) | 113/121 (0.934) |
| `bfs-bidir` | 115/115 (1.000) | 121/121 (1.000) |
| `bfs-fwd` | 108/115 (0.939) | 120/121 (0.992) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
