# Results: test_id

- split: `test_id`, problems: 243
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/0`
- git commit: `3266f433a36e9de3689bd987f0378b6b96744ded`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 142/243 | 0.584 [0.522, 0.645] | 8 | 92 | 1.066 |
| `bestfirst-size_diff-w1` | 300 | 167/243 | 0.687 [0.626, 0.742] | 11 | 134 | 1.094 |
| `bestfirst-size_diff-w1` | 1000 | 201/243 | 0.827 [0.775, 0.870] | 18 | 254 | 1.149 |
| `bestfirst-size_diff-w1` | 3000 | 218/243 | 0.897 [0.853, 0.929] | 23 | 290 | 1.157 |
| `bestfirst-size_diff-w3` | 100 | 116/243 | 0.477 [0.415, 0.540] | 6 | 88 | 1.083 |
| `bestfirst-size_diff-w3` | 300 | 142/243 | 0.584 [0.522, 0.645] | 11 | 158 | 1.133 |
| `bestfirst-size_diff-w3` | 1000 | 170/243 | 0.700 [0.639, 0.754] | 22 | 240 | 1.194 |
| `bestfirst-size_diff-w3` | 3000 | 194/243 | 0.798 [0.743, 0.844] | 35 | 460 | 1.246 |
| `bestfirst-symbol_bag-w1` | 100 | 182/243 | 0.749 [0.691, 0.799] | 6 | 90 | 1.082 |
| `bestfirst-symbol_bag-w1` | 300 | 209/243 | 0.860 [0.811, 0.898] | 9 | 121 | 1.098 |
| `bestfirst-symbol_bag-w1` | 1000 | 225/243 | 0.926 [0.886, 0.953] | 11 | 163 | 1.108 |
| `bestfirst-symbol_bag-w1` | 3000 | 231/243 | 0.951 [0.916, 0.972] | 11 | 169 | 1.115 |
| `bestfirst-symbol_bag-w3` | 100 | 167/243 | 0.687 [0.626, 0.742] | 6 | 80 | 1.122 |
| `bestfirst-symbol_bag-w3` | 300 | 187/243 | 0.770 [0.713, 0.818] | 7 | 109 | 1.145 |
| `bestfirst-symbol_bag-w3` | 1000 | 202/243 | 0.831 [0.779, 0.873] | 8 | 119 | 1.166 |
| `bestfirst-symbol_bag-w3` | 3000 | 213/243 | 0.877 [0.829, 0.912] | 9 | 148 | 1.180 |
| `bfs-bidir` | 100 | 243/243 | **1.000 [0.984, 1.000]** | 4 | 47 | 1.000 |
| `bfs-bidir` | 300 | 243/243 | **1.000 [0.984, 1.000]** | 4 | 47 | 1.000 |
| `bfs-bidir` | 1000 | 243/243 | **1.000 [0.984, 1.000]** | 4 | 47 | 1.000 |
| `bfs-bidir` | 3000 | 243/243 | **1.000 [0.984, 1.000]** | 4 | 47 | 1.000 |
| `bfs-fwd` | 100 | 160/243 | 0.658 [0.597, 0.715] | 8 | 79 | 1.000 |
| `bfs-fwd` | 300 | 183/243 | 0.753 [0.695, 0.803] | 11 | 125 | 1.000 |
| `bfs-fwd` | 1000 | 207/243 | 0.852 [0.802, 0.891] | 16 | 201 | 1.000 |
| `bfs-fwd` | 3000 | 227/243 | 0.934 [0.896, 0.959] | 20 | 245 | 1.000 |

## Fair optimality gap

Mean proof_len / dist over the **192** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.171 |
| `bestfirst-size_diff-w3` | 1.247 |
| `bestfirst-symbol_bag-w1` | 1.111 |
| `bestfirst-symbol_bag-w3` | 1.180 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 14 | 32 | 0.0114 | 0.0908 |  |
| `bestfirst-size_diff-w1` | 300 | 15 | 31 | 0.0259 | 0.181 |  |
| `bestfirst-size_diff-w1` | 1000 | 10 | 16 | 0.327 | 1 |  |
| `bestfirst-size_diff-w1` | 3000 | 7 | 16 | 0.0931 | 0.559 |  |
| `bestfirst-size_diff-w3` | 100 | 10 | 54 | 2e-08 | 3.19e-07 | * |
| `bestfirst-size_diff-w3` | 300 | 11 | 52 | 1.67e-07 | 2.34e-06 | * |
| `bestfirst-size_diff-w3` | 1000 | 6 | 43 | 5.73e-08 | 8.59e-07 | * |
| `bestfirst-size_diff-w3` | 3000 | 2 | 35 | 1.02e-08 | 1.74e-07 | * |
| `bestfirst-symbol_bag-w1` | 100 | 36 | 14 | 0.0026 | 0.026 | * |
| `bestfirst-symbol_bag-w1` | 300 | 31 | 5 | 1.29e-05 | 0.000168 | * |
| `bestfirst-symbol_bag-w1` | 1000 | 24 | 6 | 0.00143 | 0.0157 | * |
| `bestfirst-symbol_bag-w1` | 3000 | 11 | 7 | 0.481 | 1 |  |
| `bestfirst-symbol_bag-w3` | 100 | 30 | 23 | 0.41 | 1 |  |
| `bestfirst-symbol_bag-w3` | 300 | 21 | 17 | 0.627 | 1 |  |
| `bestfirst-symbol_bag-w3` | 1000 | 15 | 20 | 0.5 | 1 |  |
| `bestfirst-symbol_bag-w3` | 3000 | 6 | 20 | 0.00936 | 0.0842 |  |
| `bfs-bidir` | 100 | 83 | 0 | 2.07e-25 | 4.14e-24 | * |
| `bfs-bidir` | 300 | 60 | 0 | 1.73e-18 | 3.3e-17 | * |
| `bfs-bidir` | 1000 | 36 | 0 | 2.91e-11 | 5.24e-10 | * |
| `bfs-bidir` | 3000 | 16 | 0 | 3.05e-05 | 0.000366 | * |

## Per theory (budget 3000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 104/121 (0.860) | 114/122 (0.934) |
| `bestfirst-size_diff-w3` | 88/121 (0.727) | 106/122 (0.869) |
| `bestfirst-symbol_bag-w1` | 116/121 (0.959) | 115/122 (0.943) |
| `bestfirst-symbol_bag-w3` | 101/121 (0.835) | 112/122 (0.918) |
| `bfs-bidir` | 121/121 (1.000) | 122/122 (1.000) |
| `bfs-fwd` | 106/121 (0.876) | 121/122 (0.992) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
