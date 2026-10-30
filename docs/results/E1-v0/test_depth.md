# Results: test_depth

- split: `test_depth`, problems: 1718
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/0`
- git commit: `985d4dd5368141ffe1c58b2c47515c5f5a26c099`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 37/1718 | 0.022 [0.016, 0.030] | 66 | 954 | 1.016 |
| `bestfirst-size_diff-w1` | 300 | 75/1718 | 0.044 [0.035, 0.054] | 105 | 1665 | 1.026 |
| `bestfirst-size_diff-w1` | 1000 | 121/1718 | 0.070 [0.059, 0.084] | 219 | 3120 | 1.075 |
| `bestfirst-size_diff-w1` | 3000 | 243/1718 | 0.141 [0.126, 0.159] | 1002 | 19098 | 1.188 |
| `bestfirst-size_diff-w3` | 100 | 20/1718 | 0.012 [0.008, 0.018] | 54 | 774 | 1.080 |
| `bestfirst-size_diff-w3` | 300 | 56/1718 | 0.033 [0.025, 0.042] | 144 | 2061 | 1.105 |
| `bestfirst-size_diff-w3` | 1000 | 111/1718 | 0.065 [0.054, 0.077] | 274 | 4338 | 1.172 |
| `bestfirst-size_diff-w3` | 3000 | 212/1718 | 0.123 [0.109, 0.140] | 864 | 17048 | 1.359 |
| `bestfirst-symbol_bag-w1` | 100 | 182/1718 | 0.106 [0.092, 0.121] | 33 | 702 | 1.097 |
| `bestfirst-symbol_bag-w1` | 300 | 293/1718 | 0.171 [0.154, 0.189] | 73 | 1568 | 1.121 |
| `bestfirst-symbol_bag-w1` | 1000 | 455/1718 | 0.265 [0.245, 0.286] | 163 | 3720 | 1.147 |
| `bestfirst-symbol_bag-w1` | 3000 | 643/1718 | 0.374 [0.352, 0.397] | 397 | 9550 | 1.194 |
| `bestfirst-symbol_bag-w3` | 100 | 146/1718 | 0.085 [0.073, 0.099] | 25 | 454 | 1.151 |
| `bestfirst-symbol_bag-w3` | 300 | 239/1718 | 0.139 [0.124, 0.156] | 66 | 1303 | 1.205 |
| `bestfirst-symbol_bag-w3` | 1000 | 383/1718 | 0.223 [0.204, 0.243] | 186 | 3800 | 1.281 |
| `bestfirst-symbol_bag-w3` | 3000 | 553/1718 | 0.322 [0.300, 0.344] | 432 | 10202 | 1.379 |
| `bfs-bidir` | 100 | 553/1718 | **0.322 [0.300, 0.344]** | 45 | 698 | 1.000 |
| `bfs-bidir` | 300 | 841/1718 | **0.490 [0.466, 0.513]** | 64 | 1186 | 1.000 |
| `bfs-bidir` | 1000 | 1118/1718 | **0.651 [0.628, 0.673]** | 102 | 1964 | 1.000 |
| `bfs-bidir` | 3000 | 1305/1718 | **0.760 [0.739, 0.779]** | 146 | 2804 | 1.000 |
| `bfs-fwd` | 100 | 45/1718 | 0.026 [0.020, 0.035] | 68 | 800 | 0.991 |
| `bfs-fwd` | 300 | 107/1718 | 0.062 [0.052, 0.075] | 161 | 1813 | 0.996 |
| `bfs-fwd` | 1000 | 180/1718 | 0.105 [0.091, 0.120] | 260 | 2806 | 0.998 |
| `bfs-fwd` | 3000 | 303/1718 | 0.176 [0.159, 0.195] | 645 | 6889 | 0.999 |

## Fair optimality gap

Mean proof_len / dist over the **122** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.258 |
| `bestfirst-size_diff-w3` | 1.475 |
| `bestfirst-symbol_bag-w1` | 1.199 |
| `bestfirst-symbol_bag-w3` | 1.289 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 0.997 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 29 | 37 | 0.389 | 0.389 |  |
| `bestfirst-size_diff-w1` | 300 | 48 | 80 | 0.00593 | 0.0119 | * |
| `bestfirst-size_diff-w1` | 1000 | 70 | 129 | 3.48e-05 | 0.000174 | * |
| `bestfirst-size_diff-w1` | 3000 | 92 | 152 | 0.000148 | 0.000592 | * |
| `bestfirst-size_diff-w3` | 100 | 18 | 43 | 0.00187 | 0.0056 | * |
| `bestfirst-size_diff-w3` | 300 | 36 | 87 | 4.85e-06 | 2.91e-05 | * |
| `bestfirst-size_diff-w3` | 1000 | 67 | 136 | 1.46e-06 | 1.02e-05 | * |
| `bestfirst-size_diff-w3` | 3000 | 80 | 171 | 9.32e-09 | 7.46e-08 | * |
| `bestfirst-symbol_bag-w1` | 100 | 158 | 21 | 3.54e-27 | 3.89e-26 | * |
| `bestfirst-symbol_bag-w1` | 300 | 221 | 35 | 3.37e-34 | 4.38e-33 | * |
| `bestfirst-symbol_bag-w1` | 1000 | 316 | 41 | 9.79e-54 | 1.47e-52 | * |
| `bestfirst-symbol_bag-w1` | 3000 | 373 | 33 | 4.84e-74 | 7.75e-73 | * |
| `bestfirst-symbol_bag-w3` | 100 | 126 | 25 | 2.04e-17 | 1.84e-16 | * |
| `bestfirst-symbol_bag-w3` | 300 | 170 | 38 | 3.96e-21 | 3.96e-20 | * |
| `bestfirst-symbol_bag-w3` | 1000 | 253 | 50 | 8.18e-34 | 9.81e-33 | * |
| `bestfirst-symbol_bag-w3` | 3000 | 311 | 61 | 1.77e-41 | 2.47e-40 | * |
| `bfs-bidir` | 100 | 508 | 0 | 2.39e-153 | 4.06e-152 | * |
| `bfs-bidir` | 300 | 734 | 0 | 2.21e-221 | 3.98e-220 | * |
| `bfs-bidir` | 1000 | 938 | 0 | 8.61e-283 | 1.64e-281 | * |
| `bfs-bidir` | 3000 | 1002 | 0 | 4.67e-302 | 9.33e-301 | * |

## Per theory (budget 3000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 95/908 (0.105) | 148/810 (0.183) |
| `bestfirst-size_diff-w3` | 86/908 (0.095) | 126/810 (0.156) |
| `bestfirst-symbol_bag-w1` | 326/908 (0.359) | 317/810 (0.391) |
| `bestfirst-symbol_bag-w3` | 282/908 (0.311) | 271/810 (0.335) |
| `bfs-bidir` | 628/908 (0.692) | 677/810 (0.836) |
| `bfs-fwd` | 88/908 (0.097) | 215/810 (0.265) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
