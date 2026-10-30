# Results: test_depth

- split: `test_depth`, problems: 1711
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/1`
- git commit: `d6394351ffb69ac1f234885dc9b56953d52237ab`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 27/1711 | 0.016 [0.011, 0.023] | 64 | 808 | 1.037 |
| `bestfirst-size_diff-w1` | 300 | 74/1711 | 0.043 [0.035, 0.054] | 124 | 1954 | 1.032 |
| `bestfirst-size_diff-w1` | 1000 | 121/1711 | 0.071 [0.060, 0.084] | 218 | 3319 | 1.055 |
| `bestfirst-size_diff-w1` | 3000 | 227/1711 | 0.133 [0.117, 0.150] | 893 | 15508 | 1.125 |
| `bestfirst-size_diff-w3` | 100 | 18/1711 | 0.011 [0.007, 0.017] | 58 | 825 | 1.067 |
| `bestfirst-size_diff-w3` | 300 | 37/1711 | 0.022 [0.016, 0.030] | 102 | 1622 | 1.041 |
| `bestfirst-size_diff-w3` | 1000 | 106/1711 | 0.062 [0.051, 0.074] | 378 | 6132 | 1.154 |
| `bestfirst-size_diff-w3` | 3000 | 207/1711 | 0.121 [0.106, 0.137] | 931 | 16076 | 1.293 |
| `bestfirst-symbol_bag-w1` | 100 | 132/1711 | 0.077 [0.065, 0.091] | 36 | 771 | 1.049 |
| `bestfirst-symbol_bag-w1` | 300 | 228/1711 | 0.133 [0.118, 0.150] | 82 | 1780 | 1.083 |
| `bestfirst-symbol_bag-w1` | 1000 | 391/1711 | 0.229 [0.209, 0.249] | 219 | 4975 | 1.126 |
| `bestfirst-symbol_bag-w1` | 3000 | 582/1711 | 0.340 [0.318, 0.363] | 503 | 12302 | 1.178 |
| `bestfirst-symbol_bag-w3` | 100 | 117/1711 | 0.068 [0.057, 0.081] | 31 | 700 | 1.126 |
| `bestfirst-symbol_bag-w3` | 300 | 201/1711 | 0.117 [0.103, 0.134] | 82 | 1720 | 1.159 |
| `bestfirst-symbol_bag-w3` | 1000 | 358/1711 | 0.209 [0.191, 0.229] | 258 | 5960 | 1.281 |
| `bestfirst-symbol_bag-w3` | 3000 | 487/1711 | 0.285 [0.264, 0.306] | 409 | 9456 | 1.352 |
| `bfs-bidir` | 100 | 518/1711 | **0.303 [0.281, 0.325]** | 48 | 687 | 1.000 |
| `bfs-bidir` | 300 | 799/1711 | **0.467 [0.443, 0.491]** | 66 | 1234 | 1.000 |
| `bfs-bidir` | 1000 | 1098/1711 | **0.642 [0.619, 0.664]** | 114 | 2089 | 1.000 |
| `bfs-bidir` | 3000 | 1299/1711 | **0.759 [0.738, 0.779]** | 170 | 3203 | 1.000 |
| `bfs-fwd` | 100 | 26/1711 | 0.015 [0.010, 0.022] | 76 | 838 | 1.000 |
| `bfs-fwd` | 300 | 82/1711 | 0.048 [0.039, 0.059] | 144 | 1587 | 1.000 |
| `bfs-fwd` | 1000 | 164/1711 | 0.096 [0.083, 0.111] | 300 | 3196 | 1.000 |
| `bfs-fwd` | 3000 | 284/1711 | 0.166 [0.149, 0.184] | 812 | 8448 | 1.000 |

## Fair optimality gap

Mean proof_len / dist over the **99** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.178 |
| `bestfirst-size_diff-w3` | 1.327 |
| `bestfirst-symbol_bag-w1` | 1.168 |
| `bestfirst-symbol_bag-w3` | 1.270 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 24 | 23 | 1 | 1 |  |
| `bestfirst-size_diff-w1` | 300 | 51 | 59 | 0.505 | 1 |  |
| `bestfirst-size_diff-w1` | 1000 | 59 | 102 | 0.000876 | 0.0035 | * |
| `bestfirst-size_diff-w1` | 3000 | 104 | 161 | 0.000557 | 0.00278 | * |
| `bestfirst-size_diff-w3` | 100 | 17 | 25 | 0.28 | 0.84 |  |
| `bestfirst-size_diff-w3` | 300 | 28 | 73 | 8.64e-06 | 6.05e-05 | * |
| `bestfirst-size_diff-w3` | 1000 | 56 | 114 | 1.02e-05 | 6.14e-05 | * |
| `bestfirst-size_diff-w3` | 3000 | 82 | 159 | 7.98e-07 | 6.38e-06 | * |
| `bestfirst-symbol_bag-w1` | 100 | 125 | 19 | 2.55e-20 | 2.81e-19 | * |
| `bestfirst-symbol_bag-w1` | 300 | 186 | 40 | 1.07e-23 | 1.29e-22 | * |
| `bestfirst-symbol_bag-w1` | 1000 | 277 | 50 | 3.01e-39 | 4.51e-38 | * |
| `bestfirst-symbol_bag-w1` | 3000 | 353 | 55 | 2.4e-54 | 3.84e-53 | * |
| `bestfirst-symbol_bag-w3` | 100 | 112 | 21 | 3.32e-16 | 2.98e-15 | * |
| `bestfirst-symbol_bag-w3` | 300 | 161 | 42 | 1.28e-17 | 1.28e-16 | * |
| `bestfirst-symbol_bag-w3` | 1000 | 232 | 38 | 3.85e-35 | 5.4e-34 | * |
| `bestfirst-symbol_bag-w3` | 3000 | 265 | 62 | 4.98e-31 | 6.47e-30 | * |
| `bfs-bidir` | 100 | 492 | 0 | 1.56e-148 | 2.66e-147 | * |
| `bfs-bidir` | 300 | 717 | 0 | 2.9e-216 | 5.22e-215 | * |
| `bfs-bidir` | 1000 | 934 | 0 | 1.38e-281 | 2.62e-280 | * |
| `bfs-bidir` | 3000 | 1015 | 0 | 5.7e-306 | 1.14e-304 | * |

## Per theory (budget 3000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 94/908 (0.104) | 133/803 (0.166) |
| `bestfirst-size_diff-w3` | 80/908 (0.088) | 127/803 (0.158) |
| `bestfirst-symbol_bag-w1` | 302/908 (0.333) | 280/803 (0.349) |
| `bestfirst-symbol_bag-w3` | 229/908 (0.252) | 258/803 (0.321) |
| `bfs-bidir` | 616/908 (0.678) | 683/803 (0.851) |
| `bfs-fwd` | 61/908 (0.067) | 223/803 (0.278) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
