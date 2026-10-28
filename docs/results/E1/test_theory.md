# Results: test_theory

- split: `test_theory`, problems: 600
- methods: `bestfirst-size_diff-w1`, `bestfirst-size_diff-w3`, `bestfirst-symbol_bag-w1`, `bestfirst-symbol_bag-w3`, `bfs-bidir`, `bfs-fwd`
- budgets (node expansions): 100, 300, 1000, 3000
- dataset format: `isobench/1`
- git commit: `3ecea43b650b4310488124cd21b0e35e7b16abbc`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-size_diff-w1` | 100 | 263/600 | 0.438 [0.399, 0.478] | 12 | 322 | 1.018 |
| `bestfirst-size_diff-w1` | 300 | 289/600 | 0.482 [0.442, 0.522] | 15 | 380 | 1.023 |
| `bestfirst-size_diff-w1` | 1000 | 330/600 | 0.550 [0.510, 0.589] | 24 | 477 | 1.041 |
| `bestfirst-size_diff-w1` | 3000 | 366/600 | 0.610 [0.570, 0.648] | 29 | 712 | 1.049 |
| `bestfirst-size_diff-w3` | 100 | 187/600 | 0.312 [0.276, 0.350] | 5 | 202 | 1.024 |
| `bestfirst-size_diff-w3` | 300 | 230/600 | 0.383 [0.345, 0.423] | 11 | 338 | 1.026 |
| `bestfirst-size_diff-w3` | 1000 | 308/600 | 0.513 [0.473, 0.553] | 37 | 971 | 1.052 |
| `bestfirst-size_diff-w3` | 3000 | 337/600 | 0.562 [0.522, 0.601] | 54 | 1733 | 1.068 |
| `bestfirst-symbol_bag-w1` | 100 | 305/600 | 0.508 [0.468, 0.548] | 8 | 202 | 1.028 |
| `bestfirst-symbol_bag-w1` | 300 | 348/600 | 0.580 [0.540, 0.619] | 11 | 335 | 1.035 |
| `bestfirst-symbol_bag-w1` | 1000 | 385/600 | 0.642 [0.603, 0.679] | 15 | 420 | 1.040 |
| `bestfirst-symbol_bag-w1` | 3000 | 436/600 | 0.727 [0.690, 0.761] | 24 | 582 | 1.046 |
| `bestfirst-symbol_bag-w3` | 100 | 251/600 | 0.418 [0.380, 0.458] | 5 | 131 | 1.047 |
| `bestfirst-symbol_bag-w3` | 300 | 300/600 | 0.500 [0.460, 0.540] | 7 | 219 | 1.050 |
| `bestfirst-symbol_bag-w3` | 1000 | 364/600 | 0.607 [0.567, 0.645] | 13 | 489 | 1.073 |
| `bestfirst-symbol_bag-w3` | 3000 | 395/600 | 0.658 [0.619, 0.695] | 20 | 716 | 1.089 |
| `bfs-bidir` | 100 | 408/600 | **0.680 [0.642, 0.716]** | 4 | 103 | 1.000 |
| `bfs-bidir` | 300 | 478/600 | **0.797 [0.763, 0.827]** | 5 | 137 | 1.000 |
| `bfs-bidir` | 1000 | 510/600 | **0.850 [0.819, 0.876]** | 7 | 158 | 1.000 |
| `bfs-bidir` | 3000 | 543/600 | **0.905 [0.879, 0.926]** | 9 | 200 | 1.000 |
| `bfs-fwd` | 100 | 242/600 | 0.403 [0.365, 0.443] | 6 | 129 | 1.000 |
| `bfs-fwd` | 300 | 284/600 | 0.473 [0.434, 0.513] | 11 | 188 | 1.000 |
| `bfs-fwd` | 1000 | 314/600 | 0.523 [0.483, 0.563] | 15 | 274 | 1.000 |
| `bfs-fwd` | 3000 | 358/600 | 0.597 [0.557, 0.635] | 26 | 480 | 1.000 |

## Fair optimality gap

Mean proof_len / dist over the **300** problems with known dist solved by every method at budget 3000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-size_diff-w1` | 1.046 |
| `bestfirst-size_diff-w3` | 1.072 |
| `bestfirst-symbol_bag-w1` | 1.034 |
| `bestfirst-symbol_bag-w3` | 1.062 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |

## Paired comparison vs `bfs-fwd`

a = solved by the method but not `bfs-fwd`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bestfirst-size_diff-w1` | 100 | 43 | 22 | 0.0125 | 0.1 |  |
| `bestfirst-size_diff-w1` | 300 | 34 | 29 | 0.615 | 1 |  |
| `bestfirst-size_diff-w1` | 1000 | 42 | 26 | 0.0681 | 0.409 |  |
| `bestfirst-size_diff-w1` | 3000 | 45 | 37 | 0.44 | 1 |  |
| `bestfirst-size_diff-w3` | 100 | 26 | 81 | 9.37e-08 | 1.03e-06 | * |
| `bestfirst-size_diff-w3` | 300 | 20 | 74 | 1.85e-08 | 2.22e-07 | * |
| `bestfirst-size_diff-w3` | 1000 | 39 | 45 | 0.586 | 1 |  |
| `bestfirst-size_diff-w3` | 3000 | 36 | 57 | 0.0375 | 0.263 |  |
| `bestfirst-symbol_bag-w1` | 100 | 77 | 14 | 1.05e-11 | 1.47e-10 | * |
| `bestfirst-symbol_bag-w1` | 300 | 79 | 15 | 1.15e-11 | 1.5e-10 | * |
| `bestfirst-symbol_bag-w1` | 1000 | 86 | 15 | 2.83e-13 | 4.24e-12 | * |
| `bestfirst-symbol_bag-w1` | 3000 | 98 | 20 | 1.53e-13 | 2.46e-12 | * |
| `bestfirst-symbol_bag-w3` | 100 | 60 | 51 | 0.448 | 1 |  |
| `bestfirst-symbol_bag-w3` | 300 | 65 | 49 | 0.16 | 0.799 |  |
| `bestfirst-symbol_bag-w3` | 1000 | 76 | 26 | 7.44e-07 | 7.44e-06 | * |
| `bestfirst-symbol_bag-w3` | 3000 | 72 | 35 | 0.000445 | 0.00401 | * |
| `bfs-bidir` | 100 | 166 | 0 | 2.14e-50 | 3.63e-49 | * |
| `bfs-bidir` | 300 | 194 | 0 | 7.97e-59 | 1.51e-57 | * |
| `bfs-bidir` | 1000 | 196 | 0 | 1.99e-59 | 3.98e-58 | * |
| `bfs-bidir` | 3000 | 185 | 0 | 4.08e-56 | 7.34e-55 | * |

## Per theory (budget 3000)

| method | lattice | ring |
| --- | ---: | ---: |
| `bestfirst-size_diff-w1` | 170/300 (0.567) | 196/300 (0.653) |
| `bestfirst-size_diff-w3` | 162/300 (0.540) | 175/300 (0.583) |
| `bestfirst-symbol_bag-w1` | 201/300 (0.670) | 235/300 (0.783) |
| `bestfirst-symbol_bag-w3` | 172/300 (0.573) | 223/300 (0.743) |
| `bfs-bidir` | 264/300 (0.880) | 279/300 (0.930) |
| `bfs-fwd` | 165/300 (0.550) | 193/300 (0.643) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
