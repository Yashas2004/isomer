# Results: test_depth

- split: `test_depth`, problems: 300
- methods: `bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd`, `neural-w1@runs/h-v1/best.pt#bf16-fast`
- budgets (node expansions): 30, 100, 300, 1000
- dataset format: `isobench/2`
- git commit: `ccc97a542f1204e4951cb5a19468f9341ac9a7df`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 30 | 9/300 | 0.030 [0.016, 0.056] | 9 | 158 | 1.022 |
| `bestfirst-symbol_bag-w1` | 100 | 13/300 | 0.043 [0.025, 0.073] | 13 | 301 | 1.046 |
| `bestfirst-symbol_bag-w1` | 300 | 32/300 | 0.107 [0.077, 0.147] | 122 | 2344 | 1.074 |
| `bestfirst-symbol_bag-w1` | 1000 | 60/300 | 0.200 [0.159, 0.249] | 258 | 6887 | 1.113 |
| `bfs-bidir` | 30 | 18/300 | 0.060 [0.038, 0.093] | 18 | 262 | 1.000 |
| `bfs-bidir` | 100 | 77/300 | 0.257 [0.211, 0.309] | 44 | 629 | 1.000 |
| `bfs-bidir` | 300 | 123/300 | 0.410 [0.356, 0.466] | 69 | 1290 | 1.000 |
| `bfs-bidir` | 1000 | 175/300 | 0.583 [0.527, 0.638] | 134 | 2421 | 1.000 |
| `bfs-fwd` | 30 | 0/300 | 0.000 [0.000, 0.013] | – | – | – |
| `bfs-fwd` | 100 | 5/300 | 0.017 [0.007, 0.038] | 72 | 745 | 1.000 |
| `bfs-fwd` | 300 | 13/300 | 0.043 [0.025, 0.073] | 143 | 1693 | 1.000 |
| `bfs-fwd` | 1000 | 26/300 | 0.087 [0.060, 0.124] | 298 | 3123 | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 139/300 | **0.463 [0.408, 0.520]** | 8 | 142 | 1.015 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 173/300 | **0.577 [0.520, 0.631]** | 9 | 194 | 1.018 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 202/300 | **0.673 [0.618, 0.724]** | 11 | 279 | 1.021 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 224/300 | **0.747 [0.695, 0.793]** | 13 | 374 | 1.022 |

## Fair optimality gap

Mean proof_len / dist over the **15** problems with known dist solved by every method at budget 1000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-symbol_bag-w1` | 1.200 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1.000 |

## Paired comparison vs `bestfirst-symbol_bag-w1`

a = solved by the method but not `bestfirst-symbol_bag-w1`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bfs-bidir` | 30 | 15 | 6 | 0.0784 | 0.157 |  |
| `bfs-bidir` | 100 | 66 | 2 | 1.59e-17 | 9.54e-17 | * |
| `bfs-bidir` | 300 | 93 | 2 | 2.3e-25 | 1.61e-24 | * |
| `bfs-bidir` | 1000 | 117 | 2 | 2.15e-32 | 1.72e-31 | * |
| `bfs-fwd` | 30 | 0 | 9 | 0.00391 | 0.0117 | * |
| `bfs-fwd` | 100 | 5 | 13 | 0.0963 | 0.157 |  |
| `bfs-fwd` | 300 | 7 | 26 | 0.00132 | 0.00527 | * |
| `bfs-fwd` | 1000 | 11 | 45 | 5.38e-06 | 2.69e-05 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 130 | 0 | 1.47e-39 | 1.32e-38 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 160 | 0 | 1.37e-48 | 1.37e-47 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 170 | 0 | 1.34e-51 | 1.6e-50 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 164 | 0 | 8.55e-50 | 9.41e-49 | * |

## Per theory (budget 1000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 21/128 (0.164) | 39/172 (0.227) |
| `bfs-bidir` | 64/128 (0.500) | 111/172 (0.645) |
| `bfs-fwd` | 1/128 (0.008) | 25/172 (0.145) |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 95/128 (0.742) | 129/172 (0.750) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
