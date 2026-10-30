# Results: test_id

- split: `test_id`, problems: 300
- methods: `bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd`, `neural-w1@runs/h-v1/best.pt#bf16-fast`
- budgets (node expansions): 30, 100, 300, 1000
- dataset format: `isobench/2`
- git commit: `ccc97a542f1204e4951cb5a19468f9341ac9a7df`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 30 | 211/300 | 0.703 [0.649, 0.752] | 4 | 71 | 1.043 |
| `bestfirst-symbol_bag-w1` | 100 | 242/300 | 0.807 [0.758, 0.847] | 6 | 93 | 1.046 |
| `bestfirst-symbol_bag-w1` | 300 | 263/300 | 0.877 [0.835, 0.909] | 7 | 118 | 1.060 |
| `bestfirst-symbol_bag-w1` | 1000 | 279/300 | 0.930 [0.895, 0.954] | 8 | 132 | 1.083 |
| `bfs-bidir` | 30 | 279/300 | 0.930 [0.895, 0.954] | 3 | 41 | 1.000 |
| `bfs-bidir` | 100 | 300/300 | **1.000 [0.987, 1.000]** | 3 | 44 | 1.000 |
| `bfs-bidir` | 300 | 300/300 | **1.000 [0.987, 1.000]** | 3 | 44 | 1.000 |
| `bfs-bidir` | 1000 | 300/300 | **1.000 [0.987, 1.000]** | 3 | 44 | 1.000 |
| `bfs-fwd` | 30 | 169/300 | 0.563 [0.507, 0.618] | 4 | 42 | 1.000 |
| `bfs-fwd` | 100 | 211/300 | 0.703 [0.649, 0.752] | 7 | 95 | 1.000 |
| `bfs-fwd` | 300 | 250/300 | 0.833 [0.787, 0.871] | 10 | 144 | 1.000 |
| `bfs-fwd` | 1000 | 274/300 | 0.913 [0.876, 0.940] | 14 | 195 | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 293/300 | **0.977 [0.953, 0.989]** | 3 | 34 | 1.005 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 298/300 | 0.993 [0.976, 0.998] | 3 | 34 | 1.005 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 299/300 | 0.997 [0.981, 0.999] | 3 | 34 | 1.007 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 300/300 | **1.000 [0.987, 1.000]** | 3 | 34 | 1.007 |

## Fair optimality gap

Mean proof_len / dist over the **260** problems with known dist solved by every method at budget 1000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-symbol_bag-w1` | 1.086 |
| `bfs-bidir` | 1.000 |
| `bfs-fwd` | 1.000 |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1.003 |

## Paired comparison vs `bestfirst-symbol_bag-w1`

a = solved by the method but not `bestfirst-symbol_bag-w1`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bfs-bidir` | 30 | 69 | 1 | 1.2e-19 | 1.32e-18 | * |
| `bfs-bidir` | 100 | 58 | 0 | 6.94e-18 | 6.94e-17 | * |
| `bfs-bidir` | 300 | 37 | 0 | 1.46e-11 | 1.16e-10 | * |
| `bfs-bidir` | 1000 | 21 | 0 | 9.54e-07 | 4.77e-06 | * |
| `bfs-fwd` | 30 | 13 | 55 | 2.75e-07 | 1.65e-06 | * |
| `bfs-fwd` | 100 | 12 | 43 | 3.31e-05 | 9.92e-05 | * |
| `bfs-fwd` | 300 | 14 | 27 | 0.0596 | 0.119 |  |
| `bfs-fwd` | 1000 | 14 | 19 | 0.487 | 0.487 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 82 | 0 | 4.14e-25 | 4.96e-24 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 57 | 1 | 4.09e-16 | 3.68e-15 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 37 | 1 | 2.84e-10 | 1.99e-09 | * |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 21 | 0 | 9.54e-07 | 4.77e-06 | * |

## Per theory (budget 1000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 143/157 (0.911) | 136/143 (0.951) |
| `bfs-bidir` | 157/157 (1.000) | 143/143 (1.000) |
| `bfs-fwd` | 131/157 (0.834) | 143/143 (1.000) |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 157/157 (1.000) | 143/143 (1.000) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
