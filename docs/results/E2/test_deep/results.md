# Results: test_deep

- split: `test_deep`, problems: 100
- methods: `bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd`, `neural-w1@runs/h-v1/best.pt#bf16-fast`
- budgets (node expansions): 30, 100, 300, 1000
- dataset format: `isobench/2`
- git commit: `497f19ff91d3183cdc93ab623806dce5180a1e1f`
- smaller budgets are derived from one run at the largest budget (prefix property; see `isomer.search.common`). Solved = accepted by the trusted kernel.

## Solve rate

Rate with 95% Wilson interval; best rate per budget in bold. Medians and the optimality gap (mean proof_len / dist) are over that method's solved problems.

| method | budget | solved | rate [95% CI] | median expansions | median nodes stored | opt gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 30 | 0/100 | **0.000 [0.000, 0.037]** | – | – | – |
| `bestfirst-symbol_bag-w1` | 100 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bestfirst-symbol_bag-w1` | 300 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bestfirst-symbol_bag-w1` | 1000 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-bidir` | 30 | 0/100 | **0.000 [0.000, 0.037]** | – | – | – |
| `bfs-bidir` | 100 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-bidir` | 300 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-bidir` | 1000 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-fwd` | 30 | 0/100 | **0.000 [0.000, 0.037]** | – | – | – |
| `bfs-fwd` | 100 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-fwd` | 300 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `bfs-fwd` | 1000 | 0/100 | 0.000 [0.000, 0.037] | – | – | – |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 0/100 | **0.000 [0.000, 0.037]** | – | – | – |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 1/100 | **0.010 [0.002, 0.054]** | 79 | 5409 | – |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 1/100 | **0.010 [0.002, 0.054]** | 79 | 5409 | – |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 2/100 | **0.020 [0.006, 0.070]** | 430 | 21011 | – |

## Fair optimality gap

Mean proof_len / dist over the **0** problems with known dist solved by every method at budget 1000. 1.000 = always shortest.

| method | fair gap |
| --- | ---: |
| `bestfirst-symbol_bag-w1` | – |
| `bfs-bidir` | – |
| `bfs-fwd` | – |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | – |

## Paired comparison vs `bestfirst-symbol_bag-w1`

a = solved by the method but not `bestfirst-symbol_bag-w1`; b = the reverse. Exact two-sided McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05.

| method | budget | a | b | p | p (Holm) | sig |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 30 | 0 | 0 | 1 | 1 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 100 | 1 | 0 | 1 | 1 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 300 | 1 | 0 | 1 | 1 |  |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 1000 | 2 | 0 | 0.5 | 1 |  |

Not tested (identical to reference on every problem): `bfs-bidir`, `bfs-fwd`

## Per theory (budget 1000)

| method | bool | group |
| --- | ---: | ---: |
| `bestfirst-symbol_bag-w1` | 0/66 (0.000) | 0/34 (0.000) |
| `bfs-bidir` | 0/66 (0.000) | 0/34 (0.000) |
| `bfs-fwd` | 0/66 (0.000) | 0/34 (0.000) |
| `neural-w1@runs/h-v1/best.pt#bf16-fast` | 2/66 (0.030) | 0/34 (0.000) |

## Notes

- censored cells (wall-clock limit): 0
- error rows: 0, unsound rows: 0
- `bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd` identical on all problems
