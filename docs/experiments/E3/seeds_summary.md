# E3 seed study: h-v1 and two reseeded runs

Every number below is read from `runs/e3/scores/`, `runs/e3/labels/` and `runs/e3/bench-seeds/`. All three models have the h-v1 configuration and differ only in the training seed. "Mean" and "range" are over the 3 models; a range over 3 runs is not a confidence interval. This analysis is exploratory, on the E2 test samples.

## Verdicts

**Saturation near the training maximum appears in every one of the 3 models.**
- h-v1: mean prediction 4.56 at k = 6 and 4.77 at k = 9 (0.070 per unit of true distance); highest mean beyond the training range 4.77.
- h-v1-s1: mean prediction 4.54 at k = 6 and 4.70 at k = 9 (0.053 per unit of true distance); highest mean beyond the training range 4.70.
- h-v1-s2: mean prediction 4.55 at k = 6 and 4.63 at k = 9 (0.025 per unit of true distance); highest mean beyond the training range 4.71.

Rule: over distances k > 5 with at least 10 labelled states, the mean prediction stays below 5 (the largest training distance) and rises by less than 0.25 per unit of true distance.

**The margin collapse appears in every one of the 3 models.**
- h-v1: drop to the optimal child 1.60 at k = 1, at most 0.08 at k = 7–9; margin -1.21 at k = 1, at depth at most 20% of that.
- h-v1-s1: drop to the optimal child 1.54 at k = 1, at most 0.12 at k = 7–9; margin -1.66 at k = 1, at depth at most 18% of that.
- h-v1-s2: drop to the optimal child 1.59 at k = 1, at most 0.14 at k = 7–9; margin -0.91 at k = 1, at depth at most 17% of that.

Rule: at every distance k >= 7 with at least 10 labelled states, the drop is below 0.25 (a true distance drops by 1) and |margin| is below 30% of its value at k = 1. The thresholds were chosen after seeing these results, not pre-registered; the tables below are the evidence.

What does vary between seeds is the level: overall all-children top-1 on test_depth is 0.719 (h-v1), 0.800 (h-v1-s1), 0.645 (h-v1-s2), and the solve rate at 1000 expansions is 74.7% (h-v1), 80.0% (h-v1-s1), 65.0% (h-v1-s2).

## Population and unknown states

test_depth: all 300 problems of the E2 sample are labelled (1965 on-path states). 30 states in 22 problems hit the labelling node cap and are **unknown**; they are excluded from every diagnostic below, leaving 1935 scored states. The search results use all 300 problems.

| state distance k | on-path states | unknown | unknown share | scored |
|---:|---:|---:|---:|---:|
| 1 | 300 | 0 | 0.0% | 300 |
| 2 | 300 | 0 | 0.0% | 300 |
| 3 | 300 | 0 | 0.0% | 300 |
| 4 | 300 | 0 | 0.0% | 300 |
| 5 | 300 | 0 | 0.0% | 300 |
| 6 | 230 | 0 | 0.0% | 230 |
| 7 | 133 | 9 | 6.8% | 124 |
| 8 | 75 | 11 | 14.7% | 64 |
| 9 | 23 | 8 | 34.8% | 15 |
| 10 | 4 | 2 | 50.0% | 2 |

Unknown states by the distance of their problem: 6 at 7, 11 at 8, 9 at 9, 4 at 10; by theory: 23 bool, 7 group.

Unknown states are those whose forward search exceeded 300,000 nodes, so they are the highest-branching deep states. Their exclusion removes a growing share of the deepest rows and probably flatters top-1 there. Rows marked \* have fewer than 10 scored states and carry no weight.

test_id: 300 problems, 804 states, 0 unknown.

## test_depth, by true distance of the state

### Mean prediction

A calibrated distance would equal k.

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.477 | 0.478 | 0.796 | 0.584 | 0.477 to 0.796 |
| 2 | 300 | 1.663 | 1.703 | 1.842 | 1.736 | 1.663 to 1.842 |
| 3 | 300 | 2.782 | 2.769 | 2.813 | 2.788 | 2.769 to 2.813 |
| 4 | 300 | 3.657 | 3.617 | 3.585 | 3.620 | 3.585 to 3.657 |
| 5 | 300 | 4.238 | 4.268 | 4.201 | 4.235 | 4.201 to 4.268 |
| 6 | 230 | 4.559 | 4.542 | 4.551 | 4.551 | 4.542 to 4.559 |
| 7 | 124 | 4.688 | 4.694 | 4.685 | 4.689 | 4.685 to 4.694 |
| 8 | 64 | 4.764 | 4.703 | 4.708 | 4.725 | 4.703 to 4.764 |
| 9 | 15 | 4.769 | 4.701 | 4.625 | 4.699 | 4.625 to 4.769 |
| 10 \* | 2 | 4.838 | 4.699 | 4.452 | 4.663 | 4.452 to 4.838 |

### All-children top-1 (optimal child ranked first, ties split)

"chance" is the mean share of children that are optimal; symbol_bag is the same for every model.

| k | states | children | chance | symbol_bag | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 46.8 | 0.025 | 0.160 | 0.823 | 0.910 | 0.727 | 0.820 | 0.727 to 0.910 |
| 2 | 300 | 42.1 | 0.046 | 0.191 | 0.730 | 0.847 | 0.667 | 0.748 | 0.667 to 0.847 |
| 3 | 300 | 36.4 | 0.075 | 0.156 | 0.740 | 0.837 | 0.613 | 0.730 | 0.613 to 0.837 |
| 4 | 300 | 29.8 | 0.119 | 0.163 | 0.690 | 0.813 | 0.640 | 0.714 | 0.640 to 0.813 |
| 5 | 300 | 23.0 | 0.216 | 0.233 | 0.653 | 0.730 | 0.617 | 0.667 | 0.617 to 0.730 |
| 6 | 230 | 18.8 | 0.271 | 0.253 | 0.657 | 0.687 | 0.583 | 0.642 | 0.583 to 0.687 |
| 7 | 124 | 12.6 | 0.353 | 0.332 | 0.694 | 0.685 | 0.629 | 0.669 | 0.629 to 0.694 |
| 8 | 64 | 7.8 | 0.427 | 0.368 | 0.734 | 0.750 | 0.672 | 0.719 | 0.672 to 0.750 |
| 9 | 15 | 3.6 | 0.622 | 0.563 | 0.933 | 0.933 | 0.800 | 0.889 | 0.800 to 0.933 |
| 10 \* | 2 | 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 to 1.000 |

Overall (95% interval: cluster bootstrap over problems, within one model):

| model | top-1 | 95% CI | symbol_bag | mean rank of best optimal child | Spearman(h, distance) |
|---:|---:|---:|---:|---:|---:|
| h-v1 | 0.719 | 0.690 to 0.747 | 0.209 | 1.90 | 0.853 |
| h-v1-s1 | 0.800 | 0.773 to 0.826 | 0.209 | 1.67 | 0.861 |
| h-v1-s2 | 0.645 | 0.615 to 0.675 | 0.209 | 2.44 | 0.833 |

### Drop to the optimal child: h(state) − h(optimal child)

A true distance gives 1.

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.603 | 1.541 | 1.591 | 1.578 | 1.541 to 1.603 |
| 2 | 300 | 1.158 | 1.175 | 0.998 | 1.110 | 0.998 to 1.175 |
| 3 | 300 | 1.049 | 1.001 | 0.886 | 0.979 | 0.886 to 1.049 |
| 4 | 300 | 0.839 | 0.789 | 0.730 | 0.786 | 0.730 to 0.839 |
| 5 | 300 | 0.590 | 0.566 | 0.604 | 0.587 | 0.566 to 0.604 |
| 6 | 230 | 0.303 | 0.295 | 0.344 | 0.314 | 0.295 to 0.344 |
| 7 | 124 | 0.068 | 0.119 | 0.139 | 0.109 | 0.068 to 0.139 |
| 8 | 64 | 0.076 | 0.072 | 0.056 | 0.068 | 0.056 to 0.076 |
| 9 | 15 | 0.000 | 0.071 | -0.133 | -0.020 | -0.133 to 0.071 |
| 10 \* | 2 | 0.049 | 0.147 | -0.201 | -0.001 | -0.201 to 0.147 |

### Margin: min h over optimal children − min h over non-optimal children

Negative means the right move is preferred. Undefined where every child is optimal.

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | -1.211 | -1.661 | -0.910 | -1.261 | -1.661 to -0.910 |
| 2 | 300 | -0.762 | -1.193 | -0.517 | -0.824 | -1.193 to -0.517 |
| 3 | 300 | -0.598 | -0.876 | -0.273 | -0.582 | -0.876 to -0.273 |
| 4 | 300 | -0.448 | -0.723 | -0.287 | -0.486 | -0.723 to -0.287 |
| 5 | 300 | -0.363 | -0.555 | -0.270 | -0.396 | -0.555 to -0.270 |
| 6 | 230 | -0.196 | -0.314 | -0.094 | -0.201 | -0.314 to -0.094 |
| 7 | 124 | -0.146 | -0.177 | -0.119 | -0.147 | -0.177 to -0.119 |
| 8 | 64 | -0.239 | -0.252 | -0.154 | -0.215 | -0.252 to -0.154 |
| 9 | 15 | -0.226 | -0.294 | -0.109 | -0.210 | -0.294 to -0.109 |
| 10 \* | 2 | – | – | – | – | – |

### Edge-violation rate: share of parent–child edges with |h(parent) − h(child)| > 1

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.728 | 0.731 | 0.646 | 0.702 | 0.646 to 0.731 |
| 2 | 300 | 0.574 | 0.575 | 0.519 | 0.556 | 0.519 to 0.575 |
| 3 | 300 | 0.437 | 0.442 | 0.421 | 0.433 | 0.421 to 0.442 |
| 4 | 300 | 0.300 | 0.339 | 0.351 | 0.330 | 0.300 to 0.351 |
| 5 | 300 | 0.232 | 0.202 | 0.241 | 0.225 | 0.202 to 0.241 |
| 6 | 230 | 0.125 | 0.110 | 0.126 | 0.120 | 0.110 to 0.126 |
| 7 | 124 | 0.074 | 0.039 | 0.078 | 0.064 | 0.039 to 0.078 |
| 8 | 64 | 0.040 | 0.018 | 0.024 | 0.027 | 0.018 to 0.040 |
| 9 | 15 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 |
| 10 \* | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 to 0.000 |

| all edges | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|
| test_depth | 0.454 | 0.456 | 0.429 | 0.447 | 0.429 to 0.456 |
| test_id | 0.666 | 0.698 | 0.618 | 0.661 | 0.618 to 0.698 |

## test_id (in distribution), by true distance of the state

### Mean prediction

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.330 | 0.260 | 0.516 | 0.368 | 0.260 to 0.516 |
| 2 | 248 | 1.737 | 1.761 | 1.851 | 1.783 | 1.737 to 1.851 |
| 3 | 171 | 2.825 | 2.880 | 2.865 | 2.857 | 2.825 to 2.880 |
| 4 | 85 | 3.729 | 3.936 | 3.676 | 3.780 | 3.676 to 3.936 |

### All-children top-1

| k | states | children | chance | symbol_bag | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 26.9 | 0.053 | 0.268 | 0.973 | 1.000 | 0.980 | 0.984 | 0.973 to 1.000 |
| 2 | 248 | 23.7 | 0.110 | 0.314 | 0.903 | 0.960 | 0.899 | 0.921 | 0.899 to 0.960 |
| 3 | 171 | 20.6 | 0.180 | 0.317 | 0.871 | 0.953 | 0.848 | 0.891 | 0.848 to 0.953 |
| 4 | 85 | 17.1 | 0.281 | 0.327 | 0.765 | 0.906 | 0.835 | 0.835 | 0.765 to 0.906 |

### Drop to the optimal child

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.658 | 1.620 | 1.747 | 1.675 | 1.620 to 1.747 |
| 2 | 248 | 1.450 | 1.560 | 1.325 | 1.445 | 1.325 to 1.560 |
| 3 | 171 | 1.015 | 1.114 | 0.949 | 1.026 | 0.949 to 1.114 |
| 4 | 85 | 0.933 | 1.096 | 0.910 | 0.980 | 0.910 to 1.096 |

### Edge-violation rate

| k | states | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.846 | 0.899 | 0.798 | 0.848 | 0.798 to 0.899 |
| 2 | 248 | 0.622 | 0.666 | 0.542 | 0.610 | 0.542 to 0.666 |
| 3 | 171 | 0.450 | 0.445 | 0.410 | 0.435 | 0.410 to 0.450 |
| 4 | 85 | 0.370 | 0.332 | 0.428 | 0.377 | 0.332 to 0.428 |

## Search: neural-w1 on the 300 test_depth problems

Solved means the kernel accepted the proof (`isomer bench`). One search per problem at 1000 expansions; smaller budgets are derived from it.

| expansions | h-v1 | h-v1-s1 | h-v1-s2 | mean | range |
|---:|---:|---:|---:|---:|---:|
| 30 | 139/300 = 46.3% | 180/300 = 60.0% | 124/300 = 41.3% | 49.2% | 41.3% to 60.0% |
| 100 | 173/300 = 57.7% | 202/300 = 67.3% | 152/300 = 50.7% | 58.6% | 50.7% to 67.3% |
| 300 | 202/300 = 67.3% | 222/300 = 74.0% | 176/300 = 58.7% | 66.7% | 58.7% to 74.0% |
| 1000 | 224/300 = 74.7% | 240/300 = 80.0% | 195/300 = 65.0% | 73.2% | 65.0% to 80.0% |

Censored cells: 0. Rows with status other than ok: none.

Reproduction: the h-v1 run agrees with the E2 run in `runs/E2-test_depth` on 300 of 300 problems (solved, expansions and proof length).

## Provenance

| model | checkpoint sha256 | scoring peak VRAM (MiB) | bench commit |
|---:|---:|---:|---:|
| h-v1 | `928015f6250a8531…` | 101 | `2b5b94c` |
| h-v1-s1 | `9982b0d574d663b8…` | 101 | `2b5b94c` |
| h-v1-s2 | `7f6864613e9c8bbf…` | 101 | `2b5b94c` |

Scoring: cuda, bf16-fast. Label file sha256: `634430a1ff4a4f57…` (test_depth), `a17dd892b10bab6b…` (test_id).

## Limits

- 3 seeds of one configuration on one benchmark; no intervention is tested here.
- Diagnostics use states on one stored shortest proof per problem, not the states search visits.
- Depth is confounded with branching: deep exact-distance problems have few children per state, so raw top-1 rises again at k = 8–9 mostly because chance does.
- The deepest rows are small and lose the most states to the node cap (table above).
