# E3-d6: does training on deeper problems move the saturation? (h-d6-s0)

`h-d6-s0` has the h-v1 configuration and is trained on train problems of exact distance up to 6 (training states up to 7). The baseline is h-v1, h-v1-s1, h-v1-s2, trained on distance up to 4 (states up to 5). All models are scored on the same labelled test_depth states (1935 scored, 30 unknown and excluded) and searched on the same 300 problems.

The prediction and its decision rules are in `docs/experiments/E3/d6_prediction.md`, committed as `583e232` before any d6 training state was generated.

## Verdict

**The prediction is supported on every pre-registered quantity.**

| quantity | baseline range (3 seeds) | predicted | h-d6-s0 (95% CI) | supported / refuted if | verdict |
|---:|---:|---:|---:|---:|---:|
| mean prediction, k = 7 | 4.685 to 4.694 | about 6.2 | 6.234 (6.147 to 6.317) | >= 5.7 / <= 5.0 | supported |
| mean prediction, k = 8 | 4.703 to 4.764 | about 6.5 | 6.646 (6.564 to 6.733) | >= 5.7 / <= 5.0 | supported |
| drop to the optimal child, k = 7 | 0.068 to 0.139 | about 0.6 | 0.578 (0.493 to 0.666) | >= 0.35 / <= 0.2 | supported |
| drop to the optimal child, k = 8 | 0.056 to 0.076 | about 0.3 | 0.506 (0.422 to 0.588) | >= 0.18 / <= 0.1 | supported |
| plateau: highest mean prediction at k >= 7 | below 5 | between 6 and 7 | 6.841 | below 7 | holds |

The interval is a cluster bootstrap over problems within the one d6 model; it does not include seed-to-seed variation, for which the only yardstick is the baseline range over 3 seeds.

## test_depth, by true distance of the state

Distances 5-6 are in-distribution by distance for h-d6-s0 and out of range for the baseline; the test is at 7-8. Rows marked \* have fewer than 10 scored states. "difference" is h-d6-s0 minus the baseline mean.

### Mean prediction

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.584 | 0.477 to 0.796 | -0.121 | -0.705 |
| 2 | 300 | 1.736 | 1.663 to 1.842 | 1.334 | -0.402 |
| 3 | 300 | 2.788 | 2.769 to 2.813 | 2.469 | -0.319 |
| 4 | 300 | 3.620 | 3.585 to 3.657 | 3.681 | +0.062 |
| 5 | 300 | 4.235 | 4.201 to 4.268 | 4.798 | +0.563 |
| 6 | 230 | 4.551 | 4.542 to 4.559 | 5.617 | +1.066 |
| 7 | 124 | 4.689 | 4.685 to 4.694 | 6.234 | +1.545 |
| 8 | 64 | 4.725 | 4.703 to 4.764 | 6.646 | +1.921 |
| 9 | 15 | 4.699 | 4.625 to 4.769 | 6.841 | +2.142 |
| 10 \* | 2 | 4.663 | 4.452 to 4.838 | 6.866 | +2.203 |

### Drop to the optimal child

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.578 | 1.541 to 1.603 | 1.311 | -0.268 |
| 2 | 300 | 1.110 | 0.998 to 1.175 | 1.415 | +0.304 |
| 3 | 300 | 0.979 | 0.886 to 1.049 | 1.031 | +0.052 |
| 4 | 300 | 0.786 | 0.730 to 0.839 | 1.055 | +0.269 |
| 5 | 300 | 0.587 | 0.566 to 0.604 | 0.992 | +0.405 |
| 6 | 230 | 0.314 | 0.295 to 0.344 | 0.798 | +0.484 |
| 7 | 124 | 0.109 | 0.068 to 0.139 | 0.578 | +0.470 |
| 8 | 64 | 0.068 | 0.056 to 0.076 | 0.506 | +0.438 |
| 9 | 15 | -0.020 | -0.133 to 0.071 | 0.276 | +0.296 |
| 10 \* | 2 | -0.001 | -0.201 to 0.147 | 0.083 | +0.084 |

### All-children top-1 (exploratory)

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.820 | 0.727 to 0.910 | 0.993 | +0.173 |
| 2 | 300 | 0.748 | 0.667 to 0.847 | 0.970 | +0.222 |
| 3 | 300 | 0.730 | 0.613 to 0.837 | 0.970 | +0.240 |
| 4 | 300 | 0.714 | 0.640 to 0.813 | 0.953 | +0.239 |
| 5 | 300 | 0.667 | 0.617 to 0.730 | 0.920 | +0.253 |
| 6 | 230 | 0.642 | 0.583 to 0.687 | 0.917 | +0.275 |
| 7 | 124 | 0.669 | 0.629 to 0.694 | 0.903 | +0.234 |
| 8 | 64 | 0.719 | 0.672 to 0.750 | 0.969 | +0.250 |
| 9 | 15 | 0.889 | 0.800 to 0.933 | 1.000 | +0.111 |
| 10 \* | 2 | 1.000 | 1.000 to 1.000 | 1.000 | +0.000 |

### Margin (exploratory)

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | -1.261 | -1.661 to -0.910 | -2.145 | -0.884 |
| 2 | 300 | -0.824 | -1.193 to -0.517 | -1.893 | -1.069 |
| 3 | 300 | -0.582 | -0.876 to -0.273 | -1.451 | -0.869 |
| 4 | 300 | -0.486 | -0.723 to -0.287 | -1.423 | -0.937 |
| 5 | 300 | -0.396 | -0.555 to -0.270 | -1.358 | -0.962 |
| 6 | 230 | -0.201 | -0.314 to -0.094 | -1.207 | -1.006 |
| 7 | 124 | -0.147 | -0.177 to -0.119 | -0.876 | -0.729 |
| 8 | 64 | -0.215 | -0.252 to -0.154 | -0.883 | -0.668 |
| 9 | 15 | -0.210 | -0.294 to -0.109 | -0.671 | -0.461 |
| 10 \* | 2 | – | – | – | – |

### Edge-violation rate (exploratory)

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.702 | 0.646 to 0.731 | 0.943 | +0.241 |
| 2 | 300 | 0.556 | 0.519 to 0.575 | 0.847 | +0.291 |
| 3 | 300 | 0.433 | 0.421 to 0.442 | 0.740 | +0.307 |
| 4 | 300 | 0.330 | 0.300 to 0.351 | 0.634 | +0.304 |
| 5 | 300 | 0.225 | 0.202 to 0.241 | 0.466 | +0.241 |
| 6 | 230 | 0.120 | 0.110 to 0.126 | 0.328 | +0.207 |
| 7 | 124 | 0.064 | 0.039 to 0.078 | 0.202 | +0.138 |
| 8 | 64 | 0.027 | 0.018 to 0.040 | 0.095 | +0.067 |
| 9 | 15 | 0.000 | 0.000 to 0.000 | 0.000 | +0.000 |
| 10 \* | 2 | 0.000 | 0.000 to 0.000 | 0.000 | +0.000 |

## test_id (in distribution for every model), by true distance

### Mean prediction

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.368 | 0.260 to 0.516 | 0.106 | -0.262 |
| 2 | 248 | 1.783 | 1.737 to 1.851 | 1.619 | -0.164 |
| 3 | 171 | 2.857 | 2.825 to 2.880 | 2.842 | -0.015 |
| 4 | 85 | 3.780 | 3.676 to 3.936 | 4.021 | +0.241 |

### All-children top-1

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.984 | 0.973 to 1.000 | 1.000 | +0.016 |
| 2 | 248 | 0.921 | 0.899 to 0.960 | 0.996 | +0.075 |
| 3 | 171 | 0.891 | 0.848 to 0.953 | 0.977 | +0.086 |
| 4 | 85 | 0.835 | 0.765 to 0.906 | 0.976 | +0.141 |

### Drop to the optimal child

| k | states | baseline mean | baseline range | h-d6-s0 | difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.675 | 1.620 to 1.747 | 1.566 | -0.110 |
| 2 | 248 | 1.445 | 1.325 to 1.560 | 1.561 | +0.116 |
| 3 | 171 | 1.026 | 0.949 to 1.114 | 1.207 | +0.182 |
| 4 | 85 | 0.980 | 0.910 to 1.096 | 1.260 | +0.280 |

## Search: neural-w1 on the 300 test_depth problems (exploratory)

Solved means the kernel accepted the proof.

| expansions | baseline mean | baseline range | h-d6-s0 |
|---:|---:|---:|---:|
| 30 | 49.2% | 41.3% to 60.0% | 265/300 = 88.3% |
| 100 | 58.6% | 50.7% to 67.3% | 287/300 = 95.7% |
| 300 | 66.7% | 58.7% to 74.0% | 292/300 = 97.3% |
| 1000 | 73.2% | 65.0% to 80.0% | 298/300 = 99.3% |

Problems solved, by exact distance of the problem (baseline: range over seeds):

| problem distance | problems | baseline @30 | h-d6-s0 @30 | baseline @100 | h-d6-s0 @100 | baseline @300 | h-d6-s0 @300 | baseline @1000 | h-d6-s0 @1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 70 | 61 to 68 | 70 | 66 to 69 | 70 | 68 to 69 | 70 | 68 to 69 | 70 |
| 6 | 97 | 42 to 72 | 91 | 56 to 79 | 97 | 70 to 87 | 97 | 78 to 92 | 97 |
| 7 | 58 | 15 to 31 | 51 | 18 to 37 | 53 | 25 to 41 | 55 | 33 to 45 | 56 |
| 8 | 52 | 4 to 9 | 36 | 10 to 15 | 46 | 12 to 21 | 47 | 14 to 29 | 52 |
| 9 | 19 | 0 to 1 | 14 | 1 to 2 | 17 | 1 to 3 | 19 | 2 to 3 | 19 |
| 10 | 4 | 0 to 0 | 3 | 0 to 1 | 4 | 0 to 1 | 4 | 0 to 2 | 4 |

Rows with status other than ok: none.

## Training runs

| model | states directory | seed | train problems | train states | optimizer steps |
|---:|---:|---:|---:|---:|---:|
| h-v1 | `data/states-v2` | 0 | 29268 | 551072 | 9160 |
| h-v1-s1 | `data/states-v2` | 1 | 29268 | 551072 | 9160 |
| h-v1-s2 | `data/states-v2` | 2 | 29268 | 551072 | 9160 |
| h-d6-s0 | `data/states-v2-d6` | 0 | 36045 | 792841 | 11280 |

## What this comparison does not control

- **Amount of data.** h-d6-s0 saw more problems, more states and more optimizer steps than the baseline (table above), not only deeper ones. A size-matched control was not run.
- **One seed** of h-d6-s0 against 3 baseline seeds.
- **Dropped training states.** `isomer states` drops an off-path state whose BFS label exceeds its budget; see `dropped` in the states directory's `stats.json`.
- **Train and test pools differ in generation.** Distance 5-6 train problems come from walks of length at most 6; test_depth problems from walks of 6-24.
- Diagnostics use on-path states of one stored shortest proof; unknown states are excluded for every model alike.
