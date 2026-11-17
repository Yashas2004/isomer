# E3-d6: depth coverage or data volume? Seeds and the size-matched control

Baseline: h-v1, h-v1-s1, h-v1-s2 (train problems up to distance 4). d6: h-d6-s0, h-d6-s1, h-d6-s2 (up to distance 6, more problems). Control: `h-d6m-s0`, trained on a seeded random subset of the d6 train problems with as many problems as the baseline had. All models have the h-v1 configuration and are scored on the same labelled test_depth states (1935 scored, 30 unknown and excluded) and searched on the same 300 problems.

The control's decision rules are in `docs/experiments/E3/d6_matched_prediction.md`, committed as `a2815ff` before the control was trained.

## Control verdict

**Depth coverage is supported on every pre-registered quantity: the size-matched model keeps the d6 behaviour.**

| quantity (test_depth) | baseline range (3 seeds) | d6 (3 seeds) | h-d6m-s0 (95% CI) | depth / volume if | verdict |
|---:|---:|---:|---:|---:|---:|
| mean prediction, k = 7 | 4.685 to 4.694 | 6.186 to 6.291 | 6.006 (5.903 to 6.099) | >= 5.7 / <= 5.0 | depth |
| mean prediction, k = 8 | 4.703 to 4.764 | 6.577 to 6.646 | 6.421 (6.310 to 6.529) | >= 5.7 / <= 5.0 | depth |
| drop to the optimal child, k = 7 | 0.068 to 0.139 | 0.573 to 0.580 | 0.486 (0.416 to 0.548) | >= 0.35 / <= 0.2 | depth |
| drop to the optimal child, k = 8 | 0.056 to 0.076 | 0.351 to 0.506 | 0.417 (0.320 to 0.520) | >= 0.18 / <= 0.1 | depth |
| all-children top-1, all scored states | 0.645 to 0.800 | 0.948 to 0.964 | 0.940 (0.926 to 0.952) | >= 0.9 / <= 0.85 | depth |

The interval is a cluster bootstrap over problems within the one control model. Seed-to-seed variation is shown by the baseline range and the d6 range.

## Each d6 seed against the original prediction (`d6_prediction.md`)

| model | mean prediction, k = 7 | mean prediction, k = 8 | drop to the optimal child, k = 7 | drop to the optimal child, k = 8 | highest mean prediction at k >= 7 | overall |
|---:|---:|---:|---:|---:|---:|---:|
| h-d6-s0 | 6.234 (supported) | 6.646 (supported) | 0.578 (supported) | 0.506 (supported) | 6.841 (holds) | supported |
| h-d6-s1 | 6.291 (supported) | 6.582 (supported) | 0.573 (supported) | 0.351 (supported) | 6.671 (holds) | supported |
| h-d6-s2 | 6.186 (supported) | 6.577 (supported) | 0.580 (supported) | 0.439 (supported) | 6.737 (holds) | supported |

## test_depth, by true distance of the state

Rows marked \* have fewer than 10 scored states.

### Mean prediction

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.584 | 0.477 to 0.796 | -0.118 | -0.148 to -0.086 | 0.085 |
| 2 | 300 | 1.736 | 1.663 to 1.842 | 1.338 | 1.324 to 1.356 | 1.502 |
| 3 | 300 | 2.788 | 2.769 to 2.813 | 2.458 | 2.430 to 2.474 | 2.640 |
| 4 | 300 | 3.620 | 3.585 to 3.657 | 3.661 | 3.641 to 3.681 | 3.765 |
| 5 | 300 | 4.235 | 4.201 to 4.268 | 4.736 | 4.702 to 4.798 | 4.724 |
| 6 | 230 | 4.551 | 4.542 to 4.559 | 5.608 | 5.540 to 5.666 | 5.465 |
| 7 | 124 | 4.689 | 4.685 to 4.694 | 6.237 | 6.186 to 6.291 | 6.006 |
| 8 | 64 | 4.725 | 4.703 to 4.764 | 6.602 | 6.577 to 6.646 | 6.421 |
| 9 | 15 | 4.699 | 4.625 to 4.769 | 6.750 | 6.671 to 6.841 | 6.630 |
| 10 \* | 2 | 4.663 | 4.452 to 4.838 | 6.742 | 6.603 to 6.866 | 6.931 |

### Drop to the optimal child

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.578 | 1.541 to 1.603 | 1.322 | 1.305 to 1.351 | 1.432 |
| 2 | 300 | 1.110 | 0.998 to 1.175 | 1.415 | 1.393 to 1.437 | 1.361 |
| 3 | 300 | 0.979 | 0.886 to 1.049 | 1.031 | 1.031 to 1.032 | 1.008 |
| 4 | 300 | 0.786 | 0.730 to 0.839 | 1.056 | 1.038 to 1.074 | 0.958 |
| 5 | 300 | 0.587 | 0.566 to 0.604 | 0.985 | 0.970 to 0.992 | 0.831 |
| 6 | 230 | 0.314 | 0.295 to 0.344 | 0.840 | 0.798 to 0.904 | 0.703 |
| 7 | 124 | 0.109 | 0.068 to 0.139 | 0.577 | 0.573 to 0.580 | 0.486 |
| 8 | 64 | 0.068 | 0.056 to 0.076 | 0.432 | 0.351 to 0.506 | 0.417 |
| 9 | 15 | -0.020 | -0.133 to 0.071 | 0.185 | 0.058 to 0.276 | 0.378 |
| 10 \* | 2 | -0.001 | -0.201 to 0.147 | -0.022 | -0.120 to 0.083 | 0.484 |

### All-children top-1

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.820 | 0.727 to 0.910 | 0.994 | 0.993 to 0.997 | 0.993 |
| 2 | 300 | 0.748 | 0.667 to 0.847 | 0.971 | 0.970 to 0.973 | 0.960 |
| 3 | 300 | 0.730 | 0.613 to 0.837 | 0.972 | 0.963 to 0.983 | 0.963 |
| 4 | 300 | 0.714 | 0.640 to 0.813 | 0.962 | 0.953 to 0.973 | 0.933 |
| 5 | 300 | 0.667 | 0.617 to 0.730 | 0.927 | 0.917 to 0.943 | 0.920 |
| 6 | 230 | 0.642 | 0.583 to 0.687 | 0.914 | 0.896 to 0.930 | 0.883 |
| 7 | 124 | 0.669 | 0.629 to 0.694 | 0.898 | 0.879 to 0.911 | 0.887 |
| 8 | 64 | 0.719 | 0.672 to 0.750 | 0.958 | 0.953 to 0.969 | 0.891 |
| 9 | 15 | 0.889 | 0.800 to 0.933 | 1.000 | 1.000 to 1.000 | 1.000 |
| 10 \* | 2 | 1.000 | 1.000 to 1.000 | 1.000 | 1.000 to 1.000 | 1.000 |

### Margin

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | -1.261 | -1.661 to -0.910 | -2.200 | -2.274 to -2.145 | -2.100 |
| 2 | 300 | -0.824 | -1.193 to -0.517 | -1.923 | -1.973 to -1.893 | -1.800 |
| 3 | 300 | -0.582 | -0.876 to -0.273 | -1.478 | -1.553 to -1.430 | -1.316 |
| 4 | 300 | -0.486 | -0.723 to -0.287 | -1.442 | -1.597 to -1.305 | -1.190 |
| 5 | 300 | -0.396 | -0.555 to -0.270 | -1.356 | -1.471 to -1.240 | -1.111 |
| 6 | 230 | -0.201 | -0.314 to -0.094 | -1.184 | -1.280 to -1.067 | -0.960 |
| 7 | 124 | -0.147 | -0.177 to -0.119 | -0.881 | -0.939 to -0.829 | -0.766 |
| 8 | 64 | -0.215 | -0.252 to -0.154 | -0.804 | -0.883 to -0.759 | -0.699 |
| 9 | 15 | -0.210 | -0.294 to -0.109 | -0.579 | -0.671 to -0.511 | -0.858 |
| 10 \* | 2 | – | – | – | – | – |

### Edge-violation rate

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.702 | 0.646 to 0.731 | 0.945 | 0.931 to 0.961 | 0.894 |
| 2 | 300 | 0.556 | 0.519 to 0.575 | 0.868 | 0.847 to 0.887 | 0.770 |
| 3 | 300 | 0.433 | 0.421 to 0.442 | 0.768 | 0.740 to 0.797 | 0.675 |
| 4 | 300 | 0.330 | 0.300 to 0.351 | 0.667 | 0.634 to 0.713 | 0.538 |
| 5 | 300 | 0.225 | 0.202 to 0.241 | 0.524 | 0.466 to 0.604 | 0.410 |
| 6 | 230 | 0.120 | 0.110 to 0.126 | 0.336 | 0.328 to 0.341 | 0.289 |
| 7 | 124 | 0.064 | 0.039 to 0.078 | 0.169 | 0.150 to 0.202 | 0.196 |
| 8 | 64 | 0.027 | 0.018 to 0.040 | 0.066 | 0.044 to 0.095 | 0.101 |
| 9 | 15 | 0.000 | 0.000 to 0.000 | 0.000 | 0.000 to 0.000 | 0.056 |
| 10 \* | 2 | 0.000 | 0.000 to 0.000 | 0.000 | 0.000 to 0.000 | 0.000 |

## All-children top-1, all scored states (95% CI: cluster bootstrap over problems)

| model | test_depth | test_id |
|---:|---:|---:|
| h-v1 | 0.719 (0.690 to 0.747) | 0.908 (0.883 to 0.931) |
| h-v1-s1 | 0.800 (0.773 to 0.826) | 0.968 (0.953 to 0.980) |
| h-v1-s2 | 0.645 (0.615 to 0.675) | 0.912 (0.888 to 0.933) |
| h-d6-s0 | 0.953 (0.941 to 0.964) | 0.991 (0.984 to 0.997) |
| h-d6-s1 | 0.964 (0.952 to 0.973) | 0.996 (0.991 to 1.000) |
| h-d6-s2 | 0.948 (0.936 to 0.959) | 0.993 (0.986 to 0.998) |
| h-d6m-s0 | 0.940 (0.926 to 0.952) | 0.983 (0.972 to 0.991) |

### test_id, by true distance of the state

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.984 | 0.973 to 1.000 | 1.000 | 1.000 to 1.000 | 1.000 |
| 2 | 248 | 0.921 | 0.899 to 0.960 | 0.997 | 0.996 to 1.000 | 0.988 |
| 3 | 171 | 0.891 | 0.848 to 0.953 | 0.984 | 0.977 to 0.988 | 0.965 |
| 4 | 85 | 0.835 | 0.765 to 0.906 | 0.976 | 0.965 to 0.988 | 0.941 |

## Search: neural-w1 on the 300 test_depth problems (exploratory)

Solved means the kernel accepted the proof.

| expansions | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|
| 30 | 49.2% | 41.3% to 60.0% | 90.7% | 88.3% to 94.7% | 255/300 = 85.0% |
| 100 | 58.6% | 50.7% to 67.3% | 97.4% | 95.7% to 99.0% | 281/300 = 93.7% |
| 300 | 66.7% | 58.7% to 74.0% | 98.7% | 97.3% to 100.0% | 294/300 = 98.0% |
| 1000 | 73.2% | 65.0% to 80.0% | 99.6% | 99.3% to 100.0% | 298/300 = 99.3% |

Problems solved, by exact distance of the problem (ranges over seeds):

| problem distance | problems | baseline @30 | d6 @30 | baseline @1000 | d6 @1000 | h-d6m-s0 @30 | h-d6m-s0 @1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 70 | 61 to 68 | 69 to 70 | 68 to 69 | 70 to 70 | 68 | 70 |
| 6 | 97 | 42 to 72 | 91 to 95 | 78 to 92 | 97 to 97 | 91 | 97 |
| 7 | 58 | 15 to 31 | 51 to 55 | 33 to 45 | 56 to 58 | 51 | 57 |
| 8 | 52 | 4 to 9 | 36 to 45 | 14 to 29 | 52 to 52 | 37 | 51 |
| 9 | 19 | 0 to 1 | 10 to 16 | 2 to 3 | 18 to 19 | 8 | 19 |
| 10 | 4 | 0 to 0 | 2 to 3 | 0 to 2 | 3 to 4 | 0 | 4 |

Rows with status other than ok: none.

## Training runs

| model | states directory | seed | train problems | train states | optimizer steps |
|---:|---:|---:|---:|---:|---:|
| h-v1 | `data/states-v2` | 0 | 29268 | 551072 | 9160 |
| h-v1-s1 | `data/states-v2` | 1 | 29268 | 551072 | 9160 |
| h-v1-s2 | `data/states-v2` | 2 | 29268 | 551072 | 9160 |
| h-d6-s0 | `data/states-v2-d6` | 0 | 36045 | 792841 | 11280 |
| h-d6-s1 | `data/states-v2-d6` | 1 | 36045 | 792841 | 11280 |
| h-d6-s2 | `data/states-v2-d6` | 2 | 36045 | 792841 | 11280 |
| h-d6m-s0 | `data/states-v2-d6-matched` | 0 | 29268 | 642983 | 9160 |

## What this does and does not control

- **Matched:** the number of train problems and of optimizer steps (the trainer batches by problem). **Not matched:** the number of training states; the control's problems are deeper than the baseline's and have more states each (table above).
- The control is one seed and one random subset. The d6 range shows how much a seed alone moves each quantity.
- The control tests volume against depth coverage of the *training problems*. It does not separate the deeper problems from the thin tail of distance-7 off-path states they bring.
- Diagnostics use on-path states of one stored shortest proof; unknown states are excluded for every model alike. Search results are exploratory.
