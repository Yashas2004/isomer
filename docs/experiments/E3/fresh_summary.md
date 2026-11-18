# E3-fresh: confirmatory evaluation on newly generated problems

300 problems of `test_depth_fresh` (`docs/experiments/E3/ids-test_depth_fresh.txt`), generated and labelled after every E3 hypothesis was formed: the v2 eval-pool generator and labeller with a new seed, ids disjoint from every split of `data/isobench-v2` and `data/isobench-v2-d6`. The rules are those of `docs/experiments/E3/fresh_prediction.md` (commit `40ebab3`), committed before any model was run on this sample. Every number below is read from `runs/e3-fresh`.

## Verdicts

| hypothesis | rule | verdict |
|---:|---:|---:|
| H1 saturation | every baseline seed < 5.0 and every d6 seed >= 5.7 in mean prediction at k = 7 and 8 | **supported** |
| H2 move choice | every d6 seed's all-children top-1 > the best baseline seed's | **supported** |
| H3 search | every d6 seed solves more than the best baseline seed at 30 and 1,000 expansions, and every Holm-adjusted McNemar p < 0.05 | **supported** |
| H4 control | h-d6m-s0 is "depth" on all five rules of d6_matched_prediction.md | **supported** |

## Population and unknown states

All 300 problems are labelled (1965 on-path states). 26 states in 19 problems are **unknown** (their child labelling exceeded 300,000 nodes) and are excluded from H1, H2 and H4, leaving 1939 scored states. H3 uses all 300 problems.

| state distance k | on-path states | unknown | scored |
|---:|---:|---:|---:|
| 1 | 300 | 0 | 300 |
| 2 | 300 | 0 | 300 |
| 3 | 300 | 0 | 300 |
| 4 | 300 | 0 | 300 |
| 5 | 300 | 0 | 300 |
| 6 | 230 | 0 | 230 |
| 7 | 133 | 7 | 126 |
| 8 | 75 | 10 | 65 |
| 9 | 23 | 6 | 17 |
| 10 | 4 | 3 | 1 |

Unknown states by the distance of their problem: 3 at 7, 14 at 8, 3 at 9, 6 at 10.

## H1: saturation

| model | group | k | scored states | mean prediction | rule | holds |
|---:|---:|---:|---:|---:|---:|---:|
| h-v1 | baseline | 7 | 126 | 4.652 | < 5.0 | yes |
| h-v1 | baseline | 8 | 65 | 4.724 | < 5.0 | yes |
| h-v1-s1 | baseline | 7 | 126 | 4.651 | < 5.0 | yes |
| h-v1-s1 | baseline | 8 | 65 | 4.678 | < 5.0 | yes |
| h-v1-s2 | baseline | 7 | 126 | 4.664 | < 5.0 | yes |
| h-v1-s2 | baseline | 8 | 65 | 4.679 | < 5.0 | yes |
| h-d6-s0 | d6 | 7 | 126 | 6.162 | >= 5.7 | yes |
| h-d6-s0 | d6 | 8 | 65 | 6.565 | >= 5.7 | yes |
| h-d6-s1 | d6 | 7 | 126 | 6.284 | >= 5.7 | yes |
| h-d6-s1 | d6 | 8 | 65 | 6.565 | >= 5.7 | yes |
| h-d6-s2 | d6 | 7 | 126 | 6.025 | >= 5.7 | yes |
| h-d6-s2 | d6 | 8 | 65 | 6.470 | >= 5.7 | yes |

## H2: move choice (all-children top-1, ties split, all scored states)

| model | group | top-1 | 95% CI (cluster bootstrap) | symbol_bag |
|---:|---:|---:|---:|---:|
| h-v1 | baseline | 0.695 | 0.667 to 0.723 | 0.211 |
| h-v1-s1 | baseline | 0.803 | 0.776 to 0.829 | 0.211 |
| h-v1-s2 | baseline | 0.613 | 0.580 to 0.647 | 0.211 |
| h-d6-s0 | d6 | 0.949 | 0.936 to 0.960 | 0.211 |
| h-d6-s1 | d6 | 0.971 | 0.962 to 0.980 | 0.211 |
| h-d6-s2 | d6 | 0.956 | 0.944 to 0.967 | 0.211 |
| h-d6m-s0 | control | 0.936 | 0.922 to 0.949 | 0.211 |

Best baseline seed: h-v1-s1 (0.803). h-d6-s0 0.949: yes; h-d6-s1 0.971: yes; h-d6-s2 0.956: yes.

## H3: search (neural-w1, solved of 300, kernel-accepted)

One search per problem at 1,000 expansions; smaller budgets are derived from it. bfs-bidir and bestfirst-symbol_bag-w1 are reported, not tested.

| expansions | h-v1 | h-v1-s1 | h-v1-s2 | h-d6-s0 | h-d6-s1 | h-d6-s2 | h-d6m-s0 | bfs-bidir | bestfirst-symbol_bag-w1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 143 | 171 | 114 | 263 | 281 | 261 | 248 | 19 | 7 |
| 100 | 176 | 210 | 144 | 279 | 289 | 282 | 272 | 85 | 16 |
| 300 | 196 | 226 | 173 | 289 | 294 | 287 | 283 | 125 | 33 |
| 1000 | 216 | 234 | 192 | 290 | 294 | 292 | 290 | 178 | 55 |

Rows with a status other than ok: none.

Counts against the best baseline seed at each budget:

| expansions | d6 seed | solved | best baseline seed | more |
|---:|---:|---:|---:|---:|
| 30 | h-d6-s0 | 263 | h-v1-s1 (171) | yes |
| 30 | h-d6-s1 | 281 | h-v1-s1 (171) | yes |
| 30 | h-d6-s2 | 261 | h-v1-s1 (171) | yes |
| 1000 | h-d6-s0 | 290 | h-v1-s1 (234) | yes |
| 1000 | h-d6-s1 | 294 | h-v1-s1 (234) | yes |
| 1000 | h-d6-s2 | 292 | h-v1-s1 (234) | yes |

Exact McNemar tests, Holm-adjusted over all 18:

| expansions | d6 seed | baseline seed | solved only by d6 | solved only by baseline | p | Holm p | d6 better, Holm p < 0.05 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | h-d6-s0 | h-v1 | 120 | 0 | 1.5e-36 | 2.1e-35 | yes |
| 30 | h-d6-s0 | h-v1-s1 | 95 | 3 | 9.9e-25 | 6.9e-24 | yes |
| 30 | h-d6-s0 | h-v1-s2 | 149 | 0 | 2.8e-45 | 4.8e-44 | yes |
| 30 | h-d6-s1 | h-v1 | 139 | 1 | 2e-40 | 3e-39 | yes |
| 30 | h-d6-s1 | h-v1-s1 | 111 | 1 | 4.4e-32 | 5.2e-31 | yes |
| 30 | h-d6-s1 | h-v1-s2 | 167 | 0 | 1.1e-50 | 1.9e-49 | yes |
| 30 | h-d6-s2 | h-v1 | 118 | 0 | 6e-36 | 7.8e-35 | yes |
| 30 | h-d6-s2 | h-v1-s1 | 90 | 0 | 1.6e-27 | 1.3e-26 | yes |
| 30 | h-d6-s2 | h-v1-s2 | 147 | 0 | 1.1e-44 | 1.8e-43 | yes |
| 1000 | h-d6-s0 | h-v1 | 75 | 1 | 2e-21 | 8.2e-21 | yes |
| 1000 | h-d6-s0 | h-v1-s1 | 57 | 1 | 4.1e-16 | 4.1e-16 | yes |
| 1000 | h-d6-s0 | h-v1-s2 | 98 | 0 | 6.3e-30 | 6.3e-29 | yes |
| 1000 | h-d6-s1 | h-v1 | 78 | 0 | 6.6e-24 | 4e-23 | yes |
| 1000 | h-d6-s1 | h-v1-s1 | 60 | 0 | 1.7e-18 | 5.2e-18 | yes |
| 1000 | h-d6-s1 | h-v1-s2 | 103 | 1 | 1e-29 | 9.3e-29 | yes |
| 1000 | h-d6-s2 | h-v1 | 76 | 0 | 2.6e-23 | 1.3e-22 | yes |
| 1000 | h-d6-s2 | h-v1-s1 | 58 | 0 | 6.9e-18 | 1.4e-17 | yes |
| 1000 | h-d6-s2 | h-v1-s2 | 100 | 0 | 1.6e-30 | 1.7e-29 | yes |

Solved at 1,000 expansions by problem distance:

| problem distance | problems | h-v1 | h-v1-s1 | h-v1-s2 | h-d6-s0 | h-d6-s1 | h-d6-s2 | h-d6m-s0 | bfs-bidir | bestfirst-symbol_bag-w1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 70 | 69 | 69 | 68 | 70 | 70 | 70 | 70 | 70 | 31 |
| 6 | 97 | 83 | 88 | 73 | 97 | 97 | 97 | 97 | 78 | 20 |
| 7 | 58 | 38 | 49 | 36 | 56 | 57 | 58 | 57 | 29 | 4 |
| 8 | 52 | 24 | 27 | 15 | 49 | 51 | 51 | 49 | 1 | 0 |
| 9 | 19 | 2 | 1 | 0 | 15 | 16 | 14 | 14 | 0 | 0 |
| 10 | 4 | 0 | 0 | 0 | 3 | 3 | 2 | 3 | 0 | 0 |

## H4: the size-matched control h-d6m-s0

| quantity | h-d6m-s0 | depth if | volume if | verdict |
|---:|---:|---:|---:|---:|
| mean prediction at k = 7 | 6.032 | >= 5.7 | <= 5.0 | depth |
| mean prediction at k = 8 | 6.378 | >= 5.7 | <= 5.0 | depth |
| drop to the optimal child at k = 7 | 0.581 | >= 0.35 | <= 0.2 | depth |
| drop to the optimal child at k = 8 | 0.424 | >= 0.18 | <= 0.1 | depth |
| all-children top-1, all scored states | 0.936 | >= 0.9 | <= 0.85 | depth |

## By true distance of the state (reported, not tested)

Mean and range over the three seeds of each group; the last column is h-d6m-s0. Rows marked \* have fewer than 10 scored states.

### Mean prediction

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.573 | 0.451 to 0.776 | -0.135 | -0.137 to -0.129 | 0.100 |
| 2 | 300 | 1.738 | 1.675 to 1.845 | 1.374 | 1.329 to 1.411 | 1.549 |
| 3 | 300 | 2.730 | 2.696 to 2.761 | 2.507 | 2.459 to 2.556 | 2.647 |
| 4 | 300 | 3.574 | 3.567 to 3.578 | 3.690 | 3.645 to 3.732 | 3.736 |
| 5 | 300 | 4.219 | 4.184 to 4.238 | 4.709 | 4.667 to 4.738 | 4.748 |
| 6 | 230 | 4.514 | 4.478 to 4.540 | 5.580 | 5.495 to 5.668 | 5.495 |
| 7 | 126 | 4.655 | 4.651 to 4.664 | 6.157 | 6.025 to 6.284 | 6.032 |
| 8 | 65 | 4.694 | 4.678 to 4.724 | 6.533 | 6.470 to 6.565 | 6.378 |
| 9 | 17 | 4.645 | 4.573 to 4.717 | 6.538 | 6.490 to 6.566 | 6.540 |
| 10 \* | 1 | 4.587 | 4.479 to 4.645 | 6.610 | 6.512 to 6.692 | 6.601 |

### All-children top-1

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.818 | 0.740 to 0.900 | 0.998 | 0.997 to 1.000 | 0.997 |
| 2 | 300 | 0.766 | 0.663 to 0.853 | 0.990 | 0.977 to 1.000 | 0.967 |
| 3 | 300 | 0.692 | 0.577 to 0.837 | 0.971 | 0.970 to 0.973 | 0.937 |
| 4 | 300 | 0.674 | 0.573 to 0.807 | 0.970 | 0.963 to 0.983 | 0.937 |
| 5 | 300 | 0.641 | 0.563 to 0.733 | 0.933 | 0.923 to 0.953 | 0.917 |
| 6 | 230 | 0.639 | 0.587 to 0.730 | 0.919 | 0.896 to 0.943 | 0.896 |
| 7 | 126 | 0.624 | 0.524 to 0.675 | 0.899 | 0.865 to 0.944 | 0.865 |
| 8 | 65 | 0.728 | 0.615 to 0.785 | 0.908 | 0.892 to 0.938 | 0.892 |
| 9 | 17 | 0.765 | 0.706 to 0.824 | 0.902 | 0.824 to 1.000 | 0.882 |
| 10 \* | 1 | 1.000 | 1.000 to 1.000 | 1.000 | 1.000 to 1.000 | 1.000 |

### Drop to the optimal child

| k | states | baseline mean | baseline range | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.503 | 1.478 to 1.519 | 1.316 | 1.310 to 1.320 | 1.471 |
| 2 | 300 | 1.152 | 1.055 to 1.222 | 1.474 | 1.457 to 1.500 | 1.416 |
| 3 | 300 | 0.951 | 0.856 to 1.030 | 1.066 | 1.062 to 1.069 | 1.043 |
| 4 | 300 | 0.793 | 0.755 to 0.836 | 1.102 | 1.092 to 1.113 | 0.965 |
| 5 | 300 | 0.590 | 0.576 to 0.618 | 0.970 | 0.933 to 1.032 | 0.908 |
| 6 | 230 | 0.306 | 0.293 to 0.319 | 0.850 | 0.777 to 0.921 | 0.724 |
| 7 | 126 | 0.170 | 0.123 to 0.236 | 0.631 | 0.585 to 0.664 | 0.581 |
| 8 | 65 | 0.055 | 0.032 to 0.082 | 0.431 | 0.321 to 0.498 | 0.424 |
| 9 | 17 | -0.013 | -0.058 to 0.011 | 0.281 | 0.173 to 0.342 | 0.398 |
| 10 \* | 1 | -0.041 | -0.237 to 0.351 | 0.709 | 0.567 to 0.907 | 1.139 |

## Overlap with training states (reported, not tested)

Exact (theory, state, goal) pairs, scored states only. Problem ids are disjoint by construction; states are not.

| k | scored states | state in v2 | state in d6 | states with a child in v2 | states with a child in d6 |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 3 | 3 | 4 | 4 |
| 2 | 300 | 2 | 2 | 4 | 4 |
| 3 | 300 | 2 | 2 | 4 | 4 |
| 4 | 300 | 1 | 1 | 2 | 2 |
| 5 | 300 | 0 | 0 | 1 | 1 |
| 6 | 230 | 0 | 0 | 0 | 0 |
| 7 | 126 | 0 | 0 | 0 | 0 |
| 8 | 65 | 0 | 0 | 0 | 0 |
| 9 | 17 | 0 | 0 | 0 | 0 |
| 10 | 1 | 0 | 0 | 0 | 0 |

Problems whose start pair is a training pair: 0 (v2), 0 (d6); problems with any on-path pair in training: 4 (v2), 4 (d6), of 300.

## Provenance

| model | checkpoint sha256 | scoring peak VRAM (MiB) |
|---:|---:|---:|
| h-v1 | `928015f6250a8531…` | 110 |
| h-v1-s1 | `9982b0d574d663b8…` | 110 |
| h-v1-s2 | `7f6864613e9c8bbf…` | 110 |
| h-d6-s0 | `d5ca6f32a62c97a9…` | 110 |
| h-d6-s1 | `e1dd6b3e29b236e8…` | 110 |
| h-d6-s2 | `f870d9057871fa3e…` | 110 |
| h-d6m-s0 | `abffc4afd0d00e74…` | 110 |

Label file sha256: `dc512d8dbc72e37a…`.

## Limits

- Same generator, theories and action space as v2: this confirms the results on new problems, not on a new distribution.
- Ids that collide with v2 or d6 were replaced at generation; the number is in the datasheet.
- The H1 and H4 thresholds were chosen with the E2-sample results in view; what is new here is the problems, not the rules.
- Three seeds per group; the control is one seed.
- Unknown states are excluded from H1, H2 and H4, which removes the highest-branching deep states.
