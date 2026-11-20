# E3-d5: does the ceiling track the training range? (h-d5-s0)

h-d5-s0 has the h-v1 configuration, seed 0, trained on problems up to exact distance 5 (training states up to 6): between the baseline (problems up to 4, states up to 5) and d6 (up to 6, states up to 7). **One seed**: nothing below is a seed range. The rules are those of `docs/experiments/E3/d5_prediction.md` (commit `cef3f69`), committed before any d5 training state existed, and are judged on the E2 test_depth sample (`runs/e3`). The fresh sample (`runs/e3-fresh`) is reported, not tested.

## Verdict

**not supported** (all three rules must hold).

| rule | requirement | h-d5-s0 | holds |
|---:|---:|---:|---:|
| H-d5a: mean prediction at k = 7 | 4.764 < value < 6.186 | 5.493 | yes |
| H-d5a: mean prediction at k = 8 | 4.764 < value < 6.577 | 5.683 | yes |
| H-d5b: highest mean prediction at k >= 6 (at k = 9) | 5.2 <= value <= 6.2 | 5.794 | yes |
| H-d5c: all-children top-1, all scored states | 0.800 < value < 0.948 | 0.793 | **no** |

## test_depth, E2 sample (tested)

All 300 problems are labelled (1965 on-path states); 30 are unknown (child labelling above 300,000 nodes) and excluded, leaving 1935 scored states.

### Mean prediction by true distance k

A calibrated distance would equal k. Rows marked \* have fewer than 10 scored states.

| k | states | baseline mean | baseline range | h-d5-s0 | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.584 | 0.477 to 0.796 | 0.470 | -0.118 | -0.148 to -0.086 | 0.085 |
| 2 | 300 | 1.736 | 1.663 to 1.842 | 1.733 | 1.338 | 1.324 to 1.356 | 1.502 |
| 3 | 300 | 2.788 | 2.769 to 2.813 | 2.804 | 2.458 | 2.430 to 2.474 | 2.640 |
| 4 | 300 | 3.620 | 3.585 to 3.657 | 3.751 | 3.661 | 3.641 to 3.681 | 3.765 |
| 5 | 300 | 4.235 | 4.201 to 4.268 | 4.524 | 4.736 | 4.702 to 4.798 | 4.724 |
| 6 | 230 | 4.551 | 4.542 to 4.559 | 5.099 | 5.608 | 5.540 to 5.666 | 5.465 |
| 7 | 124 | 4.689 | 4.685 to 4.694 | 5.493 | 6.237 | 6.186 to 6.291 | 6.006 |
| 8 | 64 | 4.725 | 4.703 to 4.764 | 5.683 | 6.602 | 6.577 to 6.646 | 6.421 |
| 9 | 15 | 4.699 | 4.625 to 4.769 | 5.794 | 6.750 | 6.671 to 6.841 | 6.630 |
| 10 \* | 2 | 4.663 | 4.452 to 4.838 | 5.768 | 6.742 | 6.603 to 6.866 | 6.931 |

### The quantities of the rules

| quantity | baseline range (3 seeds) | h-d5-s0 | d6 range (3 seeds) | h-d6m-s0 | h-d5-s0 is |
|---:|---:|---:|---:|---:|---:|
| mean prediction at k = 7 | 4.685 to 4.694 | 5.493 | 6.186 to 6.291 | 6.006 | between baseline and d6 |
| mean prediction at k = 8 | 4.703 to 4.764 | 5.683 | 6.577 to 6.646 | 6.421 | between baseline and d6 |
| highest mean prediction at k >= 6 | 4.703 to 4.769 | 5.794 | 6.671 to 6.841 | 6.630 | between baseline and d6 |
| all-children top-1 | 0.645 to 0.800 | 0.793 | 0.948 to 0.964 | 0.940 | at or below the baseline range |

### Search: neural-w1, solved of 300 (kernel-accepted; exploratory, no rule)

| expansions | baseline range | h-d5-s0 | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|
| 30 | 124 to 180 | 180 | 265 to 284 | 255 |
| 100 | 152 to 202 | 212 | 287 to 297 | 281 |
| 300 | 176 to 222 | 234 | 292 to 300 | 294 |
| 1000 | 195 to 240 | 246 | 298 to 300 | 298 |

## test_depth_fresh, fresh sample (reported, not tested)

All 300 problems are labelled (1965 on-path states); 26 are unknown (child labelling above 300,000 nodes) and excluded, leaving 1939 scored states.

### Mean prediction by true distance k

A calibrated distance would equal k. Rows marked \* have fewer than 10 scored states.

| k | states | baseline mean | baseline range | h-d5-s0 | d6 mean | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.573 | 0.451 to 0.776 | 0.304 | -0.135 | -0.137 to -0.129 | 0.100 |
| 2 | 300 | 1.738 | 1.675 to 1.845 | 1.637 | 1.374 | 1.329 to 1.411 | 1.549 |
| 3 | 300 | 2.730 | 2.696 to 2.761 | 2.768 | 2.507 | 2.459 to 2.556 | 2.647 |
| 4 | 300 | 3.574 | 3.567 to 3.578 | 3.701 | 3.690 | 3.645 to 3.732 | 3.736 |
| 5 | 300 | 4.219 | 4.184 to 4.238 | 4.523 | 4.709 | 4.667 to 4.738 | 4.748 |
| 6 | 230 | 4.514 | 4.478 to 4.540 | 4.985 | 5.580 | 5.495 to 5.668 | 5.495 |
| 7 | 126 | 4.655 | 4.651 to 4.664 | 5.381 | 6.157 | 6.025 to 6.284 | 6.032 |
| 8 | 65 | 4.694 | 4.678 to 4.724 | 5.657 | 6.533 | 6.470 to 6.565 | 6.378 |
| 9 | 17 | 4.645 | 4.573 to 4.717 | 5.631 | 6.538 | 6.490 to 6.566 | 6.540 |
| 10 \* | 1 | 4.587 | 4.479 to 4.645 | 5.657 | 6.610 | 6.512 to 6.692 | 6.601 |

### The quantities of the rules

| quantity | baseline range (3 seeds) | h-d5-s0 | d6 range (3 seeds) | h-d6m-s0 | h-d5-s0 is |
|---:|---:|---:|---:|---:|---:|
| mean prediction at k = 7 | 4.651 to 4.664 | 5.381 | 6.025 to 6.284 | 6.032 | between baseline and d6 |
| mean prediction at k = 8 | 4.678 to 4.724 | 5.657 | 6.470 to 6.565 | 6.378 | between baseline and d6 |
| highest mean prediction at k >= 6 | 4.678 to 4.724 | 5.657 | 6.557 to 6.566 | 6.540 | between baseline and d6 |
| all-children top-1 | 0.613 to 0.803 | 0.790 | 0.949 to 0.971 | 0.936 | at or below the baseline range |

### Search: neural-w1, solved of 300 (kernel-accepted; exploratory, no rule)

| expansions | baseline range | h-d5-s0 | d6 range | h-d6m-s0 |
|---:|---:|---:|---:|---:|
| 30 | 114 to 171 | 181 | 261 to 281 | 248 |
| 100 | 144 to 210 | 208 | 279 to 289 | 272 |
| 300 | 173 to 226 | 221 | 287 to 294 | 283 |
| 1000 | 192 to 234 | 237 | 290 to 294 | 290 |

## Provenance

h-d5-s0 checkpoint sha256 `d881358164f2bd5a…`; label files `634430a1ff4a4f57…` (E2) and `dc512d8dbc72e37a…` (fresh).

## Limits

- One d5 seed. The baseline and d6 ranges are over three seeds; one run cannot be placed inside a seed spread.
- d5 is not size-matched: it has more training problems than the baseline and fewer than d6, so range and data volume both increase from baseline to d5 to d6. The d6 control (`d6_matched_summary.md`) separated the two for d6 only.
- Unknown states are excluded from the per-distance quantities and top-1; the deepest rows are small.
