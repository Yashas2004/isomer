# E3-d6: prediction, written before the run

Date: 2026-11-16. This file is committed before any d6 training state is
generated and before any d6 model exists. The decision rules below are also
constants in `src/isomer/e3/seeds_compare.py` (`PREDICTIONS`, `PLATEAU_BELOW`),
committed with this file.

## Question

All three seeds of the h-v1 configuration saturate just below 5, the largest
exact distance of a training state (`seeds_summary.md`): on test_depth the mean
prediction is 4.69, 4.73, 4.70 at true distance 7, 8, 9, and the predicted drop
from a state to its optimal child falls from 1.58 at distance 1 to 0.11 at
distance 7. Is the training range the cause?

## Intervention

One model, `h-d6-s0`: the h-v1 configuration (`docs/results/train-h-v1/config.json`),
seed 0, trained on `data/states-v2-d6`. That is `isomer states --offpath 6` on
`data/isobench-v2-d6`, which is IsoBench v2 with the train pool kept up to exact
distance 6 instead of 4 (training states up to 7 instead of 5). test_depth,
test_deep and test_theory are the v2 files, unchanged.

## Measurement

The diagnostics of `seeds_summary.md`, on the same 1,935 scored on-path states
of the 300 E2 test_depth problems (`runs/e3/labels/test_depth.jsonl`, 30 unknown
states excluded), with the same scripts. Baseline: h-v1, h-v1-s1, h-v1-s2.

Distances 5 and 6 are now in-distribution by distance for the new model, so
they are reported but are not a test. The test is at distances 7 and 8
(124 and 64 scored states). Distance 9 has 15 scored states and distance 10
has 2: reported, not used.

## Prediction

If the training range causes the saturation, the baseline curve should shift
by two distances: the mean prediction should saturate between 6 and 7 instead
of at 4.8, and the drop to the optimal child should stay near 1 for one or two
distances longer. The point predictions are the baseline means at distances 5
and 6 (mean prediction 4.24, 4.55; drop 0.59, 0.31), moved to distances 7 and 8.

| quantity (h-d6-s0, test_depth) | baseline range, 3 seeds | predicted | supported if | refuted if |
|---|---|---|---|---|
| mean prediction at k = 7 | 4.685 to 4.694 | about 6.2 | >= 5.7 | <= 5.0 |
| mean prediction at k = 8 | 4.703 to 4.764 | about 6.5 | >= 5.7 | <= 5.0 |
| drop to the optimal child at k = 7 | 0.068 to 0.139 | about 0.6 | >= 0.35 | <= 0.20 |
| drop to the optimal child at k = 8 | 0.056 to 0.076 | about 0.3 | >= 0.18 | <= 0.10 |
| highest mean prediction at k >= 7 (k with >= 10 states) | below 5 | between 6 and 7 | below 7 | — |

A value between the two thresholds is reported as "partial". The overall
verdict is "supported" only if all four rows are supported and the plateau
stays below 7; "refuted" only if all four are refuted; otherwise "mixed", with
the rows shown.

## Not predicted (reported as exploratory)

All-children top-1, margin and edge-violation rate at distances 7-8; solve
rate by budget and by problem distance; the test_id diagnostics.

## Known limits, stated in advance

- The d6 model sees more data, not only deeper data (about 23% more train
  problems, about 45% more states, more optimizer steps at the same 20 epochs).
  No size-matched control is run. A gain confined to distances <= 4 would point
  to data volume, not range.
- One seed of the new model against a three-seed baseline. The baseline seeds
  differ by at most 0.01 (k = 7) and 0.06 (k = 8) in mean prediction and by at
  most 0.07 in drop; the "supported" thresholds are far outside that spread.
- `isomer states` drops off-path states whose BFS label exceeds 3,000
  expansions. More will be dropped at distance 6-7 than in v2 (which dropped 7
  in total); the count will be reported.
- Train-pool problems of distance 5-6 come from walks of length at most 6;
  test_depth problems come from walks of 6-24.
- The thresholds were chosen with the baseline results in view. Nothing about
  the d6 model was known when they were written.

## Amendment before step B (2026-11-16)

Written after `data/isobench-v2-d6` was built (train 37,962 problems; all three
build checks passed) and after a pilot of the state generation, and before the
full training states were generated or any d6 model was trained. No threshold
and no point prediction above is changed.

Pilot: `isomer states --data data/isobench-v2-d6 --offpath 6 --limit 600`
(budget 3,000, the first 600 train problems in id order, output in a scratch
directory). Its mix of problem distances matches the full train split
(distance 5-6: 18.7% of the pilot, 18.8% of the split).

| parent distance | off-path states attempted | dropped (BFS budget) | share |
|---:|---:|---:|---:|
| 1 to 4 | 10,051 | 0 | 0% |
| 5 | 516 | 8 | 1.6% |
| 6 | 172 | 68 | 39.5% |

- About 40% of the off-path successors of distance-6 states exceed the
  3,000-expansion labelling budget and are dropped. The pilot dropped 76
  states in total; scaled to the full split that is about 4,400.
- The pilot kept 65 states at distance 7 and 449 at distance 6 out of 13,157.
  Scaled to the full split: about 3,700 states at distance 7, under 0.5% of
  about 830,000, and about 28,000 at distance 6.
- The distance-7 states that survive are those cheap enough to label within
  the budget, so they are a selected, low-branching sample.

Consequence: the effective training range of `h-d6-s0` is up to distance 6,
with a thin, selected tail at 7, and not "up to 7" as the Intervention section
says. A plateau near 6 is therefore the expected reading, at the lower end of
the "between 6 and 7" stated above.

Step B is run as planned, with budget 3,000 (the v2 setting), so that the
distance cut-off stays the only change from h-v1's training data. These are
pilot extrapolations; the actual counts will be taken from
`data/states-v2-d6/stats.json` and reported in the summary.
