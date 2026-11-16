# E3-d6 size-matched control: prediction, written before the run

Date: 2026-11-16. This file is committed before the control model
`h-d6m-s0` and the further d6 seeds (`h-d6-s1`, `h-d6-s2`) are trained. The
decision rules below are also constants in `src/isomer/e3/seeds_compare.py`
(`MATCHED_RULES`), committed with this file.

## What is known

`h-d6-s0` (h-v1 configuration, trained on `data/states-v2-d6`) met every
quantity of `d6_prediction.md` (`d6_summary.md`), and it is far better than
the three baseline seeds at every distance, including distances 1-4 that the
baseline was trained on:

| quantity (test_depth, 1,935 scored states) | baseline range, 3 seeds | h-d6-s0 |
|---|---|---|
| mean prediction at k = 7 | 4.685 to 4.694 | 6.234 |
| mean prediction at k = 8 | 4.703 to 4.764 | 6.646 |
| drop to the optimal child at k = 7 | 0.068 to 0.139 | 0.578 |
| drop to the optimal child at k = 8 | 0.056 to 0.076 | 0.506 |
| all-children top-1, all scored states | 0.645 to 0.800 | 0.953 |
| solved at 1,000 expansions (of 300) | 195 to 240 | 298 |

Exact train/test overlap does not explain it (`overlap_summary.md`: 5 of 1,935
scored states, one problem of 300).

But d6 differs from the baseline in two ways at once. It has deeper train
problems (exact distance up to 6 instead of 4), and it has more of them:
36,045 train problems and 792,841 states against 29,268 and 551,072, and so
23% more optimizer steps at the same 20 epochs. `d6_prediction.md` named this
in advance as uncontrolled.

## Hypothesis

The d6 gain is due to depth coverage, not data volume.

## Control

`h-d6m-s0`: the h-v1 configuration, seed 0, trained on
`data/states-v2-d6-matched`. That directory is a seeded random subset
(seed 0, `isomer.e3.matched`) of the d6 train problems with exactly as many
train problems as `data/states-v2`, every state of each kept problem, and the
d6 validation file unchanged. It already exists; its `stats.json` gives:

| | v2 (baseline) | d6 | d6-matched (control) |
|---|---:|---:|---:|
| train problems | 29,268 | 36,045 | 29,268 |
| optimizer steps, 20 epochs | 9,160 | 11,280 | 9,160 |
| train states | 551,072 | 792,841 | 642,983 |
| of which at distance 5 | 29,877 | 73,381 | 59,255 |
| of which at distance 6 | 0 | 26,768 | 21,568 |
| of which at distance 7 | 0 | 3,728 | 2,939 |
| train problems of distance 5 and 6 | 0 | not recorded | 3,772 and 1,704 |

The control is matched to the baseline in train problems and optimizer steps.
It is not matched in training states (17% more than the baseline), because
deeper problems have more states each; and it has fewer shallow problems than
the baseline (23,792 of distance 1-4 against 29,268).

## Measurement

As for `d6_prediction.md`: the same 1,935 scored on-path states of the 300 E2
test_depth problems, the same scripts. The test is at distances 7 and 8
(124 and 64 scored states), plus all-children top-1 over all scored states.

## Decision rules

| quantity (h-d6m-s0, test_depth) | baseline range | h-d6-s0 | "depth" if | "volume" if |
|---|---|---|---|---|
| mean prediction at k = 7 | 4.685 to 4.694 | 6.234 | >= 5.7 | <= 5.0 |
| mean prediction at k = 8 | 4.703 to 4.764 | 6.646 | >= 5.7 | <= 5.0 |
| drop to the optimal child at k = 7 | 0.068 to 0.139 | 0.578 | >= 0.35 | <= 0.20 |
| drop to the optimal child at k = 8 | 0.056 to 0.076 | 0.506 | >= 0.18 | <= 0.10 |
| all-children top-1, all scored states | 0.645 to 0.800 | 0.953 | >= 0.90 | <= 0.85 |

- The first four rows use the thresholds of `d6_prediction.md` unchanged:
  "depth" is its "supported" threshold, "volume" its "refuted" threshold. Every
  "depth" threshold is well above the highest baseline seed.
- The top-1 thresholds split the gap between the best baseline seed (0.800)
  and h-d6-s0 (0.953) in thirds: "depth" if the control keeps at least two
  thirds of that gain (0.902, rounded to 0.90), "volume" if it keeps at most
  one third (0.851, rounded to 0.85).
- A value between the two thresholds is "between".
- Overall: **depth** if all five rows say depth; **volume** if all five say
  volume; otherwise **mixed**, with the rows shown.

## Expectation

Depth, on all five rows. The mean-prediction rows are close to forced: a model
trained on labels up to 6-7 can output them, whatever the amount of data. The
informative rows are the drop at k = 7-8 and above all top-1, where the gain
extends to distances 1-4 and where data volume is a live alternative.

## The further d6 seeds

`h-d6-s1` and `h-d6-s2` (same data as h-d6-s0, seeds 1 and 2) are trained in
the same batch. They are each evaluated against the unchanged rules of
`d6_prediction.md`, and they give a d6 seed range. The baseline seeds differ by
0.155 in top-1 (0.645 to 0.800), so the control is read against that range as
well as against the fixed thresholds: a control inside the d6 seed range is
the cleanest "depth" outcome; a control that passes the thresholds but lies
below every d6 seed will be reported as such.

## Limits, stated in advance

- One control model: one seed and one random subset. It cannot distinguish a
  subset effect from a seed effect.
- States are not matched (see the table). If the control lands between the
  baseline and d6, the 17% extra states are one possible reason.
- The control does not separate the deeper problems from the thin, selected
  tail of distance-7 off-path states that comes with them (amendment in
  `d6_prediction.md`).
- The thresholds were chosen with the baseline and h-d6-s0 results in view.
  Nothing about h-d6m-s0, h-d6-s1 or h-d6-s2 was known when they were written.
- Search results (solve rates) are reported as exploratory, with no rule.
