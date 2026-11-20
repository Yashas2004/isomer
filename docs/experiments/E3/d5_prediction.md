# E3-d5: does the ceiling track the training range? Prediction, written before any d5 state exists

Date: 2026-11-20. This file is committed before any d5 training state is generated and before
`h-d5-s0` exists. The decision rules below are also constants in
`src/isomer/e3/d5_summary.py`, committed with this file.

## What is known

Two training ranges, both measured on the E2 test_depth sample (300 problems, 1,935 scored states;
`seeds_summary.md`, `d6_family_summary.md`):

| model group | train problems up to exact distance | training states up to | mean prediction at k = 7 | at k = 8 | all-children top-1 |
|---|---:|---:|---:|---:|---:|
| baseline h-v1 (3 seeds) | 4 | 5 | 4.685 to 4.694 | 4.703 to 4.764 | 0.645 to 0.800 |
| d6 (3 seeds) | 6 | 7 | 6.186 to 6.291 | 6.577 to 6.646 | 0.948 to 0.964 |

In both groups the mean prediction levels off *below* the largest training-state distance. At
k = 8 the baseline mean is 4.725 against a maximum of 5 (0.275 below), and the d6 mean is 6.602
against 7 (0.398 below). The size-matched control (`d6_matched_summary.md`) gave 6.006 at k = 7,
6.421 at k = 8 and top-1 0.940, so the d6 ceiling is not a data-volume effect.

## Hypothesis

The ceiling tracks the training range. A model trained on problems up to distance 5 (training
states up to 6, built by the same construction as d6 with a cap of 5) levels off between the
baseline and d6, near 6 minus 0.3 to 0.4, about 5.7.

## Training

- Data: `data/isobench-v2-d5`, built by `scripts/e3_build_d6.py --max-dist 5` with the checks of d6
  (replay of v2's train pool, eval-split identity with the E2 id files, no leakage). Its eval
  splits are v2's, byte for byte.
- States: `isomer states --offpath 6 --budget 3000` into `data/states-v2-d5`, as for d6.
- Model: the h-v1 configuration, seed 0, 20 epochs, trained by `scripts/train_seeds.py --seeds 0
  --states data/states-v2-d5 --prefix h-d5-s`, into `runs/h-d5-s0`. **One seed.**

## Measurement

The same scripts as for d6, on the same population: the 1,935 scored on-path states of the 300 E2
test_depth problems (`runs/e3` labels); states whose child labelling exceeded 300,000 nodes are
unknown and excluded, as everywhere in E3. The fresh sample (300 problems, 1,939 scored states,
`runs/e3-fresh`) is evaluated the same way and reported, **not tested**.

## Decision rules (judged on the E2 test_depth sample)

| | quantity of h-d5-s0 | supported if |
|---|---|---|
| H-d5a | mean prediction at k = 7 and at k = 8 | strictly between the highest baseline value (4.764) and the lowest d6 seed's value at that k: 4.764 < value < 6.186 at k = 7, and 4.764 < value < 6.577 at k = 8 |
| H-d5b | the highest mean prediction at any k >= 6 | in [5.2, 6.2] (inclusive); point prediction about 5.7 |
| H-d5c | all-children top-1 over all scored states | strictly between the best baseline seed and the lowest d6 seed: 0.800 < value < 0.948 |

- **Supported** if all three hold. Otherwise **not supported**, and the summary shows which rows
  failed. A row whose distance has fewer than 10 scored states is not decidable.
- 4.764 is the highest baseline-seed mean prediction at either k = 7 or k = 8 (h-v1 at k = 8); it is
  the lower bound at both distances. 0.800 and 0.948 are the best baseline seed and the lowest d6
  seed (h-v1-s1 and the lowest d6 seed in `d6_family_summary.md`).
- H-d5b ranges over k >= 6 with at least 10 scored states, as the plateau rule of `d6_prediction.md`
  did: on this sample k = 6, 7, 8 and 9 (230, 124, 64 and 15 states); k = 10 has 2 and is left out.
- Values are compared at full precision with the constants above, which are the committed
  three-decimal values.

## Expectation

All three supported, with a mean prediction of about 5.7 at its highest, 0.3 below the largest
training-state distance (6), as at the other two ranges.

## Reported, not tested

Mean prediction by true distance for baseline, h-d5-s0, d6 and the control side by side, solve
rates at 30, 100, 300 and 1,000 expansions, and the same quantities on the fresh sample with
h-d5-s0 placed against the fresh baseline and d6 ranges.

## Limits, stated in advance

- **One d5 seed.** The baseline seeds alone span 0.155 in top-1 (0.645 to 0.800), so one seed can
  fall outside the H-d5c interval for seed reasons, in either direction.
- **Range and volume rise together.** d5 has more training problems than the baseline and fewer than
  d6. The size-matched control separated depth from volume for d6 only; d5 is not size-matched.
- **Thresholds chosen with known neighbours.** The baseline and d6 results were in view when the
  rules were written; nothing about d5 was.
- **Selected deep states.** As for d6, `isomer states` drops off-path states whose label exceeds
  its budget of 3,000 expansions, so the deepest training states are a selected tail. The share
  dropped will be reported from `stats.json`; it is not predicted here.
