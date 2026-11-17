# E3-fresh: confirmatory evaluation, prediction written before the run

Date: 2026-11-17. This file is committed before the fresh evaluation pool is
built, and so before any model is run on it. The decision rules are also
constants in `src/isomer/e3/fresh_summary.py`, committed with this file; a
test ties the two together. The sample's id file and sha256 will be appended
below, in a dated section, once the pool is built and before any model sees it.

## Why

Every E3 result so far was measured on the E2 test_depth sample, which was in
view while the analyses and thresholds were chosen (REPORT.md section 13).
This experiment re-tests the main claims on 300 problems that did not exist
when the claims were formed.

## Sample

- **Pool:** `data/isobench-fresh/test_depth_fresh.jsonl`, built by
  `scripts/e3_build_fresh.py` (`isomer.e3.fresh`). Same generator and labeller
  as the v2 eval pool (theories group and bool, 1,200 problems each, walk
  lengths 6-24, labelling budget 20,000 expansions and 1,500,000 stored
  nodes), seed 1 instead of 0. Problems with exact distance > 4 are kept.
- **Disjointness:** every id of every split of `data/isobench-v2` and
  `data/isobench-v2-d6` (which includes the d6 train set) is excluded at
  generation, and the build checks the written ids against those files
  again; it aborts on any collision.
- **Sample:** 300 problems with exactly the per-distance counts of
  `docs/experiments/E2/ids-test_depth.txt` (distance 5: 70, 6: 97, 7: 58,
  8: 52, 9: 19, 10: 4), ranked within a distance by
  `sha256(1|test_depth_fresh|id)`. If a distance has too few problems the
  build fails; the counts are not changed.

## Models

| group | models | trained on |
|---|---|---|
| baseline | h-v1, h-v1-s1, h-v1-s2 | `data/states-v2` (train distance <= 4) |
| d6 | h-d6-s0, h-d6-s1, h-d6-s2 | `data/states-v2-d6` (train distance <= 6) |
| control | h-d6m-s0 | `data/states-v2-d6-matched` |

All checkpoints are the existing `runs/<model>/best.pt`; nothing is retrained.

## Measurement

The same scripts as for the E2 sample, pointed at the fresh sample: every
legal child of every on-path state is labelled (`scripts/e3_label.py`), every
model scores those states (`scripts/e3_score_seeds.py`, cuda, bf16-fast),
and neural-w1 search runs with each model at 30, 100, 300 and 1,000
expansions (`scripts/e3_bench_seeds.py`; solved means kernel-accepted).

**Unknown states:** a state whose child labelling exceeds the cap of 300,000
nodes is unknown and excluded from H1, H2 and H4, as everywhere in E3. Their
number, by distance, is reported. H3 uses all 300 problems.

## Hypotheses and decision rules

| | claim | quantity | supported if |
|---|---|---|---|
| H1 | baseline saturates below its training range; d6 does not | mean prediction at k = 7 and k = 8 | every baseline seed < 5.0 and every d6 seed >= 5.7, at both distances (12 conditions) |
| H2 | d6 chooses moves better | all-children top-1 (ties split) over all scored states | every d6 seed > the best baseline seed |
| H3 | d6 searches better | neural-w1 solved at 30 and at 1,000 expansions | every d6 seed solves more than the best baseline seed at both budgets, and every Holm-adjusted p < 0.05 |
| H4 | the d6 gain is depth coverage, not data volume | the five rules of `d6_matched_prediction.md`, unchanged | h-d6m-s0 is "depth" on all five |

- **Supported** if every condition holds; **not supported** if any fails;
  **not decidable** if none fails but one cannot be evaluated.
- **H1:** a distance with fewer than 10 scored states cannot be evaluated
  (not decidable, not passed).
- **H2, H3:** the best baseline seed is the one with the highest value on the
  fresh sample (top-1 for H2; solved count at that budget for H3).
- **H3 tests:** exact two-sided McNemar test (`bench.report.mcnemar_exact`)
  for every (d6 seed, baseline seed) pair at each of the two budgets:
  3 x 3 x 2 = 18 tests, Holm-corrected as one family
  (`bench.report.holm`). A test passes if its adjusted p < 0.05 and the d6
  seed solves more of the discordant problems.
- **H4** thresholds, copied from `d6_matched_prediction.md`: mean prediction
  at k = 7 and 8 "depth" if >= 5.7, "volume" if <= 5.0; drop to the optimal
  child at k = 7 "depth" if >= 0.35, "volume" if <= 0.20; at k = 8 "depth" if
  >= 0.18, "volume" if <= 0.10; all-children top-1 "depth" if >= 0.90,
  "volume" if <= 0.85.

## Expectation

All four supported. On the E2 sample (`d6_family_summary.md`): baseline top-1
0.645 to 0.800, d6 seeds 0.948 to 0.964, control 0.940; baseline mean
prediction at k = 7 about 4.69, d6 seeds 6.186 to 6.291. The likeliest
failure is H3 at 30 expansions, where few problems may be solved by any model
and McNemar has little power after the Holm correction.

## Reported, not tested

Per-distance tables (mean prediction, top-1, drop to the optimal child), solve
rates at 100 and 300 expansions, solve counts by problem distance,
bfs-bidir and bestfirst-symbol_bag-w1 on the same problems, and the exact
(theory, state, goal) overlap of the fresh states with `data/states-v2` and
`data/states-v2-d6`.

## Limits, stated in advance

- Same generator, theories and action space: this confirms the results on new
  problems, not on a new distribution.
- Excluding ids that collide with v2 or d6 removes some problems a pure
  re-draw would contain; the number is recorded in the datasheet.
- The H1 and H4 thresholds were chosen with E2-sample results in view; what is
  new is the problems, not the rules.
- Three seeds per group; the control is one seed.
