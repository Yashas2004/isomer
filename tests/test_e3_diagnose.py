"""E3 ranking metrics on hand-computed examples.

These functions produce every number in sections 7.1-7.6 of the E3 report, so
their conventions (lower score = preferred, ties split, average rank) are
pinned here. No model and no torch.
"""

import math

import pytest

from isomer.e3.diagnose import best_opt_rank, bootstrap_mean_ci, margin, top1, wilson

# ---------------------------------------------------------------------------------- top1


def test_top1_unique_minimum():
    assert top1([0.2, 1.0, 3.0], [True, False, False]) == 1.0
    assert top1([0.2, 1.0, 3.0], [False, True, True]) == 0.0


def test_top1_splits_ties():
    # two children share the minimum, one of them optimal: half credit
    assert top1([1.0, 1.0, 2.0], [True, False, False]) == 0.5
    # three-way tie with two optimal
    assert top1([4.0, 4.0, 4.0, 9.0], [True, True, False, True]) == pytest.approx(2 / 3)
    # a constant heuristic earns exactly the base rate
    assert top1([0.0] * 5, [True, False, False, False, False]) == pytest.approx(0.2)


def test_top1_single_child_and_negative_scores():
    assert top1([-1.1], [True]) == 1.0
    assert top1([-1.1, -0.3], [False, True]) == 0.0


# ---------------------------------------------------------------------------------- best_opt_rank


def test_best_opt_rank_counts_strictly_better_children():
    assert best_opt_rank([0.1, 0.5, 0.9], [True, False, False]) == 1.0
    assert best_opt_rank([0.5, 0.1, 0.9], [True, False, False]) == 2.0
    assert best_opt_rank([0.9, 0.1, 0.5], [True, False, False]) == 3.0


def test_best_opt_rank_uses_the_best_optimal_child():
    # optimal children score 2.0 and 5.0; only the 2.0 one matters, one child beats it
    assert best_opt_rank([1.0, 2.0, 5.0, 3.0], [False, True, True, False]) == 2.0


def test_best_opt_rank_averages_ties():
    # scores 3,1,2,2: one child below the best optimal (2), which shares ranks 2 and 3
    assert best_opt_rank([3.0, 1.0, 2.0, 2.0], [False, False, True, True]) == 2.5
    # tie with a non-optimal child counts the same way
    assert best_opt_rank([2.0, 2.0], [True, False]) == 1.5
    # all tied: the middle rank
    assert best_opt_rank([7.0] * 5, [False, False, True, False, False]) == 3.0


# ---------------------------------------------------------------------------------- margin


def test_margin_is_negative_when_the_right_move_is_preferred():
    assert margin([1.0, 3.0, 2.0], [True, False, False]) == -1.0
    assert margin([3.0, 1.0, 2.0], [True, False, False]) == 2.0


def test_margin_uses_the_minimum_of_each_side():
    # min optimal = 0.5, min non-optimal = 0.75
    assert margin([0.5, 4.0, 0.75, 6.0], [True, True, False, False]) == -0.25
    assert margin([2.0, 2.0], [True, False]) == 0.0


def test_margin_undefined_when_every_child_is_optimal():
    assert margin([1.0, 2.0], [True, True]) is None
    assert margin([1.0], [True]) is None


# ---------------------------------------------------------------------------------- wilson


def test_wilson_half():
    # p = 0.5, n = 10, z = 1.96: centre 0.5, half-width 1.96*sqrt(0.025 + 0.009604)/1.38416 = 0.26341
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-5)
    assert hi == pytest.approx(0.76341, abs=1e-5)


def test_wilson_zero_and_all_successes_stay_inside_the_unit_interval():
    # k = 0: centre and half-width are both (z^2 / 2n) / (1 + z^2 / n) = 0.19208 / 1.38416
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    # so the upper end is (z^2 / n) / (1 + z^2 / n) = 0.38416 / 1.38416
    assert hi == pytest.approx(0.27754, abs=1e-5)
    lo, hi = wilson(10, 10)
    assert lo == pytest.approx(1 - 0.27754, abs=1e-5)
    assert hi == pytest.approx(1.0, abs=1e-12)


def test_wilson_matches_the_published_interval():
    # E3 report, section 3: neural-w1 at 30 expansions, 139/300 = 46.3% (Wilson 95% 40.8-52.0)
    lo, hi = wilson(139, 300)
    assert round(100 * lo, 1) == 40.8
    assert round(100 * hi, 1) == 52.0


def test_wilson_accepts_fractional_counts_and_narrows_with_n():
    # tie-split top-1 sums are not integers
    lo, hi = wilson(2.5, 5)
    assert (lo + hi) / 2 == pytest.approx(0.5)
    lo2, hi2 = wilson(250, 500)
    assert hi2 - lo2 < hi - lo


def test_wilson_z_and_empty():
    assert wilson(3, 10, z=0.0) == (0.3, 0.3)
    assert all(math.isnan(v) for v in wilson(0, 0))


# ---------------------------------------------------------------------------------- bootstrap


def test_bootstrap_mean_ci_is_seeded_and_resamples_whole_problems():
    values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    groups = ["p", "p", "p", "q", "q", "q"]
    lo, hi = bootstrap_mean_ci(values, groups, n_boot=200)
    # with two clusters a resample is all-p, all-q or one of each: the mean is 0, 0.5 or 1
    assert (lo, hi) == (0.0, 1.0)
    assert bootstrap_mean_ci(values, groups, n_boot=200) == (lo, hi)
    assert bootstrap_mean_ci([0.25, 0.25], ["a", "b"]) == (0.25, 0.25)
    assert all(math.isnan(v) for v in bootstrap_mean_ci([], []))
