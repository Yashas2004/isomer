"""E3 analysis of scored states on a tiny hand-computed fixture.

Rows have the shape ``diagnose score`` writes; only the fields each function
reads are filled in. No model and no torch.
"""

import json

import pytest

from isomer.e3.analyze import choice_classes, edge_consistency, invariance


def _row(pid, dist, h_state, h_children, optimal, child_dist=None, theory="group", i=0):
    return {"problem_id": pid, "i": i, "theory": theory, "dist": dist, "h_state": h_state,
            "h_children": h_children, "optimal": optimal, "child_dist": child_dist,
            "n_children": len(h_children)}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------------- choice_classes

CHOICE_ROWS = [
    _row("a", 3, 0.0, [0.1, 0.5, 0.9], [True, False, False], [2, 3, 4]),    # picks the optimal child
    _row("b", 3, 0.0, [0.5, 0.2, 0.9], [True, False, False], [2, 3, 4]),    # picks a same-distance child
    _row("c", 2, 0.0, [0.3, 0.3, 1.0], [False, False, True], [3, None, 1]),  # tie: away / unresolved
    _row("d", 2, 0.0, [1.0, 0.2], [True, False], [1, 3]),                   # picks a child one step further
]


def test_choice_classes_splits_ties_and_normalises_by_states():
    got = choice_classes(CHOICE_ROWS, "h_children")
    assert got == {"away": 0.375, "non_optimal_unresolved": 0.125, "optimal": 0.25, "sideways": 0.25}
    assert sum(got.values()) == pytest.approx(1.0)
    assert list(got) == sorted(got)       # stable key order in the JSON output


def test_choice_classes_reads_the_requested_heuristic():
    rows = [dict(r, sb_children=[0.0] + [1.0] * (r["n_children"] - 1)) for r in CHOICE_ROWS]
    # symbol_bag always prefers child 0: optimal in a, b, d; away (distance 3 from a state at 2) in c
    assert choice_classes(rows, "sb_children") == {"away": 0.25, "optimal": 0.75}


def test_choice_classes_optimal_wins_even_without_a_child_distance():
    # an optimal child is classified by its flag; child_dist may be missing for it
    rows = [_row("a", 4, 0.0, [0.0, 1.0], [True, False], [None, None])]
    assert choice_classes(rows, "h_children") == {"optimal": 1.0}


# ---------------------------------------------------------------------------------- edge_consistency

def test_edge_consistency_hand_computed():
    rows = [
        _row("a", 3, 3.0, [2.0, 4.5, 3.25], [True, False, False]),   # |diff| = 1.0, 1.5, 0.25
        _row("b", 1, 1.0, [-0.5, 1.0], [True, False]),               # |diff| = 1.5, 0.0
    ]
    got = edge_consistency(rows)
    assert got["edges"] == 5
    assert got["violation_rate"] == pytest.approx(2 / 5)          # a difference of exactly 1 is consistent
    assert got["violation_rate_gt_1.5"] == 0.0                    # strict: 1.5 is not counted
    assert got["mean_squared_excess"] == pytest.approx((0.25 + 0.25) / 5)
    assert got["mean_abs_diff"] == pytest.approx((1.0 + 1.5 + 0.25 + 1.5 + 0.0) / 5)
    # optimal children only: drops of 1.0 and 1.5
    assert got["optimal_child_mean_drop"] == pytest.approx(1.25)
    assert got["optimal_child_drop_sd"] == pytest.approx(0.25)


def test_edge_consistency_of_a_true_distance_has_no_violation():
    rows = [_row("a", 3, 3.0, [2.0, 3.0, 4.0], [True, False, False]),
            _row("a", 2, 2.0, [1.0, 3.0], [True, False], i=1)]
    got = edge_consistency(rows)
    assert got["violation_rate"] == 0.0
    assert got["mean_squared_excess"] == 0.0
    assert got["optimal_child_mean_drop"] == 1.0
    assert got["optimal_child_drop_sd"] == 0.0


def test_edge_consistency_drop_is_signed():
    # the optimal child is scored ABOVE its parent: a negative drop, and a violation
    got = edge_consistency([_row("a", 2, 1.0, [3.0], [True])])
    assert got["optimal_child_mean_drop"] == -2.0
    assert got["violation_rate"] == 1.0
    assert got["mean_squared_excess"] == 1.0


# ---------------------------------------------------------------------------------- invariance

def test_invariance_pairs_states_and_reports_per_theory(tmp_path):
    base = [
        _row("p1", 2, 2.0, [1.0, 3.0], [True, False]),
        _row("p2", 1, 1.0, [0.0, 2.0], [True, False]),
        _row("q1", 2, 2.0, [1.5, 0.5], [False, True], theory="bool"),
        _row("only-in-base", 2, 2.0, [1.0, 3.0], [True, False]),
    ]
    renamed = [
        _row("p1", 2, 2.5, [3.0, 1.0], [True, False]),          # choice flips to the wrong child
        _row("p2", 1, 1.0, [0.0, 2.0], [True, False]),          # unchanged
        _row("q1", 2, 2.0, [1.5, 0.5], [False, True], theory="bool"),   # no mapping for this theory
    ]
    got = invariance(_write(tmp_path / "base.jsonl", base), _write(tmp_path / "renamed.jsonl", renamed))

    assert sorted(got) == ["bool", "group"]
    # a theory whose scores are all identical was not renamed: it must not dilute the comparison
    assert got["bool"] == {"states": 1, "renaming_applies": False}

    g = got["group"]
    assert (g["states"], g["problems"], g["scored_terms"]) == (2, 2, 6)   # the unpaired base row is dropped
    # |delta h| over state + children: p1 -> 0.5, 2, 2; p2 -> 0, 0, 0
    assert g["mean_abs_delta_h"] == pytest.approx(4.5 / 6)
    assert g["frac_delta_h_gt_0.5"] == pytest.approx(2 / 6)
    # before (2,1,3,1,0,2) vs after (2.5,3,1,1,0,2): r = 1.75 / sqrt(5.5 * 6.20833)
    assert g["pearson_h"] == pytest.approx(0.29948, abs=1e-5)
    assert g["top1_before"] == 1.0
    assert g["top1_after"] == 0.5
    assert g["top1_diff"] == -0.5
    lo, hi = g["top1_diff_ci95"]
    assert -1.0 <= lo <= hi <= 0.0
    assert g["argmin_changed"] == 0.5
    assert g["mean_h_state_before"] == 1.5
    assert g["mean_h_state_after"] == 1.75


def test_invariance_of_identical_scores_applies_nowhere(tmp_path):
    rows = [_row("p1", 2, 2.0, [1.0, 3.0], [True, False])]
    got = invariance(_write(tmp_path / "a.jsonl", rows), _write(tmp_path / "b.jsonl", rows))
    assert got == {"group": {"states": 1, "renaming_applies": False}}
