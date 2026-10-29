"""Sibling groups, batching and move metrics. numpy only: runs without torch."""

import json

import numpy as np
import pytest

from isomer.core.term import parse
from isomer.learn import data as D
from isomer.learn.metrics import heuristic_scores, set_metrics, top1_metrics
from isomer.search.heuristics import size_diff, symbol_bag


def rec(pid, state, goal, dist, on_path, parent_dist=None):
    return {"problem_id": pid, "theory": "group", "split": "train", "state": state, "goal": goal, "dist": dist,
            "on_path": on_path, "parent_dist": parent_dist}


# P1: proof (* e (* e a)) -> (* e a) -> a; off-path children of the dist-2 parent
# at dist 2 and 1 (the latter equally optimal); the dist-1 parent has no siblings.
# P2: proof (* a e) -> a; one off-path child of the dist-1 parent at dist 2.
HAND = [
    rec("P1", "(* e (* e a))", "a", 2, True), rec("P1", "(* e a)", "a", 1, True), rec("P1", "a", "a", 0, True),
    rec("P1", "(* (* e e) a)", "a", 1, False, 2), rec("P1", "(* e (* e (* e a)))", "a", 3, False, 2),
    rec("P2", "(* a e)", "a", 1, True), rec("P2", "a", "a", 0, True), rec("P2", "(* (* a e) e)", "a", 2, False, 1),
]


@pytest.fixture
def hand_file(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = sorted(HAND, key=lambda r: (r["problem_id"], r["state"]))
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def test_derive_groups_on_hand_made_problem():
    p1 = [r for r in HAND if r["problem_id"] == "P1"]
    groups, singletons = D.derive_groups([r["dist"] for r in p1], [r["on_path"] for r in p1],
                                         [r["parent_dist"] for r in p1], base=10)
    # parent dist 2: on-path child (* e a) at index 11 first, then off-path 13, 14; parent dist 1: singleton
    assert groups == [[11, 13, 14]] and singletons == 1


def test_derive_groups_rejects_bad_structure():
    with pytest.raises(AssertionError, match="two on-path"):
        D.derive_groups([0, 1, 1], [True, True, True], [None, None, None])
    with pytest.raises(AssertionError, match="parent"):
        D.derive_groups([0, 1, 2], [True, True, False], [None, None, 5])


def test_load_states_groups_and_optimal(hand_file):
    ss = D.load_states(hand_file, max_len=64, keep_text=True)
    assert ss.problem_ids == ["P1", "P2"] and ss.n_states == 8 and ss.singleton_groups == 1
    assert len(ss.groups) == 2
    for g in ss.groups:
        members = [i for i in g if i >= 0]
        assert ss.on_path[members[0]] and sum(ss.on_path[m] for m in members) == 1
    named = [[(ss.states[i], int(ss.dist[i])) for i in g if i >= 0] for g in ss.groups]
    assert named[0] == [("(* e a)", 1), ("(* (* e e) a)", 1), ("(* e (* e (* e a)))", 3)]
    assert named[1] == [("a", 0), ("(* (* a e) e)", 2)]
    opt = D.optimal_mask(ss.groups, ss.dist)
    assert opt[0].tolist()[:3] == [True, True, False] and opt[1].tolist()[:2] == [True, False]
    assert not opt[1, 2:].any()  # padding is never optimal


def test_optimal_mask_with_padding_and_int16_dist():
    """Regression: the padding sentinel once overflowed int16 and became the group minimum."""
    groups = np.array([[0, 1, -1, -1], [2, 3, 4, -1]])
    dist = np.array([1, 2, 3, 3, 4], dtype=np.int16)
    assert D.optimal_mask(groups, dist).tolist() == [[True, False, False, False], [True, True, False, False]]


def test_batches_cover_problems_and_are_seeded():
    a = [b.tolist() for b in D.epoch_batches(10, 3, seed=0, epoch=0)]
    assert a == [b.tolist() for b in D.epoch_batches(10, 3, seed=0, epoch=0)]
    assert a != [b.tolist() for b in D.epoch_batches(10, 3, seed=0, epoch=1)]
    assert sorted(x for b in a for x in b) == list(range(10)) and [len(b) for b in a] == [3, 3, 3, 1]


def test_micro_batches_keep_problems_whole(hand_file):
    ss = D.load_states(hand_file, max_len=64)
    whole = D.collate(ss, np.array([0, 1]))
    assert whole.ids.shape == (8, int(ss.lengths.max())) and len(whole.groups) == 2
    parts = D.micro_batches(ss, np.array([0, 1]), micro_tokens=1)  # tiny budget: one problem each
    assert [p.tolist() for p in parts] == [[0], [1]]
    p1 = D.collate(ss, parts[0])
    # group rows index into the micro-batch and point at the same states as the global groups
    for g_local, g_global in zip(p1.groups, ss.groups[:1], strict=True):
        assert [p1.index[i] for i in g_local if i >= 0] == [i for i in g_global if i >= 0]
    big = D.micro_batches(ss, np.array([1, 0]), micro_tokens=10**9)
    assert [p.tolist() for p in big] == [[1, 0]]
    m = D.collate(ss, big[0])
    for g in m.groups:   # local rows must be the right states after reordering
        assert m.dist[g[0]] == min(m.dist[i] for i in g if i >= 0)


def test_top1_metrics_fractional_ties():
    groups = np.array([[0, 1, 2, -1], [3, 4, 5, 6], [7, 8, -1, -1]])
    optimal = np.array([[True, False, False, False], [True, True, False, False], [True, False, False, False]])
    scores = np.array([1.0, 1.0, 5.0,      # 2-way tie at the minimum, 1 optimal -> 1/2
                       3.0, 3.0, 3.0, 3.0,  # 4-way tie, 2 optimal -> 1/2
                       0.0, 2.0])           # strict win -> 1
    m = top1_metrics(scores, groups, optimal)
    assert m["top1_optimal"] == pytest.approx((0.5 + 0.5 + 1) / 3)
    assert m["top1_optimal_strict"] == pytest.approx(1 / 3)
    assert m["top1_onpath"] == pytest.approx((0.5 + 0.25 + 1) / 3)
    assert m["top1_onpath_strict"] == pytest.approx(1 / 3)
    assert m["groups"] == 3


def test_baseline_top1_matches_direct_loop(hand_file):
    ss = D.load_states(hand_file, max_len=64, keep_text=True)
    for h in (symbol_bag, size_diff):
        total = 0.0
        for g in ss.groups:
            members = [i for i in g if i >= 0]
            s = [h(parse(ss.states[i]), parse(ss.goals[i])) for i in members]
            d = [int(ss.dist[i]) for i in members]
            tied = [i for i, v in enumerate(s) if v == min(s)]
            total += sum(d[i] == min(d) for i in tied) / len(tied)
        assert set_metrics(ss, heuristic_scores(ss, h))["top1_optimal"] == pytest.approx(total / len(ss.groups))


def test_too_long_pair_raises(hand_file):
    with pytest.raises(ValueError, match="max_len"):
        D.load_states(hand_file, max_len=8)
