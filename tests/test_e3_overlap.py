"""E3 overlap between evaluated (state, goal) pairs and training states, on a hand-made fixture.

The real measurement reads 280 MB of training states; here a handful of pairs
pin what counts as overlap (exact theory, state and goal), what the three
subsets are, and that the subset metrics only select rows and never rescore.
"""

import json

import pytest

from isomer.e3 import overlap as O


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _state(state, goal, dist, theory="bool"):
    return {"theory": theory, "state": state, "goal": goal, "dist": dist, "problem_id": "t", "on_path": True}


def _label(pid, i, dist, state, goal, children, optimal, status="ok", theory="bool"):
    return {"problem_id": pid, "i": i, "dist": dist, "theory": theory, "state": state, "goal": goal,
            "status": status, "children": children, "optimal": optimal if status == "ok" else []}


TRAIN = [_state("s2", "g", 2), _state("s2", "g", 3),        # the same pair under two train problems
         _state("c1", "g", 1), _state("x", "other", 1), _state("s9", "g", 4, theory="group")]

LABELS = [
    _label("p", 0, 2, "s2", "g", ["c1", "c2"], [True, False]),          # state and one child trained on
    _label("p", 1, 1, "c1", "g", ["g", "c3"], [True, False]),           # state trained on, children not
    _label("q", 0, 2, "u2", "g", ["c1", "u3"], [False, True]),          # only a child trained on
    _label("q", 1, 1, "u3", "g", ["g", "u4"], [True, False]),           # nothing trained on
    _label("r", 0, 1, "s9", "g", ["g"], [True]),                        # same strings, other theory: no overlap
    _label("r", 1, 3, "x", "g", ["x2"], [], status="unknown"),          # same state, other goal; not scored
]


@pytest.fixture
def pairs(tmp_path):
    return O.load_pairs(_jsonl(tmp_path / "train.jsonl", TRAIN))


def test_load_pairs_keeps_every_label_of_a_pair(pairs):
    assert len(pairs) == 4
    assert pairs[O.pair_key("bool", "s2", "g")] == (1 << 2) | (1 << 3)
    assert O.pair_key("bool", "s9", "g") not in pairs                   # theory is part of the pair
    assert O.pair_key("bool", "x", "g") not in pairs                    # so is the goal


def test_mark(pairs):
    m = O.mark(LABELS, pairs)
    assert m[("p", 0)] == {"problem_id": "p", "dist": 2, "scored": True, "state_seen": True, "same_label": True,
                           "children": 2, "children_seen": 1, "any_child_seen": True}
    assert (m[("p", 1)]["state_seen"], m[("p", 1)]["any_child_seen"]) == (True, False)
    assert (m[("q", 0)]["state_seen"], m[("q", 0)]["any_child_seen"]) == (False, True)
    assert (m[("q", 1)]["state_seen"], m[("q", 1)]["any_child_seen"]) == (False, False)
    assert m[("r", 0)]["state_seen"] is False
    assert m[("r", 1)]["scored"] is False and m[("r", 1)]["state_seen"] is False


def test_same_label_is_false_when_training_has_another_distance(tmp_path):
    pairs = O.load_pairs(_jsonl(tmp_path / "t.jsonl", [_state("s2", "g", 4)]))
    m = O.mark(LABELS[:1], pairs)[("p", 0)]
    assert m["state_seen"] is True and m["same_label"] is False


def test_overlap_by_dist_counts_scored_states_only(pairs):
    by = O.overlap_by_dist(O.mark(LABELS, pairs))
    assert list(by) == [1, 2]                                           # the unknown state at distance 3 is out
    assert by[1] == {"states": 3, "state_seen": 1, "same_label": 1, "children": 5, "children_seen": 0,
                     "any_child_seen": 0}
    assert by[2] == {"states": 2, "state_seen": 1, "same_label": 1, "children": 4, "children_seen": 2,
                     "any_child_seen": 2}


def test_subsets():
    seen = {"state_seen": True, "any_child_seen": False}
    child = {"state_seen": False, "any_child_seen": True}
    clean = {"state_seen": False, "any_child_seen": False}
    assert [O.in_subset(m, "full") for m in (seen, child, clean)] == [True, True, True]
    assert [O.in_subset(m, "unseen") for m in (seen, child, clean)] == [False, True, True]
    assert [O.in_subset(m, "strict") for m in (seen, child, clean)] == [False, False, True]


def _score(pid, i, dist, h_state, h_children, optimal):
    return {"problem_id": pid, "i": i, "dist": dist, "h_state": h_state, "h_children": h_children,
            "sb_children": [0.0] * len(h_children), "optimal": optimal}


SCORES = [
    _score("p", 0, 2, 2.0, [1.0, 3.0], [True, False]),      # right
    _score("p", 1, 1, 1.0, [0.0, 2.0], [True, False]),      # right
    _score("q", 0, 2, 2.5, [1.0, 2.0], [False, True]),      # wrong
    _score("q", 1, 1, 1.5, [0.5, 0.1], [True, False]),      # wrong
    _score("r", 0, 1, 0.9, [0.1], [True]),                  # right (single child)
]


def test_subset_metrics_select_rows_and_never_rescore(pairs):
    m = O.subset_metrics(SCORES, O.mark(LABELS, pairs))
    assert list(m["full"]) == [1, 2, "all"]                 # distances in order, then the total
    assert (m["full"]["all"]["states"], m["full"]["all"]["top1"]) == (5, pytest.approx(3 / 5))
    assert (m["unseen"]["all"]["states"], m["unseen"]["all"]["top1"]) == (3, pytest.approx(1 / 3))
    assert (m["strict"]["all"]["states"], m["strict"]["all"]["top1"]) == (2, pytest.approx(1 / 2))
    assert m["full"][2] == {"states": 2, "problems": 2, "top1": 0.5, "symbol_bag": 0.5, "mean_h": 2.25,
                            "drop": pytest.approx((1.0 + 0.5) / 2)}
    assert 2 not in m["strict"]                             # both distance-2 states touch training
    assert m["strict"][1]["mean_h"] == pytest.approx((1.5 + 0.9) / 2)
    assert m["strict"][1]["drop"] == pytest.approx((1.0 + 0.8) / 2)
    lo, hi = m["full"]["all"]["top1_ci"]
    assert 0.0 <= lo <= 3 / 5 <= hi <= 1.0


SAMPLE = [{"id": "p", "theory": "bool", "lhs": "s2", "rhs": "g", "dist": 2},
          {"id": "q", "theory": "bool", "lhs": "u2", "rhs": "g", "dist": 2},
          {"id": "r", "theory": "bool", "lhs": "s9", "rhs": "g", "dist": 4}]


def test_problem_overlap(pairs, tmp_path):
    po = O.problem_overlap(SAMPLE, O.mark(LABELS, pairs), pairs)
    assert po == {"p": {"dist": 2, "start": True, "onpath_seen": 2},
                  "q": {"dist": 2, "start": False, "onpath_seen": 0},     # a seen child is not on the path
                  "r": {"dist": 4, "start": False, "onpath_seen": 0}}
    # once r's own pairs are trained on (in its theory), start and on-path overlap both show
    more = O.load_pairs(_jsonl(tmp_path / "m.jsonl", [*TRAIN, _state("s9", "g", 4), _state("x", "g", 3)]))
    po = O.problem_overlap(SAMPLE, O.mark(LABELS[4:], more), more)
    assert po["r"] == {"dist": 4, "start": True, "onpath_seen": 2}        # the unknown on-path state counts too


def _bench(pid, solved, expansions):
    return {"id": pid, "status": "ok", "solved": solved, "expansions": expansions, "max_budget": 1000,
            "stop": "solved" if solved else "budget", "dist": 2, "proof_len": 2 if solved else None,
            "nodes_stored": 1, "method": "m"}


def test_solve_counts():
    rows = [_bench("p", True, 20), _bench("q", True, 500), _bench("r", False, 1000)]
    assert O.solve_counts(rows, {"p", "q", "r"}) == {"n": 3, 30: 1, 100: 1, 300: 1, 1000: 2}
    assert O.solve_counts(rows, {"q", "r"}) == {"n": 2, 30: 0, 100: 0, 300: 0, 1000: 1}
    assert O.solve_counts(rows, set()) == {"n": 0, 30: 0, 100: 0, 300: 0, 1000: 0}


def test_render_end_to_end(tmp_path):
    root = tmp_path
    _jsonl(root / "data/states-a/train.jsonl", TRAIN)
    _jsonl(root / "data/states-a/val.jsonl", [_state("u3", "g", 1)])
    _jsonl(root / "data/states-b/train.jsonl", [_state("zz", "g", 1)])       # nothing in common
    e3 = root / "e3"
    for split in O.SPLITS:
        _jsonl(e3 / "labels" / f"{split}.jsonl", LABELS)
        for model in ("ma", "mb"):
            _jsonl(e3 / "scores" / model / f"{split}.jsonl", SCORES)
    _jsonl(e3 / "data" / "test_depth.jsonl", SAMPLE)
    for model in ("ma", "mb"):
        _jsonl(e3 / "bench-seeds" / model / "results.jsonl",
               [_bench("p", True, 20), _bench("q", True, 500), _bench("r", False, 1000)])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    text = O.render(e3, {"ma": "a", "mb": "b"}, {"a": "data/states-a", "b": "data/states-b"}, root=root)

    assert "| a | `data/states-a` | 4 |" in text and "| b | `data/states-b` | 1 |" in text
    # distance 2: both scored states have a trained child in a, none in b; one state is itself in a
    assert "| 2 | 2 | 1 (50.0%) | 0 (0.0%) | 4 | 2 (50.0%) | 0 (0.0%) | 2 (100.0%) | 0 (0.0%) |" in text
    assert "1 are in `val.jsonl`" in text
    # model ma (trained on a) loses three states, model mb (trained on b) none
    assert "| ma | a | 5 | 0.600 (" in text and "| 3 | 0.333 (" in text and "| 2 | 0.500 (" in text
    # on two splits: ma's full set (2) and mb's three subsets, which are all the full set (6)
    assert text.count("| 5 | 0.600 (") == 8
    # search: p has its start pair in a; it is the only problem with an on-path pair there
    assert "| all | 3 | 1 (33.3%) | 1 (33.3%) | 0 (0.0%) | 0 (0.0%) |" in text
    assert "| ma | a | no on-path pair in training | 2 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (50.0%) |" in text
    assert "| mb | b | no on-path pair in training | 3 | 1 (33.3%) |" in text
    assert O.render(e3, {"ma": "a", "mb": "b"}, {"a": "data/states-a", "b": "data/states-b"}, root=root) == text
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before     # read-only
