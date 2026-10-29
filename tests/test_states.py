import json
from collections import defaultdict, deque

import pytest

from isomer.core.proof import Problem
from isomer.core.rewrite import successors
from isomer.core.term import parse, size
from isomer.core.theories import THEORIES
from isomer.learn.states import StatesConfig, build_states, split_of
from isomer.search.common import size_cap

TEST_SPLITS = ("test_id", "test_depth", "test_deep", "test_theory")


def _build(data, out):
    stats = build_states(StatesConfig(data=str(data), out=str(out), offpath=3, val_fraction=0.5, workers=1),
                         progress=False)
    recs = {s: [json.loads(line) for line in (out / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in ("train", "val")}
    return stats, recs


@pytest.fixture(scope="module")
def built(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("states") / "a"
    stats, recs = _build(tiny_data, out)
    return out, stats, recs


def originals(data):
    """problem_id -> (problem, pool, cap) in the ORIGINAL action space."""
    out = {}
    for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        p = Problem.parse(THEORIES[r["theory"]][0], r["lhs"], r["rhs"])
        out[r["id"]] = (p, p.pool(), size_cap(p, 6))
    return out


def oracle_dist(problem, pool, cap, start, limit=8):
    """Plain one-directional BFS from ``start`` to the problem's rhs, in the
    original problem's action space. Independent of bidirectional_bfs."""
    rules = list(problem.theory.rules().values())
    seen, q = {start: 0}, deque([start])
    if start == problem.rhs:
        return 0
    while q:
        t = q.popleft()
        if seen[t] >= limit:
            continue
        for _, u in successors(t, rules, pool):
            if u not in seen and size(u) <= cap:
                seen[u] = seen[t] + 1
                if u == problem.rhs:
                    return seen[u]
                q.append(u)
    return None


def test_deterministic_rebuild(tiny_data, built, tmp_path):
    out, stats, _ = built
    stats2, _ = _build(tiny_data, tmp_path / "b")
    for name in ("train.jsonl", "val.jsonl"):
        assert (out / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    def stable(s):  # timings and the output path legitimately differ
        return {k: v for k, v in s.items() if k != "seconds"} | {"config": s["config"] | {"out": None}}
    assert stable(stats) == stable(stats2)


def test_splits_and_provenance(tiny_data, built):
    _, stats, recs = built
    train_ids = {r["problem_id"] for r in recs["train"]}
    val_ids = {r["problem_id"] for r in recs["val"]}
    assert train_ids and val_ids and not train_ids & val_ids
    for split, rs in recs.items():
        assert all(r["split"] == split == split_of(r["problem_id"], 0.5) for r in rs)
    test_ids = {json.loads(line)["id"] for s in TEST_SPLITS
                for line in (tiny_data / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not (train_ids | val_ids) & test_ids
    source = {json.loads(line)["id"] for line in (tiny_data / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    assert train_ids | val_ids == source
    assert stats["counts"] == {s: len(r) for s, r in recs.items()}
    assert stats["max_tokens"] > 0


def test_structure(built):
    _, stats, recs = built
    by_problem = defaultdict(list)
    for r in recs["train"] + recs["val"]:
        by_problem[r["problem_id"]].append(r)
    for rs in by_problem.values():
        on = [r for r in rs if r["on_path"]]
        d = max(r["dist"] for r in on)
        assert sorted(r["dist"] for r in on) == list(range(d + 1))
        assert all(r["parent_dist"] is None for r in on)
        assert next(r for r in on if r["dist"] == 0)["state"] == rs[0]["goal"]
        on_states = {r["state"] for r in on}
        for r in rs:
            if not r["on_path"]:
                assert r["state"] not in on_states and 0 <= r["parent_dist"] < d + 1
        assert len({r["state"] for r in rs}) == len(rs)
    assert sum(v["off_path"] for v in stats["on_off_path"].values()) > 0


def test_triangle_property_on_every_offpath_record(built):
    """One legal step changes the exact distance by at most 1, because every
    step has a legal inverse in the same action space (ADR-003)."""
    _, _, recs = built
    off = [r for r in recs["train"] + recs["val"] if not r["on_path"]]
    assert off
    for r in off:
        assert abs(r["dist"] - r["parent_dist"]) <= 1, r


def test_labels_match_independent_bfs(tiny_data, built):
    _, _, recs = built
    orig = originals(tiny_data)
    rs = recs["train"] + recs["val"]
    on = [r for r in rs if r["on_path"]][:30]
    off = [r for r in rs if not r["on_path"] and r["dist"] <= 4][:10]
    assert len(on) == 30 and len(off) == 10 and any(r["dist"] == 0 for r in on)
    for r in on + off:
        problem, pool, cap = orig[r["problem_id"]]
        assert oracle_dist(problem, pool, cap, parse(r["state"])) == r["dist"], r
