"""Size-matched states subset: the selection is seeded and whole-problem, the copy is verbatim."""

import json
from collections import Counter

import pytest

from isomer.e3 import matched as M


def _rows(pid, dist, theory="group"):
    """A problem of distance ``dist``: its on-path states and two off-path states per step."""
    out = [{"problem_id": pid, "theory": theory, "state": f"(* a{pid}{k} e)", "goal": "e", "dist": k,
            "on_path": True, "parent_dist": None, "split": "train"} for k in range(dist, -1, -1)]
    out += [{"problem_id": pid, "theory": theory, "state": f"(* b{pid}{k}{j} e)", "goal": "e", "dist": k + 1,
             "on_path": False, "parent_dist": k, "split": "train"} for k in range(1, dist + 1) for j in (0, 1)]
    return out


PROBLEMS = {f"{i:02d}": 1 + i % 6 for i in range(40)}        # 40 problems, distances 1..6


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "states"
    d.mkdir()
    rows = sorted((r for pid, dist in PROBLEMS.items() for r in _rows(pid, dist)),
                  key=lambda r: (r["problem_id"], r["state"]))
    (d / "train.jsonl").write_bytes("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    (d / "val.jsonl").write_bytes(b'{"problem_id": "v", "dist": 1}\n')
    stats = {"format": "isomer-states/1", "config": {"data": "x", "out": "states", "offpath": 6, "budget": 3000},
             "source": {"train_sha256": "ab" * 32}, "problems": {"train": 40, "val": 1},
             "counts": {"train": len(rows), "val": 1}, "on_off_path": {"val": {"on_path": 1, "off_path": 0}},
             "by_theory_dist": {"val": {"group|1": 1}}, "dropped": {"bfs_budget": 3}}
    (d / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return d


def test_select_is_seeded_and_nested():
    ids = set(PROBLEMS)
    a = M.select_problems(ids, 10, seed=0)
    assert len(a) == 10 and a <= ids
    assert M.select_problems(set(ids), 10, seed=0) == a           # a pure function of seed and ids
    assert M.select_problems(ids, 10, seed=1) != a
    assert M.select_problems(ids, 11, seed=0) > a                 # nested: a larger subset extends a smaller one
    assert M.select_problems(ids, 40, seed=0) == ids
    with pytest.raises(M.MatchedError):
        M.select_problems(ids, 41, seed=0)


def test_build_keeps_whole_problems_verbatim_and_copies_val(src, tmp_path):
    out = tmp_path / "matched"
    stats = M.build_matched(src, out, 25, seed=0, matched_to="data/states-v2")
    lines = (out / "train.jsonl").read_bytes().split(b"\n")[:-1]
    source = (src / "train.jsonl").read_bytes().split(b"\n")[:-1]
    kept = M.select_problems(set(PROBLEMS), 25, 0)
    # exactly the source lines of the kept problems, in source order
    assert lines == [x for x in source if json.loads(x)["problem_id"] in kept]
    rows = [json.loads(x) for x in lines]
    per = Counter(r["problem_id"] for r in rows)
    assert set(per) == kept
    assert all(per[pid] == 3 * PROBLEMS[pid] + 1 for pid in kept)     # every state of a kept problem
    assert (out / "val.jsonl").read_bytes() == (src / "val.jsonl").read_bytes()

    assert stats["problems"] == {"train": 25, "val": 1}
    assert stats["counts"] == {"train": len(rows), "val": 1}
    assert stats["subset"]["seed"] == 0 and stats["subset"]["train_problems"] == 25
    assert stats["subset"]["of_train_problems"] == 40 and stats["subset"]["matched_to"] == "data/states-v2"
    assert stats["dist_histogram"]["train"] == {str(k): v for k, v in sorted(Counter(r["dist"] for r in rows).items())}
    assert stats["problem_dist_histogram"]["train"] == {
        str(k): v for k, v in sorted(Counter(PROBLEMS[p] for p in kept).items())}
    assert stats["dist_histogram"]["val"] == {"1": 1}
    assert stats["on_off_path"]["train"] == {"on_path": sum(r["on_path"] for r in rows),
                                             "off_path": sum(not r["on_path"] for r in rows)}
    assert stats["source"] == {"train_sha256": "ab" * 32}             # what the trainer records as provenance
    assert stats["config"]["out"] == out.as_posix() and stats["config"]["offpath"] == 6
    assert stats["max_tokens"] > 0
    assert json.loads((out / "stats.json").read_text(encoding="utf-8")) == stats


def test_rebuild_is_byte_identical_and_seed_changes_it(src, tmp_path):
    M.build_matched(src, tmp_path / "a", 25, seed=0)
    M.build_matched(src, tmp_path / "b", 25, seed=0)
    M.build_matched(src, tmp_path / "c", 25, seed=1)
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()
    assert (tmp_path / "a" / "train.jsonl").read_bytes() != (tmp_path / "c" / "train.jsonl").read_bytes()
    sa = json.loads((tmp_path / "a" / "stats.json").read_text(encoding="utf-8"))
    sb = json.loads((tmp_path / "b" / "stats.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in sa.items() if k != "config"} == {k: v for k, v in sb.items() if k != "config"}


def test_refusals(src, tmp_path):
    with pytest.raises(M.MatchedError, match="only 40"):
        M.build_matched(src, tmp_path / "x", 41)
    assert not (tmp_path / "x").exists()
    with pytest.raises(M.MatchedError):
        M.build_matched(src, src, 10)                                 # never into the source
    M.build_matched(src, tmp_path / "y", 10)
    with pytest.raises(M.MatchedError, match="already exists"):
        M.build_matched(src, tmp_path / "y", 10)
    (src / "val.jsonl").unlink()
    with pytest.raises(M.MatchedError, match="not found"):
        M.build_matched(src, tmp_path / "z", 10)


def test_loads_in_the_trainer(src, tmp_path):
    """The subset is a states directory the trainer's loader accepts."""
    pytest.importorskip("torch")
    from isomer.learn import data as D

    out = tmp_path / "matched"
    M.build_matched(src, out, 25)
    ss = D.load_states(out / "train.jsonl", 192)
    assert ss.n_problems == 25
