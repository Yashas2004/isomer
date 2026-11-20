"""IsoBench v2-d6: the split rule and the three checks that guard the variant.

A tiny fake "v2" directory stands in for the real one. The relabelling itself
(40,000 problems) is run by hand from ``scripts/e3_build_d6.py``; its result is
accepted only through the checks tested here.
"""

import json
from dataclasses import fields

import pytest

from isomer.data.build import BuildConfig
from isomer.e3 import d6


def _id(bucket: int, n: int) -> str:
    """An id whose hash bucket (id mod 10,000) is ``bucket``: below 500 goes to test_id."""
    return f"{n * 10_000 + bucket:016x}"


def _rec(pid: str, dist, theory="group"):
    return {"id": pid, "theory": theory, "lhs": "a", "rhs": "b", "walk_len": 6, "dist": dist, "dist_lb": dist or 3,
            "dist_ub": dist or 6, "bfs_expansions": 5, "label_stop": "solved" if dist else "budget",
            "label_nodes": 9, "proof": []}


POOL = [
    _rec(_id(9000, 1), 1), _rec(_id(9000, 2), 4), _rec(_id(9000, 3), 5), _rec(_id(9000, 4), 6, "bool"),
    _rec(_id(9000, 5), 7), _rec(_id(9000, 6), None),
    _rec(_id(10, 1), 2), _rec(_id(10, 2), 6),       # hash bucket below 5%: held out
]
EVAL = {"test_depth": [_rec(_id(7000, 50), 5), _rec(_id(7000, 51), 8)],
        "test_deep": [_rec(_id(7000, 60), None)],
        "test_theory": [_rec(_id(7000, 70), 3, "ring")]}


@pytest.fixture
def v2(tmp_path):
    """A consistent fake v2: the pool split at distance 4, eval files, id files with their hashes."""
    root = tmp_path / "v2"
    root.mkdir()
    splits, _ = d6.split_train_pool(POOL, 4)
    for name, recs in {**splits, **EVAL}.items():
        (root / f"{name}.jsonl").write_bytes(d6.serialise(recs))
    (root / "datasheet.json").write_text(json.dumps({"label_budget": {"train": 3000, "test_depth": 20000}}),
                                         encoding="utf-8")
    ids = tmp_path / "ids"
    ids.mkdir()
    for name in d6.EVAL_SPLITS:
        sha = d6._sha((root / f"{name}.jsonl").read_bytes())
        (ids / f"ids-{name}.txt").write_text(f"# dataset format: isobench/2  {name}.jsonl sha256: {sha}\n",
                                             encoding="utf-8")
    return root, ids


def test_split_rule_is_the_builds_rule_at_another_distance():
    at4, dropped4 = d6.split_train_pool(POOL, 4)
    assert sorted(r["dist"] for r in at4["train"]) == [1, 4] and [r["dist"] for r in at4["test_id"]] == [2]
    assert dropped4 == 5
    at6, dropped6 = d6.split_train_pool(POOL, 6)
    assert sorted(r["dist"] for r in at6["train"]) == [1, 4, 5, 6]
    assert sorted(r["dist"] for r in at6["test_id"]) == [2, 6]
    assert dropped6 == 2                                  # distance 7 and the unlabelled problem
    # the hash decides train vs test_id, so raising the distance moves no v2 problem across
    assert {r["id"] for r in at4["train"]} <= {r["id"] for r in at6["train"]}
    assert {r["id"] for r in at4["test_id"]} <= {r["id"] for r in at6["test_id"]}


def test_buildconfig_default_is_untouched():
    assert BuildConfig().train_max_dist == 4
    assert "train_max_dist" in {f.name for f in fields(BuildConfig)}


def test_serialise_sorts_by_id_and_is_stable():
    a, b = _rec("b" * 16, 1), _rec("a" * 16, 2)
    data = d6.serialise([a, b])
    lines = data.decode("utf-8").splitlines()
    assert [json.loads(x)["id"] for x in lines] == ["a" * 16, "b" * 16]
    assert lines[0] == json.dumps(b, sort_keys=True)
    assert d6.serialise([b, a]) == data


def test_build_writes_the_variant_and_records_the_checks(v2, tmp_path):
    root, ids = v2
    out = tmp_path / "d6"
    sheet = d6.build_d6(POOL, root, out, ids)
    assert sheet["counts"] == {"train": 4, "test_id": 2, "test_depth": 2, "test_deep": 1, "test_theory": 1}
    assert sheet["config"]["train_max_dist"] == 6 and sheet["config"]["pool_max_size"] == 3
    assert sheet["dist_histogram"]["train"] == {"1": 1, "4": 1, "5": 1, "6": 1}
    assert sheet["dropped"] == {"train (dist unknown or > train_max_dist)": 2}
    assert sheet["checks"]["leakage"]["train_added"] == 2 and sheet["checks"]["leakage"]["test_id_added"] == 1
    for name in d6.EVAL_SPLITS:                           # copied, not rewritten
        assert (out / f"{name}.jsonl").read_bytes() == (root / f"{name}.jsonl").read_bytes()
    assert (out / "train.jsonl").read_bytes() == d6.serialise(d6.split_train_pool(POOL, 6)[0]["train"])
    assert json.loads((out / "datasheet.json").read_text(encoding="utf-8")) == sheet
    # v2 itself is untouched, and a second build into the same place is refused
    assert d6.check_replay(POOL, root)["train"]["problems"] == 2
    with pytest.raises(d6.D6Error, match="already exists"):
        d6.build_d6(POOL, root, out, ids)


def test_max_dist_5_keeps_only_distances_up_to_5(v2, tmp_path):
    root, ids = v2
    sheet = d6.build_d6(POOL, root, tmp_path / "d5", ids, max_dist=5)
    assert sheet["variant"] == "v2-d5" and sheet["config"]["train_max_dist"] == 5
    assert sheet["counts"] == {"train": 3, "test_id": 1, "test_depth": 2, "test_deep": 1, "test_theory": 1}
    assert sheet["dropped"] == {"train (dist unknown or > train_max_dist)": 4}     # distances 6, 6, 7 and unknown
    for name in d6.TRAIN_SPLITS:
        lines = (tmp_path / "d5" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert max(json.loads(x)["dist"] for x in lines) <= 5
    assert "d5 test_id" in sheet["checks"]["leakage"]["train_overlap_with"]
    with pytest.raises(d6.D6Error, match="must exceed"):
        d6.build_d6(POOL, root, tmp_path / "d4", ids, max_dist=4)


def test_same_pool_gives_the_same_bytes(v2, tmp_path):
    root, ids = v2
    d6.build_d6(POOL, root, tmp_path / "a", ids)
    d6.build_d6(list(reversed(POOL)), root, tmp_path / "b", ids)
    for name in ("train.jsonl", "test_id.jsonl", "test_depth.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_a_pool_that_is_not_the_v2_pool_is_refused(v2, tmp_path):
    root, ids = v2
    relabelled = [dict(r, label_nodes=10) if r["dist"] == 1 else r for r in POOL]    # one field of one record
    with pytest.raises(d6.D6Error, match="replay check failed"):
        d6.build_d6(relabelled, root, tmp_path / "d6", ids)
    assert not (tmp_path / "d6").exists()                 # nothing written


def test_eval_files_that_are_not_the_e2_ones_are_refused(v2, tmp_path):
    root, ids = v2
    (ids / "ids-test_depth.txt").write_text(f"# x sha256: {'0' * 64}\n", encoding="utf-8")
    with pytest.raises(d6.D6Error, match="E2 id file was sampled from"):
        d6.build_d6(POOL, root, tmp_path / "d6", ids)
    (ids / "ids-test_depth.txt").write_text("abc\n", encoding="utf-8")
    with pytest.raises(d6.D6Error, match="records no dataset hash"):
        d6.build_d6(POOL, root, tmp_path / "d6", ids)
    assert not (tmp_path / "d6").exists()


def test_leakage_into_an_eval_split_is_refused(v2, tmp_path):
    root, ids = v2
    leaked = POOL[2]                                      # a distance-5 problem that d6 adds to train
    data = d6.serialise([*EVAL["test_depth"], leaked])
    (root / "test_depth.jsonl").write_bytes(data)
    (ids / "ids-test_depth.txt").write_text(f"# x sha256: {d6._sha(data)}\n", encoding="utf-8")
    with pytest.raises(d6.D6Error, match="leakage.*test_depth"):
        d6.build_d6(POOL, root, tmp_path / "d6", ids)
    assert not (tmp_path / "d6").exists()


def test_refuses_to_write_into_v2(v2):
    root, ids = v2
    with pytest.raises(d6.D6Error):
        d6.build_d6(POOL, root, root, ids)
