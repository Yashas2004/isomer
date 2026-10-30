import json
import random

import pytest

from isomer.cli import main
from isomer.data.stratify import STRATA, allocate, read_ids, stratified_sample


def recs(n_by_key: dict[str, int]):
    out = []
    for key, n in n_by_key.items():
        for i in range(n):
            out.append({"id": f"{key}-{i:04d}", "theory": "group", "dist": int(key), "dist_lb": int(key)})
    return out


def test_allocate_sums_proportional_and_covers_every_stratum():
    counts = {"5": 281, "6": 389, "7": 234, "8": 207, "9": 78, "10": 15}
    a = allocate(counts, 300)
    total = sum(counts.values())
    assert sum(a.values()) == 300 and set(a) == set(counts)
    for k, c in counts.items():
        assert abs(a[k] - 300 * c / total) < 1 and a[k] >= 1
    # a tiny stratum is still represented, and the overshoot comes back from elsewhere
    b = allocate({"7": 273, "8": 298, "9": 169, "10": 25, "11": 2}, 100)
    assert sum(b.values()) == 100 and b["11"] >= 1 and b["10"] >= 1
    assert allocate({"a": 3, "b": 1}, 4) == {"a": 3, "b": 1}
    with pytest.raises(ValueError):
        allocate({"a": 3}, 4)
    with pytest.raises(ValueError):
        allocate({"a": 3, "b": 3, "c": 3}, 2)


def test_sample_is_deterministic_and_independent_of_file_order():
    rs = recs({"5": 40, "6": 25, "7": 3})
    ids, per = stratified_sample(rs, "test_depth", 20, "dist", seed=0)
    shuffled = rs[:]
    random.Random(1).shuffle(shuffled)
    assert stratified_sample(shuffled, "test_depth", 20, "dist", seed=0) == (ids, per)
    assert ids == sorted(ids) and len(ids) == len(set(ids)) == 20
    assert {i.split("-")[0] for i in ids} == {"5", "6", "7"}
    assert sum(n for _, n in per.values()) == 20
    assert stratified_sample(rs, "test_depth", 20, "dist", seed=1)[0] != ids


def test_theory_dist_and_null_strata():
    rs = [{"id": f"x{i}", "theory": t, "dist": d, "dist_lb": 1}
          for i, (t, d) in enumerate([("ring", None)] * 4 + [("lattice", 2)] * 6 + [("ring", 2)] * 10)]
    assert STRATA["theory_dist"](rs[0]) == "ring|None"
    ids, per = stratified_sample(rs, "test_theory", 10, "theory_dist", seed=0)
    assert set(per) == {"ring|None", "lattice|2", "ring|2"} and len(ids) == 10
    assert all(n >= 1 for _, n in per.values())


def test_cli_on_tiny_build(tiny_data, tmp_path):
    out = tmp_path / "ids.txt"
    args = ["stratify", "--data", str(tiny_data), "--split", "train", "--n", "6", "--strata", "dist",
            "--seed", "0", "--out", str(out)]
    assert main(args) == 0
    first = out.read_bytes()
    assert main(args) == 0 and out.read_bytes() == first
    text = first.decode()
    assert text.startswith("# python -m isomer stratify --data") and "sha256" in text
    ids = read_ids(out)
    split = {json.loads(line)["id"] for line in (tiny_data / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(ids) == 6 and set(ids) <= split
    assert main([*args[:5], "--n", "100000", *args[7:]]) == 2
