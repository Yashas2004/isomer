"""E3-fresh pool: config, disjointness check, fixed-count sample, id-file format, resume guard.

The full build labels 2,400 problems (about 1.5 h) and is run by hand; these
tests pin the pieces that decide *which* problems end up in the sample.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from isomer.data.build import BuildConfig, problem_id
from isomer.e3 import fresh, regen, seeds

ROOT = Path(__file__).resolve().parent.parent


def test_fresh_config_changes_only_seed_and_output():
    cfg, base = asdict(fresh.fresh_config()), asdict(BuildConfig())
    assert cfg["seed"] == fresh.FRESH_SEED == 1 and base["seed"] == 0      # the default is untouched
    assert cfg["out"] == "data/isobench-fresh"
    assert {k for k in cfg if cfg[k] != base[k]} == {"seed", "out"}
    assert fresh.fresh_config(workers=3).workers == 3


def test_sample_counts_are_those_of_the_e2_sample():
    assert fresh.e2_counts(ROOT / fresh.E2_IDS) == fresh.SAMPLE_COUNTS
    assert sum(fresh.SAMPLE_COUNTS.values()) == 300


def test_keep_is_the_test_depth_rule():
    cfg = fresh.fresh_config()
    assert [fresh.keep({"dist": d}, cfg) for d in (None, 3, 4, 5, 11)] == [False, False, False, True, True]


def _records(per_dist: dict[int, int]) -> list[dict]:
    return [{"id": hashlib.sha256(f"{d}-{i}".encode()).hexdigest()[:16], "dist": d}
            for d, n in per_dist.items() for i in range(n)]


def test_sample_takes_exactly_the_fixed_counts_by_rank():
    recs = _records({5: 10, 6: 8, 7: 3, 11: 2})
    ids, per = fresh.sample(recs, {5: 4, 6: 2, 7: 3})
    dist = {r["id"]: r["dist"] for r in recs}
    assert ids == sorted(ids) and len(ids) == 9
    assert [sum(dist[i] == d for i in ids) for d in (5, 6, 7, 11)] == [4, 2, 3, 0]
    assert per == {5: (10, 4), 6: (8, 2), 7: (3, 3), 11: (2, 0)}
    five = sorted((r["id"] for r in recs if r["dist"] == 5), key=lambda p: fresh.rank(1, fresh.SPLIT, p))
    assert set(five[:4]) <= set(ids) and not set(five[4:]) & set(ids)
    assert fresh.sample(list(reversed(recs)), {5: 4, 6: 2, 7: 3}) == (ids, per)    # file order does not matter


def test_a_short_stratum_is_an_error_not_a_smaller_sample():
    with pytest.raises(fresh.FreshError, match="distance 10: 2 available, 4 needed"):
        fresh.sample(_records({5: 80, 10: 2}), {5: 70, 10: 4})


def test_id_file_is_read_by_the_existing_tools(tmp_path):
    recs = _records({5: 6, 6: 5, 8: 1})
    ids, per = fresh.sample(recs, {5: 3, 6: 2})
    path = tmp_path / "ids.txt"
    path.write_text(fresh.ids_text(ids, per, "ab" * 32, "python scripts/e3_build_fresh.py"), encoding="utf-8")
    got, strata = regen.read_ids(path)
    assert got == ids and strata == {"5": 3, "6": 2, "8": 0}
    assert seeds.header_sha256(path) == "ab" * 32
    by_id = {r["id"]: r for r in recs}
    assert regen.verify_strata([by_id[i] for i in ids], fresh.SPLIT, strata)["match"]


def test_collisions_finds_a_planted_id(tmp_path):
    for d, rows in (("a", ["x1", "x2"]), ("b", ["y1"])):
        (tmp_path / d).mkdir()
        (tmp_path / d / "train.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in rows), "utf-8")
    (tmp_path / "b" / "test_id.jsonl").write_text(json.dumps({"id": "z"}) + "\n", "utf-8")
    assert fresh.collisions({"q", "r"}, ("a", "b"), tmp_path) == {"a/train.jsonl": 0, "b/test_id.jsonl": 0,
                                                                 "b/train.jsonl": 0}
    assert fresh.collisions({"q", "z"}, ("a", "b"), tmp_path)["b/test_id.jsonl"] == 1
    with pytest.raises(fresh.FreshError, match="no \\*.jsonl"):
        fresh.collisions({"q"}, ("missing",), tmp_path)


def test_generation_is_seeded_and_replaces_external_ids():
    cfg = replace(fresh.fresh_config(), eval_per_theory=3)
    jobs, stats = fresh.generate_jobs(cfg, frozenset())
    again, _ = fresh.generate_jobs(cfg, frozenset())
    assert jobs == again and len(jobs) == 6 and stats["replaced_at_generation"] == {"group": 0, "bool": 0}
    v2, _ = fresh.generate_jobs(replace(cfg, seed=0), frozenset())
    assert {fresh.job_id(j) for j in v2}.isdisjoint({fresh.job_id(j) for j in jobs})   # another stream
    first = fresh.job_id(jobs[0])
    swapped, stats = fresh.generate_jobs(cfg, frozenset({first}))
    assert first not in {fresh.job_id(j) for j in swapped} and len(swapped) == 6
    assert stats["replaced_at_generation"][jobs[0].theory] == 1
    assert all(j.budget == 20000 and j.max_nodes == 1_500_000 for j in jobs)


def test_label_all_resumes_and_rejects_a_foreign_partial_file(tmp_path):
    cfg = replace(fresh.fresh_config(), eval_per_theory=1)
    jobs, _ = fresh.generate_jobs(cfg, frozenset())
    partial = tmp_path / "p.jsonl"
    done = [{"id": fresh.job_id(j), "dist": 5} for j in jobs]
    partial.write_text("".join(json.dumps(r) + "\n" for r in done), encoding="utf-8")
    assert fresh.label_all(jobs, 1, partial, progress=False) == sorted(done, key=lambda r: r["id"])   # nothing to do
    partial.write_text(json.dumps({"id": problem_id("bool", "x", "y")}) + "\n", encoding="utf-8")
    with pytest.raises(fresh.FreshError, match="another configuration"):
        fresh.label_all(jobs, 1, partial, progress=False)


def test_build_never_overwrites_a_pool(tmp_path):
    (tmp_path / fresh.OUT).mkdir(parents=True)
    (tmp_path / fresh.OUT / f"{fresh.SPLIT}.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(fresh.FreshError, match="never overwrites"):
        fresh.build(tmp_path, progress=False)


def test_bench_command_for_another_sample_and_for_baselines():
    assert seeds.bench_command("runs/h-v1/best.pt", "o", python="py") == [
        "py", "-m", "isomer", "bench", "--data", "data/isobench-v2", "--split", "test_depth", "--ids",
        "docs/experiments/E2/ids-test_depth.txt", "--methods", "neural-w1@runs/h-v1/best.pt#bf16-fast",
        "--budgets", "30,100,300,1000", "--workers", "2", "--out", "o"]                  # default unchanged
    cmd = seeds.bench_command(None, "o", split=fresh.SPLIT, data=fresh.OUT, ids=fresh.IDS_OUT, python="py",
                              methods=("bfs-bidir", "bestfirst-symbol_bag-w1"))
    assert cmd[4:12] == ["--data", "data/isobench-fresh", "--split", "test_depth_fresh", "--ids",
                         "docs/experiments/E3/ids-test_depth_fresh.txt", "--methods", "bfs-bidir,bestfirst-symbol_bag-w1"]
    with pytest.raises(seeds.SeedsError):
        seeds.bench_command(None, "o")
