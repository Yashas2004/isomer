import json

from isomer.data.build import BuildConfig, build, problem_id


def test_problem_id_is_symmetric():
    assert problem_id("group", "a", "(* e a)") == problem_id("group", "(* e a)", "a")


def test_small_build_is_deterministic_and_verified(tmp_path):
    def run(out):
        cfg = BuildConfig(out=str(out), per_theory=15, per_heldout_theory=5, bfs_budget=300, workers=1)
        sheet = build(cfg, progress=False)
        return sheet, {f.name: f.read_text() for f in out.glob("*.jsonl")}

    sheet1, files1 = run(tmp_path / "a")
    _, files2 = run(tmp_path / "b")
    assert files1 == files2  # byte-identical rebuild
    assert sum(sheet1["counts"].values()) == 2 * 15 + 2 * 5
    for line in files1["train.jsonl"].splitlines():
        rec = json.loads(line)
        assert rec["dist"] is not None and rec["dist"] <= 4 and rec["theory"] in ("group", "bool")
        assert len(rec["proof"]) == rec["dist"]
    ids = [json.loads(l)["id"] for f in files1.values() for l in f.splitlines()]
    assert len(ids) == len(set(ids))  # no duplicates across splits
