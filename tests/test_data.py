import json

import pytest
from conftest import tiny_v2_config

from isomer.core import kernel
from isomer.core.proof import Problem, Proof
from isomer.core.rewrite import Step
from isomer.core.theories import THEORIES
from isomer.data.build import FORMAT_VERSION, SPLITS, build, problem_id, walk_in_action_space


def test_problem_id_is_symmetric():
    assert problem_id("group", "a", "(* e a)") == problem_id("group", "(* e a)", "a")


def _build(out):
    sheet = build(tiny_v2_config(out), progress=False)
    files = {f.name: f.read_bytes() for f in sorted(out.glob("*.jsonl"))}
    recs = {s: [json.loads(line) for line in files[f"{s}.jsonl"].decode().splitlines()] for s in SPLITS}
    return sheet, files, recs


def test_small_v2_build(tmp_path):
    sheet, files1, recs = _build(tmp_path / "a")
    _, files2, _ = _build(tmp_path / "b")
    assert files1 == files2  # byte-identical rebuild
    assert set(files1) == {f"{s}.jsonl" for s in SPLITS}
    assert sheet["format"] == FORMAT_VERSION == "isobench/2"
    cfg = tiny_v2_config(tmp_path)
    assert sheet["label_budget"] == {"train": 300, "test_id": 300, "test_depth": 500, "test_deep": 500,
                                     "test_theory": 500}
    assert sheet["counts"] == {s: len(r) for s, r in recs.items()}
    assert sheet["counts"]["train"] > 0 and sheet["counts"]["test_theory"] == 6

    # A problem lands in exactly one split.
    ids = [r["id"] for rs in recs.values() for r in rs]
    assert len(ids) == len(set(ids))

    for split, rs in recs.items():
        for r in rs:
            # Bounds: dist_lb <= dist <= dist_ub, or dist_lb <= dist_ub when dist is unknown.
            if r["dist"] is not None:
                assert r["dist_lb"] <= r["dist"] <= r["dist_ub"] and len(r["proof"]) == r["dist"]
            else:
                assert r["dist_lb"] <= r["dist_ub"] == len(r["proof"]) == r["walk_len"]
            # Every stored proof is accepted by the trusted kernel.
            theory, _ = THEORIES[r["theory"]]
            proof = Proof(Problem.parse(theory, r["lhs"], r["rhs"]), tuple(Step.from_json(s) for s in r["proof"]))
            assert kernel.check_certificate(proof.certificate()), (split, r["id"])
            # Split rules.
            if split in ("train", "test_id"):
                assert r["theory"] in cfg.train_theories and r["dist"] is not None
                assert r["dist"] <= cfg.train_max_dist and r["walk_len"] <= max(cfg.train_walk_lengths)
            elif split == "test_depth":
                assert r["theory"] in cfg.train_theories and r["dist"] > cfg.train_max_dist
            elif split == "test_deep":
                assert r["theory"] in cfg.train_theories and r["dist"] is None
            else:
                assert r["theory"] in cfg.heldout_theories


def test_walk_cap_check():
    """A walk is a valid dist_ub only if every term stays within the size cap
    max(|lhs|, |rhs|) + slack; one that detours through bigger terms is rejected."""
    theory, _ = THEORIES["group"]
    up, down = Step("id_l~", ()), Step("id_l", ())
    # a -> (* e a) -> (* e (* e a)) -> (* e a): sizes 1, 3, 5, 3; cap = 3 + slack.
    walk = Proof(Problem.parse(theory, "a", "(* e a)"), (up, up, down))
    assert walk.verify()
    assert walk_in_action_space(walk, slack=2) and not walk_in_action_space(walk, slack=1)


def test_exclude_ids_from(tiny_data, tmp_path):
    """A dev build with the same seed regenerates the same walks; every one
    whose id is in the excluded dataset is skipped and replaced."""
    import dataclasses

    def ids(d):
        return {json.loads(line)["id"] for f in d.glob("*.jsonl")
                for line in f.read_text(encoding="utf-8").splitlines()}

    cfg = dataclasses.replace(tiny_v2_config(tmp_path / "dev"), exclude_ids_from=str(tiny_data))
    sheet = build(cfg, progress=False)
    assert not ids(tmp_path / "dev") & ids(tiny_data)
    skipped = sheet["excluded_ids"]["skipped_at_generation"]
    assert sum(skipped.values()) > 0 and sheet["excluded_ids"]["n_ids"] == len(ids(tiny_data))
    notes = json.loads((tmp_path / "dev" / "dev_notes.json").read_text(encoding="utf-8"))
    assert notes["skipped_at_generation"] == skipped and notes["counts"] == sheet["counts"]
    build(dataclasses.replace(cfg, out=str(tmp_path / "dev2")), progress=False)
    for name in [*(f"{s}.jsonl" for s in SPLITS), "dev_notes.json"]:
        assert (tmp_path / "dev" / name).read_bytes() == (tmp_path / "dev2" / name).read_bytes()
    with pytest.raises(FileNotFoundError):
        build(dataclasses.replace(cfg, exclude_ids_from=str(tmp_path / "empty")), progress=False)
