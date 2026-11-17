"""E3-fresh summary: the committed rules, each hypothesis on hand-made inputs, and the rendered file."""

import json
from pathlib import Path

import pytest

from isomer.e3 import fresh_summary as F
from isomer.e3 import seeds
from isomer.e3 import seeds_compare as C

ROOT = Path(__file__).resolve().parent.parent


def test_rules_are_the_committed_prediction():
    assert (F.H1_DISTS, F.H1_BASELINE_BELOW, F.H1_D6_AT_LEAST, F.MIN_STATES) == ((7, 8), 5.0, 5.7, 10)
    assert (F.H3_BUDGETS, F.ALPHA) == ((30, 1000), 0.05)
    assert F.BASELINE == ("h-v1", "h-v1-s1", "h-v1-s2") and F.D6 == ("h-d6-s0", "h-d6-s1", "h-d6-s2")
    assert F.CONTROL == "h-d6m-s0"
    text = (ROOT / "docs/experiments/E3/fresh_prediction.md").read_text(encoding="utf-8")
    for s in ("< 5.0", ">= 5.7", "k = 7 and k = 8", "fewer than 10 scored states", "30 and at 1,000",
              "p < 0.05", "3 x 3 x 2 = 18 tests", "300,000", "d6_matched_prediction.md"):
        assert s in text, s
    for _, _, _, depth, volume in C.MATCHED_RULES:                     # H4 copies these unchanged
        assert f">= {depth}" in text and f"<= {volume}" in text


def _row(states, mean_h, drop=0.5, top1=0.9):
    return {"states": states, "children": 8.0, "chance": 0.2, "symbol_bag": 0.3, "mean_h": mean_h, "top1": top1,
            "top1_ci": (0.8, 0.95), "drop": drop, "margin": -0.3, "violation": 0.1}


def _tables(base_h=4.7, d6_h=6.2, states=50):
    t = {m: {5: _row(states, 4.5), 7: _row(states, base_h), 8: _row(states, base_h)} for m in F.BASELINE}
    t |= {m: {5: _row(states, 4.9), 7: _row(states, d6_h, 0.6), 8: _row(states, d6_h, 0.5)} for m in (*F.D6, F.CONTROL)}
    return t


def test_h1_supported_not_supported_and_not_decidable():
    h = F.judge_h1(_tables())
    assert h["verdict"] == F.SUPPORTED and len(h["rows"]) == 12
    t = _tables()
    t["h-v1-s2"][8] = _row(50, 5.0)                     # "< 5.0" is strict
    assert F.judge_h1(t)["verdict"] == F.NOT_SUPPORTED
    t = _tables()
    t["h-d6-s1"][7] = _row(50, 5.69)
    assert F.judge_h1(t)["verdict"] == F.NOT_SUPPORTED
    t = _tables()
    t["h-d6-s0"][8] = _row(9, 6.5)                      # too few states: no verdict on that row
    h = F.judge_h1(t)
    assert h["verdict"] == F.UNDECIDABLE and sum(r["pass"] is None for r in h["rows"]) == 1
    t["h-v1"][7] = _row(50, 5.2)                        # a failure outranks an undecidable row
    assert F.judge_h1(t)["verdict"] == F.NOT_SUPPORTED


def test_h2_compares_with_the_best_baseline_seed():
    top1 = {"h-v1": 0.7, "h-v1-s1": 0.8, "h-v1-s2": 0.65, "h-d6-s0": 0.95, "h-d6-s1": 0.81, "h-d6-s2": 0.9}
    h = F.judge_h2(top1)
    assert h["best_baseline"] == "h-v1-s1" and h["verdict"] == F.SUPPORTED
    assert F.judge_h2(top1 | {"h-d6-s1": 0.8})["verdict"] == F.NOT_SUPPORTED     # equal is not greater


def _bench(solved_at: dict[str, int], n: int = 100) -> list[dict]:
    """``solved_at[id]`` = expansions used; ids not listed are unsolved at 1000."""
    rows = []
    for i in range(n):
        pid = f"p{i:03d}"
        e = solved_at.get(pid)
        rows.append({"id": pid, "method": "m", "status": "ok", "solved": e is not None, "expansions": e or 1000,
                     "max_budget": 1000, "stop": "solved" if e else "budget", "proof_len": 3 if e else None,
                     "nodes_stored": 1, "dist": 5 + i % 3})
    return rows


def _runs(base_k=40, d6_k=90, fast=20):
    """Baseline seeds solve the first base_k problems, d6 seeds the first d6_k; ``fast`` of them within 30."""
    def solved(k):
        return {f"p{i:03d}": (10 if i < fast else 500) for i in range(k)}
    return {m: _bench(solved(base_k)) for m in F.BASELINE} | {m: _bench(solved(d6_k)) for m in F.D6}


def test_h3_counts_and_holm_family_of_18():
    h = F.judge_h3(_runs())
    assert len(h["tests"]) == 18 and h["n"] == 100
    t1000 = [t for t in h["tests"] if t["budget"] == 1000]
    assert all(t["only_d6"] == 50 and t["only_baseline"] == 0 for t in t1000)
    assert all(t["p_holm"] >= t["p"] for t in h["tests"])
    # at 30 expansions both groups solve the same 20 problems: no difference, not supported
    assert h["verdict"] == F.NOT_SUPPORTED
    assert [c["pass"] for c in h["counts"] if c["budget"] == 30] == [False] * 3
    runs = _runs()
    runs |= {m: _bench({f"p{i:03d}": 10 for i in range(90)}) for m in F.D6}     # d6 fast as well
    assert F.judge_h3(runs)["verdict"] == F.SUPPORTED


def test_h3_needs_every_run_on_the_same_problems():
    runs = _runs()
    runs["h-d6-s2"] = runs["h-d6-s2"][:-1]
    with pytest.raises(seeds.SeedsError, match="same problems"):
        F.judge_h3(runs)


def test_h4_is_the_matched_rules_unchanged():
    t = _tables()[F.CONTROL]
    t[7], t[8] = _row(50, 6.2, 0.6), _row(50, 6.3, 0.3)
    assert F.judge_h4(t, 0.95)["verdict"] == F.SUPPORTED
    assert F.judge_h4(t, 0.88)["verdict"] == F.NOT_SUPPORTED             # top-1 between: not "depth"
    assert F.judge_h4({5: t[5]}, 0.95)["verdict"] == F.UNDECIDABLE


# ---------------------------------------------------------------------------------- end to end

def _analysis(table, top1):
    by, ec = {}, {}
    for k, r in table.items():
        by[str(k)] = {"states": r["states"], "problems": r["states"], "mean_children": r["children"],
                      "base_rate": r["chance"], "symbol_bag": {"top1_optimal": r["symbol_bag"]},
                      "h": {"top1_optimal": r["top1"], "top1_ci95": list(r["top1_ci"]), "margin_mean": r["margin"]},
                      "calibration_onpath": {"per_dist": {str(k): {"mean_pred": r["mean_h"]}}}}
        ec[str(k)] = {"optimal_child_mean_drop": r["drop"], "violation_rate": r["violation"]}
    return {"overall": {"states": sum(r["states"] for r in table.values()),
                        "h": {"top1_optimal": top1, "top1_ci95": [top1 - 0.02, top1 + 0.02]},
                        "symbol_bag": {"top1_optimal": 0.2}},
            "by_dist": by, "edge_consistency_by_dist": ec}


def _fake(tmp_path):
    e3 = tmp_path / "runs" / "e3-fresh"
    (e3 / "data").mkdir(parents=True)
    (e3 / "labels").mkdir()
    sample = [{"id": f"p{i:03d}", "theory": "bool", "lhs": "a", "rhs": "b", "dist": 5} for i in range(100)]
    (e3 / "data" / "test_depth_fresh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sample), "utf-8")
    labels = [{"problem_id": r["id"], "i": i, "dist": 5 - i, "problem_dist": 5, "theory": "bool",
               "status": "unknown" if (r["id"] == "p000" and i == 0) else "ok", "state": f"s{i}", "goal": "b",
               "children": []} for r in sample for i in range(5)]
    lpath = e3 / "labels" / "test_depth_fresh.jsonl"
    lpath.write_text("".join(json.dumps(r) + "\n" for r in labels), "utf-8")
    tables = _tables()
    top1 = {"h-v1": 0.7, "h-v1-s1": 0.8, "h-v1-s2": 0.65, "h-d6-s0": 0.95, "h-d6-s1": 0.96, "h-d6-s2": 0.94,
            "h-d6m-s0": 0.94}
    runs = _runs()
    runs |= {m: _bench({f"p{i:03d}": 10 for i in range(90)}) for m in F.D6}
    runs[F.CONTROL] = runs["h-d6-s0"]
    for m in F.MODELS:
        p = seeds.score_paths(e3 / "scores" / m, "test_depth_fresh")
        p["meta"].parent.mkdir(parents=True)
        p["meta"].write_text(json.dumps({"labels_sha256": seeds.sha256(lpath), "ckpt_sha256": "cd" * 32,
                                         "max_memory_allocated": 3 * 2**20}), "utf-8")
        p["analysis"].write_text(json.dumps(_analysis(tables[m], top1[m])), "utf-8")
        (e3 / "bench-seeds" / m).mkdir(parents=True)
        (e3 / "bench-seeds" / m / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs[m]), "utf-8")
    base = [dict(r, method=meth) for meth in F.NON_NEURAL for r in _bench({"p000": 7})]
    (e3 / "bench-seeds" / "baselines").mkdir(parents=True)
    (e3 / "bench-seeds" / "baselines" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base),
                                                                     "utf-8")
    return e3


def test_render_states_every_verdict_and_is_deterministic(tmp_path):
    e3 = _fake(tmp_path)
    text = F.render(e3, root=tmp_path, prediction_commit="abc1234")
    assert "| H1 saturation |" in text and text.count("**supported**") == 4
    assert "(commit `abc1234`)" in text and "`runs/e3-fresh`" in text
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text           # no absolute paths
    assert "1 states in 1 problems are **unknown**" in text and "300,000" in text
    assert "| 1000 | 40 | 40 | 40 | 90 | 90 | 90 | 90 | 1 | 1 |" in text          # solved table, 9 methods
    assert "Not measured: a training-state file" in text                          # no states files in tmp
    assert F.render(e3, root=tmp_path, prediction_commit="abc1234") == text


def test_render_refuses_a_missing_model(tmp_path):
    e3 = _fake(tmp_path)
    (e3 / "bench-seeds" / "h-d6-s2" / "results.jsonl").unlink()
    with pytest.raises(seeds.SeedsError, match="h-d6-s2"):
        F.render(e3, root=tmp_path)
