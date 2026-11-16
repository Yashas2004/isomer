"""E3 seed summary: the verdict rules and the tables, on hand-made inputs.

The summary only copies and averages numbers that the long jobs wrote, so the
tests pin (a) what "appears in every seed" means and (b) that a table cell is
the number in the input file, for the right model and distance.
"""

import json

import pytest

from isomer.e3 import seeds
from isomer.e3 import seeds_summary as S


def _row(states, mean_h, drop, margin, top1=0.7, violation=0.1):
    return {"states": states, "children": 10.0, "chance": 0.2, "symbol_bag": 0.3, "mean_h": mean_h, "top1": top1,
            "top1_ci": (0.6, 0.8), "drop": drop, "margin": margin, "violation": violation}


def _saturating():
    """Shaped like h-v1: tracks distance up to 5, flat near 4.7 beyond, margin shrinking."""
    return {1: _row(300, 0.5, 1.6, -1.2), 2: _row(300, 1.7, 1.1, -0.8), 5: _row(300, 4.2, 0.6, -0.4),
            6: _row(230, 4.55, 0.3, -0.2), 7: _row(124, 4.69, 0.07, -0.15), 8: _row(64, 4.76, 0.08, -0.24),
            9: _row(15, 4.77, 0.0, -0.23), 10: _row(2, 9.0, 1.0, None)}


def test_stat():
    assert S.stat([0.2, 0.4, 0.9]) == (pytest.approx(0.5), 0.2, 0.9)


def test_verdicts_on_a_saturating_model():
    v = S.verdicts(_saturating())
    assert v["usable_dists"] == [1, 2, 5, 6, 7, 8, 9]        # k = 10 has 2 states: shown, never used
    s = v["saturation"]
    assert (s["from"], s["to"], s["holds"]) == (6, 9, True)
    assert s["slope"] == pytest.approx((4.77 - 4.55) / 3)
    assert s["max_mean_h"] == 4.77
    c = v["margin_collapse"]
    assert c["dists"] == [7, 8, 9] and c["holds"] is True
    assert c["max_drop"] == 0.08
    assert c["max_margin_ratio"] == pytest.approx(0.24 / 1.2)


def test_a_calibrated_model_is_not_called_saturated_or_collapsed():
    t = {k: _row(100, float(k), 1.0, -1.0) for k in range(1, 10)}
    v = S.verdicts(t)
    assert v["saturation"]["holds"] is False and v["saturation"]["slope"] == pytest.approx(1.0)
    assert v["margin_collapse"]["holds"] is False


def test_each_condition_of_the_rules_is_necessary():
    t = _saturating()
    t[8] = _row(64, 5.2, 0.08, -0.24)              # flat, but above the training maximum
    assert S.verdicts(t)["saturation"]["holds"] is False
    t = _saturating()
    t[9] = _row(15, 4.55 + 3 * 0.3, 0.0, -0.23)    # below 5.5 but still rising 0.3 per step
    assert S.verdicts(t)["saturation"]["holds"] is False
    t = _saturating()
    t[8] = _row(64, 4.76, 0.5, -0.24)              # one deep distance keeps half a step of drop
    assert S.verdicts(t)["margin_collapse"]["holds"] is False
    t = _saturating()
    t[7] = _row(124, 4.69, 0.07, -0.6)             # margin still half of its value at k = 1
    assert S.verdicts(t)["margin_collapse"]["holds"] is False


def test_verdicts_are_undecided_without_deep_states():
    v = S.verdicts({k: _row(100, float(k), 1.0, -1.0) for k in range(1, 5)})       # test_id
    assert v["saturation"]["holds"] is None and v["margin_collapse"]["holds"] is None


def test_unknown_by_dist():
    rows = ([{"problem_id": "a", "dist": k, "problem_dist": 3, "theory": "bool", "status": "ok"} for k in (1, 2)]
            + [{"problem_id": "a", "dist": 3, "problem_dist": 3, "theory": "bool", "status": "unknown"}]
            + [{"problem_id": "b", "dist": 1, "problem_dist": 2, "theory": "group", "status": "ok"},
               {"problem_id": "b", "dist": 2, "problem_dist": 2, "theory": "group", "status": "unknown"}]
            + [{"problem_id": "c", "dist": 1, "problem_dist": 1, "theory": "group", "status": "ok"}])
    u = S.unknown_by_dist(rows)
    assert (u["states"], u["unknown"], u["problems"], u["problems_with_unknown"]) == (6, 2, 3, 2)
    assert u["by_dist"] == {1: {"states": 3, "unknown": 0}, 2: {"states": 2, "unknown": 1},
                            3: {"states": 1, "unknown": 1}}
    assert u["by_problem_dist"] == {2: 1, 3: 1}
    assert u["by_theory"] == {"bool": 1, "group": 1}


def test_same_outcomes():
    a = [{"id": "x", "solved": True, "expansions": 5, "proof_len": 4},
         {"id": "y", "solved": False, "expansions": 9, "proof_len": None}]
    b = [dict(a[0]), dict(a[1], expansions=10), {"id": "z", "solved": True, "expansions": 1, "proof_len": 1}]
    assert S.same_outcomes(a, b) == (1, 2)


def test_format_has_no_negative_zero():
    assert S._f(-0.0001) == "0.000" and S._f(-0.0006) == "-0.001" and S._f(None) == "–" and S._f(1.25, 1) == "1.2"


# ---------------------------------------------------------------------------------- end to end

def _analysis(table):
    by, ec = {}, {}
    for k, r in table.items():
        by[str(k)] = {"states": r["states"], "problems": r["states"], "mean_children": r["children"],
                      "base_rate": r["chance"], "symbol_bag": {"top1_optimal": r["symbol_bag"]},
                      "h": {"top1_optimal": r["top1"], "top1_ci95": list(r["top1_ci"]), "margin_mean": r["margin"]},
                      "calibration_onpath": {"per_dist": {str(k): {"mean_pred": r["mean_h"]}}}}
        ec[str(k)] = {"optimal_child_mean_drop": r["drop"], "violation_rate": r["violation"]}
    return {"overall": {"states": sum(r["states"] for r in table.values()),
                        "h": {"top1_optimal": 0.7, "top1_ci95": [0.6, 0.8], "mean_best_optimal_rank": 1.9},
                        "symbol_bag": {"top1_optimal": 0.2}, "calibration_onpath": {"spearman": 0.85}},
            "by_dist": by, "edge_consistency_by_dist": ec, "edge_consistency": {"violation_rate": 0.45}}


def _fake_e3(tmp_path, tables):
    e3 = tmp_path / "e3"
    for split in ("test_depth", "test_id"):
        (e3 / "data").mkdir(parents=True, exist_ok=True)
        (e3 / "labels").mkdir(exist_ok=True)
        (e3 / "data" / f"{split}.jsonl").write_text(
            json.dumps({"id": "p", "dist": 2}) + "\n" + json.dumps({"id": "q", "dist": 1}) + "\n", encoding="utf-8")
        labels = [{"problem_id": "p", "dist": 2, "problem_dist": 2, "theory": "bool", "status": "unknown"},
                  {"problem_id": "p", "dist": 1, "problem_dist": 2, "theory": "bool", "status": "ok"},
                  {"problem_id": "q", "dist": 1, "problem_dist": 1, "theory": "group", "status": "ok"}]
        lpath = e3 / "labels" / f"{split}.jsonl"
        lpath.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
        for m, table in tables.items():
            p = seeds.score_paths(e3 / "scores" / m, split)
            p["meta"].parent.mkdir(parents=True, exist_ok=True)
            p["meta"].write_text(json.dumps({"labels_sha256": seeds.sha256(lpath), "ckpt_sha256": "ab" * 32,
                                             "max_memory_allocated": 100 * 2**20, "precision": "bf16-fast"}),
                                 encoding="utf-8")
            p["analysis"].write_text(json.dumps(_analysis(table)), encoding="utf-8")
    for i, m in enumerate(tables):
        run = e3 / "bench-seeds" / m
        run.mkdir(parents=True)
        rows = [{"id": pid, "method": "neural", "status": "ok", "solved": j <= i, "expansions": 20 if j <= i else 1000,
                 "max_budget": 1000, "stop": "solved" if j <= i else "budget", "proof_len": 2 if j <= i else None,
                 "nodes_stored": 5, "dist": 2} for j, pid in enumerate(("p", "q"))]
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (run / "config.json").write_text(json.dumps({"git_commit": "abcdef0123", "git_dirty": False}),
                                         encoding="utf-8")
    return e3


def test_render_copies_the_inputs_and_states_the_verdicts(tmp_path):
    other = _saturating()
    other[6] = _row(230, 4.40, 0.35, -0.3, top1=0.5)
    models = {"m0": "runs/m0/best.pt", "m1": "runs/m1/best.pt"}
    e3 = _fake_e3(tmp_path, {"m0": _saturating(), "m1": other})
    text = S.render(e3, models)
    assert "**Saturation near the training maximum appears in every one of the 2 models.**" in text
    assert "**The margin collapse appears in every one of the 2 models.**" in text
    # k = 6 row of the mean-prediction table: both models, their mean and range
    assert "| 6 | 230 | 4.550 | 4.400 | 4.475 | 4.400 to 4.550 |" in text
    # top-1 row carries the model-independent columns
    assert "| 6 | 230 | 10.0 | 0.200 | 0.300 | 0.700 | 0.500 | 0.600 | 0.500 to 0.700 |" in text
    assert "| 10 \\* | 2 |" in text                                     # too few states: marked
    # unknown states: 1 of 3, at distance 2
    assert "1 states in 1 problems" in text and "| 2 | 1 | 1 | 100.0% | 0 |" in text
    # search: m0 solves 1 of 2, m1 both, at every budget
    assert "| 30 | 1/2 = 50.0% | 2/2 = 100.0% | 75.0% | 50.0% to 100.0% |" in text
    assert S.render(e3, models) == text                                 # no timestamps


def test_render_says_so_when_a_seed_does_not_saturate(tmp_path):
    calibrated = {k: _row(r["states"], float(k), 1.0, -1.0) for k, r in _saturating().items()}
    models = {"m0": "runs/m0/best.pt", "m1": "runs/m1/best.pt"}
    text = S.render(_fake_e3(tmp_path, {"m0": _saturating(), "m1": calibrated}), models)
    assert "Saturation near the training maximum does NOT appear in every model: it fails the rule for m1." in text
    assert "The margin collapse does NOT appear in every model: it fails the rule for m1." in text


def test_render_refuses_scores_of_another_label_file(tmp_path):
    models = {"m0": "runs/m0/best.pt"}
    e3 = _fake_e3(tmp_path, {"m0": _saturating()})
    with open(e3 / "labels" / "test_depth.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")                                                   # labels changed after scoring
    with pytest.raises(seeds.SeedsError, match="another label file"):
        S.render(e3, models)


# ---------------------------------------------------------------------------------- E3-d6 comparison

from isomer.e3 import seeds_compare as C


def _shifted():
    """What the prediction expects of a d6 model: the saturating curve moved by two distances."""
    t = _saturating()
    t[7] = _row(124, 6.2, 0.6, -0.4)
    t[8] = _row(64, 6.5, 0.3, -0.2)
    t[9] = _row(15, 6.7, 0.1, -0.2)
    return t


def test_rules_are_the_committed_prediction():
    assert C.PREDICTIONS == (("mean prediction", "mean_h", 7, 6.2, 5.7, 5.0),
                             ("mean prediction", "mean_h", 8, 6.5, 5.7, 5.0),
                             ("drop to the optimal child", "drop", 7, 0.6, 0.35, 0.20),
                             ("drop to the optimal child", "drop", 8, 0.3, 0.18, 0.10))
    assert C.PLATEAU_BELOW == 7.0
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "docs/experiments/E3/d6_prediction.md").read_text("utf-8")
    for _, _, _, point, sup, ref in C.PREDICTIONS:      # the doc states every threshold the code applies
        assert f"about {point}" in text and f">= {sup}" in text and f"<= {ref}" in text


def test_judge_boundaries():
    assert C.judge(5.7, 5.7, 5.0) == "supported" and C.judge(5.69, 5.7, 5.0) == "partial"
    assert C.judge(5.0, 5.7, 5.0) == "refuted" and C.judge(5.01, 5.7, 5.0) == "partial"


def test_evaluate_supported_refuted_and_mixed():
    base = {"a": _saturating(), "b": _saturating()}
    ev = C.evaluate(_shifted(), base)
    assert ev["overall"] == "supported" and [r["verdict"] for r in ev["rows"]] == ["supported"] * 4
    assert ev["plateau"] == {"dists": [7, 8, 9], "max_mean_h": 6.7, "holds": True}
    assert (ev["rows"][0]["baseline_min"], ev["rows"][0]["baseline_max"]) == (4.69, 4.69)
    # a d6 model that behaves like the baseline refutes it
    ev = C.evaluate(_saturating(), base)
    assert ev["overall"] == "refuted" and {r["verdict"] for r in ev["rows"]} == {"refuted"}
    # mean prediction moves but the drop does not: mixed
    t = _shifted()
    t[7] = _row(124, 6.2, 0.1, -0.1)
    ev = C.evaluate(t, base)
    assert ev["overall"] == "mixed" and [r["verdict"] for r in ev["rows"]] == [
        "supported", "supported", "refuted", "supported"]
    # no saturation at all (prediction keeps rising past 7) is not "supported" either
    t = _shifted()
    t[8] = _row(64, 7.6, 0.9, -0.5)
    ev = C.evaluate(t, base)
    assert ev["plateau"]["holds"] is False and ev["overall"] == "mixed"


def test_solved_by_dist():
    rows = [{"dist": 7, "status": "ok", "solved": True, "expansions": 20, "max_budget": 1000, "stop": "solved"},
            {"dist": 7, "status": "ok", "solved": True, "expansions": 400, "max_budget": 1000, "stop": "solved"},
            {"dist": 5, "status": "ok", "solved": False, "expansions": 1000, "max_budget": 1000, "stop": "budget"}]
    assert C.solved_by_dist(rows) == {5: {"n": 1, 30: 0, 100: 0, 300: 0, 1000: 0},
                                      7: {"n": 2, 30: 1, 100: 1, 300: 1, 1000: 2}}


def test_bootstrap_cis_on_constant_scores():
    rows = [{"problem_id": p, "dist": 7, "h_state": 6.0, "h_children": [5.5, 6.5], "optimal": [True, False]}
            for p in "abc"]
    ci = C.bootstrap_cis(rows, dists=(7,))
    assert ci[("mean_h", 7)] == (6.0, 6.0) and ci[("drop", 7)] == (0.5, 0.5)


def test_render_comparison(tmp_path):
    tables = {"m0": _saturating(), "m1": _saturating(), "d6": _shifted()}
    e3 = _fake_e3(tmp_path, tables)
    scores = [{"problem_id": p, "dist": k, "h_state": tables["d6"][k]["mean_h"],
               "h_children": [tables["d6"][k]["mean_h"] - tables["d6"][k]["drop"]], "optimal": [True]}
              for k in (7, 8) for p in "ab"]
    seeds.score_paths(e3 / "scores" / "d6", "test_depth")["scores"].write_text(
        "".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")
    base = {"m0": "runs/m0/best.pt", "m1": "runs/m1/best.pt"}
    text = C.render(e3, base, "d6", "runs/d6/best.pt", prediction_commit="abc1234")
    assert "**The prediction is supported on every pre-registered quantity.**" in text
    assert "committed as `abc1234`" in text
    assert "| mean prediction, k = 7 | 4.690 to 4.690 | about 6.2 | 6.200 (6.200 to 6.200) | >= 5.7 / <= 5.0 | supported |" in text
    # per-distance row: baseline mean, range, new model, difference
    assert "| 7 | 124 | 4.690 | 4.690 to 4.690 | 6.200 | +1.510 |" in text
    assert "| 10 \\* | 2 | – | – | – | – |" in text                     # margin undefined there
    assert C.render(e3, base, "d6", "runs/d6/best.pt", prediction_commit="abc1234") == text
    with pytest.raises(seeds.SeedsError):
        C.render(e3, base, "m0", "runs/m0/best.pt")
