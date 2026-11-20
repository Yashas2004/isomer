"""E3-d5: judge the pre-registered rules for h-d5-s0, a model trained on problems up to distance 5.

    python scripts/e3_d5_summary.py        # -> runs/e3/d5_summary.md

Question: does the ceiling of the predicted distance track the training range? Baseline models
(problems up to 4, training states up to 5) level off near 4.7; d6 models (up to 6 and 7) near 6.6.
h-d5-s0 sits between them (problems up to 5, states up to 6).

The constants below are the decision rules of ``docs/experiments/E3/d5_prediction.md``, committed
before any d5 training state existed. They are constants, not arguments, so a verdict cannot be
tuned after the run. The rules are judged on the E2 test_depth sample; the fresh sample is
reported, not tested. Everything is read from the outputs of the E3 scripts; no model is run.
One d5 seed: nothing here is a seed range.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import seeds
from . import seeds_compare as C
from . import seeds_summary as S

D5 = "h-d5-s0"
BASELINE = ("h-v1", "h-v1-s1", "h-v1-s2")
D6 = ("h-d6-s0", "h-d6-s1", "h-d6-s2")
CONTROL = "h-d6m-s0"
COMPARED = (*BASELINE, *D6, CONTROL)

# H-d5a: mean prediction at k = 7 and 8 strictly between the highest baseline value (any seed, k = 7 or 8,
# E2 sample) and the lowest d6 seed's value at that k.
H_A_DISTS = (7, 8)
H_A_LOWER = 4.764
H_A_UPPER = {7: 6.186, 8: 6.577}
# H-d5b: the highest mean prediction over k >= 6 (distances with at least S.MIN_STATES scored states).
H_B_FROM = 6
H_B_RANGE = (5.2, 6.2)          # inclusive; point prediction about 5.7
# H-d5c: all-children top-1 over all scored states, strictly between the best baseline seed and the lowest d6 seed.
H_C_LOWER, H_C_UPPER = 0.800, 0.948

SUPPORTED, NOT_SUPPORTED, UNDECIDABLE = "supported", "not supported", "not decidable"


def ceiling(table: dict[int, dict]) -> tuple[float, int] | None:
    """(highest mean prediction, its distance) over k >= H_B_FROM with enough scored states."""
    beyond = {k: r["mean_h"] for k, r in table.items() if k >= H_B_FROM and r["states"] >= S.MIN_STATES}
    if not beyond:
        return None
    k = max(beyond, key=lambda x: beyond[x])
    return beyond[k], k


def judge(table: dict[int, dict], top1: float) -> dict:
    """The three rules for one model's per-distance table and overall top-1. A rule whose distance has
    fewer than ``S.MIN_STATES`` scored states is not decidable (``pass`` is None)."""
    rows = []
    for k in H_A_DISTS:
        r = table.get(k)
        ok = None if r is None or r["states"] < S.MIN_STATES else H_A_LOWER < r["mean_h"] < H_A_UPPER[k]
        rows.append({"rule": f"H-d5a: mean prediction at k = {k}", "needs": f"{H_A_LOWER} < value < {H_A_UPPER[k]}",
                     "value": None if r is None else r["mean_h"], "pass": ok})
    top = ceiling(table)
    lo, hi = H_B_RANGE
    rows.append({"rule": f"H-d5b: highest mean prediction at k >= {H_B_FROM}", "needs": f"{lo} <= value <= {hi}",
                 "value": None if top is None else top[0], "at": None if top is None else top[1],
                 "pass": None if top is None else lo <= top[0] <= hi})
    rows.append({"rule": "H-d5c: all-children top-1, all scored states", "needs": f"{H_C_LOWER:.3f} < value < {H_C_UPPER:.3f}",
                 "value": top1, "pass": H_C_LOWER < top1 < H_C_UPPER})
    passes = [r["pass"] for r in rows]
    verdict = NOT_SUPPORTED if False in passes else UNDECIDABLE if None in passes else SUPPORTED
    return {"rows": rows, "verdict": verdict}


def position(value: float, baseline: list[float], d6: list[float]) -> str:
    """Where a value falls relative to the baseline seeds and the d6 seeds (both rise in this order)."""
    if value <= max(baseline):
        return "at or below the baseline range"
    return "between baseline and d6" if value < min(d6) else "at or above the d6 range"


def first_commit(root: Path, path: str) -> str | None:
    """The first commit that added ``path`` (later commits may only append)."""
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", path], cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out[-1] if out else None


# ---------------------------------------------------------------------------------- inputs

def load(e3: Path, split: str) -> dict:
    """Per-distance tables, overall analyses, score metadata and bench summaries of every model on one
    labelled sample. Refuses incomplete labels, scores made on another label file, missing outputs."""
    labels = e3 / "labels" / f"{split}.jsonl"
    seeds.require_complete(e3 / "data" / f"{split}.jsonl", labels)
    label_hash = seeds.sha256(labels)
    names = [*COMPARED, D5]
    missing = [m for m in names if not seeds.score_paths(e3 / "scores" / m, split)["analysis"].is_file()
               or not (e3 / "bench-seeds" / m / "results.jsonl").is_file()]
    if missing:
        raise seeds.SeedsError(f"{e3.as_posix()}: no scores or bench results for {', '.join(missing)}")
    tables, overall, metas, bsum = {}, {}, {}, {}
    for m in names:
        p = seeds.score_paths(e3 / "scores" / m, split)
        metas[m] = json.loads(p["meta"].read_text(encoding="utf-8"))
        if metas[m]["labels_sha256"] != label_hash:
            raise seeds.SeedsError(f"{p['scores']} was scored on another label file; rerun e3_score_seeds.py")
        a = json.loads(p["analysis"].read_text(encoding="utf-8"))
        tables[m], overall[m] = S.by_dist(a), a["overall"]
        bsum[m] = S.bench_summary(S._jsonl(e3 / "bench-seeds" / m / "results.jsonl"))
    S._check_same_population(tables)
    return {"tables": tables, "overall": overall, "metas": metas, "bench": bsum,
            "labels": S.unknown_by_dist(S._jsonl(labels)), "label_hash": label_hash,
            "n_problems": len(S._jsonl(e3 / "data" / f"{split}.jsonl"))}


# ---------------------------------------------------------------------------------- rendering

def _yes(ok: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "not decidable"}[ok]


def by_k_table(tables: dict[str, dict[int, dict]]) -> str:
    """Mean prediction by true distance: baseline and d6 as mean and range over their seeds, then h-d5-s0 and the control."""
    rows = []
    for k, r in tables[D5].items():
        base = [tables[m][k]["mean_h"] for m in BASELINE]
        d6 = [tables[m][k]["mean_h"] for m in D6]
        rows.append([f"{k}" + ("" if r["states"] >= S.MIN_STATES else " \\*"), str(r["states"]),
                     S._f(S.stat(base)[0]), C._range(base), S._f(r["mean_h"]),
                     S._f(S.stat(d6)[0]), C._range(d6), S._f(tables[CONTROL][k]["mean_h"])])
    return S._table(["k", "states", "baseline mean", "baseline range", D5, "d6 mean", "d6 range", CONTROL], rows)


def _quantities(d: dict) -> dict[str, dict[str, float]]:
    """k = 7, k = 8, ceiling and top-1 for every model."""
    out: dict[str, dict[str, float]] = {}
    for m in d["tables"]:
        t = d["tables"][m]
        top = ceiling(t)
        out[m] = {"mean prediction at k = 7": t[7]["mean_h"], "mean prediction at k = 8": t[8]["mean_h"],
                  f"highest mean prediction at k >= {H_B_FROM}": top[0] if top else float("nan"),
                  "all-children top-1": d["overall"][m]["h"]["top1_optimal"]}
    return out


def _sample_section(title: str, d: dict) -> list[str]:
    lab, n_problems = d["labels"], d["n_problems"]
    scored = d["overall"][D5]["states"]
    q = _quantities(d)
    rows = []
    for name in q[D5]:
        base, d6 = [q[m][name] for m in BASELINE], [q[m][name] for m in D6]
        rows.append([name, C._range(base), S._f(q[D5][name]), C._range(d6), S._f(q[CONTROL][name]),
                     position(q[D5][name], base, d6)])
    return [f"## {title}", "",
            (f"All {n_problems} problems are labelled ({lab['states']} on-path states); {lab['unknown']} are unknown "
             f"(child labelling above 300,000 nodes) and excluded, leaving {scored} scored states."), "",
            "### Mean prediction by true distance k", "", ("A calibrated distance would equal k. Rows marked \\* have "
            f"fewer than {S.MIN_STATES} scored states."), "", by_k_table(d["tables"]), "",
            "### The quantities of the rules", "",
            S._table(["quantity", "baseline range (3 seeds)", D5, "d6 range (3 seeds)", CONTROL, f"{D5} is"], rows), ""]


def _search_section(d: dict) -> list[str]:
    rows = []
    for b in seeds.BUDGETS:
        cells = {m: d["bench"][m]["budgets"][b] for m in d["bench"]}
        base, d6 = [cells[m]["solved"] for m in BASELINE], [cells[m]["solved"] for m in D6]
        rows.append([str(b), C._range(base, 0), str(cells[D5]["solved"]), C._range(d6, 0), str(cells[CONTROL]["solved"])])
    n = d["bench"][D5]["problems"]
    return [f"### Search: neural-w1, solved of {n} (kernel-accepted; exploratory, no rule)", "",
            S._table(["expansions", "baseline range", D5, "d6 range", CONTROL], rows), ""]


def render(e2: str | Path = "runs/e3", fresh: str | Path = "runs/e3-fresh",
           prediction_commit: str | None = None) -> str:
    e2, fresh = Path(e2), Path(fresh)
    a, f = load(e2, "test_depth"), load(fresh, "test_depth_fresh")
    res = judge(a["tables"][D5], a["overall"][D5]["h"]["top1_optimal"])
    sha = a["metas"][D5]["ckpt_sha256"]
    out = ["# E3-d5: does the ceiling track the training range? (h-d5-s0)", "",
           ("h-d5-s0 has the h-v1 configuration, seed 0, trained on problems up to exact distance 5 (training "
            "states up to 6): between the baseline (problems up to 4, states up to 5) and d6 (up to 6, states up "
            "to 7). **One seed**: nothing below is a seed range. The rules are those of "
            "`docs/experiments/E3/d5_prediction.md`"
            + (f" (commit `{prediction_commit}`)" if prediction_commit else "")
            + ", committed before any d5 training state existed, and are judged on the E2 test_depth sample "
            "(`runs/e3`). The fresh sample (`runs/e3-fresh`) is reported, not tested."), "",
           "## Verdict", "", f"**{res['verdict']}** (all three rules must hold).", "",
           S._table(["rule", "requirement", D5, "holds"],
                    [[r["rule"] + ("" if r.get("at") is None else f" (at k = {r['at']})"), r["needs"],
                      S._f(r["value"]), _yes(r["pass"])] for r in res["rows"]]), ""]
    out += _sample_section("test_depth, E2 sample (tested)", a) + _search_section(a)
    out += _sample_section("test_depth_fresh, fresh sample (reported, not tested)", f) + _search_section(f)
    out += ["## Provenance", "", (f"{D5} checkpoint sha256 `{sha[:16]}…`; label files `{a['label_hash'][:16]}…` (E2) "
            f"and `{f['label_hash'][:16]}…` (fresh)."), "",
            "## Limits", "",
            "- One d5 seed. The baseline and d6 ranges are over three seeds; one run cannot be placed inside a seed spread.",
            ("- d5 is not size-matched: it has more training problems than the baseline and fewer than d6, so range "
             "and data volume both increase from baseline to d5 to d6. The d6 control (`d6_matched_summary.md`) "
             "separated the two for d6 only."),
            "- Unknown states are excluded from the per-distance quantities and top-1; the deepest rows are small.", ""]
    return "\n".join(out)


def write(out: str | Path, **kw) -> Path:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render(**kw), encoding="utf-8", newline="\n")
    return Path(out)
