"""E3-d6: compare one new model with the baseline seeds, against a prediction made in advance.

    python scripts/e3_seeds_summary.py --compare h-d6-s0=runs/h-d6-s0/best.pt

The question: is the saturation of h-v1 (mean prediction flat just below 5, the
largest training-state distance) caused by the training range? h-d6 is the same
configuration trained on problems up to distance 6 (states up to 7).

``PREDICTIONS`` and ``PLATEAU_BELOW`` are the decision rules of
``docs/experiments/E3/d6_prediction.md``, committed before any d6 training
state existed. They are constants here, not arguments, so the verdict cannot be
tuned after the run. Everything is read from the outputs of the E3 scripts; no
model is run.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import seeds
from . import seeds_summary as S

# (quantity, key in the per-distance table, distance k, point prediction, supported if >=, refuted if <=)
PREDICTIONS = (
    ("mean prediction", "mean_h", 7, 6.2, 5.7, 5.0),
    ("mean prediction", "mean_h", 8, 6.5, 5.7, 5.0),
    ("drop to the optimal child", "drop", 7, 0.6, 0.35, 0.20),
    ("drop to the optimal child", "drop", 8, 0.3, 0.18, 0.10),
)
PLATEAU_BELOW = 7.0     # largest distance of a d6 training state: the new model should saturate under it
PLATEAU_FROM = 7


def judge(value: float, supported_ge: float, refuted_le: float) -> str:
    if value >= supported_ge:
        return "supported"
    if value <= refuted_le:
        return "refuted"
    return "partial"


def evaluate(new: dict[int, dict], baseline: dict[str, dict[int, dict]]) -> dict:
    """Apply the pre-registered rules to the new model's per-distance table."""
    rows = []
    for name, key, k, point, sup, ref in PREDICTIONS:
        base = [t[k][key] for t in baseline.values()]
        rows.append({"quantity": name, "key": key, "k": k, "predicted": point, "supported_ge": sup,
                     "refuted_le": ref, "baseline_min": min(base), "baseline_max": max(base),
                     "value": new[k][key], "verdict": judge(new[k][key], sup, ref)})
    deep = [k for k, r in new.items() if k >= PLATEAU_FROM and r["states"] >= S.MIN_STATES]
    top = max(new[k]["mean_h"] for k in deep)
    plateau = {"dists": deep, "max_mean_h": top, "holds": top < PLATEAU_BELOW}
    verdicts = {r["verdict"] for r in rows}
    if verdicts == {"supported"} and plateau["holds"]:
        overall = "supported"
    elif verdicts == {"refuted"}:
        overall = "refuted"
    else:
        overall = "mixed"
    return {"rows": rows, "plateau": plateau, "overall": overall}


def bootstrap_cis(score_rows: list[dict], dists: tuple[int, ...] = (7, 8)) -> dict[tuple[str, int], tuple[float, float]]:
    """95% cluster-bootstrap intervals (over problems) of the mean prediction and of the mean
    drop to an optimal child, at each distance, for one model's score rows."""
    from .diagnose import bootstrap_mean_ci

    out = {}
    for k in dists:
        rs = [r for r in score_rows if r["dist"] == k]
        out[("mean_h", k)] = bootstrap_mean_ci([r["h_state"] for r in rs], [r["problem_id"] for r in rs])
        drops, groups = [], []
        for r in rs:
            for v, o in zip(r["h_children"], r["optimal"], strict=True):
                if o:
                    drops.append(r["h_state"] - v)
                    groups.append(r["problem_id"])
        out[("drop", k)] = bootstrap_mean_ci(drops, groups)
    return out


def solved_by_dist(rows: list[dict], budgets: tuple[int, ...] = seeds.BUDGETS) -> dict[int, dict]:
    """{problem distance: {"n": problems, budget: solved}} from one bench run."""
    from ..bench.run import cell

    out: dict[int, dict] = {}
    for r in rows:
        d = out.setdefault(r["dist"], {"n": 0, **{b: 0 for b in budgets}})
        d["n"] += 1
        for b in budgets:
            d[b] += cell(r, b) == "solved"
    return dict(sorted(out.items()))


def compare_table(base: dict[str, dict[int, dict]], new: dict[int, dict], key: str, new_name: str) -> str:
    rows = []
    for k, ref in new.items():
        vals = [t[k][key] for t in base.values()]
        mark = "" if ref["states"] >= S.MIN_STATES else " \\*"
        if ref[key] is None or any(v is None for v in vals):
            rows.append([f"{k}{mark}", str(ref["states"]), "–", "–", "–", "–"])
            continue
        mean, lo, hi = S.stat(vals)
        rows.append([f"{k}{mark}", str(ref["states"]), S._f(mean), f"{S._f(lo)} to {S._f(hi)}", S._f(ref[key]),
                     f"{ref[key] - mean:+.3f}"])
    return S._table(["k", "states", "baseline mean", "baseline range", new_name, "difference"], rows)


def _train_facts(ckpt: str) -> dict | None:
    path = Path(ckpt).parent / "config.json"
    if not path.exists():
        return None
    c = json.loads(path.read_text(encoding="utf-8"))
    return {"states_dir": c["args"]["states"], "seed": c["args"]["seed"], "train_states": c["data"]["train_states"],
            "train_problems": c["data"]["train_problems"], "total_steps": c["total_steps"]}


def render(e3: str | Path, baseline: dict[str, str], new_name: str, new_ckpt: str,
           prediction_commit: str | None = None) -> str:
    e3 = Path(e3)
    base_names = list(baseline)
    if new_name in baseline:
        raise seeds.SeedsError(f"{new_name} is one of the baseline models")
    tables, overall, _, labels, bench, _, bsum = S.load(e3, [*base_names, new_name])
    td = tables["test_depth"]
    base_td = {m: td[m] for m in base_names}
    ev = evaluate(td[new_name], base_td)
    ci = bootstrap_cis(S._jsonl(seeds.score_paths(e3 / "scores" / new_name, "test_depth")["scores"]))
    lab = labels["test_depth"]
    n_base = len(base_names)

    out = [f"# E3-d6: does training on deeper problems move the saturation? ({new_name})", "",
           (f"`{new_name}` has the h-v1 configuration and is trained on train problems of exact distance up to 6 "
            f"(training states up to 7). The baseline is {', '.join(base_names)}, trained on distance up to 4 "
            "(states up to 5). All models are scored on the same labelled test_depth states "
            f"({overall['test_depth'][new_name]['overall']['states']} scored, {lab['unknown']} unknown and "
            "excluded) and searched on the same 300 problems."), "",
           ("The prediction and its decision rules are in `docs/experiments/E3/d6_prediction.md`"
            + (f", committed as `{prediction_commit}`" if prediction_commit else "")
            + " before any d6 training state was generated."), "",
           "## Verdict", ""]
    word = {"supported": "**The prediction is supported on every pre-registered quantity.**",
            "refuted": "**The prediction is refuted on every pre-registered quantity.**",
            "mixed": "**The result is mixed: see the rows below.**"}[ev["overall"]]
    out += [word, ""]
    rows = []
    for r in ev["rows"]:
        lo, hi = ci[(r["key"], r["k"])]
        rows.append([f"{r['quantity']}, k = {r['k']}", f"{S._f(r['baseline_min'])} to {S._f(r['baseline_max'])}",
                     f"about {r['predicted']}", f"{S._f(r['value'])} ({S._f(lo)} to {S._f(hi)})",
                     f">= {r['supported_ge']} / <= {r['refuted_le']}", r["verdict"]])
    p = ev["plateau"]
    rows.append([f"plateau: highest mean prediction at k >= {PLATEAU_FROM}", "below 5", "between 6 and 7",
                 S._f(p["max_mean_h"]), f"below {PLATEAU_BELOW:g}", "holds" if p["holds"] else "does not hold"])
    out += [S._table(["quantity", f"baseline range ({n_base} seeds)", "predicted", f"{new_name} (95% CI)",
                      "supported / refuted if", "verdict"], rows), "",
            ("The interval is a cluster bootstrap over problems within the one d6 model; it does not include "
             f"seed-to-seed variation, for which the only yardstick is the baseline range over {n_base} seeds."), ""]

    out += ["## test_depth, by true distance of the state", "",
            (f"Distances 5-6 are in-distribution by distance for {new_name} and out of range for the baseline; "
             "the test is at 7-8. Rows marked \\* have fewer than "
             f"{S.MIN_STATES} scored states. \"difference\" is {new_name} minus the baseline mean."), ""]
    for title, key in (("Mean prediction", "mean_h"), ("Drop to the optimal child", "drop"),
                       ("All-children top-1 (exploratory)", "top1"), ("Margin (exploratory)", "margin"),
                       ("Edge-violation rate (exploratory)", "violation")):
        out += [f"### {title}", "", compare_table(base_td, td[new_name], key, new_name), ""]
    ti = tables["test_id"]
    out += ["## test_id (in distribution for every model), by true distance", ""]
    for title, key in (("Mean prediction", "mean_h"), ("All-children top-1", "top1"),
                       ("Drop to the optimal child", "drop")):
        out += [f"### {title}", "", compare_table({m: ti[m] for m in base_names}, ti[new_name], key, new_name), ""]

    # ---- search
    rows = []
    for b in seeds.BUDGETS:
        rates = [bsum[m]["budgets"][b]["rate"] for m in base_names]
        mean, lo, hi = S.stat(rates)
        c = bsum[new_name]["budgets"][b]
        rows.append([str(b), f"{100 * mean:.1f}%", f"{100 * lo:.1f}% to {100 * hi:.1f}%",
                     f"{c['solved']}/{c['n']} = {100 * c['rate']:.1f}%"])
    out += ["## Search: neural-w1 on the 300 test_depth problems (exploratory)", "",
            "Solved means the kernel accepted the proof.", "",
            S._table(["expansions", "baseline mean", "baseline range", new_name], rows), ""]
    by = {m: solved_by_dist(bench[m]) for m in [*base_names, new_name]}
    rows = []
    for d, cellv in by[new_name].items():
        row = [str(d), str(cellv["n"])]
        for b in seeds.BUDGETS:
            base = [by[m][d][b] for m in base_names]
            row += [f"{min(base)} to {max(base)}", str(cellv[b])]
        rows.append(row)
    head = ["problem distance", "problems"]
    for b in seeds.BUDGETS:
        head += [f"baseline @{b}", f"{new_name} @{b}"]
    bad = {m: {k: c for k, c in bsum[m]["status"].items() if k != "ok"} for m in [*base_names, new_name]}
    out += ["Problems solved, by exact distance of the problem (baseline: range over seeds):", "",
            S._table(head, rows), "",
            "Rows with status other than ok: " + ("none" if not any(bad.values()) else json.dumps(bad)) + ".", ""]

    # ---- what differs in training
    facts = {m: _train_facts(c) for m, c in {**baseline, new_name: new_ckpt}.items()}
    if all(facts.values()):
        out += ["## Training runs", "",
                S._table(["model", "states directory", "seed", "train problems", "train states", "optimizer steps"],
                         [[m, f"`{f['states_dir']}`", str(f["seed"]), str(f["train_problems"]),
                           str(f["train_states"]), str(f["total_steps"])] for m, f in facts.items()]), ""]
    out += ["## What this comparison does not control", "",
            (f"- **Amount of data.** {new_name} saw more problems, more states and more optimizer steps than the "
             "baseline (table above), not only deeper ones. A size-matched control was not run."),
            f"- **One seed** of {new_name} against {n_base} baseline seeds.",
            ("- **Dropped training states.** `isomer states` drops an off-path state whose BFS label exceeds its "
             "budget; see `dropped` in the states directory's `stats.json`."),
            ("- **Train and test pools differ in generation.** Distance 5-6 train problems come from walks of "
             "length at most 6; test_depth problems from walks of 6-24."),
            ("- Diagnostics use on-path states of one stored shortest proof; unknown states are excluded for "
             "every model alike."), ""]
    return "\n".join(out)


def write(e3: str | Path, out: str | Path, baseline: dict[str, str], new_name: str, new_ckpt: str,
          prediction_commit: str | None = None) -> Path:
    Path(out).write_text(render(e3, baseline, new_name, new_ckpt, prediction_commit), encoding="utf-8", newline="\n")
    return Path(out)
