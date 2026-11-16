"""E3 seed study: one markdown summary of the diagnostics of h-v1 and its reseeded twins.

    python scripts/e3_seeds_summary.py        # -> runs/e3/seeds_summary.md

Reads only files written by ``scripts/e3_label.py``, ``e3_score_seeds.py`` and
``e3_bench_seeds.py``; every number in the summary is copied or averaged from
them, nothing is recomputed from a model. The output has no timestamps, so the
same inputs give the same file.

The two verdicts (saturation, margin collapse) are decided by the fixed rules
in ``verdicts`` and the rules are printed next to them, so a reader can see
what "appears in every seed" was taken to mean. With three models the spread
is reported as a range, not as a confidence interval.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import seeds

TRAIN_MAX_DIST = 5      # largest exact distance of a training state (report, section 2)
MIN_STATES = 10         # distances with fewer labelled states are shown but not used in a verdict
FLAT_SLOPE = 0.25       # saturation: mean prediction rises by less than this per unit of true distance
SMALL_DROP = 0.25       # margin collapse: h(state) - h(optimal child) below this (a true distance gives 1)
MARGIN_RATIO = 0.30     # margin collapse: |margin| at depth below this fraction of |margin| at distance 1


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat(values: list[float]) -> tuple[float, float, float]:
    """(mean, min, max) across models."""
    return sum(values) / len(values), min(values), max(values)


# ---------------------------------------------------------------------------------- extraction

def by_dist(analysis: dict) -> dict[int, dict]:
    """The per-distance quantities of one ``analyze base`` output, keyed by true distance k."""
    out: dict[int, dict] = {}
    for key, b in analysis["by_dist"].items():
        e = analysis["edge_consistency_by_dist"][key]
        out[int(key)] = {
            "states": b["states"], "children": b["mean_children"], "chance": b["base_rate"],
            "symbol_bag": b["symbol_bag"]["top1_optimal"],
            "mean_h": b["calibration_onpath"]["per_dist"][key]["mean_pred"],
            "top1": b["h"]["top1_optimal"], "top1_ci": tuple(b["h"]["top1_ci95"]),
            "drop": e["optimal_child_mean_drop"], "margin": b["h"]["margin_mean"],
            "violation": e["violation_rate"],
        }
    return dict(sorted(out.items()))


def unknown_by_dist(label_rows: list[dict]) -> dict:
    """Unknown (node cap hit) on-path states, by the state's own distance."""
    total = Counter(r["dist"] for r in label_rows)
    unk = [r for r in label_rows if r["status"] != "ok"]
    n_unk = Counter(r["dist"] for r in unk)
    return {"states": len(label_rows), "unknown": len(unk),
            "problems": len({r["problem_id"] for r in label_rows}),
            "problems_with_unknown": len({r["problem_id"] for r in unk}),
            "by_dist": {k: {"states": total[k], "unknown": n_unk.get(k, 0)} for k in sorted(total)},
            "by_problem_dist": dict(sorted(Counter(r["problem_dist"] for r in unk).items())),
            "by_theory": dict(sorted(Counter(r["theory"] for r in unk).items()))}


def verdicts(table: dict[int, dict]) -> dict:
    """Saturation and margin collapse for one model, from its per-distance table.

    Saturation: beyond the training range the mean prediction stays below the training
    maximum and is nearly flat. Margin collapse: at distance >= 7 the predicted drop to the
    optimal child is a small fraction of a step and the margin over the best non-optimal
    child is a small fraction of its value next to the goal."""
    usable = [k for k, r in table.items() if r["states"] >= MIN_STATES]
    beyond = [k for k in usable if k > TRAIN_MAX_DIST]
    deep = [k for k in usable if k >= 7]
    out: dict = {"usable_dists": usable}
    if len(beyond) >= 2:
        lo, hi = beyond[0], beyond[-1]
        slope = (table[hi]["mean_h"] - table[lo]["mean_h"]) / (hi - lo)
        top = max(table[k]["mean_h"] for k in beyond)
        out["saturation"] = {"from": lo, "to": hi, "slope": slope, "max_mean_h": top,
                             "holds": top < TRAIN_MAX_DIST and slope < FLAT_SLOPE}
    else:
        out["saturation"] = {"holds": None}
    if deep and 1 in table and all(table[k]["margin"] is not None for k in [1, *deep]):
        max_drop = max(table[k]["drop"] for k in deep)
        ratio = max(abs(table[k]["margin"]) for k in deep) / abs(table[1]["margin"])
        out["margin_collapse"] = {"dists": deep, "drop_at_1": table[1]["drop"], "max_drop": max_drop,
                                  "margin_at_1": table[1]["margin"], "max_margin_ratio": ratio,
                                  "holds": table[1]["drop"] > 1 - SMALL_DROP and max_drop < SMALL_DROP
                                  and ratio < MARGIN_RATIO}
    else:
        out["margin_collapse"] = {"holds": None}
    return out


def bench_summary(rows: list[dict], budgets: tuple[int, ...] = seeds.BUDGETS) -> dict:
    from ..bench.run import summarize

    return {"status": dict(sorted(Counter(r["status"] for r in rows).items())), "problems": len(rows),
            "budgets": {s["budget"]: s for s in summarize(rows, list(budgets))}}


def same_outcomes(a: list[dict], b: list[dict]) -> tuple[int, int]:
    """(problems with identical solved/expansions/proof length, problems compared)."""
    other = {r["id"]: r for r in b}
    keys = ("solved", "expansions", "proof_len")
    both = [r for r in a if r["id"] in other]
    return sum(all(r[k] == other[r["id"]][k] for k in keys) for r in both), len(both)


# ---------------------------------------------------------------------------------- rendering

def _f(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "–"
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s      # no "-0.000"


def _across(values: list[float | None], nd: int = 3) -> list[str]:
    if any(v is None for v in values):
        return ["–", "–"]
    mean, lo, hi = stat(values)       # type: ignore[arg-type]
    return [_f(mean, nd), f"{_f(lo, nd)} to {_f(hi, nd)}"]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def metric_table(tables: dict[str, dict[int, dict]], key: str, extra: tuple[tuple[str, str, int], ...] = ()) -> str:
    """One row per distance: states, model-independent columns, each model, mean and range."""
    models = list(tables)
    first = tables[models[0]]
    rows = []
    for k, ref in first.items():
        vals = [tables[m][k][key] for m in models]
        mark = "" if ref["states"] >= MIN_STATES else " \\*"
        rows.append([f"{k}{mark}", str(ref["states"]), *[_f(ref[col], nd) for _, col, nd in extra],
                     *[_f(v) for v in vals], *_across(vals)])
    return _table(["k", "states", *[name for name, _, _ in extra], *models, "mean", "range"], rows)


def _check_same_population(tables: dict[str, dict[int, dict]]) -> None:
    models = list(tables)
    for m in models[1:]:
        for k, r in tables[models[0]].items():
            o = tables[m].get(k)
            if o is None or (o["states"], o["symbol_bag"], o["chance"]) != (r["states"], r["symbol_bag"], r["chance"]):
                raise seeds.SeedsError(f"{m} was scored on a different population than {models[0]} at distance {k}")


def _verdict_sentence(name: str, per_model: dict[str, dict], key: str) -> str:
    holds = [m for m, v in per_model.items() if v[key]["holds"] is True]
    undecided = [m for m, v in per_model.items() if v[key]["holds"] is None]
    n = len(per_model)
    if len(holds) == n:
        return f"**{name} appears in every one of the {n} models.**"
    if undecided:
        return f"**{name}: not decidable for {', '.join(undecided)} (too few labelled states).**"
    missing = [m for m in per_model if m not in holds]
    return f"**{name} does NOT appear in every model: it fails the rule for {', '.join(missing)}.**"


def load(e3: Path, names: list[str]) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    """Everything the summaries read, for ``names``: per-distance tables, full analyses, score
    metadata and label statistics per split, then bench rows, bench configs and bench summaries
    per model. Refuses incomplete labels and scores made on another label file or population."""
    tables: dict[str, dict[str, dict[int, dict]]] = {}
    overall: dict[str, dict[str, dict]] = {}
    metas: dict[str, dict[str, dict]] = {}
    labels: dict[str, dict] = {}
    for split in ("test_depth", "test_id"):
        st = seeds.require_complete(e3 / "data" / f"{split}.jsonl", e3 / "labels" / f"{split}.jsonl")
        labels[split] = unknown_by_dist(_jsonl(e3 / "labels" / f"{split}.jsonl")) | {"sample_problems": st["problems"]}
        label_hash = seeds.sha256(e3 / "labels" / f"{split}.jsonl")
        tables[split], overall[split], metas[split] = {}, {}, {}
        for m in names:
            p = seeds.score_paths(e3 / "scores" / m, split)
            meta = json.loads(p["meta"].read_text(encoding="utf-8"))
            if meta["labels_sha256"] != label_hash:
                raise seeds.SeedsError(f"{p['scores']} was scored on another label file; rerun e3_score_seeds.py")
            a = json.loads(p["analysis"].read_text(encoding="utf-8"))
            tables[split][m], overall[split][m], metas[split][m] = by_dist(a), a, meta
        _check_same_population(tables[split])
    bench = {m: _jsonl(e3 / "bench-seeds" / m / "results.jsonl") for m in names}
    bcfg = {m: json.loads((e3 / "bench-seeds" / m / "config.json").read_text(encoding="utf-8")) for m in names}
    bsum = {m: bench_summary(bench[m]) for m in names}
    return tables, overall, metas, labels, bench, bcfg, bsum


def render(e3: str | Path, models: dict[str, str] | None = None, e2_run: str | Path | None = None) -> str:
    e3 = Path(e3)
    models = dict(models or seeds.MODELS)
    names = list(models)
    out: list[str] = ["# E3 seed study: h-v1 and two reseeded runs", ""]
    tables, overall, metas, labels, bench, bcfg, bsum = load(e3, names)

    td, lab = tables["test_depth"], labels["test_depth"]
    v = {m: verdicts(td[m]) for m in names}
    scored = overall["test_depth"][names[0]]["overall"]["states"]

    out += [
        ("Every number below is read from `runs/e3/scores/`, `runs/e3/labels/` and `runs/e3/bench-seeds/`. "
        "All three models have the h-v1 configuration and differ only in the training seed. "
        f"\"Mean\" and \"range\" are over the {len(names)} models; a range over {len(names)} runs is not a "
        "confidence interval. This analysis is exploratory, on the E2 test samples."), "",
        "## Verdicts", "",
        _verdict_sentence("Saturation near the training maximum", v, "saturation"),
    ]
    for m in names:
        s = v[m]["saturation"]
        if s["holds"] is not None:
            out.append(f"- {m}: mean prediction {_f(td[m][s['from']]['mean_h'], 2)} at k = {s['from']} and "
                       f"{_f(td[m][s['to']]['mean_h'], 2)} at k = {s['to']} "
                       f"({_f(s['slope'], 3)} per unit of true distance); highest mean beyond the training range "
                       f"{_f(s['max_mean_h'], 2)}.")
    out += ["", (f"Rule: over distances k > {TRAIN_MAX_DIST} with at least {MIN_STATES} labelled states, the mean "
            f"prediction stays below {TRAIN_MAX_DIST} (the largest training distance) and rises by less than "
            f"{FLAT_SLOPE} per unit of true distance."), "",
            _verdict_sentence("The margin collapse", v, "margin_collapse")]
    for m in names:
        c = v[m]["margin_collapse"]
        if c["holds"] is not None:
            out.append(f"- {m}: drop to the optimal child {_f(c['drop_at_1'], 2)} at k = 1, at most "
                       f"{_f(c['max_drop'], 2)} at k = {c['dists'][0]}–{c['dists'][-1]}; margin "
                       f"{_f(c['margin_at_1'], 2)} at k = 1, at depth at most "
                       f"{100 * c['max_margin_ratio']:.0f}% of that.")
    out += ["", (f"Rule: at every distance k >= 7 with at least {MIN_STATES} labelled states, the drop is below "
            f"{SMALL_DROP} (a true distance drops by 1) and |margin| is below {100 * MARGIN_RATIO:.0f}% of its "
            "value at k = 1. The thresholds were chosen after seeing these results, not "
            "pre-registered; the tables below are the evidence."), ""]
    t1 = [overall["test_depth"][m]["overall"]["h"]["top1_optimal"] for m in names]
    s1000 = [bsum[m]["budgets"][max(seeds.BUDGETS)]["rate"] for m in names]
    out += ["What does vary between seeds is the level: overall all-children top-1 on test_depth is "
            + ", ".join(f"{_f(x)} ({m})" for m, x in zip(names, t1, strict=True))
            + f", and the solve rate at {max(seeds.BUDGETS)} expansions is "
            + ", ".join(f"{100 * x:.1f}% ({m})" for m, x in zip(names, s1000, strict=True)) + ".", ""]

    # ---- population
    out += ["## Population and unknown states", "",
            (f"test_depth: all {lab['sample_problems']} problems of the E2 sample are labelled "
            f"({lab['states']} on-path states). {lab['unknown']} states in {lab['problems_with_unknown']} problems "
            f"hit the labelling node cap and are **unknown**; they are excluded from every diagnostic below, "
            f"leaving {scored} scored states. The search results use all {lab['sample_problems']} problems."), ""]
    out.append(_table(["state distance k", "on-path states", "unknown", "unknown share", "scored"],
                      [[str(k), str(r["states"]), str(r["unknown"]), f"{100 * r['unknown'] / r['states']:.1f}%",
                        str(r["states"] - r["unknown"])] for k, r in lab["by_dist"].items()]))
    out += ["", "Unknown states by the distance of their problem: "
            + (", ".join(f"{n} at {d}" for d, n in lab["by_problem_dist"].items()) or "none")
            + "; by theory: " + (", ".join(f"{n} {t}" for t, n in lab["by_theory"].items()) or "none") + ".",
            "", ("Unknown states are those whose forward search exceeded 300,000 nodes, so they are the "
            "highest-branching deep states. Their exclusion removes a growing share of the deepest rows and "
            f"probably flatters top-1 there. Rows marked \\* have fewer than {MIN_STATES} scored states and carry "
            "no weight."), "",
            (f"test_id: {labels['test_id']['sample_problems']} problems, {labels['test_id']['states']} states, "
            f"{labels['test_id']['unknown']} unknown."), ""]

    # ---- test_depth tables
    out += ["## test_depth, by true distance of the state", "",
            "### Mean prediction", "", "A calibrated distance would equal k.", "", metric_table(td, "mean_h"), "",
            "### All-children top-1 (optimal child ranked first, ties split)", "",
            "\"chance\" is the mean share of children that are optimal; symbol_bag is the same for every model.", "",
            metric_table(td, "top1", (("children", "children", 1), ("chance", "chance", 3), ("symbol_bag", "symbol_bag", 3))),
            ""]
    rows = []
    for m in names:
        o = overall["test_depth"][m]["overall"]
        rows.append([m, _f(o["h"]["top1_optimal"]), f"{_f(o['h']['top1_ci95'][0])} to {_f(o['h']['top1_ci95'][1])}",
                     _f(o["symbol_bag"]["top1_optimal"]), _f(o["h"]["mean_best_optimal_rank"], 2),
                     _f(o["calibration_onpath"]["spearman"])])
    out += ["Overall (95% interval: cluster bootstrap over problems, within one model):", "",
            _table(["model", "top-1", "95% CI", "symbol_bag", "mean rank of best optimal child",
                    "Spearman(h, distance)"], rows), "",
            "### Drop to the optimal child: h(state) − h(optimal child)", "", "A true distance gives 1.", "",
            metric_table(td, "drop"), "",
            "### Margin: min h over optimal children − min h over non-optimal children", "",
            "Negative means the right move is preferred. Undefined where every child is optimal.", "",
            metric_table(td, "margin"), "",
            "### Edge-violation rate: share of parent–child edges with |h(parent) − h(child)| > 1", "",
            metric_table(td, "violation"), ""]
    ev = {s: [overall[s][m]["edge_consistency"]["violation_rate"] for m in names] for s in tables}
    out += [_table(["all edges", *names, "mean", "range"],
                   [[s, *[_f(x) for x in ev[s]], *_across(ev[s])] for s in ("test_depth", "test_id")]), ""]

    # ---- test_id
    ti = tables["test_id"]
    out += ["## test_id (in distribution), by true distance of the state", "",
            "### Mean prediction", "", metric_table(ti, "mean_h"), "",
            "### All-children top-1", "",
            metric_table(ti, "top1", (("children", "children", 1), ("chance", "chance", 3), ("symbol_bag", "symbol_bag", 3))),
            "", "### Drop to the optimal child", "", metric_table(ti, "drop"), "",
            "### Edge-violation rate", "", metric_table(ti, "violation"), ""]

    # ---- search
    n = bsum[names[0]]["problems"]
    rows = []
    for b in seeds.BUDGETS:
        cells = [bsum[m]["budgets"][b] for m in names]
        rates = [c["rate"] for c in cells]
        mean, lo, hi = stat(rates)
        rows.append([str(b), *[f"{c['solved']}/{c['n']} = {100 * c['rate']:.1f}%" for c in cells],
                     f"{100 * mean:.1f}%", f"{100 * lo:.1f}% to {100 * hi:.1f}%"])
    bad = {m: {k: c for k, c in bsum[m]["status"].items() if k != "ok"} for m in names}
    censored = sum(c["censored"] for m in names for c in bsum[m]["budgets"].values())
    out += [f"## Search: neural-w1 on the {n} test_depth problems", "",
            ("Solved means the kernel accepted the proof (`isomer bench`). One search per problem at "
            f"{max(seeds.BUDGETS)} expansions; smaller budgets are derived from it."), "",
            _table(["expansions", *names, "mean", "range"], rows), "",
            f"Censored cells: {censored}. Rows with status other than ok: "
            + ("none" if not any(bad.values()) else json.dumps(bad)) + ".", ""]
    if e2_run is not None and (Path(e2_run) / "results.jsonl").exists():
        ref = [r for r in _jsonl(Path(e2_run) / "results.jsonl")
               if r["method"] == seeds.method_name(models[names[0]])]
        same, both = same_outcomes(bench[names[0]], ref)
        out += [(f"Reproduction: the {names[0]} run agrees with the E2 run in `{Path(e2_run).as_posix()}` on "
                f"{same} of {both} problems (solved, expansions and proof length)."), ""]

    # ---- provenance
    rows = [[m, f"`{metas['test_depth'][m]['ckpt_sha256'][:16]}…`",
             str(round(max(metas[s][m]["max_memory_allocated"] or 0 for s in tables) / 2**20)),
             f"`{bcfg[m]['git_commit'][:7]}`" + (" (dirty)" if bcfg[m]["git_dirty"] else "")] for m in names]
    out += ["## Provenance", "",
            _table(["model", "checkpoint sha256", "scoring peak VRAM (MiB)", "bench commit"], rows), "",
            (f"Scoring: cuda, {metas['test_depth'][names[0]]['precision']}. Label file sha256: "
            f"`{metas['test_depth'][names[0]]['labels_sha256'][:16]}…` (test_depth), "
            f"`{metas['test_id'][names[0]]['labels_sha256'][:16]}…` (test_id)."), "",
            "## Limits", "",
            f"- {len(names)} seeds of one configuration on one benchmark; no intervention is tested here.",
            "- Diagnostics use states on one stored shortest proof per problem, not the states search visits.",
            ("- Depth is confounded with branching: deep exact-distance problems have few children per state, "
            "so raw top-1 rises again at k = 8–9 mostly because chance does."),
            "- The deepest rows are small and lose the most states to the node cap (table above).", ""]
    return "\n".join(out)


def write(e3: str | Path, out: str | Path, models: dict[str, str] | None = None,
          e2_run: str | Path | None = None) -> Path:
    text = render(e3, models, e2_run)
    Path(out).write_text(text, encoding="utf-8", newline="\n")
    return Path(out)
