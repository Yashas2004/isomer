"""E3-fresh: judge the pre-registered hypotheses H1-H4 on the fresh sample.

    python scripts/e3_fresh_summary.py        # -> runs/e3/fresh_summary.md

The rules below are those of ``docs/experiments/E3/fresh_prediction.md``,
committed before any model was run on the fresh sample. They are constants,
not arguments, so a verdict cannot be tuned after the run; a test ties them to
the document. Every number is read from the outputs of the E3 scripts in
``runs/e3-fresh`` (labels, scores, bench results); no model is run here.

A hypothesis is **supported** when every one of its conditions holds,
**not supported** when one fails, and **not decidable** when none fails but one
cannot be evaluated (a distance with fewer than ``MIN_STATES`` scored states).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import seeds
from . import seeds_compare as C
from . import seeds_summary as S
from .fresh import SPLIT

BASELINE = ("h-v1", "h-v1-s1", "h-v1-s2")
D6 = ("h-d6-s0", "h-d6-s1", "h-d6-s2")
CONTROL = "h-d6m-s0"
MODELS = (*BASELINE, *D6, CONTROL)
NON_NEURAL = ("bfs-bidir", "bestfirst-symbol_bag-w1")       # reported, not tested

# H1 saturation: mean prediction at k = 7 and 8; every baseline seed below, every d6 seed at least.
H1_DISTS = (7, 8)
H1_BASELINE_BELOW = 5.0
H1_D6_AT_LEAST = 5.7
MIN_STATES = 10          # a distance with fewer scored states is not decidable
# H3 search: neural-w1 solved at these budgets; exact McNemar per (d6, baseline) pair, one Holm family.
H3_BUDGETS = (30, 1000)
ALPHA = 0.05
# H4 control: seeds_compare.MATCHED_RULES unchanged; supported if every row says "depth".

SUPPORTED, NOT_SUPPORTED, UNDECIDABLE = "supported", "not supported", "not decidable"


def combine(passes: list[bool | None]) -> str:
    if any(p is False for p in passes):
        return NOT_SUPPORTED
    if any(p is None for p in passes):
        return UNDECIDABLE
    return SUPPORTED


# ---------------------------------------------------------------------------------- the four rules

def judge_h1(tables: dict[str, dict[int, dict]]) -> dict:
    """12 conditions: each of 6 models at each of k = 7, 8."""
    rows = []
    for group, models, below in (("baseline", BASELINE, True), ("d6", D6, False)):
        for m in models:
            for k in H1_DISTS:
                r = tables[m].get(k)
                states = 0 if r is None else r["states"]
                value = None if r is None else r["mean_h"]
                if value is None or states < MIN_STATES:
                    ok = None
                else:
                    ok = value < H1_BASELINE_BELOW if below else value >= H1_D6_AT_LEAST
                rows.append({"model": m, "group": group, "k": k, "states": states, "value": value,
                             "rule": f"< {H1_BASELINE_BELOW}" if below else f">= {H1_D6_AT_LEAST}", "pass": ok})
    return {"rows": rows, "verdict": combine([r["pass"] for r in rows])}


def judge_h2(top1: dict[str, float]) -> dict:
    """Each d6 seed's all-children top-1 above the best baseline seed's."""
    best = max(BASELINE, key=lambda m: top1[m])
    rows = [{"model": m, "value": top1[m], "best_baseline": best, "baseline_value": top1[best],
             "pass": top1[m] > top1[best]} for m in D6]
    return {"rows": rows, "best_baseline": best, "verdict": combine([r["pass"] for r in rows])}


def solved_sets(rows: list[dict], b: int) -> tuple[set[str], set[str]]:
    """(ids solved at budget b, ids censored at b) of one bench run."""
    from ..bench.run import cell

    return ({r["id"] for r in rows if cell(r, b) == "solved"}, {r["id"] for r in rows if cell(r, b) == "censored"})


def judge_h3(bench: dict[str, list[dict]]) -> dict:
    """Counts: each d6 seed solves more than the best baseline seed, at each budget. Tests: exact
    McNemar for every (d6 seed, baseline seed, budget), Holm over all of them; each adjusted p < ALPHA."""
    from ..bench.report import holm, mcnemar_exact

    ids = {m: {r["id"] for r in bench[m]} for m in (*BASELINE, *D6)}
    ref = ids[BASELINE[0]]
    if any(v != ref for v in ids.values()):
        raise seeds.SeedsError("the bench runs were not made on the same problems")
    counts, tests = [], []
    for b in H3_BUDGETS:
        solved, censored = {}, set()
        for m in (*BASELINE, *D6):
            solved[m], c = solved_sets(bench[m], b)
            censored |= c
        best = max(BASELINE, key=lambda m: len(solved[m]))
        for d in D6:
            counts.append({"budget": b, "model": d, "solved": len(solved[d]), "best_baseline": best,
                           "baseline_solved": len(solved[best]), "pass": len(solved[d]) > len(solved[best])})
            for base in BASELINE:
                only_d6 = len((solved[d] - solved[base]) - censored)
                only_base = len((solved[base] - solved[d]) - censored)
                tests.append({"budget": b, "d6": d, "baseline": base, "only_d6": only_d6, "only_baseline": only_base,
                              "censored": len(censored), "p": mcnemar_exact(only_d6, only_base)})
    for t, adj in zip(tests, holm([t["p"] for t in tests]), strict=True):
        t["p_holm"] = adj
        t["pass"] = adj < ALPHA and t["only_d6"] > t["only_baseline"]
    return {"counts": counts, "tests": tests, "n": len(ref),
            "verdict": combine([c["pass"] for c in counts] + [t["pass"] for t in tests])}


def judge_h4(table: dict[int, dict], overall_top1: float) -> dict:
    """The five rules of d6_matched_prediction.md for the control, unchanged."""
    missing = [k for _, _, k, _, _ in C.MATCHED_RULES if k is not None and k not in table]
    if missing:
        return {"rows": [], "overall": None, "verdict": UNDECIDABLE, "missing": missing}
    ev = C.evaluate_matched(table, overall_top1)
    return {**ev, "verdict": SUPPORTED if ev["overall"] == "depth" else NOT_SUPPORTED}


# ---------------------------------------------------------------------------------- inputs

def load(e3: Path) -> dict:
    """Labels, scores and bench results of every model in ``MODELS``; refuses anything partial."""
    sample, labels_path = e3 / "data" / f"{SPLIT}.jsonl", e3 / "labels" / f"{SPLIT}.jsonl"
    st = seeds.require_complete(sample, labels_path)
    label_hash = seeds.sha256(labels_path)
    missing = [m for m in MODELS if not seeds.score_paths(e3 / "scores" / m, SPLIT)["analysis"].is_file()
               or not (e3 / "bench-seeds" / m / "results.jsonl").is_file()]
    if missing:
        raise seeds.SeedsError(f"no scores or bench results for {', '.join(missing)}; every model is needed")
    if not (e3 / "bench-seeds" / "baselines" / "results.jsonl").is_file():
        raise seeds.SeedsError("no bench-seeds/baselines/results.jsonl (bfs-bidir, bestfirst-symbol_bag-w1)")
    tables, overall, metas, bench = {}, {}, {}, {}
    for m in MODELS:
        p = seeds.score_paths(e3 / "scores" / m, SPLIT)
        metas[m] = json.loads(p["meta"].read_text(encoding="utf-8"))
        if metas[m]["labels_sha256"] != label_hash:
            raise seeds.SeedsError(f"{p['scores']} was scored on another label file; rerun e3_score_seeds.py")
        a = json.loads(p["analysis"].read_text(encoding="utf-8"))
        tables[m], overall[m] = S.by_dist(a), a["overall"]
        bench[m] = S._jsonl(e3 / "bench-seeds" / m / "results.jsonl")
    S._check_same_population(tables)
    base_rows = S._jsonl(e3 / "bench-seeds" / "baselines" / "results.jsonl")
    for meth in NON_NEURAL:
        bench[meth] = [r for r in base_rows if r["method"] == meth]
        if not bench[meth]:
            raise seeds.SeedsError(f"bench-seeds/baselines has no rows of {meth}")
    return {"sample": S._jsonl(sample), "sample_status": st, "labels": S._jsonl(labels_path),
            "label_hash": label_hash, "tables": tables, "overall": overall, "metas": metas, "bench": bench}


def overlap_counts(sample: list[dict], label_rows: list[dict], train_sets: dict[str, str], root: Path) -> dict | None:
    """Exact (theory, state, goal) overlap of the fresh on-path states with each training-state file,
    by state distance, and per problem. None if a training-state file is missing."""
    from . import overlap as O

    paths = {s: root / d / "train.jsonl" for s, d in train_sets.items()}
    if not all(p.is_file() for p in paths.values()):
        return None
    out = {}
    for s, p in paths.items():
        pairs = O.load_pairs(p)
        marks = O.mark(label_rows, pairs)
        po = O.problem_overlap(sample, marks, pairs)
        out[s] = {"by_dist": O.overlap_by_dist(marks),
                  "problems_start_seen": sum(v["start"] for v in po.values()),
                  "problems_onpath_seen": sum(v["onpath_seen"] > 0 for v in po.values())}
    return out


# ---------------------------------------------------------------------------------- rendering

def _p(x: float) -> str:
    return f"{x:.2g}" if x < 0.001 else f"{x:.3f}"


def _yes(ok: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "not decidable"}[ok]


def render(e3: str | Path, root: str | Path = ".", prediction_commit: str | None = None,
           ids_file: str = "docs/experiments/E3/ids-test_depth_fresh.txt") -> str:
    e3, root = Path(e3), Path(root)
    try:            # the summary is committed: name the directory relative to the repository, never absolute
        shown = e3.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = e3.name
    d = load(e3)
    tables, overall, bench = d["tables"], d["overall"], d["bench"]
    top1 = {m: overall[m]["h"]["top1_optimal"] for m in MODELS}
    h1, h2, h3 = judge_h1(tables), judge_h2(top1), judge_h3(bench)
    h4 = judge_h4(tables[CONTROL], top1[CONTROL])
    lab = S.unknown_by_dist(d["labels"])
    scored = overall[MODELS[0]]["states"]
    n = len(d["sample"])

    out = ["# E3-fresh: confirmatory evaluation on newly generated problems", "",
           (f"{n} problems of `{SPLIT}` (`{ids_file}`), generated and labelled after every E3 hypothesis was "
            "formed: the v2 eval-pool generator and labeller with a new seed, ids disjoint from every split of "
            "`data/isobench-v2` and `data/isobench-v2-d6`. The rules are those of "
            "`docs/experiments/E3/fresh_prediction.md`"
            + (f" (commit `{prediction_commit}`)" if prediction_commit else "")
            + ", committed before any model was run on this sample. Every number below is read from "
            f"`{shown}`."), "",
           "## Verdicts", "",
           S._table(["hypothesis", "rule", "verdict"], [
               ["H1 saturation", (f"every baseline seed < {H1_BASELINE_BELOW} and every d6 seed >= {H1_D6_AT_LEAST} "
                f"in mean prediction at k = {H1_DISTS[0]} and {H1_DISTS[1]}"), f"**{h1['verdict']}**"],
               ["H2 move choice", "every d6 seed's all-children top-1 > the best baseline seed's",
                f"**{h2['verdict']}**"],
               ["H3 search", (f"every d6 seed solves more than the best baseline seed at {H3_BUDGETS[0]} and "
                f"{H3_BUDGETS[1]:,} expansions, and every Holm-adjusted McNemar p < {ALPHA}"), f"**{h3['verdict']}**"],
               ["H4 control", f"{CONTROL} is \"depth\" on all five rules of d6_matched_prediction.md",
                f"**{h4['verdict']}**"]]), ""]

    # ---- population
    out += ["## Population and unknown states", "",
            (f"All {n} problems are labelled ({lab['states']} on-path states). {lab['unknown']} states in "
             f"{lab['problems_with_unknown']} problems are **unknown** (their child labelling exceeded 300,000 "
             f"nodes) and are excluded from H1, H2 and H4, leaving {scored} scored states. H3 uses all {n} "
             "problems."), "",
            S._table(["state distance k", "on-path states", "unknown", "scored"],
                     [[str(k), str(r["states"]), str(r["unknown"]), str(r["states"] - r["unknown"])]
                      for k, r in lab["by_dist"].items()]), "",
            "Unknown states by the distance of their problem: "
            + (", ".join(f"{c} at {k}" for k, c in lab["by_problem_dist"].items()) or "none") + ".", ""]

    # ---- H1
    rows = [[r["model"], r["group"], str(r["k"]), str(r["states"]), S._f(r["value"]), r["rule"], _yes(r["pass"])]
            for r in h1["rows"]]
    out += ["## H1: saturation", "", S._table(["model", "group", "k", "scored states", "mean prediction", "rule",
                                                "holds"], rows), ""]

    # ---- H2
    rows = [[m, "baseline" if m in BASELINE else "d6" if m in D6 else "control", S._f(top1[m]),
             f"{S._f(overall[m]['h']['top1_ci95'][0])} to {S._f(overall[m]['h']['top1_ci95'][1])}",
             S._f(overall[m]["symbol_bag"]["top1_optimal"])] for m in MODELS]
    out += ["## H2: move choice (all-children top-1, ties split, all scored states)", "",
            S._table(["model", "group", "top-1", "95% CI (cluster bootstrap)", "symbol_bag"], rows), "",
            f"Best baseline seed: {h2['best_baseline']} ({S._f(top1[h2['best_baseline']])}). "
            + "; ".join(f"{r['model']} {S._f(r['value'])}: {_yes(r['pass'])}" for r in h2["rows"]) + ".", ""]

    # ---- H3
    from ..bench.run import cell

    names = [*MODELS, *NON_NEURAL]
    rows = []
    for b in seeds.BUDGETS:
        rows.append([str(b), *[str(sum(cell(r, b) == "solved" for r in bench[m])) for m in names]])
    out += [f"## H3: search (neural-w1, solved of {n}, kernel-accepted)", "",
            ("One search per problem at 1,000 expansions; smaller budgets are derived from it. bfs-bidir and "
            "bestfirst-symbol_bag-w1 are reported, not tested."), "",
            S._table(["expansions", *names], rows), ""]
    bad = {m: sorted({r["status"] for r in bench[m]} - {"ok"}) for m in names}
    bad = {m: v for m, v in bad.items() if v}
    out += ["Rows with a status other than ok: " + (json.dumps(bad) if bad else "none") + ".", ""]
    rows = [[str(c["budget"]), c["model"], str(c["solved"]), f"{c['best_baseline']} ({c['baseline_solved']})",
             _yes(c["pass"])] for c in h3["counts"]]
    out += ["Counts against the best baseline seed at each budget:", "",
            S._table(["expansions", "d6 seed", "solved", "best baseline seed", "more"], rows), ""]
    rows = [[str(t["budget"]), t["d6"], t["baseline"], str(t["only_d6"]), str(t["only_baseline"]), _p(t["p"]),
             _p(t["p_holm"]), _yes(t["pass"])] for t in h3["tests"]]
    out += [f"Exact McNemar tests, Holm-adjusted over all {len(h3['tests'])}:", "",
            S._table(["expansions", "d6 seed", "baseline seed", "solved only by d6", "solved only by baseline",
                      "p", "Holm p", f"d6 better, Holm p < {ALPHA}"], rows), ""]
    by = {m: C.solved_by_dist(bench[m]) for m in names}
    top = max(seeds.BUDGETS)
    rows = [[str(k), str(by[names[0]][k]["n"]), *[str(by[m][k][top]) for m in names]] for k in by[names[0]]]
    out += [f"Solved at {top:,} expansions by problem distance:", "",
            S._table(["problem distance", "problems", *names], rows), ""]

    # ---- H4
    if h4["rows"]:
        rows = [[r["quantity"] + ("" if r["k"] is None else f" at k = {r['k']}"), S._f(r["value"]),
                 f">= {r['depth_ge']}", f"<= {r['volume_le']}", r["verdict"]] for r in h4["rows"]]
        out += [f"## H4: the size-matched control {CONTROL}", "",
                S._table(["quantity", CONTROL, "depth if", "volume if", "verdict"], rows), ""]
    else:
        out += [f"## H4: the size-matched control {CONTROL}", "",
                f"Not decidable: no scored states at distance {', '.join(map(str, h4['missing']))}.", ""]

    # ---- per-distance tables
    groups = {"baseline": {m: tables[m] for m in BASELINE}, "d6": {m: tables[m] for m in D6}}
    out += ["## By true distance of the state (reported, not tested)", "",
            (f"Mean and range over the three seeds of each group; the last column is {CONTROL}. Rows marked \\* "
            f"have fewer than {S.MIN_STATES} scored states."), ""]
    for title, key in (("Mean prediction", "mean_h"), ("All-children top-1", "top1"),
                       ("Drop to the optimal child", "drop")):
        out += [f"### {title}", "", C.family_table(groups, (CONTROL, tables[CONTROL]), key), ""]

    # ---- overlap
    ov = overlap_counts(d["sample"], d["labels"], {"v2": "data/states-v2", "d6": "data/states-v2-d6"}, root)
    out += ["## Overlap with training states (reported, not tested)", ""]
    if ov is None:
        out += [("Not measured: a training-state file (`data/states-v2/train.jsonl` or "
                "`data/states-v2-d6/train.jsonl`) is missing."), ""]
    else:
        rows = []
        for k in ov["v2"]["by_dist"]:
            ref = ov["v2"]["by_dist"][k]
            rows.append([str(k), str(ref["states"]), *[str(ov[s]["by_dist"][k]["state_seen"]) for s in ov],
                         *[str(ov[s]["by_dist"][k]["any_child_seen"]) for s in ov]])
        out += [("Exact (theory, state, goal) pairs, scored states only. Problem ids are disjoint by "
                 "construction; states are not."), "",
                S._table(["k", "scored states", *[f"state in {s}" for s in ov],
                          *[f"states with a child in {s}" for s in ov]], rows), "",
                "Problems whose start pair is a training pair: "
                + ", ".join(f"{ov[s]['problems_start_seen']} ({s})" for s in ov)
                + "; problems with any on-path pair in training: "
                + ", ".join(f"{ov[s]['problems_onpath_seen']} ({s})" for s in ov) + f", of {n}.", ""]

    # ---- provenance and limits
    rows = [[m, f"`{d['metas'][m]['ckpt_sha256'][:16]}…`",
             str(round((d["metas"][m].get("max_memory_allocated") or 0) / 2**20))] for m in MODELS]
    out += ["## Provenance", "", S._table(["model", "checkpoint sha256", "scoring peak VRAM (MiB)"], rows), "",
            f"Label file sha256: `{d['label_hash'][:16]}…`.", "",
            "## Limits", "",
            ("- Same generator, theories and action space as v2: this confirms the results on new problems, not on "
             "a new distribution."),
            "- Ids that collide with v2 or d6 were replaced at generation; the number is in the datasheet.",
            ("- The H1 and H4 thresholds were chosen with the E2-sample results in view; what is new here is the "
             "problems, not the rules."),
            "- Three seeds per group; the control is one seed.",
            "- Unknown states are excluded from H1, H2 and H4, which removes the highest-branching deep states.", ""]
    return "\n".join(out)


def write(e3: str | Path, out: str | Path, **kw) -> Path:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render(e3, **kw), encoding="utf-8", newline="\n")
    return Path(out)
