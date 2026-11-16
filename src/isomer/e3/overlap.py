"""E3: do the evaluated states occur in the training states?

    python scripts/e3_overlap.py        # -> runs/e3/overlap_summary.md

Problem ids of the test splits are disjoint from the train pool by
construction (``data.build``), but an id names a (lhs, rhs) *problem*. A model
is trained and evaluated on (state, goal) *pairs*, and nothing in the build
prevents an on-path state of a test problem, together with its goal, from also
being a state of some train problem. A model that has seen the pair with its
distance label can rank by recall, so before a result is read as
generalization this module measures:

1. how often a scored test state (state, goal), and each of its children
   (child, goal), occurs in a training-state file;
2. all-children top-1 recomputed on the states that do not overlap with the
   model's own training set;
3. per test_depth problem, whether its start pair or any on-path pair was
   trained on, and the solve rate on the problems without such a pair.

A pair is identified by theory and the exact strings of state and goal, which
is what the tokenizer sees. Nothing existing is modified: the module only reads
labels, scores, bench results and the two training-state files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import seeds
from . import seeds_summary as S

TRAIN_SETS = {"v2": "data/states-v2", "d6": "data/states-v2-d6"}
MODEL_TRAIN = {"h-v1": "v2", "h-v1-s1": "v2", "h-v1-s2": "v2", "h-d6-s0": "d6"}
SPLITS = ("test_depth", "test_id")
SUBSETS = ("full", "unseen", "strict")


def pair_key(theory: str, state: str, goal: str) -> str:
    return f"{theory}\t{state}\t{goal}"


def load_pairs(path: str | Path) -> dict[str, int]:
    """{pair: bitmask of the distances it is labelled with} for one states file.
    A pair can occur under several train problems (different pools), hence a set of labels."""
    pairs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                k = pair_key(r["theory"], r["state"], r["goal"])
                pairs[k] = pairs.get(k, 0) | (1 << r["dist"])
    return pairs


def mark(label_rows: list[dict], pairs: dict[str, int]) -> dict[tuple[str, int], dict]:
    """Per on-path state (problem_id, i): was its pair trained on, with the same distance
    label, and which of its children's pairs were."""
    out = {}
    for r in label_rows:
        bits = pairs.get(pair_key(r["theory"], r["state"], r["goal"]), 0)
        kids = [pair_key(r["theory"], c, r["goal"]) in pairs for c in r["children"]]
        out[(r["problem_id"], r["i"])] = {
            "problem_id": r["problem_id"], "dist": r["dist"], "scored": r["status"] == "ok",
            "state_seen": bool(bits), "same_label": bool(bits >> r["dist"] & 1),
            "children": len(kids), "children_seen": sum(kids), "any_child_seen": any(kids)}
    return out


def overlap_by_dist(marks: dict[tuple[str, int], dict]) -> dict[int, dict]:
    """Counts over scored states, by the state's distance."""
    out: dict[int, dict] = defaultdict(lambda: dict.fromkeys(
        ("states", "state_seen", "same_label", "children", "children_seen", "any_child_seen"), 0))
    for m in marks.values():
        if not m["scored"]:
            continue
        d = out[m["dist"]]
        d["states"] += 1
        for k in ("state_seen", "same_label", "children", "children_seen", "any_child_seen"):
            d[k] += int(m[k])
    return dict(sorted(out.items()))


def in_subset(m: dict, subset: str) -> bool:
    """full: every scored state. unseen: the state's pair was not trained on.
    strict: neither the state's pair nor any child's pair was trained on."""
    if subset == "full":
        return True
    if subset == "unseen":
        return not m["state_seen"]
    return not m["state_seen"] and not m["any_child_seen"]


def subset_metrics(score_rows: list[dict], marks: dict[tuple[str, int], dict]) -> dict[str, dict]:
    """{subset: {k or "all": metrics}} for one model: all-children top-1 of the model and of
    symbol_bag, mean prediction, and mean drop to an optimal child (edge-pooled, as ``analyze``)."""
    from .diagnose import bootstrap_mean_ci, top1

    out: dict[str, dict] = {}
    for subset in SUBSETS:
        groups: dict = defaultdict(list)
        for r in score_rows:
            if in_subset(marks[(r["problem_id"], r["i"])], subset):
                groups[r["dist"]].append(r)
                groups["all"].append(r)
        res = {}
        for k in sorted(groups, key=lambda x: (x == "all", 0 if x == "all" else x)):    # distances, then "all"
            rs = groups[k]
            t = [top1(r["h_children"], r["optimal"]) for r in rs]
            drops = [r["h_state"] - v for r in rs for v, o in zip(r["h_children"], r["optimal"], strict=True) if o]
            res[k] = {"states": len(rs), "problems": len({r["problem_id"] for r in rs}),
                      "top1": sum(t) / len(t),
                      "symbol_bag": sum(top1(r["sb_children"], r["optimal"]) for r in rs) / len(rs),
                      "mean_h": sum(r["h_state"] for r in rs) / len(rs),
                      "drop": sum(drops) / len(drops) if drops else None}
            if k == "all":
                res[k]["top1_ci"] = bootstrap_mean_ci(t, [r["problem_id"] for r in rs])
        out[subset] = res
    return out


def problem_overlap(sample: list[dict], marks: dict[tuple[str, int], dict], pairs: dict[str, int]) -> dict[str, dict]:
    """Per problem: is its start pair (lhs, rhs) in the training states, and how many of its
    on-path pairs are (every on-path state with distance >= 1, scored or not)."""
    on: dict[str, int] = defaultdict(int)
    for m in marks.values():
        on[m["problem_id"]] += int(m["state_seen"])
    return {r["id"]: {"dist": r["dist"], "start": pair_key(r["theory"], r["lhs"], r["rhs"]) in pairs,
                      "onpath_seen": on.get(r["id"], 0)} for r in sample}


def solve_counts(bench_rows: list[dict], keep: set[str], budgets: tuple[int, ...] = seeds.BUDGETS) -> dict:
    from ..bench.run import cell

    rows = [r for r in bench_rows if r["id"] in keep]
    return {"n": len(rows), **{b: sum(cell(r, b) == "solved" for r in rows) for b in budgets}}


# ---------------------------------------------------------------------------------- rendering

def _pct(k: int, n: int) -> str:
    return "–" if n == 0 else f"{k} ({100 * k / n:.1f}%)"


def _overlap_table(by: dict[str, dict[int, dict]]) -> str:
    sets = list(by)
    head = ["k", "scored states", *[f"state in {s}" for s in sets], "children",
            *[f"children in {s}" for s in sets], *[f"states with a child in {s}" for s in sets]]
    rows = []
    ks = list(by[sets[0]])
    tot = {s: dict.fromkeys(("states", "state_seen", "children", "children_seen", "any_child_seen"), 0) for s in sets}
    for k in ks:
        ref = by[sets[0]][k]
        rows.append([str(k), str(ref["states"]), *[_pct(by[s][k]["state_seen"], ref["states"]) for s in sets],
                     str(ref["children"]), *[_pct(by[s][k]["children_seen"], ref["children"]) for s in sets],
                     *[_pct(by[s][k]["any_child_seen"], ref["states"]) for s in sets]])
        for s in sets:
            for key in tot[s]:
                tot[s][key] += by[s][k][key]
    ref = tot[sets[0]]
    rows.append(["all", str(ref["states"]), *[_pct(tot[s]["state_seen"], ref["states"]) for s in sets],
                 str(ref["children"]), *[_pct(tot[s]["children_seen"], ref["children"]) for s in sets],
                 *[_pct(tot[s]["any_child_seen"], ref["states"]) for s in sets]])
    return S._table(head, rows)


def _top1_table(metrics: dict[str, dict]) -> str:
    rows = []
    for k in metrics["full"]:
        row = [str(k)]
        for subset in SUBSETS:
            m = metrics[subset].get(k)
            row += ["0", "–"] if m is None else [str(m["states"]), S._f(m["top1"])]
        strict = metrics["strict"].get(k)
        row.append("–" if strict is None else S._f(strict["symbol_bag"]))
        rows.append(row)
    return S._table(["k", "all states", "top-1", "state unseen", "top-1", "state and children unseen", "top-1",
                     "symbol_bag on the last subset"], rows)


def render(e3: str | Path, models: dict[str, str] | None = None, train_sets: dict[str, str] | None = None,
           root: str | Path = ".") -> str:
    e3, root = Path(e3), Path(root)
    models = dict(models or MODEL_TRAIN)
    train_sets = dict(train_sets or TRAIN_SETS)
    pairs = {s: load_pairs(root / d / "train.jsonl") for s, d in train_sets.items()}
    val = {s: load_pairs(root / d / "val.jsonl") for s, d in train_sets.items() if (root / d / "val.jsonl").exists()}
    labels = {sp: S._jsonl(e3 / "labels" / f"{sp}.jsonl") for sp in SPLITS}
    marks = {sp: {s: mark(labels[sp], pairs[s]) for s in train_sets} for sp in SPLITS}

    out = ["# E3: overlap between evaluated states and training states", "",
           ("A pair is (theory, state, goal) with the exact strings the tokenizer sees. \"In v2\" / \"in d6\" means "
            "the pair is a line of `train.jsonl` in "
            + " / ".join(f"`{d}`" for d in train_sets.values())
            + ". Problem ids of the test splits are disjoint from the train pool; this measures the (state, goal) "
            "pairs, which the build does not keep disjoint."), "",
           S._table(["training set", "directory", "distinct training pairs"],
                    [[s, f"`{train_sets[s]}`", str(len(pairs[s]))] for s in train_sets]), ""]

    # ---- 1. overlap rates
    out += ["## 1. Overlap of scored states and of their children", "",
            ("\"State in X\": the on-path state's own pair is a training pair. \"Children in X\": share of all "
             "(child, goal) pairs that are training pairs. \"States with a child in X\": at least one child."), ""]
    for sp in SPLITS:
        by = {s: overlap_by_dist(marks[sp][s]) for s in train_sets}
        out += [f"### {sp}, by distance of the state", "", _overlap_table(by), ""]
        notes = []
        for s in train_sets:
            seen = sum(d["state_seen"] for d in by[s].values())
            same = sum(d["same_label"] for d in by[s].values())
            extra = ""
            if s in val:
                v = sum(m["scored"] and m["state_seen"] for m in mark(labels[sp], val[s]).values())
                extra = f"; {v} are in `val.jsonl` (used to pick the checkpoint)"
            notes.append(f"- {s}: of the {seen} scored states whose pair is in training, {same} carry the same "
                         f"distance label there as in the test labels{extra}.")
        out += [*notes, ""]

    # ---- 2. top-1 on non-overlapping states
    out += ["## 2. All-children top-1 on non-overlapping states", "",
            ("Overlap is taken against each model's own training set. \"State unseen\": the state's pair is not a "
             "training pair. \"State and children unseen\": in addition no child's pair is. Top-1 is over all "
             "children as before (ties split); only the set of states changes, no score is recomputed."), ""]
    metrics: dict[str, dict[str, dict]] = {}
    for sp in SPLITS:
        metrics[sp] = {}
        rows = []
        for m, s in models.items():
            metrics[sp][m] = subset_metrics(S._jsonl(seeds.score_paths(e3 / "scores" / m, sp)["scores"]),
                                            marks[sp][s])
            row = [m, s]
            for subset in SUBSETS:
                a = metrics[sp][m][subset].get("all")
                row += (["0", "–"] if a is None else
                        [str(a["states"]), f"{S._f(a['top1'])} ({S._f(a['top1_ci'][0])} to {S._f(a['top1_ci'][1])})"])
            rows.append(row)
        out += [f"### {sp}, all distances (95% CI: cluster bootstrap over problems)", "",
                S._table(["model", "trained on", "all states", "top-1", "state unseen", "top-1",
                          "state and children unseen", "top-1"], rows), ""]
        for m in models:
            out += [f"#### {sp}, {m}, by distance of the state", "", _top1_table(
                {sub: {k: v for k, v in metrics[sp][m][sub].items() if k != "all"} for sub in SUBSETS}), ""]

    out += ["### Mean prediction and drop to the optimal child on non-overlapping test_depth states", "",
            ("The quantities of the d6 prediction, on the strictest subset (state and children unseen), next to "
             "the value on all scored states."), ""]
    rows = []
    for k in (5, 6, 7, 8):
        for m in models:
            full, strict = metrics["test_depth"][m]["full"].get(k), metrics["test_depth"][m]["strict"].get(k)
            if full is None:
                continue
            rows.append([str(k), m, str(full["states"]), S._f(full["mean_h"]), S._f(full["drop"]),
                         "0" if strict is None else str(strict["states"]),
                         "–" if strict is None else S._f(strict["mean_h"]),
                         "–" if strict is None else S._f(strict["drop"])])
    out += [S._table(["k", "model", "all states", "mean prediction", "drop", "state and children unseen",
                      "mean prediction", "drop"], rows), ""]

    # ---- 3. search
    sample = S._jsonl(e3 / "data" / "test_depth.jsonl")
    po = {s: problem_overlap(sample, marks["test_depth"][s], pairs[s]) for s in train_sets}
    out += ["## 3. Search on test_depth problems", "",
            ("Per problem: is the start pair (lhs, rhs) a training pair, and is any (state, goal) pair on its "
             "stored shortest proof (distance >= 1, including states the labeller left unknown) a training pair."),
            ""]
    dists = sorted({r["dist"] for r in sample})
    rows = []
    for d in [*dists, "all"]:
        ids = [r["id"] for r in sample if d == "all" or r["dist"] == d]
        row = [str(d), str(len(ids))]
        for s in train_sets:
            row += [_pct(sum(po[s][i]["start"] for i in ids), len(ids)),
                    _pct(sum(po[s][i]["onpath_seen"] > 0 for i in ids), len(ids))]
        rows.append(row)
    head = ["problem distance", "problems"]
    for s in train_sets:
        head += [f"start pair in {s}", f"any on-path pair in {s}"]
    out += [S._table(head, rows), ""]
    rows = []
    for m, s in models.items():
        bench = S._jsonl(e3 / "bench-seeds" / m / "results.jsonl")
        clean = {i for i, v in po[s].items() if v["onpath_seen"] == 0}
        for name, keep in (("all problems", set(po[s])), ("no on-path pair in training", clean),
                           ("some on-path pair in training", set(po[s]) - clean)):
            c = solve_counts(bench, keep)
            rows.append([m, s, name, str(c["n"]), *[_pct(c[b], c["n"]) for b in seeds.BUDGETS]])
    out += ["Solved (kernel-accepted) by neural-w1, on all problems and on the two parts:", "",
            S._table(["model", "trained on", "problems", "n", *[f"solved @{b}" for b in seeds.BUDGETS]], rows), "",
            "## Limits", "",
            ("- Only exact pairs are counted. A pair that differs by a renaming, by the direction of the equation "
             "(the goal as state), or by one rewrite is not counted as overlap."),
            ("- Search visits many states that are not on the stored proof; their overlap with training is not "
             "measured here. \"No on-path pair in training\" does not mean the search saw no training pair."),
            ("- The non-overlapping states are a different population from all scored states (they may be harder "
             "or easier), so a change in top-1 between the columns is not only the effect of recall. symbol_bag on "
             "the same subset is given for reference."),
            "- Unknown states (labelling node cap) are excluded from sections 1 and 2, as everywhere in E3.", ""]
    return "\n".join(out)


def write(e3: str | Path, out: str | Path, **kw) -> Path:
    Path(out).write_text(render(e3, **kw), encoding="utf-8", newline="\n")
    return Path(out)
