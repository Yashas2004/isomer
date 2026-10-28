"""Reports: ``results.md`` and figures, regenerated purely from a run's logs.

Nothing here re-runs search. Every number comes from ``results.jsonl`` and
``config.json`` and goes through ``run.cell`` (solved / unsolved / censored at a
budget), so the report can never disagree with the runner about what counts as
solved. One command regenerates every table and figure (DESIGN.md §9).

Statistics, chosen so that they do not flatter any method:

* **Wilson intervals** on solve rates (reused from ``run``).
* **Fair optimality gap**: mean ``proof_len / dist`` only over problems that
  *every* method solved. Otherwise a method that solves only easy problems
  looks optimal.
* **Paired exact McNemar tests** against a reference method. All methods
  run on the same problems, so the right test is on the discordant pairs, not
  a comparison of two independent rates. **Holm-Bonferroni** adjusts for the
  many (method, budget) comparisons.

matplotlib is optional (``pip install -e ".[plots]"``) and imported only
inside ``plot_all``, so the core package never depends on it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

from .run import BenchError, cell, summarize, wilson

DEFAULT_REFERENCE = "bfs-fwd"

# matplotlib's tab10 (colour-blind-friendly ordering), hard-coded so the
# method -> colour mapping is testable without matplotlib installed.
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# --------------------------------------------------------------------------
# Pure statistics
# --------------------------------------------------------------------------

def mcnemar_exact(a: int, b: int) -> float:
    """Exact two-sided McNemar test: a = solved only by the method, b = only by
    the reference. Under H0 the discordant pairs split 50/50, so this is a
    binomial test on a + b trials with p = 0.5."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.
    It controls the family-wise error rate and is never less powerful than Bonferroni."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _methods(rows: list[dict]) -> list[str]:
    return sorted({r["method"] for r in rows})


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["method"], r["id"]): r for r in rows}


def _max_budget(rows: list[dict]) -> int:
    return max(r["max_budget"] for r in rows)


def fair_gap(rows: list[dict], budget: int | None = None) -> tuple[int, dict[str, float | None]]:
    """Mean proof_len / dist per method over the problems with known dist that
    *every* method solved at ``budget`` (default: the run's max budget).
    Returns (number of such problems, {method: mean gap or None})."""
    b = _max_budget(rows) if budget is None else budget
    methods, idx = _methods(rows), _index(rows)
    ids = sorted({r["id"] for r in rows if r["dist"]})
    common = [i for i in ids
              if all((m, i) in idx and cell(idx[m, i], b) == "solved" for m in methods)]
    gaps = {m: (sum(idx[m, i]["proof_len"] / idx[m, i]["dist"] for i in common) / len(common)
                if common else None) for m in methods}
    return len(common), gaps


def identical_methods(rows: list[dict]) -> list[list[str]]:
    """Groups (size >= 2) of methods with identical outcomes on every problem.
    Equal (status, solved, expansions, proof_len) means equal cells at every
    budget, so such methods are indistinguishable in this run."""
    sig: dict[str, list] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        sig[r["method"]].append((r["id"], r["status"], r["solved"], r["expansions"], r["proof_len"]))
    groups: dict[tuple, list[str]] = defaultdict(list)
    for m in sorted(sig):
        groups[tuple(sig[m])].append(m)
    return sorted(g for g in groups.values() if len(g) > 1)


def paired(rows: list[dict], method: str, reference: str, b: int) -> tuple[int, int]:
    """(solved by method only, solved by reference only) at budget b, over
    problems where neither cell is censored."""
    idx = _index(rows)
    a = c = 0
    for (m, i), r in idx.items():
        if m != reference or (method, i) not in idx:
            continue
        x, y = cell(idx[method, i], b), cell(r, b)
        if "censored" in (x, y):
            continue
        a += x == "solved" and y != "solved"
        c += y == "solved" and x != "solved"
    return a, c


def paired_table(rows: list[dict], reference: str, budgets: list[int]) -> tuple[list[dict], list[str]]:
    """McNemar tests of every method against ``reference`` at every budget,
    Holm-adjusted as one family. Methods identical to the reference on every
    problem are left out: their p = 1 is known in advance, and including them
    would only inflate the Holm family. Returns (tests, untested methods)."""
    same = next((g for g in identical_methods(rows) if reference in g), [reference])
    untested = [m for m in same if m != reference]
    others = [m for m in _methods(rows) if m not in same]
    tests = []
    for m in others:
        for b in sorted(budgets):
            a, c = paired(rows, m, reference, b)
            tests.append({"method": m, "budget": b, "a": a, "b": c, "p": mcnemar_exact(a, c)})
    for t, q in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = q
    return tests, untested


def method_colors(methods: list[str]) -> dict[str, str]:
    """Fixed colour per method (by sorted name) so every figure agrees."""
    return {m: TAB10[k % len(TAB10)] for k, m in enumerate(sorted(set(methods)))}


# --------------------------------------------------------------------------
# Loading and markdown
# --------------------------------------------------------------------------

def load_run(run_dir: str | Path) -> tuple[list[dict], dict]:
    run = Path(run_dir)
    res, cfg = run / "results.jsonl", run / "config.json"
    for p in (res, cfg):
        if not p.is_file():
            raise BenchError(f"{p} not found; is {run} a directory written by `isomer bench`?")
    with open(res, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise BenchError(f"{res} is empty")
    return rows, json.loads(cfg.read_text(encoding="utf-8"))


def _f(x: float | None, spec: str = ".3f") -> str:
    return "–" if x is None else format(x, spec)


def _table(head: list[str], body: list[list[str]], align: str | None = None) -> list[str]:
    align = align or "l" + "r" * (len(head) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    return ["| " + " | ".join(row) + " |" for row in (head, sep, *body)]


def render_markdown(rows: list[dict], config: dict, reference: str, reference_note: str = "") -> str:
    budgets = sorted(config["budgets"])
    big = budgets[-1]
    methods = _methods(rows)
    summary = summarize(rows, budgets)
    out: list[str] = [f"# Results: {config.get('split', '?')}", ""]

    # Header ---------------------------------------------------------------
    out += [
        f"- split: `{config.get('split')}`, problems: {config.get('n_problems')}",
        f"- methods: {', '.join(f'`{m}`' for m in methods)}",
        f"- budgets (node expansions): {', '.join(map(str, budgets))}",
        f"- dataset format: `{config.get('dataset_format')}`",
        f"- git commit: `{config.get('git_commit')}`",
        ("- smaller budgets are derived from one run at the largest budget (prefix property; "
        "see `isomer.search.common`). Solved = accepted by the trusted kernel."),
    ]
    if config.get("git_dirty"):
        out += ["", ("> **WARNING:** this run was made from a dirty working tree; the commit above "
                "does not fully identify the code that produced it.")]

    # Main table -------------------------------------------------------------
    best = {b: max((s["rate"] for s in summary if s["budget"] == b and s["rate"] is not None), default=None)
            for b in budgets}
    body = []
    for s in summary:
        rate = "–" if s["rate"] is None else f"{s['rate']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
        if s["rate"] is not None and s["rate"] == best[s["budget"]]:
            rate = f"**{rate}**"
        body.append([f"`{s['method']}`", str(s["budget"]), f"{s['solved']}/{s['n']}", rate,
                     _f(s["median_expansions"], ".0f"), _f(s["median_nodes_stored"], ".0f"),
                     _f(s["opt_gap"])])
    out += ["", "## Solve rate", "",
            ("Rate with 95% Wilson interval; best rate per budget in bold. Medians and the "
            "optimality gap (mean proof_len / dist) are over that method's solved problems."), ""]
    out += _table(["method", "budget", "solved", "rate [95% CI]", "median expansions",
                   "median nodes stored", "opt gap"], body)

    # Fair optimality gap ------------------------------------------------------
    n_common, gaps = fair_gap(rows, big)
    out += ["", "## Fair optimality gap", "",
            (f"Mean proof_len / dist over the **{n_common}** problems with known dist solved by "
            f"every method at budget {big}. 1.000 = always shortest."), ""]
    out += _table(["method", "fair gap"], [[f"`{m}`", _f(gaps[m])] for m in methods])

    # Paired comparison ------------------------------------------------------
    tests, untested = paired_table(rows, reference, budgets)
    out += ["", f"## Paired comparison vs `{reference}`", "",
            (f"a = solved by the method but not `{reference}`; b = the reverse. Exact two-sided "
            "McNemar test; Holm-Bonferroni adjusted across all rows. * = adjusted p < 0.05."), ""]
    body = [[f"`{t['method']}`", str(t["budget"]), str(t["a"]), str(t["b"]), f"{t['p']:.3g}",
             f"{t['p_holm']:.3g}", "*" if t["p_holm"] < 0.05 else ""] for t in tests]
    out += _table(["method", "budget", "a", "b", "p", "p (Holm)", "sig"], body, "lrrrrrl") if body \
        else ["(no other methods)"]
    if untested:
        out += ["", "Not tested (identical to reference on every problem): "
                + ", ".join(f"`{m}`" for m in untested)]

    # Per theory -------------------------------------------------------------
    theories = sorted({r["theory"] for r in rows})
    per = {t: {s["method"]: s for s in summarize([r for r in rows if r["theory"] == t], [big])}
           for t in theories}
    body = [[f"`{m}`", *(f"{per[t][m]['solved']}/{per[t][m]['n']} ({_f(per[t][m]['rate'])})"
                         if m in per[t] else "–" for t in theories)] for m in methods]
    out += ["", f"## Per theory (budget {big})", ""]
    out += _table(["method", *theories], body)

    # Notes ------------------------------------------------------------------
    censored = sum(s["censored"] for s in summary)
    errors = sum(r["status"] == "error" for r in rows)
    unsound = sum(r["status"] == "unsound" for r in rows)
    notes = [f"censored cells (wall-clock limit): {censored}"]
    status = f"error rows: {errors}, unsound rows: {unsound}"
    notes.append(status if not (errors or unsound) else f"**NOT REPORTABLE**: {status} (must both be 0)")
    for g in identical_methods(rows):
        names = " and ".join(f"`{m}`" for m in g) if len(g) == 2 else ", ".join(f"`{m}`" for m in g)
        notes.append(f"{names} identical on all problems")
    if reference_note:
        notes.append(reference_note)
    out += ["", "## Notes", "", *(f"- {n}" for n in notes), ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# --------------------------------------------------------------------------

def _cumulative(values: list[float], x_max: float) -> tuple[list[float], list[int]]:
    """Step-curve points: how many values are <= x, for x from 1 to x_max."""
    xs = sorted(max(1.0, v) for v in values)
    return [1.0, *xs, max(x_max, xs[-1] if xs else 1.0)], [0, *range(1, len(xs) + 1), len(xs)]


def plot_all(rows: list[dict], config: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    big = max(config["budgets"])
    methods = _methods(rows)
    colors = method_colors(methods)
    n_problems = len({r["id"] for r in rows})
    title = f"{config.get('split', '?')}, max budget {big}"
    written: list[Path] = []
    # Identical methods would hide each other: draw later ones dashed and say so.
    style = {m: {"color": colors[m], "label": m} for m in methods}
    for group in identical_methods(rows):
        for m in group[1:]:
            style[m].update(linestyle="--", label=f"{m} (= {group[0]})")

    def save(fig, ax, name: str) -> None:
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # 1. Solve rate by true distance --------------------------------------------
    dists = sorted({r["dist"] for r in rows if r["dist"] is not None})
    buckets: list[int | None] = [*dists, *([None] if any(r["dist"] is None for r in rows) else [])]
    counts = [len({r["id"] for r in rows if r["dist"] == d}) for d in buckets]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, m in enumerate(methods):
        ys, lo, hi = [], [], []
        for d in buckets:
            cells = [cell(r, big) for r in rows if r["method"] == m and r["dist"] == d]
            cells = [c for c in cells if c != "censored"]
            solved = sum(c == "solved" for c in cells)
            y = solved / len(cells) if cells else float("nan")
            ci_lo, ci_hi = wilson(solved, len(cells))
            ys.append(y)
            lo.append(y - ci_lo if cells else float("nan"))
            hi.append(ci_hi - y if cells else float("nan"))
        # Offset each method slightly so the 95% Wilson bars don't overlap.
        dx = (k - (len(methods) - 1) / 2) * 0.08
        ax.errorbar([i + dx for i in range(len(buckets))], ys, yerr=[lo, hi], marker="o",
                    markersize=4, capsize=2.5, elinewidth=1, **style[m])
    ax.set_xticks(range(len(buckets)),
                  [f"{'unk' if d is None else d}\n(n={n})" for d, n in zip(buckets, counts)])
    ax.set_xlabel("true shortest proof length (steps)")
    ax.set_ylabel(f"solve rate at {big} expansions")
    ax.set_ylim(-0.03, 1.03)
    save(fig, ax, "solve_by_dist.png")

    # 2. Cactus plot ----------------------------------------------------------
    solved = {m: [r for r in rows if r["method"] == m and cell(r, big) == "solved"] for m in methods}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        xs, ys = _cumulative([r["expansions"] for r in solved[m]], big)
        ax.step(xs, ys, where="post", **style[m])
    ax.set_xscale("log")
    ax.set_xlim(1, big)
    ax.set_xlabel("expansion budget (nodes expanded, log scale)")
    ax.set_ylabel(f"problems solved (of {n_problems})")
    save(fig, ax, "cactus.png")

    # 3. Memory view ------------------------------------------------------------
    nodes_max = max((r["nodes_stored"] or 1) for r in rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        xs, ys = _cumulative([r["nodes_stored"] for r in solved[m]], nodes_max)
        ax.step(xs, ys, where="post", **style[m])
    ax.set_xscale("log")
    ax.set_xlim(1, nodes_max)
    ax.set_xlabel("nodes stored when solved (terms kept in memory, log scale)")
    ax.set_ylabel(f"problems solved (of {n_problems})")
    save(fig, ax, "nodes_vs_solved.png")
    return written


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def report(run_dir: str | Path, reference: str | None = None, plots: bool = True) -> int:
    """Write ``<run_dir>/report/``. Returns 1 if the run has error/unsound rows
    (the report is still written, and it says so), else 0."""
    rows, config = load_run(run_dir)
    methods = _methods(rows)
    note = ""
    if reference is None:
        reference = DEFAULT_REFERENCE if DEFAULT_REFERENCE in methods else methods[0]
        if reference != DEFAULT_REFERENCE:
            note = f"reference `{DEFAULT_REFERENCE}` is not in this run; compared against `{reference}` instead"
    elif reference not in methods:
        raise BenchError(f"reference {reference!r} is not in this run; methods: {', '.join(methods)}")

    out = Path(run_dir) / "report"
    out.mkdir(exist_ok=True)
    md = out / "results.md"
    md.write_text(render_markdown(rows, config, reference, note), encoding="utf-8")
    written = [md]
    if plots and importlib.util.find_spec("matplotlib") is None:
        print('plots skipped: matplotlib is not installed (pip install -e ".[plots]")')
    elif plots:
        written += plot_all(rows, config, out)
    for p in written:
        print(f"wrote {p}")

    bad = sum(r["status"] in ("error", "unsound") for r in rows)
    if bad:
        print(f"WARNING: {bad} error/unsound rows; this run is NOT reportable", flush=True)
    return 1 if bad else 0
