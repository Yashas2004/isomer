"""Equal-time comparison (E2 "Equal time"), derived purely from logged rows.

Neural rows come from the main run (expansion budget up to 1000); the
baselines come from a separate time run with a much larger expansion budget,
so that they are not cut off by their (cheap) expansions long before the
neural searcher would run out of seconds. No search is re-run.

**Agreement check first.** The time run must reproduce the main run at every
budget the main run covers (same cells, and for solved problems the same proof,
length and expansions). That follows from the prefix property, so a mismatch
means the two runs are not the same searcher on the same problems, and the
analysis aborts.

**Censoring at a time limit T** is exact, never extrapolated. A row's
``seconds`` is its time to stop: the solve if solved, else its expansion budget
or node cap. Within T a row is

* solved, if solved and seconds <= T;
* unsolved, if solved after T, or unsolved and still running at T
  (seconds >= T), or ``stop == "exhausted"`` (no proof exists in the action
  space, so more time cannot help);
* censored, if unsolved and it stopped before T on its expansion budget or node
  cap: we do not know what it would have done with the remaining time.
  Censored rows are excluded from n and counted separately.

**Times are machine-dependent** and here measured under different worker
counts (neural: workers sharing one GPU; baselines: CPU workers), so the
report states the hardware and workers up front.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .report import _f, _table, holm, load_run, mcnemar_exact, method_colors
from .run import BenchError, cell, wilson

TIME_LIMITS = (1.0, 3.0, 10.0, 30.0)
NEURAL = "neural-w1"
BASELINES = ("bestfirst-symbol_bag-w1", "bfs-fwd")
STATE_CUTOFF = 126   # longest (state, goal) the model was trained on, in tokens
STATE_METHODS = (NEURAL, "bestfirst-symbol_bag-w1")


def short(method: str) -> str:
    """``neural-w1@runs/h-v1/best.pt#bf16-fast`` -> ``neural-w1``."""
    return method.partition("@")[0]


def _by_method(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(short(r["method"]), {})[r["id"]] = r
    return out


# --------------------------------------------------------------------------
# Pure pieces
# --------------------------------------------------------------------------

def time_cell(row: dict, t: float) -> str:
    """Outcome of ``row`` within ``t`` seconds: "solved", "unsolved" or "censored"."""
    solved = row["status"] == "ok" and row["solved"]
    if solved:
        return "solved" if row["seconds"] <= t else "unsolved"
    if row["stop"] == "exhausted" or row["seconds"] >= t:
        return "unsolved"
    return "censored"


def check_agreement(main_rows: list[dict], time_rows: list[dict], budgets: list[int],
                    baselines: tuple[str, ...] = BASELINES) -> list[str]:
    """Mismatches between the time run and the main run at the main run's budgets."""
    bad: list[str] = []
    for r in (*main_rows, *time_rows):
        if r["status"] != "ok":
            bad.append(f"{r['method']} {r['id']}: status {r['status']}")
    main, tim = _by_method(main_rows), _by_method(time_rows)
    for m in baselines:
        if m not in main or m not in tim:
            bad.append(f"{m}: missing from the {'main' if m not in main else 'time'} run")
            continue
        a, b = main[m], tim[m]
        for pid in sorted(set(a) ^ set(b)):
            bad.append(f"{m} {pid}: only in the {'main' if pid in a else 'time'} run")
        for pid in sorted(set(a) & set(b)):
            x, y = a[pid], b[pid]
            for bud in sorted(budgets):
                if cell(x, bud) != cell(y, bud):
                    bad.append(f"{m} {pid}: budget {bud}: main {cell(x, bud)}, time {cell(y, bud)}")
            if cell(x, max(budgets)) == "solved":
                for k in ("proof_len", "expansions", "proof"):
                    if x[k] != y[k]:
                        bad.append(f"{m} {pid}: {k} differs (main {x[k] if k != 'proof' else '...'}, "
                                   f"time {y[k] if k != 'proof' else '...'})")
    return bad


def time_table(rows_by_method: dict[str, list[dict]], limits: tuple[float, ...] = TIME_LIMITS) -> list[dict]:
    out = []
    for m, rows in rows_by_method.items():
        for t in limits:
            cells = [time_cell(r, t) for r in rows]
            n = sum(c != "censored" for c in cells)
            k = cells.count("solved")
            lo, hi = wilson(k, n)
            out.append({"method": m, "T": t, "n": n, "solved": k, "rate": k / n if n else None,
                        "ci_lo": lo, "ci_hi": hi, "censored": len(cells) - n})
    return out


def paired_time(a_rows: list[dict], b_rows: list[dict], t: float) -> tuple[int, int, int]:
    """(solved by a only, solved by b only, pairs used) within ``t``, over
    problems uncensored for both methods."""
    b_by = {r["id"]: r for r in b_rows}
    only_a = only_b = used = 0
    for r in a_rows:
        if r["id"] not in b_by:
            continue
        x, y = time_cell(r, t), time_cell(b_by[r["id"]], t)
        if "censored" in (x, y):
            continue
        used += 1
        only_a += x == "solved" and y != "solved"
        only_b += y == "solved" and x != "solved"
    return only_a, only_b, used


def time_mcnemar(rows_by_method: dict[str, list[dict]], limits: tuple[float, ...] = TIME_LIMITS,
                 method: str = NEURAL, references: tuple[str, ...] = BASELINES) -> list[dict]:
    """Exact McNemar of ``method`` vs each reference at each limit; Holm over all rows."""
    tests = []
    for ref in references:
        for t in limits:
            a, b, n = paired_time(rows_by_method[method], rows_by_method[ref], t)
            tests.append({"method": method, "reference": ref, "T": t, "pairs": n, "a": a, "b": b,
                          "p": mcnemar_exact(a, b)})
    for test, q in zip(tests, holm([t["p"] for t in tests])):
        test["p_holm"] = q
    return tests


def censor_point(rows: list[dict]) -> float | None:
    """Smallest seconds among a method's unsolved rows: beyond it the method's
    time curve is incomplete (some runs stopped on expansions or nodes)."""
    return min((r["seconds"] for r in rows
                if not (r["status"] == "ok" and r["solved"]) and r["stop"] != "exhausted"), default=None)


def state_length_table(main_rows: list[dict], budget: int = 1000, cutoff: int = STATE_CUTOFF,
                       methods: tuple[str, ...] = STATE_METHODS) -> list[dict]:
    """Solve rate at ``budget`` by max_state_tokens <= cutoff vs > cutoff."""
    by = {m: [r for r in main_rows if short(r["method"]) == m] for m in methods}
    out = []
    for m, rows in by.items():
        for group, keep in ((f"<= {cutoff}", lambda v: v <= cutoff), (f"> {cutoff}", lambda v: v > cutoff)):
            cells = [cell(r, budget) for r in rows if keep(r["max_state_tokens"])]
            cells = [c for c in cells if c != "censored"]
            k, n = cells.count("solved"), len(cells)
            lo, hi = wilson(k, n)
            out.append({"method": m, "group": group, "n": n, "solved": k, "rate": k / n if n else None,
                        "ci_lo": lo, "ci_hi": hi})
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _rate(s: dict) -> str:
    return "–" if s["rate"] is None else f"{s['rate']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"


def _secs(t: float) -> str:
    return f"{t:g} s"


def render_time_md(res: dict) -> str:
    h = res["header"]
    out = [f"# Equal-time comparison: {h['split']}", "",
           (f"- main run (neural): `{h['main_dir']}`, commit `{h['main_commit']}`, {h['main_workers']} workers, "
           f"device `{h['neural_device']}`, max budget {h['main_max_budget']}"),
           (f"- time run (baselines): `{h['time_dir']}`, commit `{h['time_commit']}`, {h['time_workers']} workers, "
           f"max budget {h['time_max_budget']}, max nodes {h['time_max_nodes']}"),
           (f"- hardware (config.json): `{h['platform']}`, {h['cpu_count']} logical CPUs, Python {h['python']}; "
           "the GPU model is not recorded in config.json"),
           (f"- **Times are machine-dependent and measured under different worker counts** (neural: "
           f"{h['main_workers']} workers sharing one GPU; baselines: {h['time_workers']} CPU workers)."),
           ("- Neural seconds include, for the first problem in each worker process, loading the model and "
           f"starting CUDA (the searcher's clock starts before the load): at most {h['main_workers']} neural "
           "rows are inflated, against neural; the logs do not say which."),
           f"- agreement check: {res['agreement']}",
           ("- within T: solved = solved and seconds <= T; unsolved = solved after T, or unsolved and still "
            "running at T; **censored** = unsolved but stopped (expansion budget / node cap) before T, "
            "excluded from n."),
           ""]
    out += ["## Solve rate within T", "", "Rate with 95% Wilson interval over uncensored problems.", ""]
    out += _table(["method", "T", "n", "solved", "rate [95% CI]", "censored"],
                  [[f"`{s['method']}`", _secs(s["T"]), str(s["n"]), str(s["solved"]), _rate(s),
                    str(s["censored"])] for s in res["time_table"]])
    out += ["", f"## Paired comparison: `{NEURAL}` vs baselines", "",
            (f"Problems uncensored for both methods. a = solved within T by `{NEURAL}` only; b = by the "
             "baseline only. Exact two-sided McNemar; Holm-Bonferroni across all rows. * = adjusted p < 0.05."),
            ""]
    out += _table(["vs", "T", "pairs", "a", "b", "p", "p (Holm)", "sig"],
                  [[f"`{t['reference']}`", _secs(t["T"]), str(t["pairs"]), str(t["a"]), str(t["b"]),
                    f"{t['p']:.3g}", f"{t['p_holm']:.3g}", "*" if t["p_holm"] < 0.05 else ""]
                   for t in res["mcnemar"]], "lrrrrrrl")
    out += ["", "## Censoring points", "",
            "Smallest seconds among a method's unsolved rows: beyond it, its curve is incomplete.", ""]
    out += _table(["method", "censoring point (s)"],
                  [[f"`{m}`", _f(v, ".2f") if v is not None else "none (all solved)"]
                   for m, v in res["censor_points"].items()])
    return "\n".join(out) + "\n"


def render_state_md(res: dict) -> str:
    h = res["header"]
    out = [f"# Solve rate by maximum state length reached: {h['split']}", "",
           (f"- source: `{h['main_dir']}`, budget {res['state_budget']}; max_state_tokens = longest "
           "[CLS] state [SEP] goal [SEP] among the states the search stored"),
           f"- cutoff {STATE_CUTOFF} tokens: the longest pair in training (RoPE length extrapolation check)",
           ("- **Descriptive, not causal:** max_state_tokens is partly an outcome of the search itself "
            "(longer searches reach longer states), so unsolved runs, which use the whole budget, tend to "
            "have longer states."),
           ""]
    out += _table(["method", "max_state_tokens", "n", "solved", "rate [95% CI]"],
                  [[f"`{s['method']}`", s["group"], str(s["n"]), str(s["solved"]), _rate(s)]
                   for s in res["state_length"]])
    return "\n".join(out) + "\n"


def plot_time(rows_by_method: dict[str, list[dict]], colors: dict[str, str], censor: dict[str, float | None],
              split: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(rows) for rows in rows_by_method.values())
    all_secs = [r["seconds"] for rows in rows_by_method.values() for r in rows]
    lo, hi = max(1e-3, min(all_secs)), max(all_secs)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for t in TIME_LIMITS:
        ax.axvline(t, color="0.8", linewidth=0.8, zorder=0)
    for m, rows in rows_by_method.items():
        xs = sorted(r["seconds"] for r in rows if r["status"] == "ok" and r["solved"])
        ax.step([lo, *xs, hi], [0, *range(1, len(xs) + 1), len(xs)], where="post", color=colors[m], label=m)
        if censor[m] is not None:
            ax.axvline(censor[m], color=colors[m], linestyle=":", linewidth=1.2,
                       label=f"{m}: censored beyond {censor[m]:.2f} s")
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("seconds to solve (log scale; machine-dependent)")
    ax.set_ylabel(f"problems solved (of {n})")
    ax.set_title(f"{split}: equal time (grey: T = 1, 3, 10, 30 s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def timecompare(main_dir: str | Path, time_dir: str | Path, out_dir: str | Path, plots: bool = True) -> dict:
    """Verify agreement, then write timecompare.md/.json, state_length.md and
    cactus_time.png into ``out_dir``. Raises ``BenchError`` (writing nothing) on a mismatch."""
    main_rows, main_cfg = load_run(main_dir)
    time_rows, time_cfg = load_run(time_dir)
    if main_cfg.get("ids_sha256") != time_cfg.get("ids_sha256") or main_cfg.get("split") != time_cfg.get("split"):
        raise BenchError(f"{main_dir} and {time_dir} are not on the same split and ids file")
    main_budgets = sorted(main_cfg["budgets"])
    bad = check_agreement(main_rows, time_rows, main_budgets)
    if bad:
        raise BenchError(f"time run disagrees with main run at budgets <= {main_budgets[-1]} "
                         f"({len(bad)} mismatches):\n  " + "\n  ".join(bad[:10]))
    main_by, time_by = _by_method(main_rows), _by_method(time_rows)
    if NEURAL not in main_by:
        raise BenchError(f"{main_dir} has no {NEURAL} rows")
    n_ids = len(main_by[NEURAL])
    agreement = (f"OK: {n_ids} problems x {len(main_budgets)} budgets ({', '.join(map(str, main_budgets))}) "
                 f"x {len(BASELINES)} baselines agree (cell; proof, proof_len, expansions when solved)")
    print(f"agreement: {agreement}")

    rows_by_method = {NEURAL: sorted(main_by[NEURAL].values(), key=lambda r: r["id"]),
                      **{m: sorted(time_by[m].values(), key=lambda r: r["id"]) for m in BASELINES}}
    neural_full = next(r["method"] for r in main_rows if short(r["method"]) == NEURAL)
    header = {
        "split": main_cfg.get("split"), "main_dir": Path(main_dir).as_posix(), "time_dir": Path(time_dir).as_posix(),
        "main_commit": main_cfg.get("git_commit"), "time_commit": time_cfg.get("git_commit"),
        "main_dirty": main_cfg.get("git_dirty"), "time_dirty": time_cfg.get("git_dirty"),
        "main_workers": main_cfg.get("workers"), "time_workers": time_cfg.get("workers"),
        "main_max_budget": main_budgets[-1], "time_max_budget": max(time_cfg["budgets"]),
        "time_max_nodes": time_cfg.get("budget", {}).get("max_nodes"),
        "platform": main_cfg.get("platform"), "cpu_count": main_cfg.get("cpu_count"),
        "python": main_cfg.get("python"), "neural_method": neural_full,
        "neural_device": ", ".join(sorted({str(r.get("device")) for r in rows_by_method[NEURAL]})),
        "time_platform": time_cfg.get("platform"),
    }
    censor = {m: censor_point(rows) for m, rows in rows_by_method.items()}
    res = {"header": header, "agreement": agreement, "time_limits": list(TIME_LIMITS),
           "time_table": time_table(rows_by_method), "mcnemar": time_mcnemar(rows_by_method),
           "censor_points": censor, "state_budget": main_budgets[-1],
           "state_length": state_length_table(main_rows, main_budgets[-1])}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (("timecompare.md", render_time_md(res)), ("state_length.md", render_state_md(res)),
                       ("timecompare.json", json.dumps(res, indent=2, sort_keys=True) + "\n")):
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
    if plots and importlib.util.find_spec("matplotlib") is None:
        print('plot skipped: matplotlib is not installed (pip install -e ".[plots]")')
    elif plots:
        # Colours over the main run's full method names, so they match its other figures.
        colors = {short(m): c for m, c in method_colors(sorted({r["method"] for r in main_rows})).items()}
        plot_time(rows_by_method, colors, censor, header["split"], out / "cactus_time.png")
        written.append(out / "cactus_time.png")
    for p in written:
        print(f"wrote {p}")
    return res
