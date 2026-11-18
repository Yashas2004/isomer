"""E3 Figure 1: mean predicted distance vs true distance, read only from a committed summary.

    python scripts/e3_figures.py      # -> docs/experiments/E3/saturation.png

The figure is drawn from the "Mean prediction" table of a committed summary
file (default ``docs/experiments/E3/fresh_summary.md``; ``d6_family_summary.md``
has the same table for the E2 sample). No model output is read, so the figure
cannot show a number that is not in the repository. The summaries give, per
distance, the mean and min-max range over the three seeds of each family and
the control's value; individual d6 seeds are not tabulated per distance, so
each family is drawn as its mean with a band spanning its seeds.

Rows marked ``\\*`` (fewer than 10 scored states) are left out, as everywhere in E3.
matplotlib is imported only when plotting.
"""

from __future__ import annotations

import re
from pathlib import Path

SUMMARY = "docs/experiments/E3/fresh_summary.md"
OUT = "docs/experiments/E3/saturation.png"
TRAIN_MAX = {"baseline": 5, "d6": 7}      # largest training-state distance of each family
# Reference palette slots 1-3 (validated all-pairs, light mode); ink and chrome from the same reference.
COLORS = {"baseline": "#2a78d6", "d6": "#eb6834", "control": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


class FigureError(Exception):
    """The summary does not hold the table the figure needs."""


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def mean_prediction_table(text: str) -> dict:
    """The first "### Mean prediction" table of a family-style summary:
    {"control": name, "rows": [{k, states, baseline: (mean, lo, hi), d6: (mean, lo, hi), control: value}]},
    leaving out rows marked ``\\*``."""
    lines = text.splitlines()
    try:
        start = lines.index("### Mean prediction")
    except ValueError:
        raise FigureError('no "### Mean prediction" section') from None
    table = []
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(table) < 3:
        raise FigureError("the mean prediction table is empty")
    head = table[0]
    need = ["k", "states", "baseline mean", "baseline range", "d6 mean", "d6 range"]
    if head[:6] != need or len(head) != 7:
        raise FigureError(f"unexpected columns {head}; expected {need} + the control")
    rows = []
    for cells in table[2:]:
        if "\\*" in cells[0]:
            continue
        out: dict = {"k": int(cells[0]), "states": int(cells[1]), "control": _num(cells[6])}
        for fam, (mean, rng) in {"baseline": (cells[2], cells[3]), "d6": (cells[4], cells[5])}.items():
            m = re.fullmatch(r"(\S+) to (\S+)", rng)
            if not m:
                raise FigureError(f"cannot read the range {rng!r}")
            out[fam] = (_num(mean), _num(m.group(1)), _num(m.group(2)))
        rows.append(out)
    return {"control": head[6], "rows": rows}


def plot(table: dict, out: str | Path, source: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table["rows"]
    ks = [r["k"] for r in rows]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    top = max(ks) + 0.5
    ax.plot([0.5, top], [0.5, top], linestyle=":", color=MUTED, linewidth=1.2, label="calibrated (prediction = k)")
    for fam, name in (("baseline", "baseline h-v1"), ("d6", "d6")):
        c, y = COLORS[fam], TRAIN_MAX[fam]
        ax.axhline(y, linestyle="--", color=c, linewidth=1.0, alpha=0.8)
        ax.text(0.65, y + 0.08, f"{name} training maximum ({y})", color=INK_2, fontsize=8.5, va="bottom")

    series = [("baseline", "baseline h-v1, 3 seeds (mean, range)", "o"), ("d6", "d6, 3 seeds (mean, range)", "s")]
    for fam, label, marker in series:
        mean = [r[fam][0] for r in rows]
        ax.fill_between(ks, [r[fam][1] for r in rows], [r[fam][2] for r in rows], color=COLORS[fam], alpha=0.18,
                        linewidth=0)
        ax.plot(ks, mean, color=COLORS[fam], linewidth=2, marker=marker, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ctrl = [r["control"] for r in rows]
    ax.plot(ks, ctrl, color=COLORS["control"], linewidth=2, marker="D", markersize=4.5, linestyle="-",
            markeredgecolor=SURFACE, markeredgewidth=1, label=f"control {table['control']}, 1 seed")

    # direct labels at the right end (identity is never color alone)
    last = rows[-1]
    ends = [("baseline", last["baseline"][0], "baseline"), ("d6", last["d6"][0], "d6"),
            ("control", last["control"], "control")]
    placed: list[float] = []
    for fam, y, text in sorted(ends, key=lambda e: e[1]):
        y_text = max([y, *[p + 0.32 for p in placed]])
        placed.append(y_text)
        ax.text(ks[-1] + 0.15, y_text, text, color=INK, fontsize=9, va="center")

    ax.set_xlim(0.5, ks[-1] + 1.1)
    ax.set_ylim(min(0, min(r["d6"][1] for r in rows)) - 0.3, top)
    ax.set_xticks(ks)
    ax.set_xlabel("true distance to the goal, k (exact, shortest proof)")
    ax.set_ylabel("mean predicted distance")
    ax.set_title("The predicted distance levels off below each model's training maximum", color=INK, fontsize=11,
                 loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.text(0.01, 0.01, f"Source: {source}. Scored on-path states; distances with fewer than 10 states left out.",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, metadata={"Software": None})      # no version stamp: same input, same file
    plt.close(fig)
    return Path(out)
