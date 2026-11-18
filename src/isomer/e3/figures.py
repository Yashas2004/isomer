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
OUT_PAPER = "docs/experiments/E3/saturation.pdf"
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


PAPER_SIZE = (6.5, 3.4)    # inches: TMLR's single-column text block is 6.5 in wide
SCREEN_SIZE = (7.0, 4.6)


def spread(values: list[float], gap: float) -> list[float]:
    """Label positions for ``values`` (same order), at least ``gap`` apart, each group of
    crowded labels centred on the mean of its values, so no label drifts far from its line."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        groups.append([i])
        while len(groups) > 1:          # merge the last two groups while they would overlap
            a, b = groups[-2], groups[-1]
            centre_a = sum(values[j] for j in a) / len(a)
            centre_b = sum(values[j] for j in b) / len(b)
            if centre_b - (len(b) - 1) * gap / 2 - (centre_a + (len(a) - 1) * gap / 2) >= gap:
                break
            groups[-2:] = [a + b]
    pos = [0.0] * len(values)
    for g in groups:
        centre = sum(values[j] for j in g) / len(g)
        for n, j in enumerate(g):
            pos[j] = centre + (n - (len(g) - 1) / 2) * gap
    return pos


def plot(table: dict, out: str | Path, source: str, paper: bool = False) -> tuple[Path, dict[str, list[float]]]:
    """Draw the figure; return the file and the y values actually drawn per series (for checking).
    ``paper``: TMLR single-column size, no title and no source footer; the format follows the file suffix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table["rows"]
    ks = [r["k"] for r in rows]
    fs = 9 if paper else 10
    plt.rcParams.update({"font.family": "sans-serif", "font.size": fs, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                         "pdf.fonttype": 42})          # embed TrueType, as journals ask
    fig, ax = plt.subplots(figsize=PAPER_SIZE if paper else SCREEN_SIZE, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    top = max(ks) + 0.5
    (calibrated,) = ax.plot([0.5, top], [0.5, top], linestyle=":", color=MUTED, linewidth=1.2,
                            label="calibrated (prediction = k)")
    for fam, name in (("baseline", "baseline h-v1"), ("d6", "d6")):
        c, y = COLORS[fam], TRAIN_MAX[fam]
        ax.axhline(y, linestyle="--", color=c, linewidth=1.0, alpha=0.8)
        ax.text(0.65, y + 0.08, f"{name} training maximum ({y})", color=INK_2, fontsize=fs - 1.5, va="bottom")

    drawn: dict[str, list[float]] = {}
    # Markers differ in shape and fill so coinciding points stay visible: the d6 square is hollow
    # and larger, the control diamond is solid, smaller and drawn on top of it.
    style = {"baseline": {"marker": "o", "markersize": 5, "markerfacecolor": COLORS["baseline"],
                              "markeredgecolor": SURFACE, "markeredgewidth": 1, "zorder": 3},
             "d6": {"marker": "s", "markersize": 6, "markerfacecolor": SURFACE, "markeredgecolor": COLORS["d6"],
                        "markeredgewidth": 1.3, "zorder": 3},
             "control": {"marker": "D", "markersize": 4, "markerfacecolor": COLORS["control"],
                             "markeredgecolor": COLORS["control"], "markeredgewidth": 0.5, "zorder": 4}}
    for fam in ("baseline", "d6"):
        ax.fill_between(ks, [r[fam][1] for r in rows], [r[fam][2] for r in rows], color=COLORS[fam], alpha=0.18,
                        linewidth=0, zorder=1)
        (line,) = ax.plot(ks, [r[fam][0] for r in rows], color=COLORS[fam], linewidth=2, **style[fam])
        drawn[fam] = [float(v) for v in line.get_ydata()]
    (line,) = ax.plot(ks, [r["control"] for r in rows], color=COLORS["control"], linewidth=1.6, **style["control"])
    drawn["control"] = [float(v) for v in line.get_ydata()]

    ax.set_xlim(0.5, ks[-1] + 1.5)
    ax.set_ylim(min(0, min(r["d6"][1] for r in rows)) - 0.3, top)
    ax.set_xticks(ks)
    ax.set_xlabel("true distance to the goal, k (exact, shortest proof)")
    ax.set_ylabel("mean predicted distance")
    if not paper:           # in the paper the LaTeX caption carries the message
        ax.set_title("The predicted distance levels off below each model's training maximum", color=INK,
                     fontsize=fs + 1, loc="left")
    ax.legend(handles=[calibrated], loc="lower right", frameon=False, fontsize=fs - 1.5, labelcolor=INK)
    if not paper:
        fig.text(0.01, 0.01, f"Source: {source}. Scored on-path states; distances with fewer than 10 states left "
                 "out.", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1) if not paper else (0, 0, 1, 1))

    # Direct end labels, placed once the axes size is final: at least 1.4 line heights apart,
    # with a short leader where a label had to move off its point.
    names = {"baseline": "baseline", "d6": "d6", "control": "control"}
    fams = list(names)
    ys = [drawn[f][-1] for f in fams]
    y_lo, y_hi = ax.get_ylim()
    height_in = ax.get_window_extent().height / fig.dpi
    gap = 1.4 * fs / 72 * (y_hi - y_lo) / height_in
    x_end, x_text = ks[-1], ks[-1] + 0.3
    for fam, y, y_text in zip(fams, ys, spread(ys, gap), strict=True):
        if abs(y_text - y) > 0.02:
            ax.plot([x_end + 0.1, x_text - 0.04], [y, y_text], color=MUTED, linewidth=0.6, zorder=2)
        ax.text(x_text, y_text, names[fam], color=INK, fontsize=fs - 0.5, va="center", zorder=5,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta = {"Software": None} if Path(out).suffix == ".png" else {"Creator": None, "Producer": None,
                                                                  "CreationDate": None}
    fig.savefig(out, facecolor=SURFACE, metadata=meta)        # no version or date stamp: same input, same file
    plt.close(fig)
    return Path(out), drawn
