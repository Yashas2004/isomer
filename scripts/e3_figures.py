"""E3 Figure 1: mean predicted distance vs true distance (baseline seeds, d6 seeds, control).

    python scripts/e3_figures.py              # docs/experiments/E3/saturation.png (README, with source footer)
    python scripts/e3_figures.py --paper      # docs/experiments/E3/saturation.pdf (TMLR single column, no footer)
    python scripts/e3_figures.py --summary docs/experiments/E3/d6_family_summary.md --out runs/e3/saturation_e2.png

Reads only the committed summary file (see ``isomer.e3.figures``) and checks
that every value drawn is the summary's value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import figures

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default=figures.SUMMARY)
    ap.add_argument("--paper", action="store_true", help=f"PDF for the paper (default --out {figures.OUT_PAPER})")
    ap.add_argument("--out", default=None, help=f"default: {figures.OUT}, or {figures.OUT_PAPER} with --paper")
    a = ap.parse_args(argv)
    out = a.out or (figures.OUT_PAPER if a.paper else figures.OUT)
    try:
        table = figures.mean_prediction_table((ROOT / a.summary).read_text(encoding="utf-8"))
    except (OSError, figures.FigureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _, drawn = figures.plot(table, ROOT / out, Path(a.summary).as_posix(), paper=a.paper)
    want = {"baseline": [r["baseline"][0] for r in table["rows"]], "d6": [r["d6"][0] for r in table["rows"]],
            "control": [r["control"] for r in table["rows"]]}
    if drawn != want:
        print(f"error: drawn values differ from {a.summary}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({len(table['rows'])} distances; drawn values equal the summary's)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
