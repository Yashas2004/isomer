"""E3 Figure 1: mean predicted distance vs true distance (baseline seeds, d6 seeds, control).

    python scripts/e3_figures.py                       # fresh sample -> docs/experiments/E3/saturation.png
    python scripts/e3_figures.py --summary docs/experiments/E3/d6_family_summary.md --out runs/e3/saturation_e2.png

Reads only the committed summary file (see ``isomer.e3.figures``).
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
    ap.add_argument("--out", default=figures.OUT)
    a = ap.parse_args(argv)
    try:
        table = figures.mean_prediction_table((ROOT / a.summary).read_text(encoding="utf-8"))
    except (OSError, figures.FigureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    figures.plot(table, ROOT / a.out, Path(a.summary).as_posix())
    print(f"wrote {a.out} ({len(table['rows'])} distances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
