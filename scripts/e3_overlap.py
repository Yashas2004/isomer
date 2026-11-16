"""E3: overlap between evaluated (state, goal) pairs and the training states.

    python scripts/e3_overlap.py        # -> runs/e3/overlap_summary.md

Reads the labels, scores and bench results under runs/e3 and the training
states in data/states-v2 and data/states-v2-d6. Runs no model and no search,
changes no existing file; about a minute (it parses 280 MB of training states).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import overlap, seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e3", default="runs/e3")
    ap.add_argument("--out", default=None, help="default: E3/overlap_summary.md")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    e3 = Path(a.e3)
    try:
        out = overlap.write(e3, Path(a.out) if a.out else e3 / "overlap_summary.md")
    except (seeds.SeedsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
