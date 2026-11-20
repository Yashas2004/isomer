"""E3-d5, last step: judge the pre-registered rules for h-d5-s0 and write the summary.

    python scripts/e3_d5_summary.py        # -> runs/e3/d5_summary.md

Reads runs/e3 (E2 sample) and runs/e3-fresh (fresh sample): labels, scores and bench results of
the three baseline seeds, the three d6 seeds, the control and h-d5-s0. Refuses partial inputs.
The commit of docs/experiments/E3/d5_prediction.md is looked up with git and printed in the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTION = "docs/experiments/E3/d5_prediction.md"


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import d5_summary, seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e2", default="runs/e3", help="directory of the E2-sample outputs")
    ap.add_argument("--fresh", default="runs/e3-fresh", help="directory of the fresh-sample outputs")
    ap.add_argument("--out", default="runs/e3/d5_summary.md")
    a = ap.parse_args(argv)
    try:
        path = d5_summary.write(ROOT / a.out, e2=ROOT / a.e2, fresh=ROOT / a.fresh,
                                prediction_commit=d5_summary.first_commit(ROOT, PREDICTION))
    except seeds.SeedsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shown = path.relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path
    print(f"wrote {shown.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
