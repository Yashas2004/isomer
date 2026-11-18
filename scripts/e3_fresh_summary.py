"""E3-fresh, step E: judge H1-H4 by the committed rules and write the summary.

    python scripts/e3_fresh_summary.py        # -> runs/e3/fresh_summary.md

Reads runs/e3-fresh (labels, scores, bench-seeds) and the training-state files
for the overlap table. Refuses partial inputs: every one of the seven models
and the non-neural baseline run must be there. The commit of
docs/experiments/E3/fresh_prediction.md is looked up with git and printed in
the summary.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTION = "docs/experiments/E3/fresh_prediction.md"


def prediction_commit() -> str | None:
    """The first commit that added the prediction file (later commits may only append)."""
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", PREDICTION], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out[-1] if out else None


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import fresh_summary, seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e3", default="runs/e3-fresh")
    ap.add_argument("--out", default="runs/e3/fresh_summary.md")
    a = ap.parse_args(argv)
    try:
        path = fresh_summary.write(ROOT / a.e3, ROOT / a.out, root=ROOT, prediction_commit=prediction_commit())
    except seeds.SeedsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shown = path.relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path    # --out may be outside the repo
    print(f"wrote {shown.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
