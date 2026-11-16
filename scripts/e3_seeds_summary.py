"""E3 seed study, step 4: write runs/e3/seeds_summary.md from the finished runs.

    python scripts/e3_seeds_summary.py

Needs the outputs of e3_label.py, e3_score_seeds.py and e3_bench_seeds.py.
Takes seconds; runs no model and no search.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import seeds, seeds_summary

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e3", default="runs/e3")
    ap.add_argument("--out", default=None, help="default: E3/seeds_summary.md")
    ap.add_argument("--e2-run", default="runs/E2-test_depth", help="E2 run to compare the h-v1 search with")
    a = ap.parse_args(argv)
    os.chdir(ROOT)      # relative paths, so the summary names runs/..., not this machine's directories
    e3 = Path(a.e3)
    try:
        out = seeds_summary.write(e3, Path(a.out) if a.out else e3 / "seeds_summary.md", e2_run=Path(a.e2_run))
    except (seeds.SeedsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
