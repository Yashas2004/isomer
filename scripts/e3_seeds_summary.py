"""E3 seed study: write the summary of the finished runs.

    python scripts/e3_seeds_summary.py
        -> runs/e3/seeds_summary.md (h-v1 and its two reseeded runs)
    python scripts/e3_seeds_summary.py --compare h-d6-s0=runs/h-d6-s0/best.pt
        -> runs/e3/d6_summary.md (that model against the baseline seeds and the
           pre-registered prediction, docs/experiments/E3/d6_prediction.md)

Needs the outputs of e3_label.py, e3_score_seeds.py and e3_bench_seeds.py for
every model named. Takes seconds; runs no model and no search.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTION = "docs/experiments/E3/d6_prediction.md"


def _last_commit(path: str) -> str | None:
    try:
        cp = subprocess.run(["git", "log", "-1", "--format=%h", "--", path], check=False, capture_output=True,
                            text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import seeds, seeds_compare, seeds_summary

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e3", default="runs/e3")
    ap.add_argument("--out", default=None, help="default: E3/seeds_summary.md, or E3/d6_summary.md with --compare")
    ap.add_argument("--models", default=None,
                    help="comma-separated NAME or NAME=CHECKPOINT for the seed summary (default: the h-v1 seeds)")
    ap.add_argument("--compare", default=None, help="NAME=CHECKPOINT of one new model to compare with --baseline")
    ap.add_argument("--baseline", default=None, help="baseline models for --compare (default: the h-v1 seeds)")
    ap.add_argument("--e2-run", default="runs/E2-test_depth", help="E2 run to compare the h-v1 search with")
    a = ap.parse_args(argv)
    os.chdir(ROOT)      # relative paths, so the summary names runs/..., not this machine's directories
    e3 = Path(a.e3)
    try:
        if a.compare:
            new = seeds.parse_models(a.compare)
            if len(new) != 1:
                raise seeds.SeedsError("--compare takes exactly one NAME=CHECKPOINT")
            (name, ckpt), = new.items()
            out = seeds_compare.write(e3, Path(a.out) if a.out else e3 / "d6_summary.md",
                                      seeds.parse_models(a.baseline), name, ckpt, _last_commit(PREDICTION))
        else:
            out = seeds_summary.write(e3, Path(a.out) if a.out else e3 / "seeds_summary.md",
                                      seeds.parse_models(a.models), e2_run=Path(a.e2_run))
    except (seeds.SeedsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
