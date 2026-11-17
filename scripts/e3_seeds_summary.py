"""E3 seed study: write the summary of the finished runs.

    python scripts/e3_seeds_summary.py
        -> runs/e3/seeds_summary.md (h-v1 and its two reseeded runs)
    python scripts/e3_seeds_summary.py --compare h-d6-s0
        -> runs/e3/d6_summary.md (that model against the baseline seeds and the
           pre-registered prediction, docs/experiments/E3/d6_prediction.md)
    python scripts/e3_seeds_summary.py --family
        -> runs/e3/d6_family_summary.md (baseline seeds, d6 seeds and the
           size-matched control against docs/experiments/E3/d6_matched_prediction.md)
    python scripts/e3_seeds_summary.py --family --available --out runs/e3/d6_matched_summary.md
        -> the same with only the d6 seeds that have outputs; the others are named

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
MATCHED_PREDICTION = "docs/experiments/E3/d6_matched_prediction.md"


def _last_commit(path: str) -> str | None:
    try:
        cp = subprocess.run(["git", "log", "-1", "--format=%h", "--", path], check=False, capture_output=True,
                            text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout.strip() or None


def _one(spec: str, flag: str) -> tuple[str, str]:
    from isomer.e3 import seeds

    models = seeds.parse_models(spec)
    if len(models) != 1:
        raise seeds.SeedsError(f"{flag} takes exactly one model")
    (name, ckpt), = models.items()
    return name, ckpt


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import seeds, seeds_compare, seeds_summary

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e3", default="runs/e3")
    ap.add_argument("--out", default=None, help="default: seeds_summary.md, d6_summary.md or d6_family_summary.md "
                                                "in the E3 directory")
    ap.add_argument("--models", default=None,
                    help="comma-separated NAME or NAME=CHECKPOINT for the seed summary (default: the h-v1 seeds)")
    ap.add_argument("--compare", default=None, help="one new model (NAME or NAME=CHECKPOINT) against --baseline")
    ap.add_argument("--family", action="store_true", help="baseline seeds, --d6 seeds and the --matched control")
    ap.add_argument("--baseline", default=None, help="baseline models (default: the h-v1 seeds)")
    ap.add_argument("--d6", default="h-d6-s0,h-d6-s1,h-d6-s2", help="d6 seeds for --family")
    ap.add_argument("--matched", default="h-d6m-s0", help="size-matched control for --family")
    ap.add_argument("--available", action="store_true",
                    help="--family: leave out d6 seeds that have no scores or bench results yet, and name them in "
                         "the summary (baseline and control must still be complete)")
    ap.add_argument("--e2-run", default="runs/E2-test_depth", help="E2 run to compare the h-v1 search with")
    a = ap.parse_args(argv)
    os.chdir(ROOT)      # relative paths, so the summary names runs/..., not this machine's directories
    e3 = Path(a.e3)
    try:
        if a.family and a.compare:
            raise seeds.SeedsError("--family and --compare are different summaries; give one")
        if a.family:
            name, ckpt = _one(a.matched, "--matched")
            d6 = seeds.parse_models(a.d6)
            missing: tuple[str, ...] = ()
            if a.available:
                missing = tuple(m for m in d6 if not seeds_compare.has_outputs(e3, m))
                d6 = {m: c for m, c in d6.items() if m not in missing}
                if missing:
                    print(f"note: leaving out d6 seeds without outputs: {', '.join(missing)}")
            out = seeds_compare.write_family(e3, Path(a.out) if a.out else e3 / "d6_family_summary.md",
                                             seeds.parse_models(a.baseline), d6, name, ckpt,
                                             _last_commit(MATCHED_PREDICTION), missing)
        elif a.compare:
            name, ckpt = _one(a.compare, "--compare")
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
