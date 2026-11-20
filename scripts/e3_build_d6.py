"""E3-d6, step A: build data/isobench-v2-d6 (train problems up to exact distance 6).

    python scripts/e3_build_d6.py                  # data/isobench-v2-d6
    python scripts/e3_build_d6.py --max-dist 5     # data/isobench-v2-d5 (E3-d5)

Regenerates and relabels v2's train pool (about 10 minutes), checks that it
reproduces v2's train and test_id files byte for byte, and writes the variant
with test_depth, test_deep and test_theory copied unchanged from v2. Nothing is
written if a check fails. See ``isomer.e3.d6`` for why a full rebuild is not
used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import d6

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v2", default="data/isobench-v2")
    ap.add_argument("--max-dist", type=int, default=d6.TRAIN_MAX_DIST,
                    help="largest exact distance of a train problem (6 = d6, 5 = d5)")
    ap.add_argument("--out", default=None, help="default: data/isobench-v2-d<MAX_DIST>")
    ap.add_argument("--ids", default="docs/experiments/E2", help="directory of the E2 id files")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    out = Path(a.out or f"data/isobench-v2-d{a.max_dist}")
    try:        # before the 10-minute relabelling, not after it
        d6.check_max_dist(a.max_dist)
    except d6.D6Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        print(f"error: {out.as_posix()} already exists and is not empty", file=sys.stderr)
        return 2
    print(f"building {out.as_posix()} from the train pool of {a.v2} with {a.workers} worker processes ...", flush=True)
    try:
        sheet = d6.build_d6(d6.label_train_pool(a.workers), a.v2, out, a.ids, a.max_dist)
    except d6.D6Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: sheet[k] for k in ("counts", "dist_histogram", "dropped", "sha256", "checks")}, indent=2))
    print(f"all checks passed; datasheet: {out.as_posix()}/datasheet.json")
    return 0


if __name__ == "__main__":      # required on Windows: the labelling workers are spawned
    sys.exit(main())
