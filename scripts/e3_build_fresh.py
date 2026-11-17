"""E3-fresh, step A: build the fresh evaluation pool and draw its 300-problem sample.

    python scripts/e3_build_fresh.py                 # all cores but one, as the v2 build
    python scripts/e3_build_fresh.py --workers 8     # if the machine starts swapping

Writes data/isobench-fresh/test_depth_fresh.jsonl and datasheet.json, and
docs/experiments/E3/ids-test_depth_fresh.txt (see ``isomer.e3.fresh``).
Refuses to run if either output already exists.

Resumable: labelled problems are kept in data/isobench-fresh/labelled.partial.jsonl
while the build runs, so after Ctrl+C or a crash rerun the same command. The
partial file is deleted when the build finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import fresh

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=None, help="labelling processes (default: as BuildConfig)")
    a = ap.parse_args(argv)
    try:     # the worker count does not affect any output, so it is not part of the recorded command
        sheet = fresh.build(ROOT, a.workers, progress=True, command="python scripts/e3_build_fresh.py")
    except fresh.FreshError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: sheet[k] for k in ("counts", "generated", "replaced_at_generation", "dropped",
                                            "dist_histogram", "label_stop", "collisions_with_excluded_splits")},
                     indent=1))
    print(f"sample: {fresh.IDS_OUT}")
    return 0


if __name__ == "__main__":      # required on Windows: the labelling workers are spawned
    sys.exit(main())
