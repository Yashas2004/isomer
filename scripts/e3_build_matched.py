"""E3-d6 control: build data/states-v2-d6-matched (d6 depth range, v2 number of train problems).

    python scripts/e3_build_matched.py

A seeded random subset of the d6 train problems, as many as data/states-v2 has,
with every state of each kept problem; val.jsonl is copied unchanged. Takes
under a minute. See ``isomer.e3.matched``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import matched

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/states-v2-d6")
    ap.add_argument("--out", default="data/states-v2-d6-matched")
    ap.add_argument("--match", default="data/states-v2", help="states directory whose train-problem count is matched")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    try:
        n = json.loads((Path(a.match) / "stats.json").read_text(encoding="utf-8"))["problems"]["train"]
        stats = matched.build_matched(a.src, a.out, n, a.seed, matched_to=Path(a.match).as_posix())
    except (matched.MatchedError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: stats[k] for k in ("subset", "problems", "counts", "dist_histogram",
                                            "problem_dist_histogram", "max_tokens", "sha256")}, indent=2))
    print(f"stats: {a.out}/stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
