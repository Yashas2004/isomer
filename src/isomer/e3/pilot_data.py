"""A small, faithful subset of IsoBench v2's train split, for CPU-scale training pilots.

    python -m isomer.e3.pilot_data OUT_DIR N_PROBLEMS [WORKERS]

The train pool is generated exactly as ``data.build`` does (after the held-out
and eval pools claim their ids), then labelled in id order (ids are content
hashes, so this is a uniform random subset) with the train labeller budget.
Records keep the build's rules: exact dist <= train_max_dist, and test_id
problems (by the build's hash rule) are excluded. Output: OUT_DIR/train.jsonl
(the build's record format) and datasheet.json (the published config plus a
``pilot`` note). Never used for reported benchmark results.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..data.build import BuildConfig, _label, _split
from .regen import generate_jobs


def main(out: str, n: int, workers: int = 2) -> dict:
    cfg = BuildConfig()
    jobs = generate_jobs({"__all_train__"}, cfg, keep_pool="train")
    order = sorted(jobs)
    kept: list[dict] = []
    stats = {"labelled": 0, "dropped_dist": 0, "dropped_test_id": 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(order), 512):
            if len(kept) >= n:
                break
            for rec in ex.map(_label, [jobs[i] for i in order[start:start + 512]], chunksize=16):
                rec.pop("_seconds")
                stats["labelled"] += 1
                dest = _split("train", rec, cfg)
                if dest is None:
                    stats["dropped_dist"] += 1
                elif dest == "test_id":
                    stats["dropped_test_id"] += 1
                elif len(kept) < n:
                    kept.append(rec)
    kept.sort(key=lambda r: r["id"])
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    with open(o / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in kept)
    sheet = {"format": "isobench/2", "config": {k: list(v) if isinstance(v, tuple) else v
                                                for k, v in cfg.__dict__.items() if k != "workers"},
             "pilot": {"n_train": len(kept), "train_pool_generated": len(order), **stats,
                       "note": "uniform (lowest-id) subset of the v2 train split; pilots only"}}
    (o / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True), encoding="utf-8")
    return sheet["pilot"]


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2)))
