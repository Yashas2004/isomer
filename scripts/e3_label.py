"""E3 seed study, step 1: build the labelled populations once (shared by all models).

    python scripts/e3_label.py                   # test_id, then test_depth, 5 workers
    python scripts/e3_label.py --workers 3       # if the machine starts swapping
    python scripts/e3_label.py --splits test_depth_fresh --data data/isobench-fresh \
        --ids docs/experiments/E3/ids-test_depth_fresh.txt --out runs/e3-fresh      # E3-fresh

For each split: take the E2 stratified sample out of data/isobench-v2 by id
(``runs/e3/data/SPLIT.jsonl``; no relabelling, the dataset's own records), then
label every legal child of every on-path state (``runs/e3/labels/SPLIT.jsonl``).
All 300 problems of each sample, no subset.

Resumable: ``label_children`` keeps finished problems and redoes only the rest,
so after Ctrl+C or a crash rerun the same command.

Memory: a worker holds a goal ball and a forward search of up to 300,000 terms
each, and CPython does not hand that memory back, so every worker ends up at
its high-water mark. The default of 5 workers is chosen for 16 GB of RAM; see
``--workers``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import label_children, seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default=",".join(seeds.SPLITS), help="comma-separated, labelled in this order")
    ap.add_argument("--workers", type=int, default=5,
                    help="labelling processes; a test_depth problem peaked at 0.3-0.5 GB in a trial of 11, "
                         "and a long-lived worker stays at its high-water mark")
    ap.add_argument("--out", default="runs/e3", help="writes OUT/data and OUT/labels")
    ap.add_argument("--data", default=seeds.DATA, help="dataset directory holding SPLIT.jsonl")
    ap.add_argument("--ids", default=seeds.IDS, help="id file of the sample; {split} is replaced by the split name")
    ap.add_argument("--sample-only", action="store_true", help="write the samples, label nothing")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    try:
        for split in [s for s in a.splits.split(",") if s]:
            sample = out / "data" / f"{split}.jsonl"
            labels = out / "labels" / f"{split}.jsonl"
            info = seeds.write_sample(split, ROOT / a.ids.format(split=split), ROOT / a.data, sample)
            print(f"== {split}: {info['problems']} problems -> {sample.relative_to(ROOT).as_posix()} "
                  f"(dataset hash {'matches the id file' if info['dataset_hash_checked'] else 'NOT CHECKED'}, "
                  f"strata match)", flush=True)
            if a.sample_only:
                continue
            t0 = time.perf_counter()
            label_children.main(str(sample), str(labels), a.workers)
            st = seeds.label_status(sample, labels)
            print(f"== {split}: {json.dumps(st)} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if not st["complete"]:
                print(f"error: {split} is not completely labelled; rerun the same command", file=sys.stderr)
                return 1
    except seeds.SeedsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":      # required on Windows: the labelling workers are spawned
    sys.exit(main())
