"""E3 seed study, step 2: score the labelled states with each model and analyse them.

    python scripts/e3_score_seeds.py

For each of h-v1, h-v1-s1, h-v1-s2 and each of test_depth, test_id this runs
``isomer.e3.diagnose score`` (cuda, bf16-fast) and ``isomer.e3.analyze base``:

    runs/e3/scores/<model>/<split>.jsonl            scores of every state and child
    runs/e3/scores/<model>/<split>.score.json       throughput, peak VRAM, checkpoint and label hashes
    runs/e3/scores/<model>/<split>.analysis.json    the analysis (what the summary reads)

It refuses to score a split whose labels are incomplete: all three models must
see the same, whole population. Finished scores are reused when the checkpoint
and the label file are unchanged (``--redo`` to force).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=None,
                    help="comma-separated NAME or NAME=CHECKPOINT (default: the three h-v1 seeds)")
    ap.add_argument("--splits", default="test_depth,test_id")
    ap.add_argument("--e3", default="runs/e3", help="directory holding data/ and labels/; scores/ is written here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default=seeds.PRECISION)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    e3 = ROOT / a.e3
    splits = [s for s in a.splits.split(",") if s]
    try:
        models = seeds.parse_models(a.models)
        for ckpt in models.values():        # everything is checked before the first model is loaded
            if not (ROOT / ckpt).is_file():
                raise seeds.SeedsError(f"checkpoint {ckpt} not found")
        for split in splits:
            st = seeds.require_complete(e3 / "data" / f"{split}.jsonl", e3 / "labels" / f"{split}.jsonl")
            print(f"== {split}: {st['labelled']} problems, {st['states']} states "
                  f"({st['unknown_states']} unknown, excluded from every metric)", flush=True)
        if a.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise seeds.SeedsError("--device cuda but CUDA is not available")
        summary = []
        for m, ckpt in models.items():
            for split in splits:
                res = seeds.score_and_analyze(m, ROOT / ckpt, split, e3 / "labels" / f"{split}.jsonl",
                                              e3 / "scores" / m, a.device, a.precision, a.redo)
                summary.append(res)
                peak = res["max_memory_allocated"]
                print(f"== {m} {split}: top-1 {res['top1_optimal']:.4f} (symbol_bag {res['symbol_bag_top1']:.4f}) "
                      f"on {res['states']} states; {res['seq_per_second']} seq/s; "
                      f"max VRAM {None if peak is None else round(peak / 2**20)} MiB", flush=True)
    except seeds.SeedsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # index.json lists every model scored so far: entries of models not in this call are kept
    index = e3 / "scores" / "index.json"
    done = {(r["model"], r["split"]) for r in summary}
    old = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    merged = [r for r in old if (r["model"], r["split"]) not in done] + summary
    index.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
