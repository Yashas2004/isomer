"""E3 seed study, step 3: neural-w1 search with each model on the 300 test_depth problems.

    python scripts/e3_bench_seeds.py
    python scripts/e3_bench_seeds.py --dry-run     # print the commands, start nothing

One ``isomer bench`` run per model, one after the other, at budgets
30,100,300,1000 (one search per problem at 1,000 expansions; smaller budgets
are derived, see ``bench.run``). The command is E2's with only the checkpoint
changed, so the h-v1 run doubles as a reproduction of E2's test_depth row.
Every solved proof is checked by the kernel inside ``isomer bench``.

    runs/e3/bench-seeds/<model>/results.jsonl, config.json
    runs/e3/bench-seeds/<model>.log

One run per model, not one run with three methods: ``isomer bench`` writes its
results only at the end, so an interruption costs one model, not all three.
A model whose ``results.jsonl`` exists is skipped; rerun the same command to
continue. A non-zero exit of ``isomer bench`` (an error or an unsound row)
stops the script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_logged(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd`` from the repository root, copying its output to the console and to ``log_path``."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def main(argv: list[str] | None = None) -> int:
    from isomer.e3 import seeds

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=None,
                    help="comma-separated NAME or NAME=CHECKPOINT (default: the three h-v1 seeds)")
    ap.add_argument("--out", default="runs/e3/bench-seeds")
    ap.add_argument("--workers", type=int, default=2, help="as in E2; each holds its own CUDA context")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        models = seeds.parse_models(a.models)
    except seeds.SeedsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    plan = [(m, seeds.bench_command(ckpt, f"{a.out}/{m}", workers=a.workers)) for m, ckpt in models.items()]
    if a.dry_run:
        for _, cmd in plan:
            print(subprocess.list2cmdline(cmd))
        return 0
    for ckpt in models.values():
        if not (ROOT / ckpt).is_file():
            print(f"error: checkpoint {ckpt} not found", file=sys.stderr)
            return 2
    (ROOT / a.out).mkdir(parents=True, exist_ok=True)
    for m, cmd in plan:
        run = ROOT / a.out / m
        if (run / "results.jsonl").exists():
            print(f"== {m}: {a.out}/{m}/results.jsonl exists, skipping", flush=True)
            continue
        print(f"== {m}: {subprocess.list2cmdline(cmd)}", flush=True)
        t0 = time.perf_counter()
        code = run_logged(cmd, ROOT / a.out / f"{m}.log")
        print(f"== {m}: exit code {code} after {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":      # required on Windows: `isomer bench` spawns workers
    sys.exit(main())
