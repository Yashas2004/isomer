"""Train the h-v1 configuration with further seeds (E3 report, section 15, item 1).

    python scripts/train_seeds.py                # seeds 1 and 2 -> runs/h-v1-s1, runs/h-v1-s2
    python scripts/train_seeds.py --dry-run      # print the commands, start nothing

Why a script and not two shell lines: the hyper-parameters are read from the
published ``docs/results/train-h-v1/config.json`` instead of being retyped, so
a seed run cannot silently differ from h-v1 in anything but the seed. Values
the CLI cannot set (weight decay, gradient clip, max length) are checked
against the current ``TrainConfig`` defaults, and the script refuses to start
if they have drifted.

Each seed runs in its own process, one after the other, so the second run
starts from a clean CUDA context and a crash in one seed cannot corrupt the
other. Console output is copied to ``runs/h-v1-sN.log``. Peak VRAM
(``torch.cuda.max_memory_allocated``, logged per epoch by the trainer) is
reported per seed and written to ``runs/train_seeds.json``.

A finished seed (all epochs in ``metrics.jsonl`` and a ``best.pt``) is skipped,
so the command can be rerun after an interruption. A half-finished run
directory is never overwritten unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "docs" / "results" / "train-h-v1" / "config.json"

# config.json "args" key -> `isomer train` flag
CLI_FLAGS = {
    "states": "--states", "epochs": "--epochs", "batch_problems": "--batch-problems", "lr": "--lr",
    "warmup": "--warmup", "d_model": "--d-model", "layers": "--layers", "heads": "--heads",
    "lambda_rank": "--lambda-rank", "tau": "--tau", "rank_target": "--rank-target",
    "micro_tokens": "--micro-tokens", "device": "--device",
}
# set per run by this script, or bookkeeping that does not affect training
PER_RUN = {"seed", "out", "argv", "overwrite", "log_every"}
# losses added after h-v1 (E3); h-v1 is these values (tests/test_train_e3.py: same gradient)
E3_FLAGS = {"--lambda-huber": 1.0, "--lambda-consistency": 0.0}
VRAM_LIMIT = 4 * 2**30      # CLAUDE.md: GPU code must fit a 4 GB card
DEFAULT_PREFIX = "h-v1-s"


class ConfigDrift(Exception):
    """The published configuration cannot be reproduced by the current trainer."""


def base_args(path: Path = BASE_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["args"]


def check_defaults(args: dict, defaults: dict) -> None:
    """Every published argument must be settable on the command line or equal today's default."""
    for key, value in sorted(args.items()):
        if key in CLI_FLAGS or key in PER_RUN:
            continue
        if key not in defaults:
            raise ConfigDrift(f"h-v1 argument {key!r} is unknown to the current TrainConfig")
        if defaults[key] != value:
            raise ConfigDrift(f"h-v1 used {key}={value!r}; the trainer now defaults to {defaults[key]!r} "
                              f"and `isomer train` has no flag for it")
    if args.get("max_steps") is not None:
        raise ConfigDrift("h-v1 config has max_steps set; that is a truncated run")


def train_command(args: dict, seed: int, out: str, python: str = sys.executable,
                  overwrite: bool = False) -> list[str]:
    """The `isomer train` command for one seed: h-v1's arguments, spelled out in full."""
    cmd = [python, "-m", "isomer", "train", "--out", out, "--seed", str(seed)]
    for key, flag in CLI_FLAGS.items():
        cmd += [flag, str(args[key])]
    for flag, value in E3_FLAGS.items():
        cmd += [flag, str(value)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def read_metrics(run: Path) -> list[dict]:
    path = Path(run) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_complete(run: Path, epochs: int) -> bool:
    return (Path(run) / "best.pt").exists() and len(read_metrics(run)) >= epochs


def summarize_run(run: Path) -> dict:
    """Peak VRAM and the best epoch, from the trainer's own per-epoch log."""
    rows = read_metrics(run)
    peaks = [r["max_memory_allocated"] for r in rows if r.get("max_memory_allocated") is not None]
    best = max(rows, key=lambda r: r["val_top1_optimal"]) if rows else None
    return {
        "run": Path(run).as_posix(),
        "epochs_done": len(rows),
        "max_memory_allocated_bytes": max(peaks) if peaks else None,
        "max_memory_allocated_mib": round(max(peaks) / 2**20) if peaks else None,
        "best_epoch": None if best is None else best["epoch"],
        "best_val_top1_optimal": None if best is None else best["val_top1_optimal"],
        "train_seconds": round(sum(r["train_seconds"] + r["val_seconds"] for r in rows), 1),
    }


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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="1,2", help="comma-separated seeds, trained in this order")
    ap.add_argument("--runs", default="runs", help="parent directory of the run directories")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="run directory is RUNS/PREFIX<seed>")
    ap.add_argument("--states", default=None,
                    help="train on another states directory (e.g. data/states-v2-d6); every other argument "
                         "stays h-v1's. Needs its own --prefix")
    ap.add_argument("--overwrite", action="store_true", help="replace half-finished run directories")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    args = base_args()
    same_data = a.states is None or Path(a.states).as_posix() == Path(args["states"]).as_posix()
    if same_data and args["seed"] in seeds:
        print(f"error: seed {args['seed']} on {args['states']} is h-v1 itself", file=sys.stderr)
        return 2
    if not same_data:
        if a.prefix == DEFAULT_PREFIX:
            print(f"error: --states {a.states} is not h-v1's training data; give the runs another --prefix",
                  file=sys.stderr)
            return 2
        args = dict(args, states=Path(a.states).as_posix())     # the only argument that may differ from h-v1
    try:
        from dataclasses import asdict

        from isomer.learn.train import TrainConfig
        check_defaults(args, asdict(TrainConfig()))
    except ImportError as exc:
        print(f"error: cannot import the trainer ({exc}); run with the project's .venv python", file=sys.stderr)
        return 2
    except ConfigDrift as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runs = Path(a.runs)
    plan = [(seed, runs / f"{a.prefix}{seed}") for seed in seeds]
    if a.dry_run:
        for seed, run in plan:
            print(subprocess.list2cmdline(train_command(args, seed, run.as_posix(), overwrite=a.overwrite)))
        return 0

    (ROOT / runs).mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    status = 0
    for seed, run in plan:
        run_abs = ROOT / run
        if is_complete(run_abs, args["epochs"]):
            print(f"== seed {seed}: {run.as_posix()} is complete, skipping", flush=True)
        else:
            if run_abs.exists() and any(run_abs.iterdir()) and not a.overwrite:
                print(f"error: {run.as_posix()} exists but is not a finished run; inspect it, then rerun with "
                      f"--overwrite (seeds already finished are kept)", file=sys.stderr)
                status = 2
                break
            cmd = train_command(args, seed, run.as_posix(), overwrite=a.overwrite)
            print(f"== seed {seed}: {subprocess.list2cmdline(cmd)}", flush=True)
            t0 = time.perf_counter()
            code = run_logged(cmd, ROOT / runs / f"{a.prefix}{seed}.log")
            print(f"== seed {seed}: exit code {code} after {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if code != 0:
                status = code
                summary.append({"seed": seed, "exit_code": code, **summarize_run(run_abs)})
                break
        s = {"seed": seed, "exit_code": 0, **summarize_run(run_abs)}
        s["run"] = run.as_posix()
        summary.append(s)
        peak = s["max_memory_allocated_bytes"]
        print(f"== seed {seed}: max VRAM {s['max_memory_allocated_mib']} MiB "
              f"(torch.cuda.max_memory_allocated), best val top1_optimal {s['best_val_top1_optimal']} "
              f"at epoch {s['best_epoch']}", flush=True)
        if peak is not None and peak > VRAM_LIMIT:
            print(f"warning: seed {seed} peaked above 4 GiB; it would not fit the reference 4 GB card",
                  file=sys.stderr)

    # one summary file per run family, so a d6 run does not overwrite the h-v1 seeds' summary
    out = ROOT / runs / ("train_seeds.json" if a.prefix == DEFAULT_PREFIX else f"train_seeds_{a.prefix.strip('-')}.json")
    out.write_text(json.dumps({"base_config": BASE_CONFIG.relative_to(ROOT).as_posix(), "states": args["states"],
                               "runs": summary},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
