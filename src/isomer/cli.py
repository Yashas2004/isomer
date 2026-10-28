"""Command line entry point: ``python -m isomer <command>``."""

from __future__ import annotations

import argparse
import json
import platform
import sys


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"python   {platform.python_version()} ({sys.executable})")
    try:
        import torch
    except ImportError:
        print("torch    NOT INSTALLED  -> see README 'Install PyTorch with CUDA'")
        return 1
    print(f"torch    {torch.__version__}")
    if not torch.cuda.is_available():
        print("cuda     NOT AVAILABLE  -> you installed a CPU wheel; reinstall with a CUDA index URL")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"gpu      {props.name}  {props.total_memory / 2**30:.1f} GiB  sm_{props.major}{props.minor}")
    print(f"bf16     {torch.cuda.is_bf16_supported()}")
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"matmul   ok ({(x @ x).sum().item():.1f})")
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    from .core.proof import Problem, Proof
    from .core.rewrite import Step
    from .core.theories import GROUP

    problem = Problem.parse(GROUP, "(* e (* (inv a) a))", "e")
    proof = Proof(problem, (Step("id_l", ()), Step("inv_l", ())))
    print(problem)
    for t in proof.trace():
        print("   ", t)
    cert = proof.certificate()
    print("kernel:", "ACCEPTED" if proof.verify() else "REJECTED", "digest", cert["digest"][:16])

    bad = Proof(problem, (Step("inv_l", ()),))
    v = bad.verify()
    print("tampered proof:", "ACCEPTED" if v else f"REJECTED ({v.reason})")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .core.generate import sample_problems
    from .core.theories import THEORIES

    theory, sig = THEORIES[args.theory]
    for p in sample_problems(theory, sig, n=args.n, k=args.k, seed=args.seed):
        ok = p.verify()
        print(json.dumps({"lhs": str(p.problem.lhs), "rhs": str(p.problem.rhs),
                          "steps": len(p.steps), "kernel": bool(ok)}))
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from .data.build import BuildConfig, build

    cfg = BuildConfig(out=args.out, per_theory=args.per_theory, per_heldout_theory=args.per_heldout,
                      bfs_budget=args.bfs_budget, seed=args.seed)
    if args.workers:
        cfg.workers = args.workers
    print(f"building {cfg.out} with {cfg.workers} worker processes ...")
    sheet = build(cfg)
    print(json.dumps({k: sheet[k] for k in ("counts", "unlabelled_fraction", "seconds")}, indent=2))
    print(f"datasheet: {cfg.out}/datasheet.json")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench.run import BenchConfig, BenchError, run

    def csv(s: str) -> list[str]:
        return list(dict.fromkeys(x.strip() for x in s.split(",") if x.strip()))

    try:
        budgets = [int(b) for b in csv(args.budgets)]
    except ValueError:
        print(f"error: --budgets must be comma-separated integers, got {args.budgets!r}", file=sys.stderr)
        return 2
    cfg = BenchConfig(data=args.data, split=args.split, methods=csv(args.methods), budgets=budgets,
                      out=args.out, limit=args.limit, workers=args.workers, max_seconds=args.max_seconds,
                      overwrite=args.overwrite, argv=tuple(args.argv))
    try:
        return run(cfg)
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_report(args: argparse.Namespace) -> int:
    from .bench.report import report
    from .bench.run import BenchError

    try:
        return report(args.run, reference=args.reference)
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="isomer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check Python/PyTorch/CUDA").set_defaults(fn=cmd_doctor)
    sub.add_parser("demo", help="prove a tiny group identity and check it").set_defaults(fn=cmd_demo)
    sp = sub.add_parser("sample", help="generate random-walk problems")
    sp.add_argument("--theory", default="group", choices=["group", "bool", "lattice", "ring"])
    sp.add_argument("-n", type=int, default=5)
    sp.add_argument("-k", type=int, default=6)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_sample)
    dp = sub.add_parser("data", help="build the IsoBench dataset")
    dp.add_argument("--out", default="data/isobench-v1")
    dp.add_argument("--per-theory", type=int, default=2000)
    dp.add_argument("--per-heldout", type=int, default=300)
    dp.add_argument("--bfs-budget", type=int, default=3000)
    dp.add_argument("--seed", type=int, default=0)
    dp.add_argument("--workers", type=int, default=0, help="0 = all cores but one")
    dp.set_defaults(fn=cmd_data)
    bp = sub.add_parser("bench", help="run searchers on an IsoBench split, kernel-check, summarize")
    bp.add_argument("--data", required=True, help="dataset directory, e.g. data/isobench-v1")
    bp.add_argument("--split", default="test_id")
    bp.add_argument("--methods", required=True, help="comma-separated searcher names")
    bp.add_argument("--budgets", required=True, help="comma-separated expansion budgets")
    bp.add_argument("--limit", type=int, default=None, help="first N problems in id order")
    bp.add_argument("--workers", type=int, default=0, help="0 = all cores but one; 1 = in-process")
    bp.add_argument("--max-seconds", type=float, default=None,
                    help="per-run wall-clock cap (breaks the prefix property; cells above it are censored)")
    bp.add_argument("--out", required=True, help="run directory, e.g. runs/NAME")
    bp.add_argument("--overwrite", action="store_true")
    bp.set_defaults(fn=cmd_bench)
    rp = sub.add_parser("report", help="write results.md and figures for a bench run (no re-running)")
    rp.add_argument("run", help="run directory written by `isomer bench`, e.g. runs/NAME")
    rp.add_argument("--reference", default=None,
                    help="method for paired comparisons (default: bfs-fwd if it is in the run)")
    rp.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.argv = list(sys.argv) if argv is None else ["isomer", *argv]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
