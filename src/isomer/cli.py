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

    cfg = BuildConfig(out=args.out, train_per_theory=args.train_per_theory, eval_per_theory=args.eval_per_theory,
                      heldout_per_theory=args.heldout_per_theory, train_label_budget=args.train_label_budget,
                      label_budget=args.label_budget, label_max_nodes=args.label_max_nodes, seed=args.seed,
                      exclude_ids_from=args.exclude_ids_from)
    if args.workers:
        cfg.workers = args.workers
    print(f"building {cfg.out} with {cfg.workers} worker processes ...")
    sheet = build(cfg)
    print(json.dumps({k: sheet[k] for k in ("counts", "dropped", "label_stop", "max_nodes_stored", "seconds")},
                     indent=2))
    print(f"datasheet: {cfg.out}/datasheet.json")
    return 0


def cmd_states(args: argparse.Namespace) -> int:
    from .learn.states import StatesConfig, build_states

    cfg = StatesConfig(data=args.data, out=args.out, offpath=args.offpath, budget=args.budget, limit=args.limit)
    if args.workers:
        cfg.workers = args.workers
    print(f"building {cfg.out} from {cfg.data}/train.jsonl with {cfg.workers} worker processes ...")
    stats = build_states(cfg)
    print(json.dumps({k: stats[k] for k in ("problems", "counts", "on_off_path", "dropped", "max_tokens", "seconds")},
                     indent=2))
    print(f"stats: {cfg.out}/stats.json")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:  # torch is optional for the rest of isomer: import it only here
        from .learn.train import TrainConfig, main_train
    except ImportError as exc:
        print(f"error: training needs torch ({exc}); pip install -e \".[learn]\" or see README", file=sys.stderr)
        return 2
    cfg = TrainConfig(states=args.states, out=args.out, epochs=args.epochs, batch_problems=args.batch_problems,
                      lr=args.lr, warmup=args.warmup, d_model=args.d_model, layers=args.layers, heads=args.heads,
                      lambda_rank=args.lambda_rank, tau=args.tau, rank_target=args.rank_target,
                      micro_tokens=args.micro_tokens, seed=args.seed, device=args.device, max_steps=args.max_steps,
                      overwrite=args.overwrite, argv=tuple(args.argv))
    return main_train(cfg)


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
                      max_nodes=args.max_nodes, overwrite=args.overwrite, argv=tuple(args.argv))
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
    dp.add_argument("--out", default="data/isobench-v2")
    dp.add_argument("--train-per-theory", type=int, default=20000)
    dp.add_argument("--eval-per-theory", type=int, default=1200)
    dp.add_argument("--heldout-per-theory", type=int, default=300)
    dp.add_argument("--train-label-budget", type=int, default=3000)
    dp.add_argument("--label-budget", type=int, default=20000, help="eval/held-out labelling expansions")
    dp.add_argument("--label-max-nodes", type=int, default=1_500_000, help="eval/held-out node cap")
    dp.add_argument("--seed", type=int, default=0)
    dp.add_argument("--exclude-ids-from", default=None,
                    help="dataset dir: never generate a problem whose id is in any of its splits (dev sets)")
    dp.add_argument("--workers", type=int, default=0, help="0 = all cores but one")
    dp.set_defaults(fn=cmd_data)
    stp = sub.add_parser("states", help="distance-labelled (state, goal) examples from train.jsonl")
    stp.add_argument("--data", default="data/isobench-v2")
    stp.add_argument("--out", default="data/states-v2")
    stp.add_argument("--offpath", type=int, default=3, help="off-path successors per on-path state")
    stp.add_argument("--budget", type=int, default=3000, help="BFS expansions per off-path label")
    stp.add_argument("--limit", type=int, default=None, help="first N problems in id order")
    stp.add_argument("--workers", type=int, default=0, help="0 = all cores but one; 1 = in-process")
    stp.set_defaults(fn=cmd_states)
    tp = sub.add_parser("train", help="train the learned distance heuristic h(state, goal)")
    tp.add_argument("--states", default="data/states-v2")
    tp.add_argument("--out", required=True, help="run directory, e.g. runs/NAME")
    tp.add_argument("--epochs", type=int, default=20)
    tp.add_argument("--batch-problems", type=int, default=64, help="problems (all their states) per batch")
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--warmup", type=int, default=500, help="linear warmup steps before cosine decay")
    tp.add_argument("--d-model", type=int, default=256)
    tp.add_argument("--layers", type=int, default=6)
    tp.add_argument("--heads", type=int, default=8)
    tp.add_argument("--lambda-rank", type=float, default=1.0)
    tp.add_argument("--tau", type=float, default=1.0)
    tp.add_argument("--rank-target", default="optimal", choices=["optimal", "onpath"])
    tp.add_argument("--micro-tokens", type=int, default=32_768, help="padded tokens per micro-batch (GPU memory)")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--device", default="cuda")
    tp.add_argument("--max-steps", type=int, default=None)
    tp.add_argument("--overwrite", action="store_true")
    tp.set_defaults(fn=cmd_train)
    bp = sub.add_parser("bench", help="run searchers on an IsoBench split, kernel-check, summarize")
    bp.add_argument("--data", required=True, help="dataset directory, e.g. data/isobench-v2")
    bp.add_argument("--split", default="test_id")
    bp.add_argument("--methods", required=True, help="comma-separated searcher names")
    bp.add_argument("--budgets", required=True, help="comma-separated expansion budgets")
    bp.add_argument("--limit", type=int, default=None, help="first N problems in id order")
    bp.add_argument("--workers", type=int, default=0,
                    help="0 = all cores but one (2 if a neural method is present: each worker holds its own "
                         "CUDA context, ~0.5 GB); 1 = in-process")
    bp.add_argument("--max-seconds", type=float, default=None,
                    help="per-run wall-clock cap (breaks the prefix property; cells above it are censored)")
    bp.add_argument("--max-nodes", type=int, default=None,
                    help="per-run cap on stored nodes (stop = memory; cells above it are censored)")
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
