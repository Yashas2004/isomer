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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="isomer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check Python/PyTorch/CUDA").set_defaults(fn=cmd_doctor)
    sub.add_parser("demo", help="prove a tiny group identity and check it").set_defaults(fn=cmd_demo)
    sp = sub.add_parser("sample", help="generate random-walk problems")
    sp.add_argument("--theory", default="group", choices=["group", "bool"])
    sp.add_argument("-n", type=int, default=5)
    sp.add_argument("-k", type=int, default=6)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_sample)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
