"""``isomer prove``: find a proof for one equation and have the kernel check it.

A demo, not an evaluation: it runs one searcher on one problem and shows every
step, then replays the certificate through ``kernel.check_certificate``. The
evidence for how well the searchers work is in docs/results/E2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .core import kernel
from .core.proof import Problem, Proof
from .core.theories import THEORIES
from .search.common import PARAMETRIC, SEARCHERS, Budget, get_searcher

DEFAULT_CKPT = "runs/h-v1/best.pt"
RELEASE_URL = "https://github.com/Yashas2004/isomer/releases/download/v0.2-e2/best.pt"
RELEASE_SHA256 = "928015f6250a8531e6249e799507d4ca8bd433e50e28b7639aae6fa885376c00"


@dataclass(frozen=True)
class Example:
    theory: str
    lhs: str
    rhs: str
    dist: int
    note: str


# Illustrative only, NOT evidence. Hand-picked from the dev set data/isobench-v2-dev
# (never the v2 test splits) because neural-w1 solves them within 1000 expansions
# (and, for bool-deep, bestfirst-symbol_bag-w1 does not). Among the problems meeting
# that criterion, the smallest lhs + rhs was taken. The evidence is docs/results/E2.
EXAMPLES: dict[str, Example] = {
    # dev train 33ad263486799b8b
    "group-short": Example("group", "e", "(* (* e e) e)", 2, "a short group identity"),
    # dev test_depth 5496f35f766d7b46: neural-w1 solves it in bf16-fast (GPU) and fp32 (CPU);
    # bestfirst-symbol_bag-w1 does not within 1000 expansions
    "bool-deep": Example("bool", "(~ p)", "(~ (& p (| (| F (| F T)) F)))", 6,
                         "a 6-step boolean-algebra proof, deeper than any training problem (dist <= 4)"),
    # dev test_theory 74af9d86cd9ec6d8
    "lattice": Example("lattice", "a", "(join (meet (join (meet a a) a) a) a)", 4,
                       "a lattice problem: a theory held out from training"),
}
EXAMPLE_DISCLAIMER = ("Illustrative example, hand-picked from the dev set because neural-w1 succeeds on it; "
                      "not evidence. The evidence is docs/results/E2.")


class DemoError(Exception):
    """A usage error: printed, exit code 2."""


def missing_checkpoint_message(ckpt: str) -> str:
    return (f"checkpoint {ckpt} not found. Download the trained model (release v0.2-e2):\n"
            f"  {RELEASE_URL}\n"
            f"  sha256 {RELEASE_SHA256}\n"
            f"PowerShell:  New-Item -ItemType Directory -Force runs/h-v1; "
            f"Invoke-WebRequest {RELEASE_URL} -OutFile {DEFAULT_CKPT}; certutil -hashfile {DEFAULT_CKPT} SHA256\n"
            f"bash:        mkdir -p runs/h-v1 && curl -L -o {DEFAULT_CKPT} {RELEASE_URL} && sha256sum {DEFAULT_CKPT}\n"
            "Or run without the model: --method bestfirst-symbol_bag-w1")


def resolve_method(method: str, ckpt: str, precision: str | None) -> tuple[str, dict]:
    """Full searcher name and display info. Neural prefixes get ``@ckpt#precision``;
    the checkpoint and torch are checked here, before any search."""
    if method in SEARCHERS:
        return method, {}
    if method not in PARAMETRIC:
        raise DemoError(f"unknown method {method!r}; valid: {', '.join(sorted(SEARCHERS))}, "
                        f"{', '.join(sorted(PARAMETRIC))}")
    if not Path(ckpt).is_file():
        raise DemoError(missing_checkpoint_message(ckpt))
    try:
        import torch
    except ImportError:
        raise DemoError("neural methods need torch (see README: install PyTorch with CUDA); "
                        "or run without the model: --method bestfirst-symbol_bag-w1") from None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = precision or ("bf16-fast" if device == "cuda" else "fp32")
    return f"{method}@{ckpt}#{precision}", {"checkpoint": ckpt, "precision": precision, "device": device}


def _step_text(k: int, step, term) -> str:
    inst = "".join(f" [{v} := {t}]" for v, t in step.inst)
    pos = ".".join(map(str, step.pos)) or "root"
    return f"  {k:2d}. {step.rule} @ {pos}{inst}\n      -> {term}"


def prove(theory: str, lhs: str, rhs: str, method: str = "neural-w1", ckpt: str = DEFAULT_CKPT,
          precision: str | None = None, budget: int = 1000, cert_out: str | None = None,
          print_fn=print) -> int:
    """Search, print the proof and the kernel verdict. Returns the exit code:
    0 kernel-accepted, 1 no proof or rejected."""
    if theory not in THEORIES:
        raise DemoError(f"unknown theory {theory!r}; valid: {', '.join(sorted(THEORIES))}")
    try:
        problem = Problem.parse(THEORIES[theory][0], lhs, rhs)
    except Exception as exc:  # noqa: BLE001 -- any parse failure is a usage error
        raise DemoError(f"cannot parse the problem: {exc}") from None
    name, info = resolve_method(method, ckpt, precision)
    print_fn(f"problem  {problem}")
    print_fn(f"method   {method}" + (f"  (checkpoint {info['checkpoint']}, {info['precision']}, {info['device']})"
                                     if info else ""))
    if info:
        # Load the model and warm up CUDA before the search clock starts, so the search
        # time is not dominated by a one-time cost. The searcher reuses this cached model:
        # the cache key (checkpoint, device=None, precision) is the one it looks up.
        from .search.neural import _load

        t0 = time.perf_counter()
        _load(info["checkpoint"], None, info["precision"]).scores([problem.lhs], problem.rhs)
        print_fn(f"model    loaded in {time.perf_counter() - t0:.2f} s (one-time)")
    t0 = time.perf_counter()
    result = get_searcher(name)(problem, Budget(max_expansions=budget))
    seconds = time.perf_counter() - t0
    if result.proof is None:
        print_fn(f"no proof within {budget} expansions ({result.expansions} expansions, {seconds:.2f} s)")
        return 1
    proof = Proof(problem, result.proof.steps)
    print_fn(f"proof    {len(proof.steps)} steps")
    for k, (step, term) in enumerate(zip(proof.steps, proof.trace()[1:]), 1):
        print_fn(_step_text(k, step, term))
    print_fn(f"search   {result.expansions} expansions, {seconds:.2f} s")
    cert = proof.certificate()
    verdict = kernel.check_certificate(cert)
    print_fn(f"kernel   {'ACCEPTED' if verdict else f'REJECTED ({verdict.reason})'}  digest {cert['digest']}")
    if cert_out:
        Path(cert_out).parent.mkdir(parents=True, exist_ok=True)
        with open(cert_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(cert, indent=2, sort_keys=True) + "\n")
        print_fn(f"certificate written to {cert_out}")
    return 0 if verdict else 1
