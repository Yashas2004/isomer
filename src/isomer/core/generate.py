"""Synthetic problem generation by random walks (first version).

A walk of k rewrites from a random term t gives a problem ``t = t'`` with a
*known* proof of length k. Caveat that matters for every benchmark number we
ever report: k is an UPPER bound on proof length -- a shorter proof may exist.
Phase 1b adds BFS-verified shortest lengths for the depth-extrapolation split.
"""

from __future__ import annotations

import math
import random

from .proof import Problem, Proof
from .rewrite import Theory, successors
from .term import App, Term, size
from .theories import Signature


def random_term(sig: Signature, max_depth: int, rng: random.Random) -> Term:
    leaves = sig.constants + sig.skolems
    if max_depth <= 1 or rng.random() < 0.3:
        return App(rng.choice(leaves))
    op = rng.choice(sorted(sig.ops))
    return App(op, tuple(random_term(sig, max_depth - 1, rng) for _ in range(sig.ops[op])))


def random_walk(theory: Theory, start: Term, k: int, rng: random.Random,
                max_size: int = 40, growth_temp: float = 1.5) -> Proof:
    rules = theory.rules().values()
    cur, seen, steps = start, {start}, []
    for _ in range(k):
        options = [(s, t) for s, t in successors(cur, rules) if t not in seen and size(t) <= max_size]
        if not options:
            break
        # Size-aware sampling: growing rewrites (e.g. e -> (* (inv x) x)) are
        # always applicable, so uniform sampling produces bloated junk terms.
        base = size(cur)
        weights = [math.exp(-max(0, size(t) - base) / growth_temp) for _, t in options]
        step, cur = rng.choices(options, weights=weights, k=1)[0]
        seen.add(cur)
        steps.append(step)
    return Proof(Problem(theory, start, cur, {"walk_len": len(steps)}), tuple(steps))


def sample_problems(theory: Theory, sig: Signature, n: int, k: int, seed: int,
                    term_depth: int = 4) -> list[Proof]:
    rng = random.Random(seed)
    out: list[Proof] = []
    while len(out) < n:
        p = random_walk(theory, random_term(sig, term_depth, rng), k, rng)
        if p.steps and p.problem.lhs != p.problem.rhs:
            out.append(p)
    return out
