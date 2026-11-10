"""Exact labels for EVERY legal child of the states on a shortest proof.

Training validation (``learn.metrics``) ranks *sampled* sibling groups: the
on-path child plus at most ``offpath`` random off-path children. A searcher
instead chooses among *all* legal children. This module labels all of them,
in the original problem's action space (its pool and size cap, as every
searcher uses), so move-choice accuracy can be measured as search sees it.

Method (exact, no sampling):

1. **Goal ball.** Full BFS layers backward from the goal, B_0..B_R, so
   ``ball[x]`` is the exact distance of every x with dist(x, goal) <= R. The
   action space is symmetric (ADR-003), so backward successors are ordinary
   successors. R is as large as the node cap allows (and <= d - 1).
2. **A state s with exact distance k.** Its children C (distinct terms, size
   within the cap, as in search). Every child has dist in {k-1, k, k+1}.
   * If k - 1 <= R: child u is optimal iff ``ball.get(u) == k - 1``.
   * Else: forward BFS layers from s, where layer 1 is C and every node
     carries a bitmask of the children it is reachable from *at shortest
     distance* (the OR over all its predecessors in the previous layer).
     With a = k - R, a shortest proof from s passes through some x with
     dist(s, x) = a and dist(x, goal) = R. So the optimal children are
     exactly the union of mask(x) over x in F_a with ball[x] == R.
3. If either search exceeds its node cap the state is returned with
   ``status = "unknown"``; nothing is guessed.

The distance of a non-optimal child (k or k+1) is reported only when the
ball contains it; otherwise it is ``None`` ("> R").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.proof import Problem
from ..core.rewrite import Rule, Step, apply_step, successors
from ..core.term import Term, size
from ..search.common import size_cap


@dataclass
class StateLabel:
    state: Term
    dist: int                       # exact distance of the state
    status: str                     # "ok" or "unknown"
    children: list[Term] = field(default_factory=list)
    optimal: list[bool] = field(default_factory=list)
    child_dist: list[int | None] = field(default_factory=list)   # exact when known (inside the ball)
    forward_nodes: int = 0


def path_terms(problem: Problem, steps: list[Step]) -> list[Term]:
    rules = problem.theory.rules()
    terms = [problem.lhs]
    for s in steps:
        t = apply_step(rules[s.rule], terms[-1], s)
        assert t is not None, f"stored proof does not replay at {s}"
        terms.append(t)
    assert terms[-1] == problem.rhs
    return terms


def children_of(t: Term, rules: list[Rule], pool: tuple[Term, ...], cap: int) -> list[Term]:
    """Distinct legal children in generation order, size-capped (as every searcher sees them).
    The state itself is excluded (a rewrite can map a term to itself)."""
    out = dict.fromkeys(u for _, u in successors(t, rules, pool) if size(u) <= cap and u != t)
    return list(out)


def goal_ball(goal: Term, rules: list[Rule], pool: tuple[Term, ...], cap: int, max_radius: int,
              max_nodes: int) -> dict[Term, int]:
    """Exact distances to ``goal`` for every term within the largest complete radius <= max_radius
    whose ball fits in ``max_nodes``. A layer that would overflow is discarded entirely."""
    ball = {goal: 0}
    frontier = [goal]
    for r in range(1, max_radius + 1):
        nxt: dict[Term, None] = {}
        for t in frontier:
            for _, u in successors(t, rules, pool):
                if u not in ball and u not in nxt and size(u) <= cap:
                    nxt[u] = None
            if len(ball) + len(nxt) > max_nodes:
                return ball
        for u in nxt:
            ball[u] = r
        frontier = list(nxt)
        if not frontier:
            break
    return ball


def ball_radius(ball: dict[Term, int]) -> int:
    return max(ball.values())


def label_state(s: Term, k: int, rules: list[Rule], pool: tuple[Term, ...], cap: int,
                ball: dict[Term, int], R: int, max_nodes: int) -> StateLabel:
    kids = children_of(s, rules, pool, cap)
    child_dist = [ball.get(u) for u in kids]
    if k - 1 <= R:
        opt = [ball.get(u) == k - 1 for u in kids]
        return StateLabel(s, k, "ok", kids, opt, child_dist)
    a = k - R
    seen: set[Term] = {s}
    layer: dict[Term, int] = {}
    for i, u in enumerate(kids):
        layer[u] = layer.get(u, 0) | (1 << i)
    seen.update(layer)
    nodes = len(seen)
    for _depth in range(2, a + 1):
        nxt: dict[Term, int] = {}
        for t, m in layer.items():
            for _, u in successors(t, rules, pool):
                if u in seen or size(u) > cap:
                    continue
                nxt[u] = nxt.get(u, 0) | m
            if nodes + len(nxt) > max_nodes:
                return StateLabel(s, k, "unknown", kids, [], child_dist, nodes + len(nxt))
        seen.update(nxt)
        nodes += len(nxt)
        layer = nxt
    mask = 0
    for x, m in layer.items():
        if ball.get(x) == R:
            mask |= m
    opt = [bool(mask >> i & 1) for i in range(len(kids))]
    return StateLabel(s, k, "ok", kids, opt, child_dist, nodes)


def label_problem(problem: Problem, steps: list[Step], pool_max_size: int = 3, slack: int = 6,
                  ball_nodes: int = 400_000, forward_nodes: int = 400_000) -> tuple[list[StateLabel], int]:
    """Labels for every on-path state t_0 .. t_{d-1} of a stored shortest proof.
    Returns (labels, ball radius R)."""
    path = path_terms(problem, steps)
    d = len(path) - 1
    rules = list(problem.theory.rules().values())
    pool = problem.pool(pool_max_size)
    cap = size_cap(problem, slack)
    ball = goal_ball(problem.rhs, rules, pool, cap, max(0, d - 1), ball_nodes)
    R = ball_radius(ball)
    for i, t in enumerate(path):     # consistency: on-path states inside the ball have the right distance
        if t in ball:
            assert ball[t] == d - i, f"ball distance {ball[t]} != {d - i}: stored proof is not shortest"
    out = [label_state(path[i], d - i, rules, pool, cap, ball, R, forward_nodes) for i in range(d)]
    for i, lab in enumerate(out):    # the stored next step is always an optimal child
        if lab.status == "ok":
            assert lab.optimal[lab.children.index(path[i + 1])], "on-path child not labelled optimal"
    return out, R
