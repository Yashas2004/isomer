"""Breadth-first search: exact shortest proofs (within budget).

Two uses:
1. **Labels.** The shortest-proof length ``dist`` is the honest difficulty of
   a generated problem (a random walk only gives an upper bound).
2. **Baseline.** Uninformed search that every learned method must beat at an
   equal expansion budget.

"Shortest" is relative to Isomer's action space (ADR-003): values of
introduced and consumed variables come from the problem's pool
(``Problem.pool``), and terms larger than ``max_size`` are not explored. The
pool makes the action space symmetric, so the backward half replays exactly
the forward steps and bidirectional distances are exact.
"""

from __future__ import annotations

import time

from ..core.proof import Problem, Proof
from ..core.rewrite import Rule, Step, invert_step, successors
from ..core.term import Term, size
from .common import SEARCHERS, Budget, Parents, SearchResult, out_of_time, size_cap, steps_from_parents

__all__ = ["SearchResult", "bidir_bfs", "bidirectional_bfs", "forward_bfs"]


def bidirectional_bfs(problem: Problem, max_expansions: int = 20_000, max_size: int | None = None,
                      max_depth: int = 64, pool_max_size: int = 3,
                      max_seconds: float | None = None) -> SearchResult:
    """Grow BFS layers from both ends, always expanding the smaller frontier,
    until the two explored sets meet. Returns a shortest proof under the
    action-space limits above (layer-complete expansion guarantees this)."""
    t0 = time.perf_counter()
    rules = problem.theory.rules()
    rule_list = list(rules.values())
    pool = problem.pool(pool_max_size)
    if max_size is None:
        max_size = size_cap(problem, 6)

    if problem.lhs == problem.rhs:
        return SearchResult(Proof(problem, ()), 0, 0, time.perf_counter() - t0,
                            stats={"nodes_stored": 1, "stop": "solved"})

    fwd: Parents = {problem.lhs: None}
    bwd: Parents = {problem.rhs: None}
    fwd_frontier, bwd_frontier = [problem.lhs], [problem.rhs]
    expansions = generated = 0
    fwd_depth = bwd_depth = 0   # number of fully completed layers on each side

    def fail(stop: str, exhausted: bool = False) -> SearchResult:
        # Every term within fwd_depth of lhs and bwd_depth of rhs has been seen
        # and the sets are disjoint, so dist > fwd_depth + bwd_depth.
        return SearchResult(None, expansions, generated, time.perf_counter() - t0,
                            exhausted=exhausted, lower_bound=fwd_depth + bwd_depth + 1,
                            stats={"fwd_seen": len(fwd), "bwd_seen": len(bwd),
                                   "nodes_stored": len(fwd) + len(bwd), "stop": stop})

    while fwd_frontier and bwd_frontier and fwd_depth + bwd_depth < max_depth:
        forward = len(fwd_frontier) <= len(bwd_frontier)
        frontier, seen, other = (fwd_frontier, fwd, bwd) if forward else (bwd_frontier, bwd, fwd)
        next_frontier: list[Term] = []
        meet: Term | None = None
        for t in frontier:
            if expansions >= max_expansions:
                return fail("budget")
            if out_of_time(t0, max_seconds):
                return fail("time")
            expansions += 1
            for step, u in successors(t, rule_list, pool):
                generated += 1
                if u in seen or size(u) > max_size:
                    continue
                seen[u] = (t, step)
                if u in other:
                    meet = u
                    break
                next_frontier.append(u)
            if meet is not None:
                break
        if meet is not None:
            steps = _join(rules, fwd, bwd, meet)
            proof = Proof(problem, tuple(steps))
            return SearchResult(proof, expansions, generated, time.perf_counter() - t0,
                                lower_bound=len(steps),
                                stats={"fwd_seen": len(fwd), "bwd_seen": len(bwd),
                                       "nodes_stored": len(fwd) + len(bwd), "stop": "solved"})
        if forward:
            fwd_frontier, fwd_depth = next_frontier, fwd_depth + 1
        else:
            bwd_frontier, bwd_depth = next_frontier, bwd_depth + 1

    exhausted = not (fwd_frontier and bwd_frontier)
    return fail("exhausted" if exhausted else "budget", exhausted=exhausted)


def _join(rules: dict[str, Rule], fwd: Parents, bwd: Parents, meet: Term) -> list[Step]:
    # lhs ... -> meet: follow fwd parents back to lhs.
    head = steps_from_parents(fwd, meet)
    # meet -> ... rhs: each bwd link says "prev --step--> t" was explored from
    # the rhs side, so we traverse it backwards with the inverse step.
    tail: list[Step] = []
    t = meet
    while (link := bwd[t]) is not None:
        prev, step = link
        tail.append(invert_step(rules[step.rule], prev, step))
        t = prev
    return head + tail


def bidir_bfs(problem: Problem, budget: Budget) -> SearchResult:
    """``bidirectional_bfs`` behind the shared ``Budget`` interface."""
    return bidirectional_bfs(problem, max_expansions=budget.max_expansions,
                             max_size=size_cap(problem, budget.max_size_slack),
                             pool_max_size=budget.pool_max_size, max_seconds=budget.max_seconds)


def forward_bfs(problem: Problem, budget: Budget) -> SearchResult:
    """One-directional BFS from lhs. The goal test happens on generation (the
    first time rhs is produced, its layer is minimal), so the proof is
    shortest within the action space. Slower than bidirectional search, but it
    is the textbook baseline and the direction a learned policy searches in."""
    t0 = time.perf_counter()
    rule_list = list(problem.theory.rules().values())
    pool = problem.pool(budget.pool_max_size)
    max_size = size_cap(problem, budget.max_size_slack)
    parents: Parents = {problem.lhs: None}

    def result(stop: str, proof: Proof | None = None, depth: int = 0) -> SearchResult:
        lb = len(proof.steps) if proof else depth + 1
        return SearchResult(proof, expansions, generated, time.perf_counter() - t0,
                            exhausted=stop == "exhausted", lower_bound=lb,
                            stats={"nodes_stored": len(parents), "stop": stop})

    expansions = generated = 0
    if problem.lhs == problem.rhs:
        return result("solved", Proof(problem, ()))

    frontier = [problem.lhs]
    depth = 0   # number of fully completed layers: every term within depth of lhs is seen
    while frontier:
        next_frontier: list[Term] = []
        for t in frontier:
            if expansions >= budget.max_expansions:
                return result("budget", depth=depth)
            if out_of_time(t0, budget.max_seconds):
                return result("time", depth=depth)
            expansions += 1
            for step, u in successors(t, rule_list, pool):
                generated += 1
                if u in parents or size(u) > max_size:
                    continue
                parents[u] = (t, step)
                if u == problem.rhs:
                    return result("solved", Proof(problem, tuple(steps_from_parents(parents, u))))
                next_frontier.append(u)
        frontier, depth = next_frontier, depth + 1
    return result("exhausted", depth=depth)


SEARCHERS["bfs-fwd"] = forward_bfs
SEARCHERS["bfs-bidir"] = bidir_bfs
