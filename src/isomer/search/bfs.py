"""Bidirectional breadth-first search: exact shortest proofs (within budget).

Two uses:
1. **Labels.** The shortest-proof length ``dist`` is the honest difficulty of
   a generated problem (a random walk only gives an upper bound).
2. **Baseline.** Uninformed search that every learned method must beat at an
   equal expansion budget.

"Shortest" is relative to Isomer's action space: fresh variables are
instantiated from the current term's subterms (see
``rewrite.candidate_instantiations``) and terms larger than ``max_size`` are
not explored. Both limits are reported with every result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.proof import Problem, Proof
from ..core.rewrite import Rule, Step, invert_step, successors
from ..core.term import Term, size


@dataclass
class SearchResult:
    proof: Proof | None
    expansions: int           # number of terms whose successors were generated
    generated: int            # number of successor terms produced
    seconds: float
    exhausted: bool = False   # True if the reachable space was fully explored
    lower_bound: int = 0      # proven: no proof shorter than this exists (within limits)
    stats: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.proof is not None


# parent map: term -> (previous term, step that turned previous into term)
_Parents = dict[Term, tuple[Term, Step] | None]


def bidirectional_bfs(problem: Problem, max_expansions: int = 20_000, max_size: int | None = None,
                      max_depth: int = 64, inst_limit: int = 8) -> SearchResult:
    """Grow BFS layers from both ends, always expanding the smaller frontier,
    until the two explored sets meet. Returns a shortest proof under the
    action-space limits above (layer-complete expansion guarantees this)."""
    t0 = time.perf_counter()
    rules = problem.theory.rules()
    rule_list = list(rules.values())
    if max_size is None:
        max_size = max(size(problem.lhs), size(problem.rhs)) + 6

    if problem.lhs == problem.rhs:
        return SearchResult(Proof(problem, ()), 0, 0, time.perf_counter() - t0)

    fwd: _Parents = {problem.lhs: None}
    bwd: _Parents = {problem.rhs: None}
    fwd_frontier, bwd_frontier = [problem.lhs], [problem.rhs]
    expansions = generated = 0
    fwd_depth = bwd_depth = 0   # number of fully completed layers on each side

    def fail(exhausted: bool = False) -> SearchResult:
        # Every term within fwd_depth of lhs and bwd_depth of rhs has been seen
        # and the sets are disjoint, so dist > fwd_depth + bwd_depth.
        return SearchResult(None, expansions, generated, time.perf_counter() - t0,
                            exhausted=exhausted, lower_bound=fwd_depth + bwd_depth + 1)

    while fwd_frontier and bwd_frontier and fwd_depth + bwd_depth < max_depth:
        forward = len(fwd_frontier) <= len(bwd_frontier)
        frontier, seen, other = (fwd_frontier, fwd, bwd) if forward else (bwd_frontier, bwd, fwd)
        next_frontier: list[Term] = []
        meet: Term | None = None
        for t in frontier:
            if expansions >= max_expansions:
                return fail()
            expansions += 1
            for step, u in successors(t, rule_list, inst_limit):
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
                                stats={"fwd_seen": len(fwd), "bwd_seen": len(bwd)})
        if forward:
            fwd_frontier, fwd_depth = next_frontier, fwd_depth + 1
        else:
            bwd_frontier, bwd_depth = next_frontier, bwd_depth + 1

    return fail(exhausted=not (fwd_frontier and bwd_frontier))


def _join(rules: dict[str, Rule], fwd: _Parents, bwd: _Parents, meet: Term) -> list[Step]:
    # lhs ... -> meet: walk fwd parents back to lhs, then reverse.
    head: list[Step] = []
    t = meet
    while (link := fwd[t]) is not None:
        prev, step = link
        head.append(step)
        t = prev
    head.reverse()
    # meet -> ... rhs: each bwd link says "prev --step--> t" was explored from
    # the rhs side, so we traverse it backwards with the inverse step.
    tail: list[Step] = []
    t = meet
    while (link := bwd[t]) is not None:
        prev, step = link
        tail.append(invert_step(rules[step.rule], prev, step))
        t = prev
    return head + tail
