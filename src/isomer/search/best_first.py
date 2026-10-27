"""Weighted best-first search from lhs toward rhs.

Priority is ``g + w * h(t, rhs)``, where ``g`` is the number of steps from lhs.
Ties are broken by an insertion counter (FIFO), so runs are deterministic.

Design choices, all made so that the benchmark numbers mean something:

* **Duplicates are detected on generation, with no reopening.** Each term
  keeps the first parent that produced it. With ``h = zero`` (uniform-cost
  search) and unit step costs, the FIFO tie-break pops terms in nondecreasing
  ``g``, so that first parent already has minimal ``g`` and UCS proofs are
  shortest. With a heuristic the proofs may be longer; that is the trade-off
  being measured.
* **The goal test happens on generation**, as in ``bfs.forward_bfs``. Without
  reopening, the path returned is the one a goal test on pop would return,
  but it costs fewer expansions, and expansion counts stay comparable with BFS.
* The size cap and budget checks are the shared ones (``common``), so the
  prefix property holds.
"""

from __future__ import annotations

import heapq
import time
from functools import partial

from ..core.proof import Problem, Proof
from ..core.rewrite import successors
from ..core.term import Term, size
from .common import SEARCHERS, Budget, Parents, SearchResult, out_of_time, size_cap, steps_from_parents
from .heuristics import Heuristic, size_diff, symbol_bag, zero


def best_first(problem: Problem, budget: Budget, h: Heuristic = zero, w: float = 1.0) -> SearchResult:
    t0 = time.perf_counter()
    rule_list = list(problem.theory.rules().values())
    max_size = size_cap(problem, budget.max_size_slack)
    goal = problem.rhs
    parents: Parents = {problem.lhs: None}
    expansions = generated = 0

    def result(stop: str, proof: Proof | None = None) -> SearchResult:
        return SearchResult(proof, expansions, generated, time.perf_counter() - t0,
                            exhausted=stop == "exhausted",
                            lower_bound=len(proof.steps) if proof else 0,
                            stats={"nodes_stored": len(parents), "stop": stop})

    if problem.lhs == goal:
        return result("solved", Proof(problem, ()))

    counter = 0
    heap: list[tuple[float, int, int, Term]] = [(w * h(problem.lhs, goal), counter, 0, problem.lhs)]
    while heap:
        if expansions >= budget.max_expansions:
            return result("budget")
        if out_of_time(t0, budget.max_seconds):
            return result("time")
        _, _, g, t = heapq.heappop(heap)
        expansions += 1
        for step, u in successors(t, rule_list, budget.inst_limit):
            generated += 1
            if u in parents or size(u) > max_size:
                continue
            parents[u] = (t, step)
            if u == goal:
                return result("solved", Proof(problem, tuple(steps_from_parents(parents, u))))
            counter += 1
            heapq.heappush(heap, (g + 1 + w * h(u, goal), counter, g + 1, u))
    return result("exhausted")


# functools.partial of module-level functions: picklable for worker processes.
SEARCHERS["ucs"] = partial(best_first, h=zero, w=1.0)
for _name, _h in (("size_diff", size_diff), ("symbol_bag", symbol_bag)):
    for _w in (1, 3):
        SEARCHERS[f"bestfirst-{_name}-w{_w}"] = partial(best_first, h=_h, w=float(_w))
