"""The interface every searcher shares: budget in, ``SearchResult`` out.

Searchers are plain functions ``(problem, budget) -> SearchResult`` held in the
``SEARCHERS`` registry, so the benchmark runner can pick them by name and ship
them to worker processes. Registry entries must therefore be picklable:
module-level functions or ``functools.partial`` of them, never lambdas.

Every result carries two required stats:

* ``stats["nodes_stored"]``: the size of the searcher's seen/parent maps when
  it returns. It is a memory proxy that does not depend on the machine, for
  the memory-matched comparisons in DESIGN.md §4.
* ``stats["stop"]``: why the search ended: ``"solved"``, ``"budget"``,
  ``"time"`` or ``"exhausted"``.

**Prefix property.** Every searcher makes the same choices whatever the
budget, and it checks the expansion budget only *before* an expansion. So a
run that solves during expansion ``e`` returns the identical proof under any
``max_expansions >= e``, and fails under any smaller budget. One run at
budget B therefore gives the results at every smaller budget. This only holds
with ``max_seconds=None``, because wall-clock limits are not reproducible.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..core.proof import Problem, Proof
from ..core.rewrite import Step
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


@dataclass(frozen=True)
class Budget:
    """Limits shared by all searchers. ``max_size_slack`` and ``inst_limit``
    define the action space (ADR 002 §4); the other two cap compute.
    Keep ``max_seconds=None`` for reproducible runs (see module docstring)."""
    max_expansions: int = 1000
    max_seconds: float | None = None
    max_size_slack: int = 6
    inst_limit: int = 8


class Searcher(Protocol):
    def __call__(self, problem: Problem, budget: Budget) -> SearchResult: ...


SEARCHERS: dict[str, Callable[[Problem, Budget], SearchResult]] = {}


def get_searcher(name: str) -> Callable[[Problem, Budget], SearchResult]:
    try:
        return SEARCHERS[name]
    except KeyError:
        raise KeyError(f"unknown searcher {name!r}; valid: {', '.join(sorted(SEARCHERS))}") from None


def size_cap(problem: Problem, slack: int) -> int:
    """Terms larger than this are not explored: one definition for all searchers."""
    return max(size(problem.lhs), size(problem.rhs)) + slack


def out_of_time(t0: float, max_seconds: float | None) -> bool:
    return max_seconds is not None and time.perf_counter() - t0 > max_seconds


# parent map: term -> (previous term, step that turned previous into term)
Parents = dict[Term, tuple[Term, Step] | None]


def steps_from_parents(parents: Parents, t: Term) -> list[Step]:
    """Follow parent pointers from ``t`` back to the root; return the steps root -> t."""
    steps: list[Step] = []
    while (link := parents[t]) is not None:
        t, step = link
        steps.append(step)
    steps.reverse()
    return steps
