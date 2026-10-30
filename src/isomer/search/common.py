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
  ``"time"``, ``"memory"`` or ``"exhausted"``.

**Memory cap.** ``Budget.max_nodes`` bounds ``nodes_stored``. Like the
expansion budget it is checked only *before* an expansion, so the final count
can overshoot by one expansion's successors. A node stop is deterministic
(unlike a time stop), but the benchmark still censors it above the expansions
reached: the budgets we report are expansion budgets, and an uncapped run
might have solved the problem there.

**Prefix property.** Every searcher makes the same choices whatever the
budget, and it checks the expansion budget only *before* an expansion. So a
run that solves during expansion ``e`` returns the identical proof under any
``max_expansions >= e``, and fails under any smaller budget. One run at
budget B therefore gives the results at every smaller budget. This only holds
with ``max_seconds=None``, because wall-clock limits are not reproducible.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
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
    """Limits shared by all searchers. ``max_size_slack`` and ``pool_max_size``
    define the action space (ADR-002, ADR-003); the other three cap compute.
    Keep ``max_seconds=None`` for reproducible runs (see module docstring)."""
    max_expansions: int = 1000
    max_seconds: float | None = None
    max_size_slack: int = 6
    pool_max_size: int = 3
    max_nodes: int | None = None   # cap on nodes_stored (ADR-004)
    # Report stats["max_state_tokens"]: the longest encoded (state, goal) pair among stored
    # terms. Computed after the clock stops, so it never changes ``seconds``.
    track_state_tokens: bool = False


class Searcher(Protocol):
    def __call__(self, problem: Problem, budget: Budget) -> SearchResult: ...


SEARCHERS: dict[str, Callable[[Problem, Budget], SearchResult]] = {}

# Searchers parameterized by a path, named ``prefix@path`` (e.g.
# ``neural-w1@runs/h-v1/best.pt``): prefix -> factory(path) -> searcher. The
# factory must not do heavy work (no model loading): it also validates names.
PARAMETRIC: dict[str, Callable[[str], Callable[[Problem, Budget], SearchResult]]] = {}


def get_searcher(name: str) -> Callable[[Problem, Budget], SearchResult]:
    if name in SEARCHERS:
        return SEARCHERS[name]
    prefix, sep, arg = name.partition("@")
    if sep and arg and prefix in PARAMETRIC:
        return PARAMETRIC[prefix](arg)
    raise KeyError(f"unknown searcher {name!r}; valid: {', '.join(sorted(SEARCHERS))}, "
                   f"or {', '.join(f'{p}@PATH' for p in sorted(PARAMETRIC))}")


def size_cap(problem: Problem, slack: int) -> int:
    """Terms larger than this are not explored: one definition for all searchers."""
    return max(size(problem.lhs), size(problem.rhs)) + slack


def out_of_time(t0: float, max_seconds: float | None) -> bool:
    return max_seconds is not None and time.perf_counter() - t0 > max_seconds


def out_of_memory(nodes: int, max_nodes: int | None) -> bool:
    return max_nodes is not None and nodes >= max_nodes


# parent map: term -> (previous term, step that turned previous into term)
Parents = dict[Term, tuple[Term, Step] | None]


def token_count(t: Term, memo: dict[Term, int]) -> int:
    """Tokens of ``t`` as ``learn.tokenizer.term_tokens`` emits them: a leaf is 1,
    an application ``( op args )`` is 3 plus its arguments. Torch-free."""
    n = memo.get(t)
    if n is None:
        args = getattr(t, "args", ())
        n = memo[t] = 3 + sum(token_count(a, memo) for a in args) if args else 1
    return n


def max_state_tokens(terms: Iterable[Term], goal: Term) -> int:
    """Longest ``tokenizer.n_tokens(state, goal)`` over ``terms``:
    [CLS] state [SEP] goal [SEP]. Used to check model length extrapolation."""
    memo: dict[Term, int] = {}
    return 3 + max((token_count(t, memo) for t in terms), default=0) + token_count(goal, memo)


def steps_from_parents(parents: Parents, t: Term) -> list[Step]:
    """Follow parent pointers from ``t`` back to the root; return the steps root -> t."""
    steps: list[Step] = []
    while (link := parents[t]) is not None:
        t, step = link
        steps.append(step)
    steps.reverse()
    return steps
