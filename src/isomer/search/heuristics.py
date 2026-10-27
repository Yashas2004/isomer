"""Hand-written distance estimates ``h(term, goal) -> float`` for best-first search.

These are the non-neural baselines a learned value head has to beat, and the
reference points for RQ5 (does the value head compute something like these?).
None is admissible in general: a rewrite can change size and symbols by more
than one. They are pure functions of the two terms, so results are reproducible
and the functions can be pickled for worker processes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..core.term import App, Term, size

Heuristic = Callable[[Term, Term], float]


def zero(t: Term, goal: Term) -> float:
    """No information: turns best-first into uniform-cost search."""
    return 0.0


def size_diff(t: Term, goal: Term) -> float:
    return float(abs(size(t) - size(goal)))


def _symbols(t: Term) -> Counter[str]:
    """Multiset of operator and constant symbols (problem terms are ground)."""
    out: Counter[str] = Counter()
    stack = [t]
    while stack:
        u = stack.pop()
        if isinstance(u, App):
            out[u.op] += 1
            stack.extend(u.args)
    return out


def symbol_bag(t: Term, goal: Term) -> float:
    """L1 distance between symbol multisets: catches a wrong symbol even when
    the sizes already agree, which ``size_diff`` cannot see."""
    a, b = _symbols(t), _symbols(goal)
    return float(sum(abs(a[k] - b[k]) for k in a.keys() | b.keys()))


HEURISTICS: dict[str, Heuristic] = {
    "zero": zero,
    "size_diff": size_diff,
    "symbol_bag": symbol_bag,
}
