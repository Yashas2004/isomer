"""Offline move-choice metrics on sibling groups. numpy only, and identical
for the model and for hand-written heuristics, so the comparison is fair.

A searcher guided by h expands the child with the lowest score. For each
group of children of one parent:
* **top1_optimal** (primary). If k children tie for the lowest score and j
  of them have the group's minimum true distance, the group scores j / k.
  That is the chance that random tie-breaking picks a shortest move. Equally
  short moves are all correct, and a heuristic that ties everything earns
  only the base rate.
* ``top1_optimal_strict``: 1 only for a unique lowest score that is optimal.
* ``top1_onpath`` / ``top1_onpath_strict``: the same with "the stored
  proof's move" in place of "a shortest move". For reference only, because
  it counts equally short alternatives as wrong.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.term import Term, parse
from .data import StateSet, optimal_mask


def top1_metrics(scores: np.ndarray, groups: np.ndarray, optimal: np.ndarray) -> dict[str, float]:
    """``scores`` [n] (lower = closer), ``groups`` [G, K] indices into scores
    (-1 padded, on-path child in column 0), ``optimal`` [G, K] bool."""
    if len(groups) == 0:
        return {"top1_optimal": float("nan"), "top1_optimal_strict": float("nan"),
                "top1_onpath": float("nan"), "top1_onpath_strict": float("nan"), "groups": 0}
    valid = groups >= 0
    s = np.where(valid, np.asarray(scores, dtype=np.float64)[np.where(valid, groups, 0)], np.inf)
    tie = valid & (s == s.min(axis=1, keepdims=True))
    k = tie.sum(axis=1)
    j = (tie & optimal).sum(axis=1)
    return {"top1_optimal": float(np.mean(j / k)),
            "top1_optimal_strict": float(np.mean((k == 1) & (j == 1))),
            "top1_onpath": float(np.mean(tie[:, 0] / k)),
            "top1_onpath_strict": float(np.mean(tie[:, 0] & (k == 1))),
            "groups": len(groups)}


def heuristic_scores(ss: StateSet, h: Callable[[Term, Term], float]) -> np.ndarray:
    """h(state, goal) for every state of a set loaded with ``keep_text=True``."""
    assert ss.states is not None and ss.goals is not None, "load_states(..., keep_text=True) needed"
    goals: dict[str, Term] = {}
    out = np.empty(ss.n_states, dtype=np.float64)
    for i, (s, g) in enumerate(zip(ss.states, ss.goals, strict=True)):
        if g not in goals:
            goals[g] = parse(g)
        out[i] = h(parse(s), goals[g])
    return out


def set_metrics(ss: StateSet, scores: np.ndarray) -> dict[str, float]:
    return top1_metrics(scores, ss.groups, optimal_mask(ss.groups, ss.dist))


def mae_by_dist(pred: np.ndarray, dist: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(pred, dtype=np.float64) - dist)
    out = {"all": float(err.mean())} if len(err) else {"all": float("nan")}
    for d in np.unique(dist):
        out[str(int(d))] = float(err[dist == d].mean())
    return out
