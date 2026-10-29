"""Training data for the distance model: tokenized states, sibling groups and
problem-level batches. numpy only, so it imports and tests without torch.

**Sibling groups.** Within one problem, the on-path state with dist p is a
*parent*. Its group is every child that one legal step reaches and that the
states build sampled:
* the on-path state with dist p - 1 (the move the stored proof makes);
* every off-path state with ``parent_dist == p``.
Groups are derived from ``(problem_id, parent_dist)``. That is unambiguous
because a problem's on-path distances are unique. The on-path child is
stored in column 0. A group with no sampled siblings is a *singleton*: it
cannot rank anything, so it is excluded from the ranking loss and from the
metrics, but counted.

**Batches are whole problems.** All states of k problems form one batch, so
every sibling group is complete inside it. A batch is split into
micro-batches at problem boundaries, so that a 64-problem batch (about 1,200
sequences) fits a 4 GB GPU; see ``micro_batches``. Problem order is
shuffled per epoch by ``np.random.default_rng([seed, epoch])``, so batch
composition is reproducible.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..core.term import parse
from .tokenizer import PAD_ID, encode


@dataclass
class StateSet:
    tokens: np.ndarray          # uint8, all encoded sequences concatenated
    offsets: np.ndarray         # int64 [n + 1]
    dist: np.ndarray            # int16 [n]
    on_path: np.ndarray         # bool [n]
    problem_ids: list[str]      # [P], sorted
    problem_starts: np.ndarray  # int64 [P + 1]; a problem's states are contiguous
    groups: np.ndarray          # int64 [G, K]; global state indices, -1 padded, on-path child in column 0
    group_starts: np.ndarray    # int64 [P + 1]; a problem's groups are contiguous
    singleton_groups: int
    states: list[str] | None = None   # text, kept only when asked (heuristic baselines)
    goals: list[str] | None = None

    @property
    def n_states(self) -> int:
        return len(self.dist)

    @property
    def n_problems(self) -> int:
        return len(self.problem_ids)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def derive_groups(dist: list[int], on_path: list[bool], parent_dist: list[int | None],
                  base: int = 0) -> tuple[list[list[int]], int]:
    """Sibling groups of ONE problem whose states are listed in the arguments
    (indices offset by ``base``). Returns (groups with >= 2 members, number of
    singleton groups). The on-path child comes first in every group."""
    on_by_dist: dict[int, int] = {}
    off_by_parent: dict[int, list[int]] = {}
    for i, (d, on, pd) in enumerate(zip(dist, on_path, parent_dist, strict=True)):
        if on:
            assert d not in on_by_dist, f"two on-path states with dist {d}"
            on_by_dist[d] = base + i
        else:
            assert pd is not None, "off-path state without parent_dist"
            off_by_parent.setdefault(pd, []).append(base + i)
    assert sorted(on_by_dist) == list(range(len(on_by_dist))), "on-path distances must be 0..d"
    assert all(1 <= p < len(on_by_dist) for p in off_by_parent), "off-path parent is not an on-path state"
    groups, singletons = [], 0
    for p in range(1, len(on_by_dist)):
        members = [on_by_dist[p - 1], *off_by_parent.get(p, [])]
        assert sum(on_path[m - base] for m in members) == 1, "a group needs exactly one on-path child"
        if len(members) == 1:
            singletons += 1
        else:
            groups.append(members)
    return groups, singletons


def load_states(path: str | Path, max_len: int, keep_text: bool = False) -> StateSet:
    """Read one ``states`` jsonl file (train or val) and tokenize every pair once.
    A pair longer than ``max_len`` raises the tokenizer's error."""
    tokens, offsets = array("B"), [0]
    dist: list[int] = []
    on_path: list[bool] = []
    parent: list[int | None] = []
    pids: list[str] = []
    starts: list[int] = []
    states: list[str] = []
    goals: list[str] = []
    goal_terms: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not pids or r["problem_id"] != pids[-1]:
                assert not pids or r["problem_id"] > pids[-1], "states file must be sorted by problem_id"
                pids.append(r["problem_id"])
                starts.append(len(dist))
            goal = goal_terms.get(r["goal"])
            if goal is None:
                goal = goal_terms[r["goal"]] = parse(r["goal"])
            ids = encode(parse(r["state"]), goal, max_len=max_len)
            tokens.extend(ids)
            offsets.append(offsets[-1] + len(ids))
            dist.append(r["dist"])
            on_path.append(r["on_path"])
            parent.append(r["parent_dist"])
            if keep_text:
                states.append(r["state"])
                goals.append(r["goal"])
    starts.append(len(dist))

    groups: list[list[int]] = []
    group_starts = [0]
    singletons = 0
    for a, b in pairwise(starts):
        g, s = derive_groups(dist[a:b], on_path[a:b], parent[a:b], base=a)
        groups.extend(g)
        group_starts.append(len(groups))
        singletons += s
    width = max((len(g) for g in groups), default=1)
    garr = np.full((len(groups), width), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        garr[i, :len(g)] = g
    return StateSet(tokens=np.frombuffer(tokens, dtype=np.uint8).copy(), offsets=np.array(offsets, dtype=np.int64),
                    dist=np.array(dist, dtype=np.int16), on_path=np.array(on_path, dtype=bool), problem_ids=pids,
                    problem_starts=np.array(starts, dtype=np.int64), groups=garr,
                    group_starts=np.array(group_starts, dtype=np.int64), singleton_groups=singletons,
                    states=states if keep_text else None, goals=goals if keep_text else None)


def optimal_mask(groups: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """[G, K] bool: member is a child whose true dist equals the group's minimum.
    Equally short moves are all correct; the training target and the primary
    metric treat them alike."""
    valid = groups >= 0
    # cast before filling: the sentinel would overflow dist's int16 and wrap negative
    d = np.where(valid, dist.astype(np.int64)[np.where(valid, groups, 0)], np.iinfo(np.int64).max)
    return valid & (d == d.min(axis=1, keepdims=True))


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def epoch_batches(n_problems: int, k: int, seed: int, epoch: int) -> Iterator[np.ndarray]:
    """Problem indices of each batch: a seeded permutation cut into chunks of k."""
    order = np.random.default_rng([seed, epoch]).permutation(n_problems)
    for i in range(0, n_problems, k):
        yield order[i:i + k]


def micro_batches(ss: StateSet, problems: np.ndarray, micro_tokens: int) -> list[np.ndarray]:
    """Split a batch at problem boundaries so that each part's padded size
    (states x longest sequence) stays within ``micro_tokens``. A single
    problem larger than the budget forms its own micro-batch."""
    out: list[np.ndarray] = []
    cur: list[int] = []
    n_cur = len_cur = 0
    lengths = ss.lengths
    for p in problems:
        a, b = ss.problem_starts[p], ss.problem_starts[p + 1]
        n_p, len_p = int(b - a), int(lengths[a:b].max())
        if cur and (n_cur + n_p) * max(len_cur, len_p) > micro_tokens:
            out.append(np.array(cur, dtype=np.int64))
            cur, n_cur, len_cur = [], 0, 0
        cur.append(int(p))
        n_cur += n_p
        len_cur = max(len_cur, len_p)
    if cur:
        out.append(np.array(cur, dtype=np.int64))
    return out


@dataclass
class Micro:
    ids: np.ndarray        # int64 [B, L], PAD-padded to the micro-batch's longest
    dist: np.ndarray       # float32 [B]
    index: np.ndarray      # int64 [B], global state indices
    groups: np.ndarray     # int64 [G, K], local row indices, -1 padded, on-path child in column 0
    optimal: np.ndarray    # bool [G, K]


def collate(ss: StateSet, problems: np.ndarray) -> Micro:
    index = np.concatenate([np.arange(ss.problem_starts[p], ss.problem_starts[p + 1]) for p in problems])
    lengths = ss.lengths[index]
    ids = np.full((len(index), int(lengths.max())), PAD_ID, dtype=np.int64)
    for row, (i, n) in enumerate(zip(index, lengths, strict=True)):
        ids[row, :n] = ss.tokens[ss.offsets[i]:ss.offsets[i] + n]
    glob, local, base = [], [], 0
    for p in problems:   # global -> local row: shift each problem's indices to where its rows start
        gp = ss.groups[ss.group_starts[p]:ss.group_starts[p + 1]]
        glob.append(gp)
        local.append(np.where(gp >= 0, gp - ss.problem_starts[p] + base, -1))
        base += int(ss.problem_starts[p + 1] - ss.problem_starts[p])
    g = np.concatenate(glob)
    return Micro(ids=ids, dist=ss.dist[index].astype(np.float32), index=index, groups=np.concatenate(local),
                 optimal=optimal_mask(g, ss.dist))
