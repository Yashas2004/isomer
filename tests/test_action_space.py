"""ADR-003: the problem-level pool makes the action space symmetric."""

from collections import deque

import pytest

from isomer.core.generate import sample_problems
from isomer.core.proof import Problem
from isomer.core.rewrite import invert_step, problem_pool, successors
from isomer.core.term import App, parse, size
from isomer.core.theories import BOOL, GROUP, THEORIES
from isomer.search.bfs import bidirectional_bfs
from isomer.search.common import SEARCHERS, Budget


@pytest.mark.parametrize("name", sorted(THEORIES))
def test_symmetry_every_legal_step_has_a_legal_inverse(name):
    """For every legal step t -> u, t is a successor of u under the same pool.
    This is the property that makes bidirectional BFS labels exact."""
    theory, sig = THEORIES[name]
    rules = theory.rules()
    rule_list = list(rules.values())
    checked = 0
    for walk in sample_problems(theory, sig, n=4, k=3, seed=21, term_depth=3):
        pool = walk.problem.pool()
        for t in walk.trace():
            for step, u in successors(t, rule_list, pool):
                back = dict(successors(u, rule_list, pool))
                inverse = invert_step(rules[step.rule], t, step)
                assert back.get(inverse) == t, (name, str(t), step, str(u))
                checked += 1
    assert checked > 100


@pytest.mark.parametrize("name", sorted(THEORIES))
def test_every_walk_step_is_legal_in_the_final_problem(name):
    theory, sig = THEORIES[name]
    rule_list = list(theory.rules().values())
    for walk in sample_problems(theory, sig, n=25, k=6, seed=4):
        pool = walk.problem.pool()
        terms = walk.trace()
        for before, after, step in zip(terms, terms[1:], walk.steps):
            assert (step, after) in set(successors(before, rule_list, pool)), (name, str(before), step)


def test_v0_label_bug_regression():
    """bool (~ F) = T: forward proof of 3 steps (or_id~, or_comm, or_compl
    consuming x=F). Under ADR-002 the backward half could not replay the last
    step, so the v0 label said 5."""
    p = Problem.parse(BOOL, "(~ F)", "T")
    for name in ("bfs-fwd", "ucs", "bfs-bidir"):
        r = SEARCHERS[name](p, Budget())
        assert r.solved and len(r.proof.steps) == 3 and r.proof.verify(), name
    assert len(bidirectional_bfs(p).proof.steps) == 3


def test_problem_pool_definition():
    lhs, rhs = parse("(* (inv a) (* a (* b b)))"), parse("e")
    pool = problem_pool(("e",), ("a", "b", "c"), lhs, rhs, max_size=3)
    strs = [str(t) for t in pool]
    assert strs == ["a", "b", "e", "(inv a)", "(* b b)"]  # exactly the mentioned skolems, plus constant e
    assert all(size(t) <= 3 for t in pool) and len(set(pool)) == len(pool)
    assert pool == problem_pool(("e",), ("c", "b", "a"), lhs, rhs, max_size=3)  # order-independent
    assert Problem.parse(GROUP, str(lhs), str(rhs)).pool() == pool


def test_unmentioned_skolem_is_not_in_pool():
    pool = Problem.parse(GROUP, "(* a e)", "a").pool()
    assert App("b") not in pool and App("c") not in pool
    assert App("e") in pool  # theory constants are always included
    # A mentioned skolem is in the pool even when max_size excludes everything else.
    assert App("a") in problem_pool(("e",), ("a", "b"), parse("(* a e)"), parse("a"), max_size=0)


def _bfs_dist(problem, pool, limit):
    """Plain forward BFS distance with an explicit pool (the same size cap as
    the searchers); None if not found within ``limit`` steps."""
    rules = list(problem.theory.rules().values())
    cap = max(size(problem.lhs), size(problem.rhs)) + Budget().max_size_slack
    seen, q = {problem.lhs: 0}, deque([problem.lhs])
    while q:
        t = q.popleft()
        if seen[t] >= limit:
            continue
        for _, u in successors(t, rules, pool):
            if u not in seen and size(u) <= cap:
                seen[u] = seen[t] + 1
                if u == problem.rhs:
                    return seen[u]
                q.append(u)
    return None


@pytest.mark.slow
@pytest.mark.parametrize("name", ["group", "lattice"])
def test_mentioned_skolems_pool_is_lossless(name, k=3):
    """ADR-003: leaving unmentioned skolems out of the pool never changes the
    shortest distance (they carry no axioms, so a proof can rename them)."""
    theory, sig = THEORIES[name]
    walks = sample_problems(theory, sig, n=10, k=k, seed=8, term_depth=3)
    unmentioned = 0
    for w in walks:
        p = w.problem
        pool = p.pool()
        full = tuple(sorted(set(pool) | {App(s) for s in sig.skolems}, key=lambda u: (size(u), str(u))))
        unmentioned += len(full) > len(pool)
        d = _bfs_dist(p, pool, limit=k)
        assert d is not None and d == _bfs_dist(p, full, limit=k), str(p)
        fwd = SEARCHERS["bfs-fwd"](p, Budget(max_expansions=50_000))
        assert fwd.solved and len(fwd.proof.steps) == d  # the helper agrees with the real searcher
    assert unmentioned >= 3  # the comparison is not vacuous
