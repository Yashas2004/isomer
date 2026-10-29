from collections import deque

import pytest

from isomer.core.generate import sample_problems
from isomer.core.proof import Problem, Proof
from isomer.core.rewrite import invert_step, successors
from isomer.core.term import size
from isomer.core.theories import GROUP, THEORIES
from isomer.search.bfs import bidirectional_bfs


def oracle_dist(problem, max_size, limit=6):
    """Plain one-directional BFS: slow but obviously correct."""
    rules = list(problem.theory.rules().values())
    pool = problem.pool()
    seen, q = {problem.lhs: 0}, deque([problem.lhs])
    while q:
        t = q.popleft()
        if seen[t] >= limit:
            continue
        for _, u in successors(t, rules, pool):
            if u not in seen and size(u) <= max_size:
                seen[u] = seen[t] + 1
                if u == problem.rhs:
                    return seen[u]
                q.append(u)
    return None


def test_demo_problem_has_distance_two():
    r = bidirectional_bfs(Problem.parse(GROUP, "(* e (* (inv a) a))", "e"))
    assert r.solved and len(r.proof.steps) == 2 and r.proof.verify()


def test_trivial_problem():
    r = bidirectional_bfs(Problem.parse(GROUP, "a", "a"))
    assert r.solved and r.proof.steps == ()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["group", "lattice", "bool"])
def test_bidirectional_matches_oracle(name):
    theory, sig = THEORIES[name]
    for p in sample_problems(theory, sig, n=12, k=3, seed=5, term_depth=3):
        ms = max(size(p.problem.lhs), size(p.problem.rhs)) + 6
        r = bidirectional_bfs(p.problem, max_expansions=50_000, max_size=ms)
        assert r.solved and r.proof.verify()
        assert len(r.proof.steps) == oracle_dist(p.problem, ms)


@pytest.mark.parametrize("name", sorted(THEORIES))
def test_bfs_never_longer_than_walk_and_kernel_accepts(name):
    theory, sig = THEORIES[name]
    for p in sample_problems(theory, sig, n=15, k=4, seed=11, term_depth=3):
        r = bidirectional_bfs(p.problem, max_expansions=2000)
        if r.solved:
            assert len(r.proof.steps) <= len(p.steps)
            assert r.proof.verify()
        else:
            assert r.lower_bound >= 1


@pytest.mark.parametrize("name", sorted(THEORIES))
def test_every_step_is_invertible(name):
    theory, sig = THEORIES[name]
    rules = theory.rules()
    for p in sample_problems(theory, sig, n=20, k=4, seed=3):
        terms = p.trace()
        for before, after, step in zip(terms, terms[1:], p.steps):
            back = invert_step(rules[step.rule], before, step)
            assert Proof(Problem(theory, after, before), (back,)).verify()


def test_explicit_pool_equals_default():
    """Passing the problem's own pool changes nothing (ADR-003 default)."""
    for lhs, rhs in [("(* e (* (inv a) a))", "e"), ("(* (inv (inv a)) b)", "(* a b)")]:
        p = Problem.parse(GROUP, lhs, rhs)
        a, b = bidirectional_bfs(p, max_expansions=500), bidirectional_bfs(p, max_expansions=500, pool=p.pool())
        assert (a.proof.steps if a.proof else None) == (b.proof.steps if b.proof else None)
        assert (a.expansions, a.generated, a.lower_bound, a.stats) == (b.expansions, b.generated, b.lower_bound,
                                                                         b.stats)
