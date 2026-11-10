"""E3 child labeller: every label agrees with an independent per-child BFS."""

import random

import pytest

from isomer.core.generate import random_term, random_walk
from isomer.core.proof import Problem
from isomer.core.theories import THEORIES
from isomer.e3.children import label_problem
from isomer.search.bfs import bidirectional_bfs
from isomer.search.common import size_cap


def _shortest(theory_name: str, seed: int):
    theory, sig = THEORIES[theory_name]
    rng = random.Random(seed)
    for _ in range(200):
        p = random_walk(theory, random_term(sig, 3, rng), rng.choice([3, 4, 5]), rng, max_size=20)
        if p.problem.lhs == p.problem.rhs:
            continue
        res = bidirectional_bfs(p.problem, max_expansions=3000)
        if res.proof is not None and len(res.proof.steps) >= 3:
            return p.problem, list(res.proof.steps)
    pytest.skip("no suitable problem")


@pytest.mark.parametrize("theory_name,seed", [("group", 1), ("group", 7), ("bool", 3), ("bool", 11)])
@pytest.mark.parametrize("ball_nodes", [1, 60, 400_000])   # small balls force the forward-mask branch
def test_child_labels_match_bfs(theory_name, seed, ball_nodes):
    problem, steps = _shortest(theory_name, seed)
    labels, _ = label_problem(problem, steps, ball_nodes=ball_nodes)
    pool = problem.pool(3)
    cap = size_cap(problem, 6)
    checked = 0
    for lab in labels:
        if lab.status != "ok":       # node cap hit: reported as unknown, never guessed
            assert lab.optimal == []
            continue
        assert any(lab.optimal), "a state with dist >= 1 always has an optimal child"
        for u, opt, cd in zip(lab.children, lab.optimal, lab.child_dist, strict=True):
            res = bidirectional_bfs(Problem(problem.theory, u, problem.rhs), max_expansions=20_000,
                                    max_size=cap, pool=pool)
            assert res.proof is not None
            du = len(res.proof.steps)
            assert abs(du - lab.dist) <= 1          # symmetric unit-cost graph
            assert opt == (du == lab.dist - 1)
            if cd is not None:
                assert cd == du
            checked += 1
    assert checked > 0
