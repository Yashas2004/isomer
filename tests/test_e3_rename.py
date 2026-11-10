"""E3 renamings: each renamed problem is certified equivalent by the kernel and has the same distance."""

import random

import pytest

from isomer.core.generate import random_term, random_walk
from isomer.core.theories import THEORIES
from isomer.e3 import rename as RN
from isomer.search.bfs import bidirectional_bfs
from isomer.search.common import size_cap


def _problems(theory_name: str, n: int, seed: int = 0):
    theory, sig = THEORIES[theory_name]
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = random_walk(theory, random_term(sig, 3, rng), rng.choice([2, 3, 4]), rng, max_size=20)
        if p.problem.lhs != p.problem.rhs:
            res = bidirectional_bfs(p.problem, max_expansions=3000)
            if res.proof is not None:
                out.append(res.proof)
    return out


CASES = [
    ("group", RN.SKOLEM_PERM["group"]),
    ("bool", RN.SKOLEM_PERM["bool"]),
    ("group", RN.SKOLEM_FRESH["seen_elsewhere"]["group"]),
    ("group", RN.SKOLEM_FRESH["never_trained"]["group"]),
    ("bool", RN.SKOLEM_FRESH["never_trained"]["bool"]),
    ("bool", RN.DUAL["bool"]),
    ("lattice", RN.DUAL["lattice"]),
    ("group", RN.RESIGN["group->bool_symbols"]),
    ("bool", RN.RESIGN["bool->ring_symbols"]),
]


@pytest.mark.parametrize("theory_name,m", CASES)
def test_renamed_proof_is_kernel_checked_and_distance_unchanged(theory_name, m):
    theory, _ = THEORIES[theory_name]
    new_theory = RN.rename_theory(theory, m)
    for proof in _problems(theory_name, 6):
        p2 = RN.rename_problem(proof.problem, m, new_theory)
        v = RN.rename_proof(proof, m, p2).verify()
        assert v, v.reason
        res = bidirectional_bfs(p2, max_expansions=3000, max_size=size_cap(p2, 6))
        assert res.proof is not None and len(res.proof.steps) == len(proof.steps)
        assert len(p2.pool(3)) == len(proof.problem.pool(3))


@pytest.mark.parametrize("theory_name", ["bool", "lattice"])
def test_dual_theories_are_self_dual(theory_name):
    theory, _ = THEORIES[theory_name]
    assert RN.same_equations(RN.rename_theory(theory, RN.DUAL[theory_name]), theory)


def test_invalid_renamings_rejected():
    group, _ = THEORIES["group"]
    with pytest.raises(ValueError):
        RN.rename_theory(group, {"a": "b"})         # collides with unrenamed b
    with pytest.raises(ValueError):
        RN.rename_theory(group, {"zz": "a"})        # not a symbol of the theory
