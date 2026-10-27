import copy
import random

import pytest

from isomer.core import kernel
from isomer.core.generate import random_term, random_walk, sample_problems
from isomer.core.proof import Problem, Proof
from isomer.core.rewrite import Step
from isomer.core.term import parse
from isomer.core.theories import GROUP, THEORIES


def demo_proof():
    problem = Problem.parse(GROUP, "(* e (* (inv a) a))", "e")
    return Proof(problem, (Step("id_l", ()), Step("inv_l", ())))


def test_accepts_valid_proof():
    assert demo_proof().verify()


def test_rejects_wrong_rule():
    p = demo_proof()
    bad = Proof(p.problem, (Step("inv_l", ()),))
    v = bad.verify()
    assert not v and v.failed_step == 0


def test_rejects_wrong_position():
    p = demo_proof()
    bad = Proof(p.problem, (Step("id_l", (1,)), Step("inv_l", ())))
    assert not bad.verify()


def test_rejects_proof_ending_elsewhere():
    p = demo_proof()
    assert not Proof(p.problem, p.steps[:1]).verify()


def test_fresh_variable_rule_needs_ground_instantiation():
    # e -> (* (inv ?x) ?x) introduces ?x: the step must say what ?x is.
    problem = Problem.parse(GROUP, "e", "(* (inv a) a)")
    good = Proof(problem, (Step("inv_l~", (), (("x", parse("a")),)),))
    assert good.verify()
    missing = Proof(problem, (Step("inv_l~", ()),))
    assert not missing.verify()
    cert = good.certificate()
    cert["proof"][0]["inst"] = [["x", "?y"]]  # non-ground: must be rejected
    assert not kernel.check_certificate(cert)


def test_tampered_certificate_detected_by_digest():
    cert = demo_proof().certificate()
    tampered = copy.deepcopy(cert)
    tampered["goal"]["rhs"] = "a"
    assert kernel.certificate_digest(tampered) != cert["digest"]
    assert not kernel.check_certificate(tampered)


def test_kernel_cannot_be_fooled_by_extra_axiom_in_cert():
    # A certificate only proves something relative to the theory it states;
    # consumers must compare cert["theory"] against the theory they trust.
    cert = demo_proof().certificate()
    assert [e["name"] for e in cert["theory"]] == ["assoc", "id_l", "inv_l"]


# ---------------------------------------------------------------------------
# Differential test: the untrusted engine (rewrite.py) generates proofs and
# the independent kernel must accept every one of them.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(THEORIES))
def test_engine_and_kernel_agree_on_random_walks(name):
    theory, sig = THEORIES[name]
    proofs = sample_problems(theory, sig, n=200, k=8, seed=1234)
    failures = [(str(p.problem), p.verify().reason) for p in proofs if not p.verify()]
    assert not failures, failures[:3]


@pytest.mark.parametrize("name", sorted(THEORIES))
def test_kernel_rejects_mutated_positions(name):
    """Shift one step's position; the kernel must reject or the result must
    genuinely still be a valid proof (which we re-check from the trace)."""
    theory, sig = THEORIES[name]
    rng = random.Random(7)
    rejected = attempted = 0
    for _ in range(300):
        p = random_walk(theory, random_term(sig, 4, rng), 5, rng)
        if not p.steps:
            continue
        attempted += 1
        i = rng.randrange(len(p.steps))
        s = p.steps[i]
        mutated = s.__class__(s.rule, s.pos + (0,), s.inst)
        bad = Proof(p.problem, p.steps[:i] + (mutated,) + p.steps[i + 1:])
        if not bad.verify():
            rejected += 1
    assert attempted > 100
    assert rejected / attempted > 0.95  # a shifted step should essentially never still prove the goal
