"""Problems, proofs, and certificates (the bridge between search and kernel)."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import kernel
from .rewrite import Step, Theory, apply_step, problem_pool
from .term import Term, parse


@dataclass(frozen=True)
class Problem:
    theory: Theory
    lhs: Term
    rhs: Term
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @staticmethod
    def parse(theory: Theory, lhs: str, rhs: str, **meta) -> Problem:
        return Problem(theory, parse(lhs), parse(rhs), dict(meta))

    def __str__(self) -> str:
        return f"[{self.theory.name}] {self.lhs} = {self.rhs}"

    def pool(self, max_size: int = 3) -> tuple[Term, ...]:
        """This problem's instantiation pool (ADR-003): compute once per search."""
        sig = self.theory.signature
        return problem_pool(sig.constants, sig.skolems, self.lhs, self.rhs, max_size)


@dataclass(frozen=True)
class Proof:
    problem: Problem
    steps: tuple[Step, ...]

    def trace(self) -> list[Term]:
        """Intermediate terms (untrusted replay, for display/debugging)."""
        rules = self.problem.theory.rules()
        terms = [self.problem.lhs]
        for st in self.steps:
            nxt = apply_step(rules[st.rule], terms[-1], st)
            if nxt is None:
                raise ValueError(f"step {st} does not apply")
            terms.append(nxt)
        return terms

    def certificate(self) -> dict:
        cert = {
            "format": "isomer-cert/1",
            "theory": [
                {"name": e.name, "lhs": str(e.lhs), "rhs": str(e.rhs)}
                for e in self.problem.theory.equations
            ],
            "goal": {"lhs": str(self.problem.lhs), "rhs": str(self.problem.rhs)},
            "proof": [s.to_json() for s in self.steps],
        }
        cert["digest"] = kernel.certificate_digest(cert)
        return cert

    def verify(self) -> kernel.Verdict:
        """Round-trip through JSON so the kernel sees exactly what a third
        party would see."""
        return kernel.check_certificate(self.certificate())
