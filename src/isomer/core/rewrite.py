"""Equations, oriented rules, proof steps, and the rewrite engine.

An *equation* ``l = r`` from a theory gives up to two *oriented rules*:
``name`` (l -> r) and ``name~`` (r -> l).

Some rules introduce variables on the right that do not occur on the left,
e.g. ``e -> (* (inv ?y) ?y)``. Such a rule needs an explicit instantiation
for ``?y`` chosen by the prover; the choice is recorded in the proof step so
that the kernel can replay it. This is what makes equational proofs hard:
the action space is infinite, and the search must invent the right term.

**Action space (ADR-003).** Each problem has a finite *pool* of values
(``problem_pool``): theory constants, the skolems it mentions, and the small subterms of
its two sides. A step is legal iff every value bound to a *one-sided*
variable is in the pool. One-sided means *introduced* (in the rule's rhs only)
or *consumed* (in its lhs only). Restricting both directions makes the action
space symmetric: every legal step has a legal inverse under the same pool, so
bidirectional search computes exact shortest distances.

This module is *untrusted*: it is used by search and data generation. The
trusted checker in :mod:`isomer.core.kernel` re-implements replay on its own.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from .term import App, Position, Term, match, parse, positions, replace, size, substitute, subterm, variables


@dataclass(frozen=True, slots=True)
class Equation:
    name: str
    lhs: Term
    rhs: Term

    @staticmethod
    def parse(name: str, src: str) -> Equation:
        left, sep, right = src.partition("=")
        if not sep:
            raise ValueError(f"equation {name!r} has no '='")
        return Equation(name, parse(left), parse(right))

    def __str__(self) -> str:
        return f"{self.name}: {self.lhs} = {self.rhs}"


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    lhs: Term
    rhs: Term

    @property
    def fresh_vars(self) -> tuple[str, ...]:
        """Variables of the rhs that the lhs does not bind (sorted)."""
        return tuple(sorted(variables(self.rhs) - variables(self.lhs)))

    def __str__(self) -> str:
        return f"{self.name}: {self.lhs} -> {self.rhs}"


@dataclass(frozen=True, slots=True)
class Step:
    """One rewrite: apply ``rule`` at ``pos``; ``inst`` binds the rule's
    fresh variables (sorted by name) for rules that need it."""
    rule: str
    pos: Position
    inst: tuple[tuple[str, Term], ...] = ()

    def to_json(self) -> dict:
        return {
            "rule": self.rule,
            "pos": list(self.pos),
            "inst": [[v, str(t)] for v, t in self.inst],
        }

    @staticmethod
    def from_json(d: dict) -> Step:
        return Step(d["rule"], tuple(d["pos"]), tuple((v, parse(t)) for v, t in d["inst"]))


@dataclass(frozen=True)
class Signature:
    ops: dict[str, int] = field(hash=False)  # non-constant operators and their arities
    constants: tuple[str, ...]   # theory constants (e.g. the identity e)
    skolems: tuple[str, ...]     # free constants allowed in problems


@dataclass(frozen=True)
class Theory:
    """Axioms plus the signature problems are written in. The signature lives
    here (not in a name-keyed registry) so that a ``Problem`` alone determines
    its action space: the pool needs the constants and skolems (ADR-003)."""
    name: str
    equations: tuple[Equation, ...]
    signature: Signature

    def rules(self) -> dict[str, Rule]:
        """Both orientations of every equation, always. A rule whose lhs is a
        bare variable (``?x -> (* e ?x)``) matches at every position; that is
        a big branching factor but it is required for completeness and so that
        every step has an inverse (bidirectional search depends on this)."""
        out: dict[str, Rule] = {}
        for eq in self.equations:
            out[eq.name] = Rule(eq.name, eq.lhs, eq.rhs)
            out[inverse_name(eq.name)] = Rule(inverse_name(eq.name), eq.rhs, eq.lhs)
        return out


def inverse_name(rule: str) -> str:
    return rule[:-1] if rule.endswith("~") else rule + "~"


def invert_step(rule: Rule, before: Term, step: Step) -> Step:
    """Given that ``step`` (using ``rule``) rewrites ``before`` into some
    ``after``, return the step that rewrites ``after`` back into ``before``.
    The inverse rule's fresh variables are exactly the variables the forward
    rule *consumed*; their values come from matching the forward lhs.

    If the forward step is legal under a pool, so is the inverse: it introduces
    exactly the values the forward step consumed, and consumes exactly the
    values it introduced (ADR-003)."""
    s = match(rule.lhs, subterm(before, step.pos))
    if s is None:
        raise ValueError(f"{step} does not apply")
    consumed = sorted(variables(rule.lhs) - variables(rule.rhs))
    return Step(inverse_name(rule.name), step.pos, tuple((v, s[v]) for v in consumed))


def apply_step(rule: Rule, t: Term, step: Step) -> Term | None:
    """Apply ``rule`` to ``t`` as described by ``step``; ``None`` if it does not apply."""
    try:
        target = subterm(t, step.pos)
    except IndexError:
        return None
    s = match(rule.lhs, target)
    if s is None:
        return None
    if tuple(v for v, _ in step.inst) != rule.fresh_vars:
        return None
    for v, val in step.inst:
        s[v] = val
    return replace(t, step.pos, substitute(rule.rhs, s))


def problem_pool(theory_constants: Iterable[str], skolems: Iterable[str], lhs: Term, rhs: Term,
                 max_size: int = 3) -> tuple[Term, ...]:
    """The values one-sided variables may take in this problem: the theory
    constants, the skolems that *occur* in lhs or rhs, and every distinct
    subterm of lhs or rhs with size <= ``max_size``. Sorted by (size, str), so
    it is deterministic. It depends only on the problem, never on the current
    term, which is what makes the action space symmetric.

    Leaving out unmentioned skolems is lossless (ADR-003): skolems carry no
    axioms, so any proof that uses an unmentioned one can rename it to a
    mentioned atom, giving a proof of the same length."""
    subterms = {subterm(side, p) for side in (lhs, rhs) for p in positions(side)}
    pool: set[Term] = {App(c) for c in theory_constants}
    pool |= {App(s) for s in skolems if App(s) in subterms}
    pool |= {s for s in subterms if size(s) <= max_size}
    return tuple(sorted(pool, key=lambda u: (size(u), str(u))))


def successors(t: Term, rules: Iterable[Rule], pool: Sequence[Term]) -> Iterator[tuple[Step, Term]]:
    """Every legal one-step rewrite of ``t`` (deterministic order).

    Introduced variables range over the whole pool (the full product when a
    rule introduces several). Steps whose consumed variables are bound
    outside the pool are skipped: they would have no legal inverse."""
    allowed = frozenset(pool)
    rule_list = [(r, r.fresh_vars, tuple(sorted(variables(r.lhs) - variables(r.rhs)))) for r in rules]
    for pos in positions(t):
        target = subterm(t, pos)
        for rule, fresh, consumed in rule_list:
            s = match(rule.lhs, target)
            if s is None or any(s[v] not in allowed for v in consumed):
                continue
            for values in itertools.product(pool, repeat=len(fresh)):
                inst = tuple(zip(fresh, values))
                s2 = dict(s)
                s2.update(inst)
                yield Step(rule.name, pos, inst), replace(t, pos, substitute(rule.rhs, s2))
