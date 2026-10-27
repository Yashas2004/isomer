"""Equations, oriented rules, proof steps, and the rewrite engine.

An *equation* ``l = r`` from a theory gives up to two *oriented rules*:
``name`` (l -> r) and ``name~`` (r -> l).

Some rules introduce variables on the right that do not occur on the left,
e.g. ``e -> (* (inv ?y) ?y)``. Such a rule needs an explicit instantiation
for ``?y`` chosen by the prover; the choice is recorded in the proof step so
that the kernel can replay it. This is what makes equational proofs hard:
the action space is infinite, and the search must invent the right term.

This module is *untrusted*: it is used by search and data generation. The
trusted checker in :mod:`isomer.core.kernel` re-implements replay on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .term import Position, Term, match, parse, positions, replace, size, substitute, subterm, variables


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
class Theory:
    name: str
    equations: tuple[Equation, ...]

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
    rule *consumed*; their values come from matching the forward lhs."""
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


def instantiation_pool(t: Term, limit: int = 8) -> list[Term]:
    """Heuristic pool for fresh variables: the ``limit`` smallest distinct
    subterms of the current term. Complete search would need *all* terms;
    this is the classic subterm heuristic, and choosing well from a larger
    pool is one of the things the neural policy is meant to learn."""
    seen: set[Term] = set()
    pool: list[Term] = []
    for p in positions(t):
        s = subterm(t, p)
        if s not in seen:
            seen.add(s)
            pool.append(s)
    pool.sort(key=lambda u: (size(u), str(u)))
    return pool[:limit]


def candidate_instantiations(t: Term, fresh: tuple[str, ...], limit: int = 8,
                             pool: list[Term] | None = None) -> Iterator[tuple[tuple[str, Term], ...]]:
    if not fresh:
        yield ()
        return
    if pool is None:
        pool = instantiation_pool(t, limit)
    for u in pool:  # theories rarely need >1 fresh var; keep the product small
        yield tuple((v, u) for v in fresh)


def successors(t: Term, rules: Iterable[Rule], inst_limit: int = 8) -> Iterator[tuple[Step, Term]]:
    """Every one-step rewrite of ``t`` (deterministic order)."""
    rule_list = [(r, r.fresh_vars) for r in rules]
    pool: list[Term] | None = None
    for pos in positions(t):
        target = subterm(t, pos)
        for rule, fresh in rule_list:
            s = match(rule.lhs, target)
            if s is None:
                continue
            if fresh and pool is None:
                pool = instantiation_pool(t, inst_limit)
            for inst in candidate_instantiations(t, fresh, inst_limit, pool):
                s2 = dict(s)
                s2.update(inst)
                yield Step(rule.name, pos, inst), replace(t, pos, substitute(rule.rhs, s2))
