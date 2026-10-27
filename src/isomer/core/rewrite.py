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

from dataclasses import dataclass
from typing import Iterable, Iterator

from .term import App, Position, Term, match, parse, positions, replace, substitute, subterm, variables


@dataclass(frozen=True, slots=True)
class Equation:
    name: str
    lhs: Term
    rhs: Term

    @staticmethod
    def parse(name: str, src: str) -> "Equation":
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
    def from_json(d: dict) -> "Step":
        return Step(d["rule"], tuple(d["pos"]), tuple((v, parse(t)) for v, t in d["inst"]))


@dataclass(frozen=True)
class Theory:
    name: str
    equations: tuple[Equation, ...]

    def rules(self) -> dict[str, Rule]:
        """Both orientations of every equation. A direction is dropped only if
        its lhs is a bare variable (it would match everywhere)."""
        out: dict[str, Rule] = {}
        for eq in self.equations:
            for name, l, r in ((eq.name, eq.lhs, eq.rhs), (eq.name + "~", eq.rhs, eq.lhs)):
                if isinstance(l, App):
                    out[name] = Rule(name, l, r)
        return out


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


def candidate_instantiations(t: Term, fresh: tuple[str, ...], limit: int = 8) -> Iterator[tuple[tuple[str, Term], ...]]:
    """Heuristic pool for fresh variables: small distinct ground subterms of
    the current term. Complete search would need *all* terms; this is the
    classic subterm heuristic and is one of the things the neural policy
    is meant to improve on."""
    if not fresh:
        yield ()
        return
    pool: list[Term] = []
    seen: set[Term] = set()
    for p in positions(t):
        s = subterm(t, p)
        if s not in seen:
            seen.add(s)
            pool.append(s)
    pool.sort(key=lambda u: (len(str(u)), str(u)))
    pool = pool[:limit]
    if len(fresh) == 1:
        for u in pool:
            yield ((fresh[0], u),)
    else:  # keep the product small; theories rarely need >1 fresh var
        for u in pool:
            yield tuple((v, u) for v in fresh)


def successors(t: Term, rules: Iterable[Rule], inst_limit: int = 8) -> Iterator[tuple[Step, Term]]:
    """Every one-step rewrite of ``t`` (deterministic order)."""
    rule_list = list(rules)
    for pos in positions(t):
        target = subterm(t, pos)
        for rule in rule_list:
            s = match(rule.lhs, target)
            if s is None:
                continue
            for inst in candidate_instantiations(t, rule.fresh_vars, inst_limit):
                step = Step(rule.name, pos, inst)
                s2 = dict(s)
                s2.update(inst)
                yield step, replace(t, pos, substitute(rule.rhs, s2))
