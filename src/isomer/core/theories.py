"""Equational theories used by IsoBench.

Each theory has a signature (operator -> arity), the skolem constants that
problems may mention, and its axioms. Start with two; ring and lattice
theories are added in Phase 1b.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rewrite import Equation, Theory


@dataclass(frozen=True)
class Signature:
    ops: dict[str, int]          # non-constant operators and their arities
    constants: tuple[str, ...]   # theory constants (e.g. the identity e)
    skolems: tuple[str, ...]     # free constants allowed in problems


def _theory(name: str, axioms: dict[str, str]) -> Theory:
    return Theory(name, tuple(Equation.parse(n, s) for n, s in axioms.items()))


# Groups from the classic *left* axioms. Right identity and right inverse are
# theorems, not axioms -- deriving them takes real search (a famous small
# benchmark for equational provers).
GROUP = _theory("group", {
    "assoc": "(* (* ?x ?y) ?z) = (* ?x (* ?y ?z))",
    "id_l":  "(* e ?x) = ?x",
    "inv_l": "(* (inv ?x) ?x) = e",
})
GROUP_SIG = Signature(ops={"*": 2, "inv": 1}, constants=("e",), skolems=("a", "b", "c"))


BOOL = _theory("bool", {
    "and_comm":   "(& ?x ?y) = (& ?y ?x)",
    "or_comm":    "(| ?x ?y) = (| ?y ?x)",
    "and_assoc":  "(& (& ?x ?y) ?z) = (& ?x (& ?y ?z))",
    "or_assoc":   "(| (| ?x ?y) ?z) = (| ?x (| ?y ?z))",
    "and_dist":   "(& ?x (| ?y ?z)) = (| (& ?x ?y) (& ?x ?z))",
    "or_dist":    "(| ?x (& ?y ?z)) = (& (| ?x ?y) (| ?x ?z))",
    "and_id":     "(& ?x T) = ?x",
    "or_id":      "(| ?x F) = ?x",
    "and_compl":  "(& ?x (~ ?x)) = F",
    "or_compl":   "(| ?x (~ ?x)) = T",
})
BOOL_SIG = Signature(ops={"&": 2, "|": 2, "~": 1}, constants=("T", "F"), skolems=("p", "q", "r"))


THEORIES: dict[str, tuple[Theory, Signature]] = {
    "group": (GROUP, GROUP_SIG),
    "bool": (BOOL, BOOL_SIG),
}
