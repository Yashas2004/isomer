"""Equational theories used by IsoBench.

Each theory has a signature (operator -> arity), the skolem constants that
problems may mention, and its axioms. Group and bool are training
theories; lattice and ring are held out for zero-shot transfer tests.
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


# Lattices: the classic 8 axioms (idempotence is derivable from absorption).
LATTICE = _theory("lattice", {
    "meet_comm":  "(meet ?x ?y) = (meet ?y ?x)",
    "join_comm":  "(join ?x ?y) = (join ?y ?x)",
    "meet_assoc": "(meet (meet ?x ?y) ?z) = (meet ?x (meet ?y ?z))",
    "join_assoc": "(join (join ?x ?y) ?z) = (join ?x (join ?y ?z))",
    "absorb_mj":  "(meet ?x (join ?x ?y)) = ?x",
    "absorb_jm":  "(join ?x (meet ?x ?y)) = ?x",
})
LATTICE_SIG = Signature(ops={"meet": 2, "join": 2}, constants=(), skolems=("a", "b", "c"))


# Commutative rings with unit.
RING = _theory("ring", {
    "add_comm":  "(+ ?x ?y) = (+ ?y ?x)",
    "add_assoc": "(+ (+ ?x ?y) ?z) = (+ ?x (+ ?y ?z))",
    "add_zero":  "(+ ?x 0) = ?x",
    "add_neg":   "(+ ?x (- ?x)) = 0",
    "mul_comm":  "(* ?x ?y) = (* ?y ?x)",
    "mul_assoc": "(* (* ?x ?y) ?z) = (* ?x (* ?y ?z))",
    "mul_one":   "(* ?x 1) = ?x",
    "distrib":   "(* ?x (+ ?y ?z)) = (+ (* ?x ?y) (* ?x ?z))",
})
RING_SIG = Signature(ops={"+": 2, "*": 2, "-": 1}, constants=("0", "1"), skolems=("x", "y", "z"))


THEORIES: dict[str, tuple[Theory, Signature]] = {
    "group": (GROUP, GROUP_SIG),
    "bool": (BOOL, BOOL_SIG),
    "lattice": (LATTICE, LATTICE_SIG),
    "ring": (RING, RING_SIG),
}
