"""Meaning-preserving renamings of problems, for invariance tests (E3 Phase C).

A renaming is an injective map on symbols that preserves arity and is applied
to the THEORY'S AXIOMS AND SIGNATURE as well as to the problem. The rewrite
graph of the renamed problem is then isomorphic to the original one: every
term, step, pool value and size maps one-to-one, so every shortest distance is
unchanged. ``rename_proof`` maps a proof across, and the trusted kernel checks
it against the renamed theory, which certifies the equivalence per problem.

Renamings used in E3:

* ``skolem_perm``: permute the theory's skolems (e.g. a->b->c->a). Skolems carry
  no axioms.
* ``skolem_fresh``: replace skolems with other symbols the tokenizer knows from a
  different theory (e.g. group a,b,c -> bool's p,q,r, or ring's x,y,z, which
  never occur in training).
* ``dual``: bool (& <-> |, T <-> F) and lattice (meet <-> join). Both axiom sets
  are self-dual: the renamed axioms are the same set of equations, so the
  renamed problem is a problem of the SAME theory with the same distance.
* ``resign``: write a whole theory in another theory's symbols with matching
  arities (group (*, inv, e) -> (&, ~, T)). Same rewrite graph, unfamiliar
  pairing of tokens and dynamics (the model has no theory token).

Renaming changes the sort order of the pool, hence the order successors are
generated in. Distances are unchanged; search tie-breaking may differ.
"""

from __future__ import annotations

from ..core.proof import Problem, Proof
from ..core.rewrite import Equation, Signature, Step, Theory
from ..core.term import App, Term, Var

Mapping = dict[str, str]


def rename_term(t: Term, m: Mapping) -> Term:
    if isinstance(t, Var):
        return t
    return App(m.get(t.op, t.op), tuple(rename_term(a, m) for a in t.args))


def _check(theory: Theory, m: Mapping) -> None:
    sig = theory.signature
    arity = {**sig.ops, **{c: 0 for c in sig.constants}, **{s: 0 for s in sig.skolems}}
    src = set(arity)
    image = {m.get(s, s) for s in src}
    if len(image) != len(src):
        raise ValueError(f"renaming {m} is not injective on {sorted(src)}")
    for s, new in m.items():
        if s not in arity:
            raise ValueError(f"{s!r} is not a symbol of {theory.name}")
        if new in arity and new not in m and new != s:
            raise ValueError(f"{s!r} -> {new!r} collides with an unrenamed symbol")


def rename_theory(theory: Theory, m: Mapping, name: str | None = None) -> Theory:
    _check(theory, m)
    sig = theory.signature
    new_sig = Signature(ops={m.get(o, o): a for o, a in sig.ops.items()},
                        constants=tuple(m.get(c, c) for c in sig.constants),
                        skolems=tuple(m.get(s, s) for s in sig.skolems))
    eqs = tuple(Equation(e.name, rename_term(e.lhs, m), rename_term(e.rhs, m)) for e in theory.equations)
    return Theory(name or f"{theory.name}[renamed]", eqs, new_sig)


def rename_problem(problem: Problem, m: Mapping, theory: Theory | None = None) -> Problem:
    th = theory or rename_theory(problem.theory, m)
    return Problem(th, rename_term(problem.lhs, m), rename_term(problem.rhs, m), dict(problem.meta))


def rename_proof(proof: Proof, m: Mapping, problem: Problem) -> Proof:
    """The same steps (same rule names and positions) with renamed instantiations."""
    steps = tuple(Step(s.rule, s.pos, tuple((v, rename_term(t, m)) for v, t in s.inst)) for s in proof.steps)
    return Proof(problem, steps)


# ---- the renamings used in E3 -------------------------------------------------------------

SKOLEM_PERM: dict[str, Mapping] = {
    "group": {"a": "b", "b": "c", "c": "a"},
    "bool": {"p": "q", "q": "r", "r": "p"},
    "lattice": {"a": "b", "b": "c", "c": "a"},
    "ring": {"x": "y", "y": "z", "z": "x"},
}
SKOLEM_FRESH: dict[str, dict[str, Mapping]] = {
    # to bool's skolems (trained, but only ever with bool's operators)
    "seen_elsewhere": {"group": {"a": "p", "b": "q", "c": "r"}, "bool": {"p": "a", "q": "b", "r": "c"}},
    # to ring's skolems (never in any training state)
    "never_trained": {"group": {"a": "x", "b": "y", "c": "z"}, "bool": {"p": "x", "q": "y", "r": "z"}},
}
DUAL: dict[str, Mapping] = {
    "bool": {"&": "|", "|": "&", "T": "F", "F": "T"},
    "lattice": {"meet": "join", "join": "meet"},
}
RESIGN: dict[str, Mapping] = {
    # group's (*, inv, e) written with bool's (&, ~, T); skolems kept
    "group->bool_symbols": {"*": "&", "inv": "~", "e": "T"},
    # bool written with ring's (+, *, -, 0, 1) -- & -> *, | -> +, ~ -> -, T -> 1, F -> 0
    "bool->ring_symbols": {"&": "*", "|": "+", "~": "-", "T": "1", "F": "0"},
}


def same_equations(a: Theory, b: Theory) -> bool:
    """True if both theories have the same set of equations up to orientation and names."""
    def key(t: Theory) -> set[frozenset[str]]:
        return {frozenset((str(e.lhs), str(e.rhs))) for e in t.equations}
    return key(a) == key(b)
