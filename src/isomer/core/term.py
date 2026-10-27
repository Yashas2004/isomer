"""Terms: the single data type everything in Isomer is built on.

Conventions (borrowed from egg/egglog so our e-graph baseline can share syntax):

* ``App(op, args)`` is a function application. Constants are 0-ary apps,
  e.g. ``e``, ``0``, ``x``. In a *problem*, ``x`` and ``y`` are fixed
  (skolem) constants, not variables.
* ``Var(name)`` is a *pattern* variable and only appears inside rules.
  Its surface syntax is ``?name``.

Surface syntax is S-expressions: ``(* (inv ?x) ?x)``. S-expressions are
unambiguous, trivial to parse, and are what the neural model tokenises,
so there is exactly one canonical string per term.

Positions are tuples of argument indices from the root: ``()`` is the root,
``(1, 0)`` is "first argument of the second argument".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Union

Position = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Var:
    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True, slots=True)
class App:
    op: str
    args: tuple["Term", ...] = ()
    # Cached structural hash: terms are hashed constantly during search.
    _hash: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.op, self.args)))

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        if not self.args:
            return self.op
        return "(" + " ".join([self.op, *map(str, self.args)]) + ")"


Term = Union[Var, App]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class ParseError(ValueError):
    pass


def _tokenize(src: str) -> list[str]:
    return src.replace("(", " ( ").replace(")", " ) ").split()


def parse(src: str) -> Term:
    """Parse one S-expression. ``?x`` is a pattern variable."""
    tokens = _tokenize(src)
    if not tokens:
        raise ParseError("empty input")
    term, rest = _parse(tokens, 0)
    if rest != len(tokens):
        raise ParseError(f"trailing tokens: {' '.join(tokens[rest:])}")
    return term


def _parse(tokens: list[str], i: int) -> tuple[Term, int]:
    if i >= len(tokens):
        raise ParseError("unexpected end of input")
    tok = tokens[i]
    if tok == ")":
        raise ParseError("unexpected ')'")
    if tok != "(":
        if tok.startswith("?"):
            if len(tok) == 1:
                raise ParseError("bare '?' is not a variable")
            return Var(tok[1:]), i + 1
        return App(tok), i + 1
    # tok == "("
    i += 1
    if i >= len(tokens) or tokens[i] in "()":
        raise ParseError("expected operator after '('")
    op = tokens[i]
    if op.startswith("?"):
        raise ParseError("pattern variables cannot be operators")
    i += 1
    args: list[Term] = []
    while True:
        if i >= len(tokens):
            raise ParseError("missing ')'")
        if tokens[i] == ")":
            break
        arg, i = _parse(tokens, i)
        args.append(arg)
    if not args:
        raise ParseError(f"'({op})' has no arguments; write constants without parens")
    return App(op, tuple(args)), i + 1


# --------------------------------------------------------------------------
# Structural utilities
# --------------------------------------------------------------------------

def size(t: Term) -> int:
    if isinstance(t, Var):
        return 1
    return 1 + sum(size(a) for a in t.args)


def depth(t: Term) -> int:
    if isinstance(t, Var) or not t.args:
        return 1
    return 1 + max(depth(a) for a in t.args)


def positions(t: Term, prefix: Position = ()) -> Iterator[Position]:
    """All positions in pre-order (root first)."""
    yield prefix
    if isinstance(t, App):
        for i, a in enumerate(t.args):
            yield from positions(a, prefix + (i,))


def subterm(t: Term, pos: Position) -> Term:
    for i in pos:
        if not isinstance(t, App) or i >= len(t.args):
            raise IndexError(f"invalid position {pos}")
        t = t.args[i]
    return t


def replace(t: Term, pos: Position, new: Term) -> Term:
    if not pos:
        return new
    if not isinstance(t, App) or pos[0] >= len(t.args):
        raise IndexError(f"invalid position {pos}")
    i = pos[0]
    args = t.args[:i] + (replace(t.args[i], pos[1:], new),) + t.args[i + 1:]
    return App(t.op, args)


def variables(t: Term) -> frozenset[str]:
    if isinstance(t, Var):
        return frozenset((t.name,))
    out: set[str] = set()
    for a in t.args:
        out |= variables(a)
    return frozenset(out)


def is_ground(t: Term) -> bool:
    return not variables(t)


# --------------------------------------------------------------------------
# Matching and substitution
# --------------------------------------------------------------------------

Subst = dict[str, Term]


def match(pattern: Term, t: Term, subst: Subst | None = None) -> Subst | None:
    """Syntactic one-way matching. Non-linear patterns (``?x`` twice) require
    both occurrences to bind to equal terms. Returns a new dict or ``None``."""
    s: Subst = dict(subst) if subst else {}
    stack = [(pattern, t)]
    while stack:
        p, u = stack.pop()
        if isinstance(p, Var):
            bound = s.get(p.name)
            if bound is None:
                s[p.name] = u
            elif bound != u:
                return None
        elif isinstance(u, App) and p.op == u.op and len(p.args) == len(u.args):
            stack.extend(zip(p.args, u.args))
        else:
            return None
    return s


def substitute(t: Term, s: Subst) -> Term:
    if isinstance(t, Var):
        if t.name not in s:
            raise KeyError(f"unbound pattern variable ?{t.name}")
        return s[t.name]
    if not t.args:
        return t
    return App(t.op, tuple(substitute(a, s) for a in t.args))
