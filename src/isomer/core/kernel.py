"""The trusted kernel: replays a proof and says yes or no.

Design rule (de Bruijn criterion): nothing the search or the neural model
does is trusted. A proof counts only if this file accepts it.

To keep the trusted base small and independent, the kernel deliberately
does NOT import :mod:`isomer.core.rewrite`. It re-derives rule orientations
from the raw equations and has its own matcher. Tests run the untrusted
engine and this kernel side by side (differential testing), so a bug has to
appear identically in two separate implementations to slip through.

Trusted base = this file + the ``Var``/``App`` dataclasses and ``parse`` in
``term.py``. Keep it under ~200 lines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .term import App, Term, Var, parse


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""
    failed_step: int | None = None

    def __bool__(self) -> bool:
        return self.ok


def _vars(t: Term, acc: set[str]) -> set[str]:
    if isinstance(t, Var):
        acc.add(t.name)
    else:
        for a in t.args:
            _vars(a, acc)
    return acc


def _match(p: Term, t: Term, s: dict[str, Term]) -> bool:
    if isinstance(p, Var):
        if p.name in s:
            return s[p.name] == t
        s[p.name] = t
        return True
    if not isinstance(t, App) or p.op != t.op or len(p.args) != len(t.args):
        return False
    return all(_match(pa, ta, s) for pa, ta in zip(p.args, t.args))


def _inst(t: Term, s: dict[str, Term]) -> Term:
    if isinstance(t, Var):
        return s[t.name]
    return App(t.op, tuple(_inst(a, s) for a in t.args))


def _at(t: Term, pos: tuple[int, ...]) -> Term | None:
    for i in pos:
        if not isinstance(t, App) or not 0 <= i < len(t.args):
            return None
        t = t.args[i]
    return t


def _put(t: Term, pos: tuple[int, ...], new: Term) -> Term:
    if not pos:
        return new
    assert isinstance(t, App)
    i = pos[0]
    return App(t.op, t.args[:i] + (_put(t.args[i], pos[1:], new),) + t.args[i + 1:])


def _oriented(equations: list[tuple[str, Term, Term]]) -> dict[str, tuple[Term, Term]]:
    rules: dict[str, tuple[Term, Term]] = {}
    for name, l, r in equations:
        rules[name] = (l, r)
        rules[name + "~"] = (r, l)
    return rules


def check(
    equations: list[tuple[str, Term, Term]],
    lhs: Term,
    rhs: Term,
    steps: list[tuple[str, tuple[int, ...], list[tuple[str, Term]]]],
) -> Verdict:
    """Check that ``steps`` rewrite ``lhs`` into exactly ``rhs`` using only
    ``equations`` (in either direction). Each step is (rule, position, inst)."""
    if _vars(lhs, set()) or _vars(rhs, set()):
        return Verdict(False, "goal must be ground")
    rules = _oriented(equations)
    cur = lhs
    for k, (name, pos, inst) in enumerate(steps):
        if name not in rules:
            return Verdict(False, f"unknown rule {name!r}", k)
        l, r = rules[name]
        target = _at(cur, tuple(pos))
        if target is None:
            return Verdict(False, f"invalid position {list(pos)}", k)
        s: dict[str, Term] = {}
        if not _match(l, target, s):
            return Verdict(False, f"rule {name!r} does not match at {list(pos)}", k)
        fresh = sorted(_vars(r, set()) - _vars(l, set()))
        if sorted(v for v, _ in inst) != fresh or len(inst) != len(fresh):
            return Verdict(False, f"instantiation must bind exactly {fresh}", k)
        for v, val in inst:
            if _vars(val, set()):
                return Verdict(False, f"instantiation of ?{v} is not ground", k)
            s[v] = val
        cur = _put(cur, tuple(pos), _inst(r, s))
    if cur != rhs:
        return Verdict(False, "proof ends at a different term", None)
    return Verdict(True)


# --------------------------------------------------------------------------
# Certificates: a self-contained JSON object anyone can re-check.
# --------------------------------------------------------------------------

def check_certificate(cert: dict) -> Verdict:
    try:
        eqs = [(e["name"], parse(e["lhs"]), parse(e["rhs"])) for e in cert["theory"]]
        lhs, rhs = parse(cert["goal"]["lhs"]), parse(cert["goal"]["rhs"])
        steps = [
            (s["rule"], tuple(int(i) for i in s["pos"]), [(v, parse(t)) for v, t in s["inst"]])
            for s in cert["proof"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return Verdict(False, f"malformed certificate: {exc}")
    return check(eqs, lhs, rhs, steps)


def certificate_digest(cert: dict) -> str:
    """SHA-256 over canonical JSON (excluding any existing digest field)."""
    body = {k: v for k, v in cert.items() if k != "digest"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
