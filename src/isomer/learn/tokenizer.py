"""Fixed tokenizer for (state, goal) pairs fed to a learned heuristic.

The vocabulary is the union of every theory's symbols (operators, constants,
skolems) plus special tokens. It is built from sorted collections only, so it
is identical in every process and on every machine: a saved model's
embedding rows always mean the same symbols.

A symbol shared by two theories gets one token (``*`` in group and ring;
``a b c`` in group and lattice). The encoding carries no theory token yet;
adding one later would be a vocabulary change and needs a version bump.

Terms are tokenized from their structure, not their string: an application
``(op a b)`` becomes ``( op a b )`` and a constant becomes itself. Unknown
symbols (and pattern variables, which never occur in states) map to [UNK].
"""

from __future__ import annotations

from ..core.term import App, Term, parse
from ..core.theories import THEORIES

SPECIALS: tuple[str, ...] = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "(", ")")


def build_vocab() -> tuple[str, ...]:
    """Specials first (ids 0-5), then all theory symbols in sorted order."""
    symbols: set[str] = set()
    for name in sorted(THEORIES):
        sig = THEORIES[name][0].signature
        symbols.update(sorted(sig.ops))
        symbols.update(sig.constants)
        symbols.update(sig.skolems)
    return SPECIALS + tuple(sorted(symbols - set(SPECIALS)))


VOCAB: tuple[str, ...] = build_vocab()
TOKEN_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID, CLS_ID, SEP_ID, UNK_ID, LPAREN_ID, RPAREN_ID = range(len(SPECIALS))


def term_tokens(t: Term) -> list[str]:
    """Structural tokens of ``t``; unknown symbols become [UNK]."""
    if not isinstance(t, App):
        return ["[UNK]"]
    op = t.op if t.op in TOKEN_ID and t.op not in SPECIALS else "[UNK]"
    if not t.args:
        return [op]
    out = ["(", op]
    for a in t.args:
        out.extend(term_tokens(a))
    out.append(")")
    return out


def _as_term(x: Term | str) -> Term:
    return parse(x) if isinstance(x, str) else x


def n_tokens(state: Term | str, goal: Term | str) -> int:
    """Length of ``encode(state, goal)`` without the length check."""
    return 3 + len(term_tokens(_as_term(state))) + len(term_tokens(_as_term(goal)))


def encode(state: Term | str, goal: Term | str, max_len: int = 256) -> list[int]:
    """``[CLS] state [SEP] goal [SEP]`` as ids. Raises ``ValueError`` if longer
    than ``max_len``: silently truncating a term would change its meaning."""
    toks = ["[CLS]", *term_tokens(_as_term(state)), "[SEP]", *term_tokens(_as_term(goal)), "[SEP]"]
    if len(toks) > max_len:
        raise ValueError(f"encoded length {len(toks)} > max_len {max_len} for state={state!s} goal={goal!s}; "
                         "raise max_len (the model's context) rather than truncating")
    return [TOKEN_ID[t] for t in toks]


def _render(tokens: list[str]) -> str:
    return " ".join(tokens).replace("( ", "(").replace(" )", ")")


def decode(ids: list[int]) -> tuple[str, str]:
    """Inverse of ``encode`` for debugging: the two terms in ``str(Term)``
    format ([UNK] shown literally). Padding is ignored."""
    toks = [VOCAB[i] for i in ids if i != PAD_ID]
    if len(toks) < 3 or toks[0] != "[CLS]" or toks.count("[SEP]") != 2 or toks[-1] != "[SEP]":
        raise ValueError(f"not an encoded (state, goal) pair: {toks}")
    sep = toks.index("[SEP]")
    return _render(toks[1:sep]), _render(toks[sep + 1:-1])
