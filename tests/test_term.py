import pytest
from hypothesis import given, strategies as st

from isomer.core.term import (App, ParseError, Var, match, parse, positions, replace,
                              size, substitute, subterm, variables)

ATOMS = st.sampled_from(["e", "a", "b", "T", "F"])


def terms(max_leaves=12):
    return st.recursive(
        ATOMS.map(App),
        lambda kids: st.one_of(
            st.tuples(st.sampled_from(["inv", "~"]), kids).map(lambda x: App(x[0], (x[1],))),
            st.tuples(st.sampled_from(["*", "&", "|"]), kids, kids).map(lambda x: App(x[0], (x[1], x[2]))),
        ),
        max_leaves=max_leaves,
    )


@given(terms())
def test_print_parse_roundtrip(t):
    assert parse(str(t)) == t


@given(terms())
def test_replace_with_own_subterm_is_identity(t):
    for p in positions(t):
        assert replace(t, p, subterm(t, p)) == t


@given(terms())
def test_positions_count_equals_size(t):
    assert len(list(positions(t))) == size(t)


def test_parse_variables_and_constants():
    t = parse("(* (inv ?x) ?x)")
    assert t == App("*", (App("inv", (Var("x"),)), Var("x")))
    assert variables(t) == {"x"}


@pytest.mark.parametrize("bad", ["", "(", ")", "(*)", "(* a", "a b", "(?f a)", "?"])
def test_parse_errors(bad):
    with pytest.raises(ParseError):
        parse(bad)


def test_nonlinear_match():
    p = parse("(* ?x ?x)")
    assert match(p, parse("(* a a)")) == {"x": App("a")}
    assert match(p, parse("(* a b)")) is None


def test_substitute_roundtrip():
    p = parse("(* (inv ?x) ?y)")
    t = parse("(* (inv (* a b)) e)")
    s = match(p, t)
    assert s is not None and substitute(p, s) == t
