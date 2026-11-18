import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from isomer.core.term import (
    App,
    ParseError,
    Var,
    match,
    parse,
    positions,
    replace,
    size,
    substitute,
    subterm,
    variables,
)

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


# A Hypothesis health check (FailedHealthCheck, most likely too_slow; the log line was cut off)
# tripped once while the machine was under load (2026-11-18), and the test passed when rerun idle.
# too_slow measures how fast Hypothesis generates terms, not whether printing and parsing
# round-trip, so it is suppressed here; the property tested is unchanged.
@settings(suppress_health_check=[HealthCheck.too_slow])
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
