import pickle

import pytest

from isomer.core.generate import sample_problems
from isomer.core.term import parse
from isomer.core.theories import GROUP, GROUP_SIG, LATTICE, LATTICE_SIG
from isomer.search.common import SEARCHERS, Budget, get_searcher
from isomer.search.heuristics import size_diff, symbol_bag, zero

BUDGET = Budget(max_expansions=300)
NAMES = sorted(SEARCHERS)
PROBLEMS = (
    [p.problem for p in sample_problems(GROUP, GROUP_SIG, n=20, k=3, seed=7, term_depth=3)]
    + [p.problem for p in sample_problems(LATTICE, LATTICE_SIG, n=10, k=2, seed=7, term_depth=3)]
)


@pytest.fixture(scope="module")
def results():
    """Every searcher on every problem, once: (name, index) -> SearchResult."""
    return {(name, i): SEARCHERS[name](p, BUDGET) for name in NAMES for i, p in enumerate(PROBLEMS)}


def test_registry_names_and_errors():
    assert set(NAMES) == {
        "bfs-fwd", "bfs-bidir", "ucs",
        "bestfirst-size_diff-w1", "bestfirst-size_diff-w3",
        "bestfirst-symbol_bag-w1", "bestfirst-symbol_bag-w3",
    }
    assert get_searcher("ucs") is SEARCHERS["ucs"]
    with pytest.raises(KeyError, match="bfs-fwd"):
        get_searcher("no-such-searcher")


@pytest.mark.parametrize("name", NAMES)
def test_searchers_are_picklable(name):
    # Worker processes on Windows receive searchers by pickling.
    assert pickle.loads(pickle.dumps(SEARCHERS[name])) is not None


def test_heuristics():
    a, t = parse("a"), parse("(* a e)")
    assert zero(t, a) == 0.0
    assert size_diff(t, a) == 2.0 and size_diff(a, t) == 2.0
    assert symbol_bag(t, a) == 2.0
    # Same size, different symbols: only symbol_bag notices.
    u, v = parse("(* a b)"), parse("(* a a)")
    assert size_diff(u, v) == 0.0 and symbol_bag(u, v) == 2.0
    assert symbol_bag(t, t) == 0.0


@pytest.mark.parametrize("name", NAMES)
def test_proofs_verify(results, name):
    solved = 0
    for i, p in enumerate(PROBLEMS):
        r = results[name, i]
        if r.solved:
            solved += 1
            assert r.proof.problem == p
            v = r.proof.verify()
            assert v, v.reason
    assert solved >= 1


def test_bidir_solves_everything(results):
    # Guards against the other tests passing vacuously.
    assert all(results["bfs-bidir", i].solved for i in range(len(PROBLEMS)))


def test_optimal_searchers_agree(results):
    for i in range(len(PROBLEMS)):
        fwd = results["bfs-fwd", i]
        for other in ("ucs", "bfs-bidir"):
            r = results[other, i]
            if fwd.solved and r.solved:
                assert len(r.proof.steps) == len(fwd.proof.steps), (other, i)


@pytest.mark.parametrize("name", NAMES)
def test_budget_and_stats(results, name):
    for i in range(len(PROBLEMS)):
        r = results[name, i]
        assert r.expansions <= BUDGET.max_expansions
        assert isinstance(r.stats["nodes_stored"], int) and r.stats["nodes_stored"] >= 1
        assert r.stats["stop"] == "solved" if r.solved else r.stats["stop"] != "solved"
    tiny = Budget(max_expansions=5)
    for p in PROBLEMS:
        r = SEARCHERS[name](p, tiny)
        assert r.expansions <= 5
        assert "nodes_stored" in r.stats


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(results, name):
    for i, p in enumerate(PROBLEMS):
        a, b = results[name, i], SEARCHERS[name](p, BUDGET)
        assert a.expansions == b.expansions
        assert (a.proof.steps if a.proof else None) == (b.proof.steps if b.proof else None)


@pytest.mark.parametrize("name", NAMES)
def test_prefix_property(results, name):
    """Solved at budget B with e expansions => identical proof at any b >= e,
    and unsolved at any b < e. The benchmark runner relies on this to read
    off solve rates at every smaller budget from one run."""
    for i, p in enumerate(PROBLEMS):
        r = results[name, i]
        if not r.solved:
            continue
        e = r.expansions
        for b in (e, e + 5):
            rb = SEARCHERS[name](p, Budget(max_expansions=b))
            assert rb.solved and rb.proof.steps == r.proof.steps and rb.expansions == e, (i, b)
        if e > 0:
            assert not SEARCHERS[name](p, Budget(max_expansions=e - 1)).solved, i


@pytest.mark.parametrize("name", NAMES)
def test_node_cap(results, name):
    """A node-capped run is the uncapped run cut off at the first expansion
    that starts with >= max_nodes stored, so it is reproducible and its lower
    bound is as valid as a budget stop's."""
    cap = 50
    stops = 0
    for i, p in enumerate(PROBLEMS):
        c = SEARCHERS[name](p, Budget(max_expansions=BUDGET.max_expansions, max_nodes=cap))
        if c.stats["stop"] != "memory":
            assert c.solved == results[name, i].solved and c.expansions == results[name, i].expansions
            continue
        stops += 1
        e = c.expansions
        cut = SEARCHERS[name](p, Budget(max_expansions=e))
        assert (cut.expansions, cut.stats["nodes_stored"], cut.lower_bound, cut.solved) == \
               (e, c.stats["nodes_stored"], c.lower_bound, False)
        assert c.stats["nodes_stored"] >= cap
        if e > 0:  # the cap fired at the first check that saw >= cap nodes
            assert SEARCHERS[name](p, Budget(max_expansions=e - 1)).stats["nodes_stored"] < cap
        full = results[name, i]
        if name.startswith("bfs") and full.solved:
            assert c.lower_bound <= len(full.proof.steps)
    assert stops > 0


@pytest.mark.parametrize("name", NAMES)
def test_huge_node_cap_changes_nothing(results, name):
    """The prefix property and every result are untouched by a cap that never binds."""
    for i, p in enumerate(PROBLEMS):
        a, b = results[name, i], SEARCHERS[name](p, Budget(max_expansions=BUDGET.max_expansions, max_nodes=10**9))
        assert (a.expansions, a.stats, a.lower_bound) == (b.expansions, b.stats, b.lower_bound)
        assert (a.proof.steps if a.proof else None) == (b.proof.steps if b.proof else None)
