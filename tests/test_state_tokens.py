"""stats["max_state_tokens"]: the longest encoded (state, goal) pair a search
stored. Needed by E2's length-extrapolation analysis; must equal the
tokenizer's count and must not change search behaviour."""

import json
import random

import pytest

from isomer.core.generate import random_term
from isomer.core.proof import Problem
from isomer.core.theories import THEORIES
from isomer.learn.tokenizer import n_tokens
from isomer.search.common import SEARCHERS, Budget, max_state_tokens, token_count


@pytest.mark.parametrize("theory", sorted(THEORIES))
def test_token_count_matches_tokenizer(theory):
    _, sig = THEORIES[theory]
    rng = random.Random(0)
    for _ in range(200):
        s, g = random_term(sig, 4, rng), random_term(sig, 3, rng)
        assert max_state_tokens([s], g) == n_tokens(s, g)
        assert token_count(s, {}) == n_tokens(s, g) - n_tokens(g, g) + token_count(g, {})


def problems(data, n=8):
    recs = sorted((json.loads(line) for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()),
                  key=lambda r: r["id"])[:n]
    return [Problem.parse(THEORIES[r["theory"]][0], r["lhs"], r["rhs"]) for r in recs]


@pytest.mark.parametrize("method", sorted(SEARCHERS))
def test_every_searcher_reports_it_without_changing_the_search(tiny_data, method):
    search = SEARCHERS[method]
    for p in problems(tiny_data):
        off = search(p, Budget(max_expansions=100))
        on = search(p, Budget(max_expansions=100, track_state_tokens=True))
        assert "max_state_tokens" not in off.stats
        assert on.stats["max_state_tokens"] >= n_tokens(p.lhs, p.rhs)
        assert (on.proof.steps if on.proof else None, on.expansions, on.generated) == \
               (off.proof.steps if off.proof else None, off.expansions, off.generated)
