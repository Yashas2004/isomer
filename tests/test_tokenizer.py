import os
import subprocess
import sys

import pytest

from isomer.core.term import parse
from isomer.learn.tokenizer import (
    SPECIALS,
    TOKEN_ID,
    UNK_ID,
    VOCAB,
    build_vocab,
    decode,
    encode,
    n_tokens,
)


def test_specials_and_vocab_contents():
    assert VOCAB[:6] == SPECIALS == ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "(", ")")
    for sym in ("*", "inv", "e", "a", "&", "|", "~", "T", "F", "p", "meet", "join", "+", "-", "0", "1", "x"):
        assert sym in TOKEN_ID
    assert len(set(VOCAB)) == len(VOCAB) and build_vocab() == VOCAB


def test_vocab_identical_across_processes():
    """Embedding rows must mean the same symbols in every process: the vocab
    may not depend on set or dict iteration order (PYTHONHASHSEED)."""
    code = "from isomer.learn.tokenizer import VOCAB; print('|'.join(VOCAB))"
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                check=True).stdout.strip())
    assert outs == {"|".join(VOCAB)}


@pytest.mark.parametrize(("state", "goal"), [
    ("(* e (* (inv a) a))", "e"),
    ("(| (& q (| r T)) (& (~ p) (| q T)))", "T"),
    ("(meet a (join a b))", "a"),
    ("(+ (- x) (* 0 1))", "(+ x y)"),
    ("e", "e"),
])
def test_round_trip(state, goal):
    ids = encode(state, goal)
    assert decode(ids) == (state, goal)
    assert decode(ids + [TOKEN_ID["[PAD]"]] * 5) == (state, goal)
    assert len(ids) == n_tokens(state, goal)
    assert encode(parse(state), parse(goal)) == ids


def test_round_trip_on_built_states(tiny_data, tmp_path):
    import json

    from isomer.learn.states import StatesConfig, build_states
    build_states(StatesConfig(data=str(tiny_data), out=str(tmp_path), workers=1), progress=False)
    for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ids = encode(r["state"], r["goal"])
        assert UNK_ID not in ids and decode(ids) == (r["state"], r["goal"])


def test_unknown_symbol_and_max_len():
    ids = encode("(zzz a)", "qqq")
    assert ids.count(UNK_ID) == 2
    assert decode(ids) == ("([UNK] a)", "[UNK]")
    # [CLS] ( * a b ) [SEP] e [SEP] = 9 tokens
    with pytest.raises(ValueError, match=r"encoded length 9 > max_len 8"):
        encode("(* a b)", "e", max_len=8)
    assert len(encode("(* a b)", "e", max_len=9)) == 9
