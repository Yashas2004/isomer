import pytest

torch = pytest.importorskip("torch")

from isomer.learn.model import (
    DistanceModel,
    ModelConfig,
    apply_rope,
    default_ffn_hidden,
    load_checkpoint,
    n_params,
    rope_tables,
    save_checkpoint,
    segment_ids,
)
from isomer.learn.tokenizer import PAD_ID, VOCAB, encode

TINY = ModelConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=default_ffn_hidden(32), max_len=64)


def batch(*pairs):
    seqs = [encode(s, g) for s, g in pairs]
    ids = torch.full((len(seqs), max(map(len, seqs))), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    return ids


PAIRS = [("(* e (* (inv a) a))", "e"), ("(* a b)", "(* b a)"), ("e", "e")]


def test_forward_shape_and_padding_invariance():
    torch.manual_seed(0)
    model = DistanceModel(TINY).eval()
    ids = batch(*PAIRS)
    out = model(ids)
    assert out.shape == (3,) and out.dtype == torch.float32
    padded = torch.cat([ids, torch.full((3, 9), PAD_ID)], dim=1)
    assert torch.allclose(model(padded), out, atol=1e-5)
    # each row alone (no padding at all) gives the same prediction as inside the batch
    for i, p in enumerate(PAIRS):
        assert torch.allclose(model(batch(p)), out[i:i + 1], atol=1e-5)


def test_segment_ids():
    ids = batch(("(* a b)", "e"))[0]   # [CLS] ( * a b ) [SEP] e [SEP]
    assert segment_ids(ids[None])[0].tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]


def test_rope_scores_depend_only_on_relative_offset():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    cos, sin = rope_tables(20, 8, 10_000.0, torch.device("cpu"))

    def score(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum().item()
    assert score(3, 1) == pytest.approx(score(12, 10), abs=1e-5)
    assert score(5, 5) == pytest.approx(score(17, 17), abs=1e-5)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-3)


def test_parameter_count_formula():
    d, L, f, v = 256, 6, 683, len(VOCAB)
    expected = v * d + 2 * d + L * (4 * d * d + 3 * d * f + 2 * d) + d + (d + 1)
    assert n_params(DistanceModel(ModelConfig())) == expected
    assert default_ffn_hidden(256) == 683


def test_checkpoint_round_trip_and_vocab_check(tmp_path):
    torch.manual_seed(0)
    model = DistanceModel(TINY).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, {"epoch": 3})
    loaded, extra = load_checkpoint(path)
    ids = batch(*PAIRS)
    assert torch.equal(loaded(ids), model(ids)) and extra == {"epoch": 3} and loaded.cfg == TINY

    ck = torch.load(path, weights_only=True)
    ck["vocab"] = [*ck["vocab"][:-1], "zzz"]
    torch.save(ck, tmp_path / "bad_vocab.pt")
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(tmp_path / "bad_vocab.pt")
    ck["format"] = "other/0"
    torch.save(ck, tmp_path / "bad_format.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path / "bad_format.pt")


def test_too_long_input_rejected():
    model = DistanceModel(TINY)
    with pytest.raises(ValueError, match="max_len"):
        model(torch.ones(1, TINY.max_len + 1, dtype=torch.long))
