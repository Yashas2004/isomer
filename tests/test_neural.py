"""Neural best-first search: batching, determinism, the prefix property, names, bench.

The CPU tests use a tiny randomly initialized model so CI covers everything.
The CUDA tests repeat determinism and prefix checks with the real checkpoint on
dev problems; they skip when CUDA, the checkpoint or the dev set is missing.
"""

import json
from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from isomer.bench.run import BenchConfig, BenchError, run
from isomer.core.proof import Problem
from isomer.core.theories import THEORIES
from isomer.learn.model import DistanceModel, ModelConfig, save_checkpoint
from isomer.search.best_first import best_first
from isomer.search.common import Budget, get_searcher
from isomer.search.heuristics import zero
from isomer.search.neural import _load, neural_best_first

REAL_CKPT = Path("runs/h-v1/best.pt")
DEV = Path("data/isobench-v2-dev")


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    model = DistanceModel(ModelConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=86))
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model)
    return str(path)


def problems(data, split, n=10):
    recs = sorted((json.loads(line) for line in (Path(data) / f"{split}.jsonl").read_text(encoding="utf-8")
                   .splitlines()), key=lambda r: r["id"])[:n]
    return [Problem.parse(THEORIES[r["theory"]][0], r["lhs"], r["rhs"]) for r in recs]


def sig(r):
    return (r.proof.steps if r.proof else None, r.expansions, r.generated, r.stats["stop"], r.stats["model_calls"])


def check_determinism_and_prefix(search, probs, big=200):
    solved = 0
    for p in probs:
        r = search(p, Budget(max_expansions=big))
        assert sig(search(p, Budget(max_expansions=big))) == sig(r), "same search twice differs"
        if r.solved:
            solved += 1
            e = r.expansions
            for b in (e, e + 5):
                assert sig(search(p, Budget(max_expansions=b))) == sig(r), f"prefix property broken at {b}"
            r_less = search(p, Budget(max_expansions=e - 1))
            assert not r_less.solved and r_less.stats["stop"] == "budget" and r_less.expansions == e - 1
        elif r.stats["stop"] == "budget":
            r_less = search(p, Budget(max_expansions=big - 5))
            assert not r_less.solved and r_less.expansions == big - 5
    return solved


# ---------------------------------------------------------------- CPU, tiny model

def test_cpu_determinism_and_prefix_property(tiny_ckpt, tiny_data):
    search = partial(neural_best_first, ckpt=tiny_ckpt, w=1.0, device="cpu")
    probs = problems(tiny_data, "train") + problems(tiny_data, "test_theory", 3)
    assert check_determinism_and_prefix(search, probs) > 0
    greedy = partial(neural_best_first, ckpt=tiny_ckpt, greedy=True, device="cpu")
    check_determinism_and_prefix(greedy, probs[:5])


def test_one_batched_call_per_expansion_with_only_new_successors(tiny_ckpt, tiny_data, monkeypatch):
    h = _load(tiny_ckpt, "cpu")
    batches = []
    real = h.scores

    def spy(states, goal):
        batches.append(list(states))
        return real(states, goal)
    monkeypatch.setattr(h, "scores", spy)
    for p in problems(tiny_data, "train", 5):
        batches.clear()
        r = neural_best_first(p, Budget(max_expansions=50), ckpt=tiny_ckpt, device="cpu")
        assert r.stats["model_calls"] == len(batches) <= 1 + r.expansions
        assert batches[0] == [p.lhs]
        flat = [t for b in batches for t in b]
        assert len(flat) == len(set(flat)), "a state was scored twice"
        assert all(b for b in batches)


def test_constant_scores_reproduce_uniform_cost_search(tiny_ckpt, tiny_data, monkeypatch):
    """With h == 0 the priority is g and ties go FIFO, exactly as best_first(h=zero):
    same expansions, same proof. Checks ordering and tie-breaking are unchanged."""
    h = _load(tiny_ckpt, "cpu")
    monkeypatch.setattr(h, "scores", lambda states, goal: [0.0] * len(states))
    for p in problems(tiny_data, "train") + problems(tiny_data, "test_theory", 3):
        a = neural_best_first(p, Budget(max_expansions=300), ckpt=tiny_ckpt, device="cpu")
        b = best_first(p, Budget(max_expansions=300), h=zero, w=1.0)
        assert (a.proof.steps if a.proof else None, a.expansions, a.generated) == \
               (b.proof.steps if b.proof else None, b.expansions, b.generated)


def test_method_names():
    s = get_searcher("neural-w3@some/dir/x.pt")
    assert s.func is neural_best_first and s.keywords == {"ckpt": "some/dir/x.pt", "w": 3.0, "greedy": False}
    assert get_searcher("neural-greedy@x.pt").keywords["greedy"] is True
    assert get_searcher("bfs-fwd") is not None
    for bad in ("neural-w2@x.pt", "neural-w1@", "neural-w1", "nope"):
        with pytest.raises(KeyError, match="neural-w1@PATH"):
            get_searcher(bad)


def test_bench_with_neural_method(tiny_ckpt, tiny_data, tmp_path):
    method = f"neural-greedy@{tiny_ckpt}"
    out = tmp_path / "run"
    code = run(BenchConfig(data=str(tiny_data), split="test_theory", methods=["bfs-fwd", method],
                           budgets=[10, 100], out=str(out), workers=1))
    assert code == 0
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    import hashlib
    assert cfg["checkpoints"] == {method: {"path": tiny_ckpt,
                                           "sha256": hashlib.sha256(Path(tiny_ckpt).read_bytes()).hexdigest()}}
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    neural = [r for r in rows if r["method"] == method]
    assert neural and all(r["model_calls"] >= 1 and r["status"] == "ok" for r in neural)
    assert all("model_calls" not in r for r in rows if r["method"] == "bfs-fwd")
    with pytest.raises(BenchError, match="not found"):
        run(BenchConfig(data=str(tiny_data), split="test_theory", methods=["neural-w1@missing.pt"],
                        budgets=[10], out=str(tmp_path / "x"), workers=1))


# ---------------------------------------------------------------- CUDA, real checkpoint, dev set

@pytest.mark.skipif(not torch.cuda.is_available() or not REAL_CKPT.is_file()
                    or not (DEV / "test_depth.jsonl").is_file(),
                    reason="needs CUDA, runs/h-v1/best.pt and data/isobench-v2-dev")
def test_cuda_determinism_and_prefix_property_on_dev():
    search = partial(neural_best_first, ckpt=str(REAL_CKPT), w=1.0, device="cuda")
    probs = problems(DEV, "test_depth", 10)
    assert len(probs) == 10
    check_determinism_and_prefix(search, probs, big=300)
