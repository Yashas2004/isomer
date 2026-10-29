import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isomer.core.term import parse
from isomer.learn import data as D
from isomer.learn.infer import LearnedHeuristic
from isomer.learn.model import DistanceModel, ModelConfig
from isomer.learn.states import StatesConfig, build_states
from isomer.learn.train import (
    TrainConfig,
    TrainError,
    accumulate,
    lr_factor,
    ranking_loss_sum,
    train,
)

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_states(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("states")
    build_states(StatesConfig(data=str(tiny_data), out=str(out), val_fraction=0.3, workers=1), progress=False)
    return out


def tiny_cfg(states, out, **kw):
    base = {"states": str(states), "out": str(out), "epochs": 10, "batch_problems": 2, "lr": 3e-3, "warmup": 5,
            "d_model": 32, "layers": 2, "heads": 4, "device": "cpu", "micro_tokens": 2048, "log_every": 10**9}
    return TrainConfig(**(base | kw))


def test_ranking_loss_near_zero_when_right_and_large_when_reversed():
    groups = torch.tensor([[0, 1, 2]])
    optimal = torch.tensor([[True, False, False]])
    good = torch.tensor([-20.0, 0.0, 0.0])
    bad = torch.tensor([20.0, 0.0, 0.0])
    for target in ("optimal", "onpath"):
        assert ranking_loss_sum(good, groups, optimal, 1.0, target).item() < 1e-6
        assert ranking_loss_sum(bad, groups, optimal, 1.0, target).item() > 19.0


def test_soft_target_spreads_over_all_optimal_children():
    groups = torch.tensor([[0, 1, 2, -1]])
    optimal = torch.tensor([[True, True, False, False]])
    pred = torch.tensor([0.0, 0.0, 30.0])     # two optimal children tied, the bad one far away
    # optimal target: uniform over the two tied optimal children -> CE = log 2 (the minimum)
    assert ranking_loss_sum(pred, groups, optimal, 1.0, "optimal").item() == pytest.approx(np.log(2), abs=1e-6)
    # preferring the other optimal child is not penalized relative to the on-path one
    swapped = torch.tensor([5.0, -5.0, 30.0])
    on_first = torch.tensor([-5.0, 5.0, 30.0])
    assert ranking_loss_sum(swapped, groups, optimal, 1.0, "optimal").item() == pytest.approx(
        ranking_loss_sum(on_first, groups, optimal, 1.0, "optimal").item(), abs=1e-6)
    assert ranking_loss_sum(swapped, groups, optimal, 1.0, "onpath").item() > 9.0


def test_lr_schedule():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1) and lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0) and lr_factor(100, 10, 100) == pytest.approx(0.0)
    assert 0 < lr_factor(55, 10, 100) < 1


def test_micro_batching_does_not_change_the_gradient(tiny_states):
    ss = D.load_states(tiny_states / "train.jsonl", 192)
    problems = np.arange(min(6, ss.n_problems))

    def grads(micro_tokens):
        torch.manual_seed(0)
        model = DistanceModel(ModelConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=86))
        micros = [D.collate(ss, mb) for mb in D.micro_batches(ss, problems, micro_tokens)]
        losses = accumulate(model, micros, CPU, 1.0, 1.0, "optimal")
        return len(micros), losses, [p.grad.clone() for p in model.parameters()]
    n1, l1, g1 = grads(10**9)
    n2, l2, g2 = grads(1)
    assert n1 == 1 and n2 == len(problems)
    assert l1 == pytest.approx(l2, abs=1e-5)
    for a, b in zip(g1, g2, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_tiny_training_run_reduces_loss_and_writes_run(tiny_states, tmp_path):
    out = tmp_path / "run"
    res = train(tiny_cfg(tiny_states, out), progress=False)
    h = res["loss_history"]
    assert len(h) >= 40
    assert np.mean(h[-10:]) < np.mean(h[:10])
    for name in ("config.json", "metrics.jsonl", "best.pt", "last.pt"):
        assert (out / name).exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["param_count"] > 0 and cfg["args"]["rank_target"] == "optimal"
    assert {"symbol_bag", "size_diff"} <= set(cfg["val_baselines"])
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and all("val_top1_optimal" in r and "train_rank" in r for r in rows)
    assert res["best_top1_optimal"] == max(r["val_top1_optimal"] for r in rows)

    with pytest.raises(TrainError, match="overwrite"):
        train(tiny_cfg(tiny_states, out), progress=False)
    train(tiny_cfg(tiny_states, out, overwrite=True, max_steps=3), progress=False)
    rows = (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["step"] == 3   # stopped mid-epoch, evaluated, old log replaced


def test_learned_heuristic_matches_model_and_caches(tiny_states, tmp_path):
    out = tmp_path / "run"
    train(tiny_cfg(tiny_states, out, max_steps=2), progress=False)
    h = LearnedHeuristic(str(out / "best.pt"), device="cpu", batch_size=2)
    goal = parse("e")
    states = [parse(s) for s in ("(* e e)", "(inv e)", "(* a (inv a))", "e")]
    vals = h.h_batch(states, goal)
    assert len(vals) == 4 and h.misses == 4 and h.hits == 0
    again = h.h_batch(states[::-1], goal)
    assert again == vals[::-1] and h.hits == 4
    assert h(states[0], goal) == vals[0]
    from isomer.learn.tokenizer import encode
    direct = h.model(torch.tensor([encode(states[2], goal)])).item()
    assert vals[2] == pytest.approx(direct, abs=1e-5)


def test_cli_train_refuses_missing_cuda_or_reports_cleanly(tiny_states, tmp_path):
    from isomer.cli import main
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    assert main(["train", "--states", str(tiny_states), "--out", str(tmp_path / "x")]) == 2
