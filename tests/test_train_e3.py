"""E3 training options: defaults reproduce the h-v1 objective; consistency penalty and parents are correct."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isomer.learn import data as D
from isomer.learn import train as T
from isomer.learn.model import DistanceModel, ModelConfig


def test_consistency_sum_hand_example():
    pred = torch.tensor([3.0, 2.5, 5.0, 1.0])          # parent 0; children 1, 2, 3
    groups = torch.tensor([[1, 2, 3, -1]])
    parents = torch.tensor([0])
    # |3-2.5|=0.5 -> 0 ; |3-5|=2 -> 1 ; |3-1|=2 -> 1
    assert T.consistency_sum(pred, groups, parents).item() == pytest.approx(2.0)


def _tiny_stateset(tmp_path):
    import json
    rows = []
    for pid, goal in (("p0", "(* a b)"), ("p1", "(& p q)")):
        rows += [
            {"problem_id": pid, "state": goal, "goal": goal, "dist": 0, "on_path": True, "parent_dist": None},
            {"problem_id": pid, "state": f"(* e {goal})" if pid == "p0" else f"(& {goal} T)", "goal": goal,
             "dist": 1, "on_path": True, "parent_dist": None},
            {"problem_id": pid, "state": "a" if pid == "p0" else "p", "goal": goal, "dist": 2, "on_path": False,
             "parent_dist": 1},
            {"problem_id": pid, "state": "b" if pid == "p0" else "q", "goal": goal, "dist": 1, "on_path": False,
             "parent_dist": 1},
        ]
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return D.load_states(path, 64)


def test_group_parents_are_on_path_one_step_further(tmp_path):
    ss = _tiny_stateset(tmp_path)
    m = D.collate(ss, np.arange(ss.n_problems))
    assert m.parents is not None and len(m.parents) == len(m.groups)
    for row, par in zip(m.groups, m.parents, strict=True):
        child0 = m.index[row[0]]
        assert ss.on_path[m.index[par]]
        assert ss.dist[m.index[par]] == ss.dist[child0] + 1


def test_defaults_match_original_objective(tmp_path):
    ss = _tiny_stateset(tmp_path)
    micros = [D.collate(ss, np.arange(ss.n_problems))]
    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2, ffn_hidden=64)
    grads = []
    for kwargs in ({}, {"lambda_huber": 1.0, "lambda_consistency": 0.0}):
        torch.manual_seed(0)
        model = DistanceModel(cfg)
        T.accumulate(model, micros, torch.device("cpu"), 1.0, 1.0, "optimal", **kwargs)
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.equal(grads[0], grads[1])
    # the h-v1 formula written out by hand: Huber/n_states + rank/n_groups
    torch.manual_seed(0)
    model = DistanceModel(cfg)
    m = micros[0]
    pred = model(torch.from_numpy(m.ids))
    loss = (T.huber_sum(pred, torch.from_numpy(m.dist)) / len(m.dist)
            + T.ranking_loss_sum(pred, torch.from_numpy(m.groups), torch.from_numpy(m.optimal), 1.0) / len(m.groups))
    loss.backward()
    ref = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    assert torch.allclose(grads[0], ref, atol=1e-7)
    torch.manual_seed(0)
    model = DistanceModel(cfg)
    stats: dict = {}
    T.accumulate(model, micros, torch.device("cpu"), 1.0, 1.0, "optimal", 0.0, 0.5, stats)
    assert stats["consistency"] >= 0.0
