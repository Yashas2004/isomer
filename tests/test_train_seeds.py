"""scripts/train_seeds.py: the seed runs must be h-v1 in everything but the seed.

Nothing is trained here; only the command construction, the drift check and
the reading of the trainer's metrics log are tested.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ts():
    spec = importlib.util.spec_from_file_location("train_seeds", ROOT / "scripts" / "train_seeds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_train(cmd, monkeypatch):
    """What the real CLI parser makes of the command, without training anything."""
    from isomer import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_train", lambda ns: seen.append(ns) or 0)
    assert cmd[1:4] == ["-m", "isomer", "train"]
    assert cli.main(cmd[3:]) == 0
    return seen[0]


def test_command_reproduces_every_published_argument(ts, monkeypatch):
    args = ts.base_args()
    ns = _parse_train(ts.train_command(args, seed=1, out="runs/h-v1-s1", python="python"), monkeypatch)
    assert ns.seed == 1 and ns.out == "runs/h-v1-s1" and ns.overwrite is False
    for key in ts.CLI_FLAGS:
        assert getattr(ns, key) == args[key], key
    # the E3 loss terms are pinned to the values that reproduce the h-v1 objective
    assert ns.lambda_huber == 1.0 and ns.lambda_consistency == 0.0
    assert ns.max_steps is None


def test_seeds_differ_only_in_seed_and_out(ts):
    args = ts.base_args()
    a = ts.train_command(args, 1, "runs/h-v1-s1", python="python")
    b = ts.train_command(args, 2, "runs/h-v1-s2", python="python")
    assert [(x, y) for x, y in zip(a, b, strict=True) if x != y] == [("runs/h-v1-s1", "runs/h-v1-s2"), ("1", "2")]
    assert ts.train_command(args, 1, "runs/h-v1-s1", python="python", overwrite=True) == [*a, "--overwrite"]


def test_every_published_argument_is_accounted_for(ts):
    pytest.importorskip("torch")
    from dataclasses import asdict

    from isomer.learn.train import TrainConfig

    ts.check_defaults(ts.base_args(), asdict(TrainConfig()))    # raises on drift


def test_drift_in_an_unsettable_default_is_refused(ts):
    args = ts.base_args()
    defaults = {"weight_decay": 0.01, "grad_clip": 1.0, "max_len": 192, "max_steps": None}
    ts.check_defaults(args, defaults)
    with pytest.raises(ts.ConfigDrift, match="weight_decay"):
        ts.check_defaults(args, dict(defaults, weight_decay=0.1))
    with pytest.raises(ts.ConfigDrift, match="unknown"):
        ts.check_defaults(dict(args, dropout=0.1), defaults)
    with pytest.raises(ts.ConfigDrift, match="max_steps"):
        ts.check_defaults(dict(args, max_steps=10), dict(defaults, max_steps=10))


def _metrics(run, rows):
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_summarize_run_reports_the_peak_over_epochs(ts, tmp_path):
    run = tmp_path / "h-v1-s1"
    _metrics(run, [
        {"epoch": 0, "max_memory_allocated": 3 * 2**30, "val_top1_optimal": 0.80, "train_seconds": 10.0,
         "val_seconds": 1.0},
        {"epoch": 1, "max_memory_allocated": 3 * 2**30 + 2**20, "val_top1_optimal": 0.95, "train_seconds": 10.0,
         "val_seconds": 1.5},
        {"epoch": 2, "max_memory_allocated": 2 * 2**30, "val_top1_optimal": 0.90, "train_seconds": 10.0,
         "val_seconds": 1.0},
    ])
    s = ts.summarize_run(run)
    assert s["epochs_done"] == 3
    assert s["max_memory_allocated_bytes"] == 3 * 2**30 + 2**20
    assert s["max_memory_allocated_mib"] == 3073
    assert (s["best_epoch"], s["best_val_top1_optimal"]) == (1, 0.95)
    assert s["train_seconds"] == 33.5


def test_summarize_run_without_gpu_or_metrics(ts, tmp_path):
    assert ts.summarize_run(tmp_path / "missing")["epochs_done"] == 0
    run = tmp_path / "cpu"
    _metrics(run, [{"epoch": 0, "max_memory_allocated": None, "val_top1_optimal": 0.5, "train_seconds": 1.0,
                    "val_seconds": 1.0}])
    assert ts.summarize_run(run)["max_memory_allocated_bytes"] is None


def test_is_complete_needs_all_epochs_and_a_checkpoint(ts, tmp_path):
    run = tmp_path / "run"
    row = {"epoch": 0, "max_memory_allocated": 1, "val_top1_optimal": 0.5, "train_seconds": 1.0, "val_seconds": 1.0}
    _metrics(run, [row, row])
    assert not ts.is_complete(run, 2)                 # no checkpoint yet
    (run / "best.pt").write_bytes(b"")
    assert ts.is_complete(run, 2)
    assert not ts.is_complete(run, 20)                # interrupted after two epochs


def test_dry_run_starts_nothing(ts, tmp_path, capsys):
    pytest.importorskip("torch")
    assert ts.main(["--dry-run", "--runs", str(tmp_path / "runs")]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and "--seed 1" in lines[0] and "--seed 2" in lines[1]
    assert not (tmp_path / "runs").exists()


def test_refuses_to_retrain_h_v1_itself(ts):
    assert ts.main(["--dry-run", "--seeds", "0"]) == 2
