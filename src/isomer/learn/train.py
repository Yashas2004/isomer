"""Supervised training of the distance model on ``isomer states`` output.

Loss per batch (a batch is all states of k problems):
* **Huber** (delta 1) between the prediction and the true distance, averaged
  over the batch's states;
* plus ``lambda_rank`` times a **listwise ranking loss**, averaged over the
  batch's sibling groups (``data.derive_groups``). It is the cross-entropy
  of softmax(-pred / tau) over a group against a target distribution:
  * ``optimal`` (default): uniform over every child whose true distance is
    the group's minimum. Equally short moves are never trained as wrong.
  * ``onpath``: one-hot on the stored proof's move.

Micro-batching (gradient accumulation) keeps a 64-problem batch within a
4 GB GPU. Both loss terms are *sums* divided by the whole batch's state and
group counts, so the gradient is the same as one big batch.

Validation, after every epoch and at ``--max-steps``: distance MAE (overall
and per true distance) and top-1 move metrics (``metrics.py``), next to
symbol_bag and size_diff on the same groups. ``best.pt`` is selected by
fractional ``top1_optimal``.

**Reproducibility.** python, numpy and torch are seeded and batch
composition is fixed by the seed. CUDA kernels (SDPA backward, atomics in
reductions) are *not* bit-deterministic, so GPU runs repeat only
statistically. CPU runs repeat exactly on one machine.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..bench.run import _git
from ..search.heuristics import size_diff, symbol_bag
from . import data as D
from .metrics import heuristic_scores, mae_by_dist, set_metrics
from .model import DistanceModel, ModelConfig, default_ffn_hidden, n_params, save_checkpoint


class TrainError(Exception):
    pass


@dataclass
class TrainConfig:
    states: str = "data/states-v2"
    out: str = "runs/heuristic"
    epochs: int = 20
    batch_problems: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    max_len: int = 192
    lambda_rank: float = 1.0
    tau: float = 1.0
    rank_target: str = "optimal"   # or "onpath"
    micro_tokens: int = 32_768
    seed: int = 0
    device: str = "cuda"
    max_steps: int | None = None
    overwrite: bool = False
    log_every: int = 50
    argv: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def ranking_loss_sum(pred: torch.Tensor, groups: torch.Tensor, optimal: torch.Tensor, tau: float,
                     target: str = "optimal") -> torch.Tensor:
    """Sum over groups of CE(softmax(-pred[group] / tau), target distribution).
    ``groups`` [G, K] row indices into ``pred`` (-1 padded, on-path child in column 0)."""
    if groups.numel() == 0:
        return pred.sum() * 0.0
    valid = groups >= 0
    logits = (-pred.float()[groups.clamp(min=0)] / tau).masked_fill(~valid, float("-inf"))
    logp = F.log_softmax(logits, dim=1).masked_fill(~valid, 0.0)
    if target == "optimal":
        t = optimal.float()
        t = t / t.sum(dim=1, keepdim=True)
    elif target == "onpath":
        t = torch.zeros_like(logp)
        t[:, 0] = 1.0
    else:
        raise ValueError(f"unknown rank target {target!r}")
    return -(t * logp).sum()


def huber_sum(pred: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred.float(), dist, reduction="sum", delta=1.0)


def accumulate(model: DistanceModel, micros: list[D.Micro], device: torch.device, lambda_rank: float, tau: float,
               target: str) -> tuple[float, float]:
    """Backward over one batch given as micro-batches; returns (huber, rank).
    Both are normalized by the WHOLE batch's state and group counts, so the
    gradient does not depend on how the batch was split."""
    n_states = sum(len(m.dist) for m in micros)
    n_groups = max(1, sum(len(m.groups) for m in micros))
    hub = rank = 0.0
    for m in micros:
        ids, dist, groups, optimal = _to(m, device)
        with _autocast(device):
            pred = model(ids)
        h = huber_sum(pred, dist) / n_states
        r = ranking_loss_sum(pred, groups, optimal, tau, target) / n_groups
        (h + lambda_rank * r).backward()
        hub += h.item()
        rank += r.item()
    return hub, rank


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to 0 at ``total``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _to(m: D.Micro, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (torch.from_numpy(m.ids).to(device), torch.from_numpy(m.dist).to(device),
            torch.from_numpy(m.groups).to(device), torch.from_numpy(m.optimal).to(device))


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(model: DistanceModel, ss: D.StateSet, device: torch.device, micro_tokens: int) -> np.ndarray:
    """Predictions for every state of ``ss``, in state order."""
    model.eval()
    out = np.empty(ss.n_states, dtype=np.float64)
    for mb in D.micro_batches(ss, np.arange(ss.n_problems), micro_tokens):
        m = D.collate(ss, mb)
        with _autocast(device):
            out[m.index] = model(torch.from_numpy(m.ids).to(device)).float().cpu().numpy()
    return out


def evaluate(model: DistanceModel, ss: D.StateSet, device: torch.device, micro_tokens: int) -> dict:
    pred = predict(model, ss, device, micro_tokens)
    return {"mae": mae_by_dist(pred, ss.dist), **set_metrics(ss, pred)}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model: DistanceModel, wd: float) -> list[dict]:
    """Weight decay on weight matrices only, not on norms, embeddings or biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "emb" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(cfg: TrainConfig, progress: bool = True) -> dict:
    out = Path(cfg.out)
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        raise TrainError(f"{out} already exists; pass --overwrite to replace it")
    if cfg.rank_target not in ("optimal", "onpath"):
        raise TrainError(f"--rank-target must be optimal or onpath, got {cfg.rank_target!r}")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainError("--device cuda but CUDA is not available (python -m isomer doctor); use --device cpu")
    device = torch.device(cfg.device)
    log = functools.partial(print, flush=True) if progress else (lambda *a, **k: None)
    _seed_all(cfg.seed)

    states_dir = Path(cfg.states)
    t0 = time.perf_counter()
    train_ss = D.load_states(states_dir / "train.jsonl", cfg.max_len)
    val_ss = D.load_states(states_dir / "val.jsonl", cfg.max_len, keep_text=True)
    load_seconds = time.perf_counter() - t0
    log(f"loaded {train_ss.n_states} train / {val_ss.n_states} val states "
        f"({train_ss.n_problems} / {val_ss.n_problems} problems) in {load_seconds:.0f}s")

    baselines = {name: set_metrics(val_ss, heuristic_scores(val_ss, h))
                 for name, h in (("symbol_bag", symbol_bag), ("size_diff", size_diff))}

    mcfg = ModelConfig(max_len=cfg.max_len, d_model=cfg.d_model, n_layers=cfg.layers, n_heads=cfg.heads,
                       ffn_hidden=default_ffn_hidden(cfg.d_model))
    model = DistanceModel(mcfg).to(device)
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(train_ss.n_problems / cfg.batch_problems)
    total = cfg.epochs * steps_per_epoch if cfg.max_steps is None else min(cfg.max_steps, cfg.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, functools.partial(lr_factor, warmup=cfg.warmup, total=total))

    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.jsonl", "best.pt", "last.pt"):
        (out / name).unlink(missing_ok=True)
    status = _git("status", "--porcelain")
    stats_path = states_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    config = {
        "args": asdict(cfg),
        "argv": list(cfg.argv),
        "model": asdict(mcfg),
        "param_count": n_params(model),
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic": "seeded; CUDA kernels are not bit-deterministic, GPU runs repeat only statistically",
        "data": {"states_stats_sha256": _sha256(stats_path),
                 "source_train_sha256": stats.get("source", {}).get("train_sha256"),
                 "train_states": train_ss.n_states, "val_states": val_ss.n_states,
                 "train_problems": train_ss.n_problems, "val_problems": val_ss.n_problems,
                 "train_groups": len(train_ss.groups), "val_groups": len(val_ss.groups),
                 "train_singleton_groups": train_ss.singleton_groups,
                 "val_singleton_groups": val_ss.singleton_groups,
                 "load_seconds": round(load_seconds, 1)},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total,
        "val_baselines": baselines,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    log(f"model {config['param_count']:,} params on {device}; {steps_per_epoch} steps/epoch, {total} steps; "
        f"baselines top1_optimal: " + ", ".join(f"{k} {v['top1_optimal']:.4f}" for k, v in baselines.items()))

    step, best, history = 0, -1.0, []
    for epoch in range(cfg.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        te = time.perf_counter()
        sums = {"loss": 0.0, "huber": 0.0, "rank": 0.0}
        n_steps = 0
        for batch in D.epoch_batches(train_ss.n_problems, cfg.batch_problems, cfg.seed, epoch):
            micros = [D.collate(train_ss, mb) for mb in D.micro_batches(train_ss, batch, cfg.micro_tokens)]
            opt.zero_grad(set_to_none=True)
            hub, rank = accumulate(model, micros, device, cfg.lambda_rank, cfg.tau, cfg.rank_target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            n_steps += 1
            sums["huber"] += hub
            sums["rank"] += rank
            sums["loss"] += hub + cfg.lambda_rank * rank
            history.append(hub + cfg.lambda_rank * rank)
            if progress and step % cfg.log_every == 0:
                log(f"  step {step}/{total}  loss {history[-1]:.4f}  huber {hub:.4f}  rank {rank:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {n_steps / (time.perf_counter() - te):.2f} steps/s")
            if step >= total:
                break
        train_seconds = time.perf_counter() - te
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        tv = time.perf_counter()
        val = evaluate(model, val_ss, device, cfg.micro_tokens)
        row = {"epoch": epoch, "step": step, "lr": sched.get_last_lr()[0],
               **{f"train_{k}": v / max(1, n_steps) for k, v in sums.items()},
               "val_mae": val["mae"], **{f"val_{k}": v for k, v in val.items() if k != "mae"},
               "train_seconds": round(train_seconds, 1), "val_seconds": round(time.perf_counter() - tv, 1),
               "steps_per_second": round(n_steps / max(train_seconds, 1e-9), 3),
               "max_memory_allocated": peak}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        extra = {"epoch": epoch, "step": step, "val_top1_optimal": val["top1_optimal"]}
        save_checkpoint(out / "last.pt", model, extra)
        if val["top1_optimal"] > best:
            best = val["top1_optimal"]
            save_checkpoint(out / "best.pt", model, extra)
        log(f"epoch {epoch}: train loss {row['train_loss']:.4f}  val MAE {val['mae']['all']:.3f}  "
            f"top1_optimal {val['top1_optimal']:.4f}  (strict {val['top1_optimal_strict']:.4f}, "
            f"onpath {val['top1_onpath']:.4f})  peak {None if peak is None else round(peak / 2**20)} MiB")
        if step >= total:
            break
    return {"config": config, "best_top1_optimal": best, "steps": step, "loss_history": history}


def main_train(cfg: TrainConfig) -> int:
    try:
        train(cfg)
    except TrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
