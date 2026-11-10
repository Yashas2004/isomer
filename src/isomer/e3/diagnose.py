"""Phase B: score labelled states with a heuristic and compute calibration / ranking metrics.

    python -m isomer.e3.diagnose score LABELS.jsonl SCORES.jsonl --ckpt runs/h-v1/best.pt [--rename KIND]
    python -m isomer.e3.diagnose metrics SCORES.jsonl

Scores: the model h, symbol_bag and size_diff for each on-path state and all of
its children (fp32 on CPU unless --device/--precision say otherwise).

Metrics (all on states whose children are exactly labelled):
* calibration of h on exact-distance states (the on-path states, and every
  optimal child, whose distance is exactly k-1): MAE, RMSE, signed bias,
  Spearman, mean prediction per true distance (saturation);
* move choice over ALL legal children: fractional top-1 optimal (ties split,
  as ``learn.metrics``), the base rate (fraction of children that are
  optimal), the rank of the best optimal child, and the margin
  min h(optimal) - min h(non-optimal) (negative = the right move is preferred).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core.term import parse
from ..search.heuristics import size_diff, symbol_bag


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------------- scoring

def score(labels: str, out: str, ckpt: str, device: str = "cpu", precision: str = "fp32",
          rename: str | None = None, batch_size: int = 1024) -> dict:
    import torch

    from ..learn.infer import LearnedHeuristic
    from . import rename as RN

    torch.set_num_threads(max(1, torch.get_num_threads()))
    h = LearnedHeuristic(ckpt, device=device, precision=precision, batch_size=batch_size)
    rows = [r for r in _rows(labels) if r["status"] == "ok"]
    t0 = time.perf_counter()
    n_seq = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            m = _rename_map(rename, r["theory"])
            state, goal = parse(r["state"]), parse(r["goal"])
            kids = [parse(c) for c in r["children"]]
            if m is not None:
                state, goal = RN.rename_term(state, m), RN.rename_term(goal, m)
                kids = [RN.rename_term(c, m) for c in kids]
            vals = h.scores([state, *kids], goal)
            n_seq += len(vals)
            r2 = {k: r[k] for k in ("problem_id", "theory", "problem_dist", "i", "dist", "state_size", "n_children",
                                    "optimal", "child_dist")}
            r2.update(h_state=vals[0], h_children=vals[1:],
                      sb_state=symbol_bag(state, goal), sb_children=[symbol_bag(c, goal) for c in kids],
                      sd_children=[size_diff(c, goal) for c in kids], rename=rename)
            f.write(json.dumps(r2) + "\n")
    secs = time.perf_counter() - t0
    return {"rows": len(rows), "sequences": n_seq, "seconds": round(secs, 1),
            "seq_per_second": round(n_seq / max(secs, 1e-9), 1), "device": device, "precision": precision,
            "threads": torch.get_num_threads()}


def _rename_map(kind: str | None, theory: str) -> dict | None:
    from . import rename as RN

    if kind is None:
        return None
    if kind == "skolem_perm":
        return RN.SKOLEM_PERM.get(theory)
    if kind.startswith("skolem_fresh:"):
        return RN.SKOLEM_FRESH[kind.split(":", 1)[1]].get(theory)
    if kind == "dual":
        return RN.DUAL.get(theory)
    if kind.startswith("resign:"):
        name = kind.split(":", 1)[1]
        return RN.RESIGN[name] if name.startswith(theory + "->") else None
    raise ValueError(f"unknown rename {kind!r}")


# ---------------------------------------------------------------------------------- metrics

def top1(scores: list[float], optimal: list[bool]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    tie = s == s.min()
    return float((tie & np.asarray(optimal)).sum() / tie.sum())


def best_opt_rank(scores: list[float], optimal: list[bool]) -> float:
    """1-based rank of the best optimal child (ties: average rank)."""
    s = np.asarray(scores, dtype=np.float64)
    best = s[np.asarray(optimal)].min()
    return float((s < best).sum() + ((s == best).sum() + 1) / 2)


def margin(scores: list[float], optimal: list[bool]) -> float | None:
    s = np.asarray(scores, dtype=np.float64)
    o = np.asarray(optimal)
    if o.all():
        return None
    return float(s[o].min() - s[~o].min())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic) if len(x) > 2 else float("nan")


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - w, c + w)


def bootstrap_mean_ci(values: list[float], groups: list[str], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap over problems (states of one problem are not independent)."""
    if not values:
        return (float("nan"), float("nan"))
    by: dict[str, list[float]] = defaultdict(list)
    for v, g in zip(values, groups, strict=True):
        by[g].append(v)
    keys = list(by)
    sums = np.array([sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def metrics(scores_path: str, by: tuple[str, ...] = ("dist",)) -> dict:
    rows = _rows(scores_path)
    return summarize(rows, by)


def summarize(rows: list[dict], by: tuple[str, ...] = ("dist",)) -> dict:
    """Ranking metrics for h / symbol_bag / size_diff and calibration for h, overall and per group."""
    def block(rs: list[dict]) -> dict:
        if not rs:
            return {"states": 0}
        pid = [r["problem_id"] for r in rs]
        out: dict = {"states": len(rs), "problems": len(set(pid)),
                     "mean_children": float(np.mean([r["n_children"] for r in rs])),
                     "base_rate": float(np.mean([np.mean(r["optimal"]) for r in rs]))}
        for name, key in (("h", "h_children"), ("symbol_bag", "sb_children"), ("size_diff", "sd_children")):
            t = [top1(r[key], r["optimal"]) for r in rs]
            rank = [best_opt_rank(r[key], r["optimal"]) for r in rs]
            mg = [margin(r[key], r["optimal"]) for r in rs]
            mgv = [m for m in mg if m is not None]
            out[name] = {"top1_optimal": float(np.mean(t)), "top1_ci95": bootstrap_mean_ci(t, pid),
                         "mean_best_optimal_rank": float(np.mean(rank)),
                         "median_best_optimal_rank": float(np.median(rank)),
                         "margin_mean": float(np.mean(mgv)) if mgv else None,
                         "margin_frac_negative": float(np.mean([m < 0 for m in mgv])) if mgv else None}
        # calibration of h on exact-distance states: on-path states (k) and optimal children (k-1)
        true, pred = [], []
        for r in rs:
            true.append(r["dist"])
            pred.append(r["h_state"])
            for v, o in zip(r["h_children"], r["optimal"], strict=True):
                if o:
                    true.append(r["dist"] - 1)
                    pred.append(v)
        t_, p_ = np.asarray(true, float), np.asarray(pred, float)
        err = p_ - t_
        out["calibration_onpath"] = _calib([r["dist"] for r in rs], [r["h_state"] for r in rs])
        out["calibration_exact_all"] = {"n": len(t_), "mae": float(np.abs(err).mean()),
                                        "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
                                        "spearman": spearman(t_, p_)}
        # does h agree with symbol_bag on the move? (fraction of states whose argmin sets intersect)
        agree = []
        for r in rs:
            a = np.asarray(r["h_children"]); b = np.asarray(r["sb_children"])
            agree.append(bool(((a == a.min()) & (b == b.min())).any()))
        out["h_symbol_bag_argmin_overlap"] = float(np.mean(agree))
        return out

    res = {"overall": block(rows)}
    for key in by:
        groups: dict = defaultdict(list)
        for r in rows:
            groups[_bucket(key, r)].append(r)
        res[f"by_{key}"] = {str(k): block(v) for k, v in sorted(groups.items(), key=lambda kv: _sortkey(kv[0]))}
    return res


def _calib(true: list[int], pred: list[float]) -> dict:
    t, p = np.asarray(true, float), np.asarray(pred, float)
    err = p - t
    per = {}
    for d in sorted(set(true)):
        e = err[t == d]
        per[str(d)] = {"n": len(e), "mean_pred": float(p[t == d].mean()), "mae": float(np.abs(e).mean()),
                       "bias": float(e.mean())}
    return {"n": len(t), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "spearman": spearman(t, p), "per_dist": per}


def _bucket(key: str, r: dict):
    if key == "children_bin":
        n = r["n_children"]
        return "<25" if n < 25 else "25-49" if n < 50 else "50-99" if n < 100 else ">=100"
    if key == "size_bin":
        s = r["state_size"]
        return "<10" if s < 10 else "10-14" if s < 15 else "15-19" if s < 20 else ">=20"
    return r[key]


def _sortkey(k):
    return (0, k) if isinstance(k, (int, float)) else (1, str(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("labels")
    s.add_argument("out")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--precision", default="fp32")
    s.add_argument("--rename", default=None)
    m = sub.add_parser("metrics")
    m.add_argument("scores")
    m.add_argument("--by", default="dist,theory,children_bin,size_bin")
    a = ap.parse_args()
    if a.cmd == "score":
        print(json.dumps(score(a.labels, a.out, a.ckpt, a.device, a.precision, a.rename)))
    else:
        print(json.dumps(metrics(a.scores, tuple(a.by.split(","))), indent=1))


if __name__ == "__main__":
    main()
