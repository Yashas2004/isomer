"""E3 analysis of scored states: move-choice errors, edge consistency, renaming invariance.

    python -m isomer.e3.analyze base SCORES.jsonl
    python -m isomer.e3.analyze invariance BASE.jsonl RENAMED.jsonl

Reads the output of ``diagnose score``; prints JSON. No model, no search.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np

from .diagnose import _rows, bootstrap_mean_ci, summarize, top1


def choice_classes(rows: list[dict], key: str) -> dict[str, float]:
    """Where the lowest-scored child leads (ties split): optimal (k-1), sideways (k),
    away (k+1), or non-optimal with distance outside the goal ball (k or k+1, not resolved)."""
    c: Counter[str] = Counter()
    for r in rows:
        s = np.asarray(r[key], dtype=np.float64)
        ties = np.flatnonzero(s == s.min())
        for i in ties:
            w = 1.0 / len(ties)
            if r["optimal"][i]:
                c["optimal"] += w
            else:
                cd = r["child_dist"][i]
                c["sideways" if cd == r["dist"] else "away" if cd == r["dist"] + 1 else "non_optimal_unresolved"] += w
    return {k: v / len(rows) for k, v in sorted(c.items())}


def edge_consistency(rows: list[dict]) -> dict:
    """Parent-child edges: true distances differ by at most 1. How often does h break that?"""
    diffs = np.concatenate([np.abs(np.asarray(r["h_children"]) - r["h_state"]) for r in rows])
    excess = np.clip(diffs - 1.0, 0.0, None)
    # signed: optimal children should be ~1 lower than the parent
    step = np.concatenate([[r["h_state"] - v for v, o in zip(r["h_children"], r["optimal"], strict=True) if o]
                           for r in rows])
    return {"edges": len(diffs), "violation_rate": float((diffs > 1.0).mean()),
            "violation_rate_gt_1.5": float((diffs > 1.5).mean()),
            "mean_squared_excess": float((excess ** 2).mean()),
            "mean_abs_diff": float(diffs.mean()),
            "optimal_child_mean_drop": float(step.mean()), "optimal_child_drop_sd": float(step.std())}


def paired_vs_symbol_bag(rows: list[dict]) -> dict:
    """Per-state top-1 of h minus symbol_bag, cluster bootstrap over problems."""
    d = [top1(r["h_children"], r["optimal"]) - top1(r["sb_children"], r["optimal"]) for r in rows]
    pid = [r["problem_id"] for r in rows]
    return {"mean_diff": float(np.mean(d)), "ci95": bootstrap_mean_ci(d, pid)}


def base(path: str) -> dict:
    rows = _rows(path)
    out = summarize(rows, ("dist", "theory", "children_bin", "size_bin"))
    out["choice_classes"] = {name: choice_classes(rows, key) for name, key in
                             (("h", "h_children"), ("symbol_bag", "sb_children"), ("size_diff", "sd_children"))}
    out["edge_consistency"] = edge_consistency(rows)
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["dist"]].append(r)
    out["edge_consistency_by_dist"] = {str(k): edge_consistency(v) for k, v in sorted(by.items())}
    out["h_minus_symbol_bag_top1"] = paired_vs_symbol_bag(rows)
    return out


def invariance(base_path: str, renamed_path: str) -> dict:
    """Same states and children, scored before and after a renaming (paired)."""
    a = {(r["problem_id"], r["i"]): r for r in _rows(base_path)}
    b = {(r["problem_id"], r["i"]): r for r in _rows(renamed_path)}

    def block(keys: list) -> dict:
        if not keys:
            return {"states": 0}
        dh, t_a, t_b, flips, pid = [], [], [], [], []
        ha_all, hb_all = [], []
        for k in keys:
            ra, rb = a[k], b[k]
            xa = np.asarray([ra["h_state"], *ra["h_children"]])
            xb = np.asarray([rb["h_state"], *rb["h_children"]])
            dh.append(np.abs(xa - xb))
            ha_all.append(xa)
            hb_all.append(xb)
            t_a.append(top1(ra["h_children"], ra["optimal"]))
            t_b.append(top1(rb["h_children"], rb["optimal"]))
            ca, cb = np.asarray(ra["h_children"]), np.asarray(rb["h_children"])
            flips.append(float(int(ca.argmin()) != int(cb.argmin())))
            pid.append(k[0])
        dh_ = np.concatenate(dh)
        ha, hb = np.concatenate(ha_all), np.concatenate(hb_all)
        diff = [y - x for x, y in zip(t_a, t_b, strict=True)]
        return {"states": len(keys), "problems": len(set(pid)), "scored_terms": len(dh_),
                "mean_abs_delta_h": float(dh_.mean()), "p95_abs_delta_h": float(np.quantile(dh_, 0.95)),
                "frac_delta_h_gt_0.5": float((dh_ > 0.5).mean()),
                "pearson_h": float(np.corrcoef(ha, hb)[0, 1]),
                "top1_before": float(np.mean(t_a)), "top1_after": float(np.mean(t_b)),
                "top1_diff": float(np.mean(diff)), "top1_diff_ci95": bootstrap_mean_ci(diff, pid),
                "argmin_changed": float(np.mean(flips)),
                "mean_h_state_before": float(np.mean([a[k]["h_state"] for k in keys])),
                "mean_h_state_after": float(np.mean([b[k]["h_state"] for k in keys]))}

    # only states the renaming actually changed (a theory without a mapping is scored unrenamed)
    theories = sorted({r["theory"] for r in b.values()})
    out = {}
    for th in theories:
        keys = sorted(k for k in b if k in a and b[k]["theory"] == th)
        changed = any(not np.array_equal(np.asarray(a[k]["h_children"]), np.asarray(b[k]["h_children"])) or
                      a[k]["h_state"] != b[k]["h_state"] for k in keys)
        out[th] = block(keys) if changed else {"states": len(keys), "renaming_applies": False}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "base":
        print(json.dumps(base(sys.argv[2]), indent=1))
    else:
        print(json.dumps(invariance(sys.argv[2], sys.argv[3]), indent=1))
