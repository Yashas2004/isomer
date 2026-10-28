"""IsoBench builder: generate -> dedup -> label with exact BFS -> split -> write.

Output layout (one directory per benchmark version):

    data/isobench-v1/
        train.jsonl         group+bool, dist <= TRAIN_MAX_DIST (or walk-labelled)
        test_id.jsonl       same distribution as train, held out by hash
        test_depth.jsonl    group+bool, dist > TRAIN_MAX_DIST  (depth extrapolation)
        test_theory.jsonl   lattice+ring, never seen in training (zero-shot)
        datasheet.json      config, counts, histograms, timings

Every record carries a kernel-checked proof. ``dist`` is the exact shortest
length when BFS found it; otherwise ``dist`` is null and ``dist_lb`` is a
proven lower bound. Splits are a pure function of the problem's content hash,
so rebuilding with the same config is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.generate import random_term, random_walk
from ..core.proof import Problem, Proof
from ..core.theories import THEORIES
from ..search.bfs import bidirectional_bfs

FORMAT_VERSION = "isobench/1"   # v1: symmetric problem-level action space (ADR-003)


@dataclass
class BuildConfig:
    out: str = "data/isobench-v1"
    train_theories: tuple[str, ...] = ("group", "bool")
    heldout_theories: tuple[str, ...] = ("lattice", "ring")
    per_theory: int = 2000          # unique problems generated per training theory
    per_heldout_theory: int = 300
    walk_lengths: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)
    term_depth: int = 4
    max_term_size: int = 30
    bfs_budget: int = 3000          # expansions per problem for labelling
    pool_max_size: int = 3          # problem pool: subterms of lhs/rhs up to this size (ADR-003)
    train_max_dist: int = 4         # dist above this goes to test_depth
    test_id_fraction: float = 0.1
    seed: int = 0
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))


def problem_id(theory: str, lhs: str, rhs: str) -> str:
    """Order-independent: (a = b) and (b = a) are the same problem."""
    a, b = sorted((lhs, rhs))
    return hashlib.sha256(f"{theory}|{a}|{b}".encode()).hexdigest()[:16]


def _generate(theory_name: str, n: int, cfg: BuildConfig, seed: int) -> list[Proof]:
    theory, sig = THEORIES[theory_name]
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[Proof] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        k = rng.choice(cfg.walk_lengths)
        p = random_walk(theory, random_term(sig, cfg.term_depth, rng), k, rng, max_size=cfg.max_term_size,
                        pool_max_size=cfg.pool_max_size)
        if not p.steps or p.problem.lhs == p.problem.rhs:
            continue
        pid = problem_id(theory_name, str(p.problem.lhs), str(p.problem.rhs))
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _label(args: tuple[str, str, str, list[dict], int, int]) -> dict:
    """Runs in a worker process. Returns one JSON-ready record."""
    theory_name, lhs, rhs, walk_steps, budget, pool_max_size = args
    from ..core.rewrite import Step  # local import keeps worker start cheap

    theory, _ = THEORIES[theory_name]
    problem = Problem.parse(theory, lhs, rhs)
    walk = Proof(problem, tuple(Step.from_json(s) for s in walk_steps))
    assert walk.verify(), "generator produced a proof the kernel rejects"
    res = bidirectional_bfs(problem, max_expansions=budget, pool_max_size=pool_max_size)
    rec = {
        "id": problem_id(theory_name, lhs, rhs),
        "theory": theory_name,
        "lhs": lhs,
        "rhs": rhs,
        "walk_len": len(walk_steps),
        "dist": len(res.proof.steps) if res.proof else None,
        "dist_lb": res.lower_bound,
        "bfs_expansions": res.expansions,
    }
    # Timing is machine-dependent, so it must not go into the (deterministic)
    # records; it is returned separately and aggregated in the datasheet.
    rec["_seconds"] = res.seconds
    best = res.proof if res.proof else walk
    v = best.verify()
    assert v, f"kernel rejected labelled proof: {v.reason}"
    rec["proof"] = [s.to_json() for s in best.steps]
    return rec


def _split(rec: dict, cfg: BuildConfig) -> str:
    if rec["theory"] in cfg.heldout_theories:
        return "test_theory"
    dist = rec["dist"]
    if dist is None or dist > cfg.train_max_dist:
        return "test_depth"
    # Deterministic hash-based hold-out, independent of generation order.
    bucket = int(rec["id"], 16) % 10_000 / 10_000
    return "test_id" if bucket < cfg.test_id_fraction else "train"


def build(cfg: BuildConfig, progress: bool = True) -> dict:
    t0 = time.perf_counter()
    jobs: list[tuple[str, str, str, list[dict], int, int]] = []
    plan = [(t, cfg.per_theory) for t in cfg.train_theories] + \
           [(t, cfg.per_heldout_theory) for t in cfg.heldout_theories]
    for i, (name, n) in enumerate(plan):
        for p in _generate(name, n, cfg, seed=cfg.seed * 1000 + i):
            jobs.append((name, str(p.problem.lhs), str(p.problem.rhs),
                         [s.to_json() for s in p.steps], cfg.bfs_budget, cfg.pool_max_size))
    gen_seconds = time.perf_counter() - t0

    records: list[dict] = []
    label_seconds: list[float] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for k, rec in enumerate(ex.map(_label, jobs, chunksize=16), 1):
            label_seconds.append(rec.pop("_seconds"))
            records.append(rec)
            if progress and (k % 200 == 0 or k == len(jobs)):
                print(f"  labelled {k}/{len(jobs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "test_id": [], "test_depth": [], "test_theory": []}
    for rec in sorted(records, key=lambda r: r["id"]):
        splits[_split(rec, cfg)].append(rec)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in recs)

    def hist(recs: list[dict], key: str) -> dict[str, int]:
        return {str(k): v for k, v in sorted(Counter(r[key] for r in recs).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}

    sheet = {
        "format": FORMAT_VERSION,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items() if k != "workers"},
        "counts": {k: len(v) for k, v in splits.items()},
        "by_theory": {k: dict(Counter(r["theory"] for r in v)) for k, v in splits.items()},
        "dist_histogram": {k: hist(v, "dist") for k, v in splits.items()},
        "walk_minus_dist": dict(Counter(r["walk_len"] - r["dist"] for r in records if r["dist"] is not None)),
        "unlabelled_fraction": round(sum(r["dist"] is None for r in records) / max(1, len(records)), 4),
        "seconds": {"generate": round(gen_seconds, 1), "total": round(time.perf_counter() - t0, 1),
                    "bfs_mean": round(sum(label_seconds) / max(1, len(label_seconds)), 3),
                    "bfs_max": round(max(label_seconds, default=0.0), 3)},
        "notes": [
            ("dist = exact shortest proof length in Isomer's action space (ADR-003): fresh and consumed "
             "variable values come from the problem pool (theory constants, skolems occurring in lhs/rhs, "
             f"subterms of lhs/rhs of size <= pool_max_size = {cfg.pool_max_size}); term size <= max(|lhs|,|rhs|)+6. The action "
             "space is symmetric, so bidirectional BFS distances are exact. null when the BFS budget ran out; "
             "then dist_lb is a proven lower bound."),
            "walk_len is only an upper bound on proof length.",
            "every stored proof was accepted by the trusted kernel at build time.",
        ],
    }
    (out / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True), encoding="utf-8")
    return sheet
