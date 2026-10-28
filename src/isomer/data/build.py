"""IsoBench builder: generate -> dedup -> label with exact BFS -> split -> write.

v2 (ADR-004) builds three pools of problems, each with its own walk lengths
and labelling budget:

    pool       theories       walks k        labeller (bidirectional BFS)
    train      group, bool    1..6           3000 expansions
    eval       group, bool    6..24          20000 expansions, node cap
    held-out   lattice, ring  2..12          20000 expansions, node cap

Output layout (one directory per benchmark version):

    data/isobench-v2/
        train.jsonl         train pool, exact dist <= train_max_dist
        test_id.jsonl       same distribution as train, held out by hash
        test_depth.jsonl    eval pool, exact dist > train_max_dist
        test_deep.jsonl     eval pool, no exact dist: graded by the proven dist_lb
        test_theory.jsonl   held-out pool (zero-shot theories)
        datasheet.json      config, label budgets, counts, histograms, timings

Why separate pools: in E1 the labeller's budget equalled the largest benchmark
budget, so bfs-bidir's results were circular; deep problems were too rare to
study depth generalization; and the training set was too small. Eval problems
are labelled at 20x the largest recommended benchmark budget.

Every record carries a kernel-checked proof and the bounds
``dist_lb <= dist <= dist_ub`` (``dist`` is null when BFS stopped first; the
bounds are always present). Walks that leave the labeller's size cap are
rejected, so every walk is a proof *in the action space* and its length is a
valid ``dist_ub``.
Splits are a pure function of content hashes, so rebuilding with the same
config is byte-identical.
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
from typing import NamedTuple

from ..core.generate import random_term, random_walk
from ..core.proof import Problem, Proof
from ..core.rewrite import apply_step
from ..core.term import size
from ..core.theories import THEORIES
from ..search.bfs import bidirectional_bfs
from ..search.common import size_cap

FORMAT_VERSION = "isobench/2"   # v2: three pools, eval labelled at 20x benchmark budgets (ADR-004)

SPLITS = ("train", "test_id", "test_depth", "test_deep", "test_theory")
POOLS = ("train", "eval", "heldout")


@dataclass
class BuildConfig:
    out: str = "data/isobench-v2"
    train_theories: tuple[str, ...] = ("group", "bool")
    heldout_theories: tuple[str, ...] = ("lattice", "ring")
    # train pool: many cheap, shallow problems
    train_per_theory: int = 20000       # problems generated (before the dist filter) per theory
    train_walk_lengths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    train_label_budget: int = 3000
    # eval pool: deep problems, same theories as train
    eval_per_theory: int = 1200
    eval_walk_lengths: tuple[int, ...] = (6, 8, 10, 12, 16, 20, 24)
    # held-out pool: unseen theories
    heldout_per_theory: int = 300
    heldout_walk_lengths: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    # eval + held-out labeller
    label_budget: int = 20000           # expansions; 20x the largest recommended benchmark budget (1000)
    label_max_nodes: int = 1_500_000    # cap on stored nodes per problem (bounds worker RAM)
    train_max_dist: int = 4
    test_id_fraction: float = 0.05
    term_depth: int = 4
    max_term_size: int = 30
    max_size_slack: int = 6             # action space: terms up to max(|lhs|, |rhs|) + slack (ADR-002)
    pool_max_size: int = 3              # problem pool: subterms of lhs/rhs up to this size (ADR-003)
    seed: int = 0
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))


class LabelJob(NamedTuple):
    """Everything a worker needs, as plain picklable values."""
    pool: str
    theory: str
    lhs: str
    rhs: str
    walk_steps: list[dict]
    budget: int
    max_nodes: int | None
    pool_max_size: int
    max_size_slack: int


def problem_id(theory: str, lhs: str, rhs: str) -> str:
    """Order-independent: (a = b) and (b = a) are the same problem."""
    a, b = sorted((lhs, rhs))
    return hashlib.sha256(f"{theory}|{a}|{b}".encode()).hexdigest()[:16]


def _seed(seed: int, pool: str, theory: str) -> int:
    """One independent, stable RNG stream per (pool, theory)."""
    return int(hashlib.sha256(f"{seed}|{pool}|{theory}".encode()).hexdigest()[:16], 16)


def walk_in_action_space(walk: Proof, slack: int) -> bool:
    """True if every term on the walk is within the labeller's size cap.

    Only then is the walk a legal proof in the action space (the pool is legal
    by construction, ADR-003), so its length bounds ``dist`` from above."""
    cap = size_cap(walk.problem, slack)
    rules = walk.problem.theory.rules()
    t = walk.problem.lhs
    for step in walk.steps:
        t = apply_step(rules[step.rule], t, step)
        if t is None or size(t) > cap:
            return False
    return True


def _generate(theory_name: str, n: int, walk_lengths: tuple[int, ...], cfg: BuildConfig, seed: int,
              exclude: set[str]) -> tuple[list[Proof], int]:
    """``n`` unique problems from random walks, skipping ids in ``exclude``.
    Also returns how many walks were rejected for leaving the size cap.

    Rejecting such walks (instead of confining walks to ``size(start) + slack``
    while they run) keeps deep problems: the cap also counts the *final* term,
    and confined walks mostly collapse back to short distances (ADR-004)."""
    theory, sig = THEORIES[theory_name]
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[Proof] = []
    attempts = rejected = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        k = rng.choice(walk_lengths)
        p = random_walk(theory, random_term(sig, cfg.term_depth, rng), k, rng, max_size=cfg.max_term_size,
                        pool_max_size=cfg.pool_max_size)
        if not p.steps or p.problem.lhs == p.problem.rhs:
            continue
        if not walk_in_action_space(p, cfg.max_size_slack):
            rejected += 1
            continue
        pid = problem_id(theory_name, str(p.problem.lhs), str(p.problem.rhs))
        if pid in seen or pid in exclude:
            continue
        seen.add(pid)
        out.append(p)
    return out, rejected


def _label(job: LabelJob) -> dict:
    """Runs in a worker process. Returns one JSON-ready record plus private
    ``_``-prefixed measurements that do not go into the records."""
    from ..core.rewrite import Step  # local import keeps worker start cheap

    theory, _ = THEORIES[job.theory]
    problem = Problem.parse(theory, job.lhs, job.rhs)
    walk = Proof(problem, tuple(Step.from_json(s) for s in job.walk_steps))
    assert walk.verify(), "generator produced a proof the kernel rejects"
    res = bidirectional_bfs(problem, max_expansions=job.budget, max_nodes=job.max_nodes,
                            max_size=size_cap(problem, job.max_size_slack), pool_max_size=job.pool_max_size)
    best = res.proof if res.proof else walk
    v = best.verify()
    assert v, f"kernel rejected labelled proof: {v.reason}"
    dist = len(res.proof.steps) if res.proof else None
    dist_ub = len(best.steps)
    assert res.lower_bound <= dist_ub, f"dist_lb {res.lower_bound} > dist_ub {dist_ub}: walk left the action space?"
    return {
        "id": problem_id(job.theory, job.lhs, job.rhs),
        "theory": job.theory,
        "lhs": job.lhs,
        "rhs": job.rhs,
        "walk_len": len(job.walk_steps),
        "dist": dist,
        "dist_lb": res.lower_bound,
        "dist_ub": dist_ub,
        "bfs_expansions": res.expansions,
        "label_stop": res.stats["stop"],
        "label_nodes": res.stats["nodes_stored"],   # deterministic, unlike time
        "proof": [s.to_json() for s in best.steps],
        # Timing is machine-dependent: returned separately, aggregated in the datasheet.
        "_seconds": res.seconds,
    }


def _split(pool: str, rec: dict, cfg: BuildConfig) -> str | None:
    """Destination split, or None if the problem is dropped."""
    dist = rec["dist"]
    if pool == "heldout":
        return "test_theory"
    if pool == "eval":
        if dist is None:
            return "test_deep"
        # Shallow eval problems would duplicate test_id's distribution.
        return "test_depth" if dist > cfg.train_max_dist else None
    if dist is None or dist > cfg.train_max_dist:
        return None
    # Deterministic hash-based hold-out, independent of generation order.
    bucket = int(rec["id"], 16) % 10_000 / 10_000
    return "test_id" if bucket < cfg.test_id_fraction else "train"


def _label_pool(name: str, jobs: list[LabelJob], cfg: BuildConfig, chunksize: int,
                progress: bool) -> tuple[list[dict], list[float], float]:
    t0 = time.perf_counter()
    records: list[dict] = []
    seconds: list[float] = []
    every = max(1, len(jobs) // 20)
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for k, rec in enumerate(ex.map(_label, jobs, chunksize=chunksize), 1):
            seconds.append(rec.pop("_seconds"))
            records.append(rec)
            if progress and (k % every == 0 or k == len(jobs)):
                print(f"  [{name}] labelled {k}/{len(jobs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0
    if progress:
        print(f"  [{name}] done: {len(jobs)} problems in {wall:.0f}s", flush=True)
    return records, seconds, wall


def _hist(values: list) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}


def build(cfg: BuildConfig, progress: bool = True) -> dict:
    t0 = time.perf_counter()
    gen_seconds: dict[str, float] = {}
    jobs: dict[str, list[LabelJob]] = {p: [] for p in POOLS}
    plan = {
        "heldout": (cfg.heldout_theories, cfg.heldout_per_theory, cfg.heldout_walk_lengths, cfg.label_budget),
        "eval": (cfg.train_theories, cfg.eval_per_theory, cfg.eval_walk_lengths, cfg.label_budget),
        "train": (cfg.train_theories, cfg.train_per_theory, cfg.train_walk_lengths, cfg.train_label_budget),
    }
    # Test pools are generated first and their ids are excluded from the train
    # pool, so no test problem is ever seen in training (ADR-004).
    claimed: set[str] = set()
    rejected: dict[str, int] = {}
    for pool in ("heldout", "eval", "train"):
        tp = time.perf_counter()
        theories, n, walks, budget = plan[pool]
        rejected[pool] = 0
        for name in theories:
            problems, rej = _generate(name, n, walks, cfg, _seed(cfg.seed, pool, name), exclude=claimed)
            rejected[pool] += rej
            for p in problems:
                lhs, rhs = str(p.problem.lhs), str(p.problem.rhs)
                claimed.add(problem_id(name, lhs, rhs))
                jobs[pool].append(LabelJob(pool, name, lhs, rhs, [s.to_json() for s in p.steps], budget,
                                           cfg.label_max_nodes, cfg.pool_max_size, cfg.max_size_slack))
        gen_seconds[pool] = time.perf_counter() - tp
        if progress:
            print(f"  [{pool}] generated {len(jobs[pool])} problems ({gen_seconds[pool]:.0f}s)", flush=True)

    # Cheap train pool first; then the memory-heavy eval + held-out labelling,
    # one problem per task so long problems do not queue behind each other.
    labelled: dict[str, list[dict]] = {}
    label_seconds: dict[str, list[float]] = {}
    walls: dict[str, float] = {}
    labelled["train"], label_seconds["train"], walls["train"] = _label_pool("train", jobs["train"], cfg, 16, progress)
    hard = jobs["eval"] + jobs["heldout"]
    recs, secs, walls["eval+heldout"] = _label_pool("eval+heldout", hard, cfg, 1, progress)
    n_eval = len(jobs["eval"])
    labelled["eval"], labelled["heldout"] = recs[:n_eval], recs[n_eval:]
    label_seconds["eval"], label_seconds["heldout"] = secs[:n_eval], secs[n_eval:]

    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    dropped: Counter[str] = Counter()
    for pool in POOLS:
        for rec in labelled[pool]:
            dest = _split(pool, rec, cfg)
            if dest is None:
                dropped[pool] += 1
            else:
                splits[dest].append(rec)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs_ in splits.items():
        recs_.sort(key=lambda r: r["id"])
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in recs_)

    pool_of_split = {"train": "train", "test_id": "train", "test_depth": "eval", "test_deep": "eval",
                     "test_theory": "heldout"}
    budget_of_pool = {"train": cfg.train_label_budget, "eval": cfg.label_budget, "heldout": cfg.label_budget}
    sheet = {
        "format": FORMAT_VERSION,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items() if k != "workers"},
        # Read by `isomer bench` for its circularity guard.
        "label_budget": {s: budget_of_pool[p] for s, p in pool_of_split.items()},
        "counts": {k: len(v) for k, v in splits.items()},
        "by_theory": {k: dict(sorted(Counter(r["theory"] for r in v).items())) for k, v in splits.items()},
        "generated": {p: len(jobs[p]) for p in POOLS},
        "walks_rejected_outside_size_cap": rejected,
        "dropped": {"train (dist unknown or > train_max_dist)": dropped["train"],
                    "eval (dist <= train_max_dist)": dropped["eval"]},
        "dist_histogram": {k: _hist([r["dist"] for r in v]) for k, v in splits.items() if k != "test_deep"},
        "dist_lb_histogram": {"test_deep": _hist([r["dist_lb"] for r in splits["test_deep"]])},
        "dist_ub_histogram": {k: _hist([r["dist_ub"] for r in v]) for k, v in splits.items()},
        "walk_minus_dist": {p: _hist([r["walk_len"] - r["dist"] for r in labelled[p] if r["dist"] is not None])
                            for p in POOLS},
        "label_stop": {p: dict(sorted(Counter(r["label_stop"] for r in labelled[p]).items())) for p in POOLS},
        "max_nodes_stored": {p: max((r["label_nodes"] for r in labelled[p]), default=0) for p in POOLS},
        "seconds": {
            "generate": {p: round(s, 1) for p, s in gen_seconds.items()},
            "label_wall": {p: round(s, 1) for p, s in walls.items()},
            "label_mean": {p: round(sum(s) / max(1, len(s)), 3) for p, s in label_seconds.items()},
            "label_max": {p: round(max(s, default=0.0), 3) for p, s in label_seconds.items()},
            "total": round(time.perf_counter() - t0, 1),
        },
        "notes": [
            ("dist = exact shortest proof length in Isomer's action space (ADR-003): fresh and consumed "
             "variable values come from the problem pool (theory constants, skolems occurring in lhs/rhs, "
             f"subterms of lhs/rhs of size <= pool_max_size = {cfg.pool_max_size}); term size <= "
             f"max(|lhs|,|rhs|)+{cfg.max_size_slack}. The action space is symmetric, so bidirectional BFS "
             "distances are exact. dist is null when the labeller stopped first."),
            (f"eval and held-out problems (test_depth, test_deep, test_theory) were labelled with "
             f"{cfg.label_budget} expansions, 20x the largest recommended benchmark budget (1000), and a cap of "
             f"{cfg.label_max_nodes} stored nodes; label_stop says which limit ended the search. Train and "
             f"test_id were labelled with {cfg.train_label_budget} expansions and contain only problems "
             f"bidirectional BFS solved within that budget with dist <= {cfg.train_max_dist}."),
            ("test_deep holds eval problems with no exact dist. Grade them by the certified lower bound dist_lb "
             "(proven from completed BFS layers), never by dist_ub."),
            ("dist_ub = length of the best known proof (the BFS proof if found, else the random walk). Walks "
             "that leave the size cap are rejected at generation, so dist_lb <= dist <= dist_ub always holds."),
            "every stored proof was accepted by the trusted kernel at build time.",
        ],
    }
    (out / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True), encoding="utf-8")
    return sheet
