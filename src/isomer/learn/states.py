"""Distance-labelled (state, goal) examples for a learned heuristic h(state, goal).

Input is IsoBench's ``train.jsonl`` only; no test split is ever read. Every
example's ``dist`` is exact *in the original problem's action space*: its pool
(ADR-003) and its size cap (ADR-002).

* **On-path states.** A record with shortest proof t_0 .. t_d gives t_i with
  dist d - i, free: a sub-path of a shortest path is itself shortest.
* **Off-path states.** For each t_i with i < d, up to ``offpath`` legal
  successors u that are not on the path, sampled deterministically (seeded
  by problem id and i), labelled by bidirectional BFS run with the original
  pool and cap. A fresh ``Problem(u, goal)`` would derive a different pool and
  cap from its own sides, so both are passed explicitly. If BFS fails within
  the budget the state is dropped and counted. Because the action space is
  symmetric, |dist(u) - dist(t_i)| <= 1 for every off-path state; the tests
  check this triangle property on every record.

A problem's states all go to one split, ``train`` or ``val`` (a salted hash of
the problem id), so validation measures generalization to unseen problems.
Output is sorted and contains no timings, so rebuilding is byte-identical.
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

from ..core.proof import Problem
from ..core.rewrite import Step, apply_step, successors
from ..core.term import Term, size
from ..core.theories import THEORIES
from ..search.bfs import bidirectional_bfs
from ..search.common import size_cap
from .tokenizer import n_tokens

FORMAT_VERSION = "isomer-states/1"

# (problem_id, theory, lhs, rhs, proof steps as JSON, dist, offpath, budget, pool_max_size, size slack)
Job = tuple[str, str, str, str, list[dict], int, int, int, int, int]


@dataclass
class StatesConfig:
    data: str = "data/isobench-v2"
    out: str = "data/states-v2"
    offpath: int = 3            # off-path successors sampled per on-path state
    budget: int = 3000          # BFS expansions to label one off-path state
    val_fraction: float = 0.05  # of train *problems*
    limit: int | None = None    # first N problems in id order (measurement, tests)
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))


def split_of(problem_id: str, val_fraction: float) -> str:
    """Salted, so independent of the (unsalted) rule that made test_id."""
    h = int(hashlib.sha256(f"states-val|{problem_id}".encode()).hexdigest(), 16)
    return "val" if h % 10_000 / 10_000 < val_fraction else "train"


def _rng(problem_id: str, i: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"offpath|{problem_id}|{i}".encode()).hexdigest()[:16], 16))


def _path(problem: Problem, steps: list[Step]) -> list[Term]:
    rules = problem.theory.rules()
    terms = [problem.lhs]
    for s in steps:
        t = apply_step(rules[s.rule], terms[-1], s)
        assert t is not None, f"stored proof does not replay at {s}"
        terms.append(t)
    assert terms[-1] == problem.rhs, "stored proof does not end at rhs"
    return terms


def _states_for(job: Job) -> tuple[list[dict], dict[str, int], float]:
    """Worker: all examples of one problem. Plain tuple in, plain data out
    (picklable on Windows)."""
    t0 = time.perf_counter()
    pid, theory_name, lhs, rhs, proof, dist, offpath, budget, pool_max_size, slack = job
    theory, _ = THEORIES[theory_name]
    problem = Problem.parse(theory, lhs, rhs)
    path = _path(problem, [Step.from_json(s) for s in proof])
    d = len(path) - 1
    assert d == dist, f"{pid}: proof length {d} != dist {dist}"
    pool = problem.pool(pool_max_size)
    cap = size_cap(problem, slack)
    rules = list(theory.rules().values())
    goal = problem.rhs

    def rec(state: Term, dist_: int, on_path: bool, parent_dist: int | None) -> dict:
        return {"problem_id": pid, "theory": theory_name, "state": str(state), "goal": rhs, "dist": dist_,
                "on_path": on_path, "parent_dist": parent_dist}

    out = [rec(t, d - i, True, None) for i, t in enumerate(path)]
    counts: Counter[str] = Counter()
    taken: set[Term] = set(path)
    for i in range(d):
        candidates = list(dict.fromkeys(u for _, u in successors(path[i], rules, pool)
                                        if size(u) <= cap and u not in taken))
        if not candidates:
            counts["no_offpath_candidates"] += 1
            continue
        for u in _rng(pid, i).sample(candidates, min(offpath, len(candidates))):
            taken.add(u)
            res = bidirectional_bfs(Problem(theory, u, goal), max_expansions=budget, max_size=cap, pool=pool)
            if res.proof is None:
                counts[f"bfs_{res.stats['stop']}"] += 1
                continue
            out.append(rec(u, len(res.proof.steps), False, d - i))
    return out, dict(counts), time.perf_counter() - t0


def _source_info(data: Path) -> tuple[dict, int, int]:
    """Datasheet format and action-space settings, so states use exactly the
    action space the labels were computed in."""
    try:
        sheet = json.loads((data / "datasheet.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sheet = {}
    cfg = sheet.get("config", {})
    return sheet, int(cfg.get("pool_max_size", 3)), int(cfg.get("max_size_slack", 6))


def build_states(cfg: StatesConfig, progress: bool = True) -> dict:
    t0 = time.perf_counter()
    data = Path(cfg.data)
    raw = (data / "train.jsonl").read_bytes()   # train only: never a test split
    records = sorted((json.loads(line) for line in raw.decode("utf-8").splitlines()), key=lambda r: r["id"])
    if cfg.limit is not None:
        records = records[:cfg.limit]
    sheet, pool_max_size, slack = _source_info(data)
    jobs: list[Job] = [(r["id"], r["theory"], r["lhs"], r["rhs"], r["proof"], r["dist"], cfg.offpath, cfg.budget,
                        pool_max_size, slack) for r in records]

    results: list[tuple[list[dict], dict[str, int], float]] = []
    every = max(1, len(jobs) // 20)

    def tick(k: int) -> None:
        if progress and (k % every == 0 or k == len(jobs)):
            print(f"  states: {k}/{len(jobs)} problems ({time.perf_counter() - t0:.0f}s)", flush=True)

    if cfg.workers == 1:
        for k, job in enumerate(jobs, 1):
            results.append(_states_for(job))
            tick(k)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            for k, res in enumerate(ex.map(_states_for, jobs, chunksize=8), 1):
                results.append(res)
                tick(k)

    splits: dict[str, list[dict]] = {"train": [], "val": []}
    dropped: Counter[str] = Counter()
    problems: Counter[str] = Counter()
    seconds = []
    for (recs, counts, secs), job in zip(results, jobs, strict=True):
        split = split_of(job[0], cfg.val_fraction)
        problems[split] += 1
        dropped.update(counts)
        seconds.append(secs)
        for r in recs:
            r["split"] = split
            splits[split].append(r)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        recs.sort(key=lambda r: (r["problem_id"], r["state"]))
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in recs)

    def by(recs: list[dict], *keys: str) -> dict:
        c = Counter(tuple(str(r[k]) for k in keys) for r in recs)
        return {"|".join(k): v for k, v in sorted(c.items())}

    stats = {
        "format": FORMAT_VERSION,
        "config": {k: v for k, v in asdict(cfg).items() if k != "workers"},
        "source": {"format": sheet.get("format", "unknown"),
                   "train_sha256": hashlib.sha256(raw).hexdigest(),
                   "pool_max_size": pool_max_size, "max_size_slack": slack},
        "problems": dict(sorted(problems.items())),
        "counts": {s: len(r) for s, r in splits.items()},
        "on_off_path": {s: {"on_path": sum(r["on_path"] for r in v), "off_path": sum(not r["on_path"] for r in v)}
                        for s, v in splits.items()},
        "by_theory_dist": {s: by(v, "theory", "dist") for s, v in splits.items()},
        "dropped": dict(sorted(dropped.items())),
        "max_tokens": max((n_tokens(r["state"], r["goal"]) for v in splits.values() for r in v), default=0),
        "seconds": {"total": round(time.perf_counter() - t0, 1),
                    "per_problem_mean": round(sum(seconds) / max(1, len(seconds)), 4),
                    "per_problem_max": round(max(seconds, default=0.0), 3)},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats
