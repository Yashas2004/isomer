"""Driver: label every legal child of every on-path state of a regenerated split.

    python -m isomer.e3.label_children runs/e3/data/test_depth.jsonl runs/e3/labels/test_depth.jsonl [WORKERS]

Only records with an exact ``dist`` are used (test_deep lower bounds are never
treated as exact). Output: one JSON line per on-path state with dist >= 1.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..core.proof import Problem
from ..core.rewrite import Step
from ..core.term import size
from ..core.theories import THEORIES
from .children import label_problem

BALL_NODES = 300_000
FORWARD_NODES = 300_000


def _job(rec: dict) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    theory, _ = THEORIES[rec["theory"]]
    problem = Problem.parse(theory, rec["lhs"], rec["rhs"])
    steps = [Step.from_json(s) for s in rec["proof"]]
    labels, R = label_problem(problem, steps, ball_nodes=BALL_NODES, forward_nodes=FORWARD_NODES)
    rows = []
    for i, lab in enumerate(labels):
        rows.append({"problem_id": rec["id"], "theory": rec["theory"], "problem_dist": rec["dist"], "i": i,
                     "dist": lab.dist, "state": str(lab.state), "goal": rec["rhs"], "state_size": size(lab.state),
                     "status": lab.status, "R": R, "n_children": len(lab.children),
                     "children": [str(u) for u in lab.children], "optimal": lab.optimal,
                     "child_dist": lab.child_dist, "forward_nodes": lab.forward_nodes})
    return rows, {"problem_id": rec["id"], "R": R, "seconds": round(time.perf_counter() - t0, 2),
                  "unknown": sum(r["status"] != "ok" for r in rows)}


def main(src: str, out: str, workers: int = 2) -> None:
    recs = [json.loads(line) for line in Path(src).read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if r["dist"] is not None]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    done = _complete(out, {r["id"]: r["dist"] for r in recs})
    todo = _stratified_order([r for r in recs if r["id"] not in done])
    print(f"  resuming: {len(done)} problems already labelled, {len(todo)} to go", flush=True)
    t0 = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f, open(out + ".meta.jsonl", "a", encoding="utf-8") as mf, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (rows, m) in enumerate(ex.map(_job, todo, chunksize=1), 1):
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()
            m.update(ball_nodes=BALL_NODES, forward_nodes=FORWARD_NODES)
            mf.write(json.dumps(m) + "\n")
            mf.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} problems ({time.perf_counter() - t0:.0f}s)", flush=True)


def _stratified_order(recs: list[dict], seed: int = 0) -> list[dict]:
    """Seeded round-robin over (theory, dist) strata, so that any prefix of the
    work (e.g. when a time budget stops the run) is close to stratified."""
    import random

    by: dict[tuple, list[dict]] = {}
    for r in sorted(recs, key=lambda r: r["id"]):
        by.setdefault((r["theory"], r["dist"]), []).append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out: list[dict] = []
    keys = sorted(by)
    while any(by.values()):
        for k in keys:
            if by[k]:
                out.append(by[k].pop())
    return out


def _complete(out: str, dist: dict[str, int]) -> set[str]:
    """Problems whose rows are all in ``out`` (a problem with dist d has exactly d rows).
    Rewrites ``out`` without any partial problem left by an interrupted run."""
    path = Path(out)
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count: dict[str, int] = {}
    for r in rows:
        count[r["problem_id"]] = count.get(r["problem_id"], 0) + 1
    done = {pid for pid, n in count.items() if n == dist.get(pid)}
    keep = [r for r in rows if r["problem_id"] in done]
    if len(keep) != len(rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    return done


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
