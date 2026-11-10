"""Regenerate the problems of a published stratified sample (``docs/experiments/E2/ids-*.txt``)
without rebuilding all of IsoBench v2.

Generation is cheap and deterministic; labelling is the expensive part. This
module replays ``data.build``'s generation in exactly the same order
(held-out, then eval, then train, with the same per-(pool, theory) seeds and
the same ``claimed`` exclusion), keeps only the requested ids, and labels just
those with the same labeller settings as the build (``build._label``).

The check that this reproduces the published problems is strong but indirect:
every requested id must be generated (ids are content hashes of the two
sides), and every labelled problem must fall in the stratum the id file
records for it (``verify_strata``). A record is byte-for-byte what
``build._label`` would write.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..data.build import BuildConfig, LabelJob, _generate, _label, _seed, problem_id

SPLIT_POOL = {"test_id": "train", "test_depth": "eval", "test_deep": "eval", "test_theory": "heldout"}


def read_ids(path: str | Path) -> tuple[list[str], dict[str, int]]:
    """Problem ids and the stratum header (stratum -> number sampled)."""
    ids: list[str] = []
    strata: dict[str, int] = {}
    in_strata = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# stratum:"):
            in_strata = True
        elif line.startswith("#") and in_strata and "->" in line:
            key, _, rest = line[1:].strip().partition(":")
            strata[key.strip()] = int(rest.split("->")[1])
        elif line.strip() and not line.startswith("#"):
            ids.append(line.strip())
    return ids, strata


def generate_jobs(wanted: set[str], cfg: BuildConfig | None = None,
                  keep_pool: str | None = None) -> dict[str, LabelJob]:
    """LabelJobs for every wanted id, replaying the build's generation order.
    With ``keep_pool``, every problem of that pool is returned instead (``wanted`` ignored)."""
    cfg = cfg or BuildConfig()
    plan = {
        "heldout": (cfg.heldout_theories, cfg.heldout_per_theory, cfg.heldout_walk_lengths, cfg.label_budget),
        "eval": (cfg.train_theories, cfg.eval_per_theory, cfg.eval_walk_lengths, cfg.label_budget),
        "train": (cfg.train_theories, cfg.train_per_theory, cfg.train_walk_lengths, cfg.train_label_budget),
    }
    claimed: set[str] = set()
    found: dict[str, LabelJob] = {}
    for pool in ("heldout", "eval", "train"):
        if keep_pool is None and not wanted - set(found):
            break
        theories, n, walks, budget = plan[pool]
        for name in theories:
            problems, _, _ = _generate(name, n, walks, cfg, _seed(cfg.seed, pool, name), exclude=claimed)
            for p in problems:
                lhs, rhs = str(p.problem.lhs), str(p.problem.rhs)
                pid = problem_id(name, lhs, rhs)
                claimed.add(pid)
                if (pool == keep_pool) if keep_pool is not None else (pid in wanted):
                    found[pid] = LabelJob(pool, name, lhs, rhs, [s.to_json() for s in p.steps], budget,
                                          cfg.label_max_nodes, cfg.pool_max_size, cfg.max_size_slack)
    return found


def stratum_of(rec: dict, split: str) -> str:
    if split == "test_deep":
        return str(rec["dist_lb"])
    if split == "test_theory":
        return f"{rec['theory']}|{rec['dist']}"
    return str(rec["dist"])


def verify_strata(records: list[dict], split: str, strata: dict[str, int]) -> dict:
    got = Counter(stratum_of(r, split) for r in records)
    mismatch = {k: {"expected": strata.get(k, 0), "got": got.get(k, 0)}
                for k in set(strata) | set(got) if strata.get(k, 0) != got.get(k, 0)}
    return {"match": not mismatch, "mismatch": mismatch, "counts": dict(sorted(got.items()))}


def regen_split(split: str, ids_path: str | Path, out: str | Path, workers: int = 2) -> dict:
    ids, strata = read_ids(ids_path)
    jobs = generate_jobs(set(ids))
    missing = sorted(set(ids) - set(jobs))
    ordered = [jobs[i] for i in ids if i in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        records = []
        for rec in ex.map(_label, ordered, chunksize=1):
            rec["_label_seconds"] = rec.pop("_seconds")
            records.append(rec)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in records)
    return {"split": split, "requested": len(ids), "missing": missing,
            "strata": verify_strata(records, split, strata)}


if __name__ == "__main__":   # python -m isomer.e3.regen SPLIT IDS OUT [WORKERS]
    import sys

    res = regen_split(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    print(json.dumps(res, indent=1))
