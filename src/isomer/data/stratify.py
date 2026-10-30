"""Stratified id samples of a dataset split, for pre-registered experiments.

The sample is fixed before any method runs and committed as a text file, so
every method in an experiment sees exactly the same problems
(``isomer bench --ids``).

* **Strata:** ``dist`` (exact distance), ``dist_lb`` (certified lower bound,
  for test_deep) or ``theory_dist`` (theory x dist). A null value is its own
  stratum, ``None``.
* **Allocation:** proportional with largest remainder, at least 1 per
  non-empty stratum, never more than a stratum holds. Ties break by stratum
  key, so the result is deterministic.
* **Within a stratum,** ids are ranked by ``sha256(seed|split|id)`` and the
  first k are taken, so the choice does not depend on file order.

Only the split's ``id`` and stratum fields are read; no labels or proofs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

STRATA = {
    "dist": lambda r: str(r["dist"]),
    "dist_lb": lambda r: str(r["dist_lb"]),
    "theory_dist": lambda r: f"{r['theory']}|{r['dist']}",
}


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of ``n`` over strata with largest remainder,
    >= 1 per non-empty stratum and <= its population."""
    counts = {k: c for k, c in counts.items() if c > 0}
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"cannot sample {n} from {total} problems")
    if n < len(counts):
        raise ValueError(f"n = {n} is smaller than the number of non-empty strata ({len(counts)})")
    quota = {k: n * c / total for k, c in counts.items()}
    alloc = {k: max(1, math.floor(q)) for k, q in quota.items()}
    rem = {k: quota[k] - math.floor(quota[k]) for k in counts}
    left = n - sum(alloc.values())
    # hand out what is left by largest remainder; take back an overshoot (from the
    # >= 1 minimum) from the smallest remainders, never below 1
    for k in sorted(counts, key=lambda k: (-rem[k], k)):
        if left <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            left -= 1
    for k in sorted(counts, key=lambda k: (rem[k], k)):
        if left >= 0:
            break
        if alloc[k] > 1:
            alloc[k] -= 1
            left += 1
    while left > 0:   # every remainder used and still short: fill wherever there is room, by key
        k = next(k for k in sorted(counts) if alloc[k] < counts[k])
        alloc[k] += 1
        left -= 1
    assert sum(alloc.values()) == n
    return dict(sorted(alloc.items()))


def _rank(seed: int, split: str, pid: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|{pid}".encode()).hexdigest()


def stratified_sample(records: list[dict], split: str, n: int, strata: str, seed: int) -> tuple[list[str], dict]:
    """Sorted sampled ids and per-stratum ``{key: (population, sampled)}``."""
    key = STRATA[strata]
    groups: dict[str, list[str]] = {}
    for r in records:
        groups.setdefault(key(r), []).append(r["id"])
    alloc = allocate({k: len(v) for k, v in groups.items()}, n)
    ids: list[str] = []
    for k, k_n in alloc.items():
        ids += sorted(groups[k], key=lambda pid: _rank(seed, split, pid))[:k_n]
    return sorted(ids), {k: (len(groups[k]), alloc[k]) for k in sorted(groups)}


def write_sample(data: str, split: str, n: int, strata: str, seed: int, out: str, command: str) -> dict:
    path = Path(data) / f"{split}.jsonl"
    raw = path.read_bytes()
    fields = ("id", "theory", "dist", "dist_lb")
    records = [{k: r.get(k) for k in fields}
               for r in (json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())]
    ids, per = stratified_sample(records, split, n, strata, seed)
    try:
        fmt = json.loads((Path(data) / "datasheet.json").read_text(encoding="utf-8")).get("format", "unknown")
    except (OSError, ValueError):
        fmt = "unknown"
    header = [f"# {command}", f"# split: {split}  n: {n}  strata: {strata}  seed: {seed}",
              f"# dataset format: {fmt}  {split}.jsonl sha256: {hashlib.sha256(raw).hexdigest()}",
              "# stratum: population -> sampled"]
    header += [f"#   {k}: {pop} -> {s}" for k, (pop, s) in per.items()]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header + ids) + "\n")
    return {"ids": ids, "strata": per}


def read_ids(path: str | Path) -> list[str]:
    """Ids from a sample file: one per line; ``#`` comments and blank lines skipped."""
    return [s for line in Path(path).read_text(encoding="utf-8").splitlines()
            if (s := line.strip()) and not s.startswith("#")]
