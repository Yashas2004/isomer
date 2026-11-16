"""A size-matched subset of a training-states directory (E3-d6 control).

    python scripts/e3_build_matched.py      # -> data/states-v2-d6-matched

h-d6 was trained on deeper problems than h-v1, but also on more of them
(36,045 train problems against 29,268). To separate depth coverage from data
volume, this builds a states directory with the d6 distance range and the v2
number of train problems: a seeded random subset of the d6 train problems,
every state of each kept problem, validation file unchanged.

The subset is the ``n`` problems with the smallest ``sha256(seed|purpose|id)``.
That is a uniform random sample that depends only on the seed and the ids, not
on file order, so a rebuild is byte-identical. Kept lines are copied verbatim.

What is matched is the number of train problems, and with it the number of
optimizer steps (the trainer batches by problem). The number of *states* is
not matched: d6 problems are deeper and have more states each. ``stats.json``
records both, so the summary can say so.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

PURPOSE = "states-matched"


class MatchedError(Exception):
    """A user-facing error: missing input, too few problems, non-empty output."""


def _rank(seed: int, problem_id: str) -> str:
    return hashlib.sha256(f"{seed}|{PURPOSE}|{problem_id}".encode()).hexdigest()


def select_problems(ids: set[str], n: int, seed: int) -> set[str]:
    """``n`` of ``ids``, uniformly at random, as a pure function of (seed, ids)."""
    if n > len(ids):
        raise MatchedError(f"cannot keep {n} problems: the source has only {len(ids)}")
    return set(sorted(ids, key=lambda i: (_rank(seed, i), i))[:n])


def _hist(counter: Counter) -> dict[str, int]:
    return {str(k): v for k, v in sorted(counter.items())}


def subset_stats(rows: list[dict]) -> dict:
    """Counts of the kept train states. A problem's distance is that of its start state,
    the deepest on-path state."""
    from ..learn.tokenizer import n_tokens

    problem_dist: dict[str, int] = {}
    for r in rows:
        if r["on_path"]:
            problem_dist[r["problem_id"]] = max(problem_dist.get(r["problem_id"], 0), r["dist"])
    return {"problems": len({r["problem_id"] for r in rows}), "states": len(rows),
            "on_path": sum(r["on_path"] for r in rows), "off_path": sum(not r["on_path"] for r in rows),
            "by_theory_dist": {f"{t}|{d}": v for (t, d), v in sorted(Counter((r["theory"], r["dist"])
                                                                              for r in rows).items())},
            "dist_histogram": _hist(Counter(r["dist"] for r in rows)),
            "problem_dist_histogram": _hist(Counter(problem_dist.values())),
            "max_tokens": max((n_tokens(r["state"], r["goal"]) for r in rows), default=0)}


def build_matched(src: str | Path, out: str | Path, n_problems: int, seed: int = 0,
                  matched_to: str | None = None) -> dict:
    """Write ``out`` (train.jsonl, val.jsonl, stats.json) from the states directory ``src``."""
    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()):
        raise MatchedError(f"{out} already exists and is not empty")
    if out.resolve() == src.resolve():
        raise MatchedError("refusing to write into the source directory")
    for name in ("train.jsonl", "val.jsonl", "stats.json"):
        if not (src / name).is_file():
            raise MatchedError(f"{src / name} not found")
    raw = (src / "train.jsonl").read_bytes()
    lines = [line for line in raw.split(b"\n") if line.strip()]
    ids = [json.loads(line)["problem_id"] for line in lines]
    keep = select_problems(set(ids), n_problems, seed)
    kept = [line for line, pid in zip(lines, ids, strict=True) if pid in keep]
    train = b"".join(line + b"\n" for line in kept)
    val = (src / "val.jsonl").read_bytes()
    st = subset_stats([json.loads(line) for line in kept])
    assert st["problems"] == n_problems
    src_stats = json.loads((src / "stats.json").read_text(encoding="utf-8"))
    val_by = src_stats["by_theory_dist"]["val"]
    val_hist: Counter = Counter()
    for key, v in val_by.items():
        val_hist[int(key.split("|")[1])] += v
    stats = {
        "format": src_stats["format"],
        "config": {**src_stats["config"], "out": out.as_posix()},
        "source": src_stats["source"],      # the IsoBench split the states were generated from (read by the trainer)
        "subset": {"of": src.as_posix(), "of_train_sha256": hashlib.sha256(raw).hexdigest(), "seed": seed,
                   "train_problems": n_problems, "of_train_problems": len(set(ids)), "matched_to": matched_to,
                   "rule": f"the train problems with the {n_problems} smallest sha256('<seed>|{PURPOSE}|<id>'); "
                           "every state of a kept problem is kept; val.jsonl is copied unchanged",
                   "kept_ids_sha256": hashlib.sha256("\n".join(sorted(keep)).encode()).hexdigest()},
        "problems": {"train": st["problems"], "val": src_stats["problems"]["val"]},
        "counts": {"train": st["states"], "val": src_stats["counts"]["val"]},
        "on_off_path": {"train": {"on_path": st["on_path"], "off_path": st["off_path"]},
                        "val": src_stats["on_off_path"]["val"]},
        "by_theory_dist": {"train": st["by_theory_dist"], "val": val_by},
        "dist_histogram": {"train": st["dist_histogram"], "val": _hist(val_hist)},
        "problem_dist_histogram": {"train": st["problem_dist_histogram"]},
        "dropped_in_source": src_stats.get("dropped", {}),
        "max_tokens": max(st["max_tokens"], 0),
        "sha256": {"train": hashlib.sha256(train).hexdigest(), "val": hashlib.sha256(val).hexdigest()},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_bytes(train)
    (out / "val.jsonl").write_bytes(val)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats
