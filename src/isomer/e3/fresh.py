"""E3-fresh: a new evaluation pool for a confirmatory test of the E3 claims.

    python scripts/e3_build_fresh.py      # -> data/isobench-fresh/, docs/experiments/E3/ids-test_depth_fresh.txt

Every E3 result so far was measured on the E2 test_depth sample, which was in
view while the analyses and thresholds were chosen. This module draws problems
that did not exist then:

* **Generation:** ``data.build._generate`` with the v2 eval-pool settings of
  ``BuildConfig()`` (theories group and bool, ``eval_per_theory`` each, walk
  lengths ``eval_walk_lengths``) and the stream ``_seed(FRESH_SEED, "eval",
  theory)``. v2 and d6 used seed 0. Every id in ``EXCLUDE_FROM`` is passed as
  ``external``, so a colliding walk is replaced by the next one (counted).
* **Labelling:** ``data.build._label`` with the eval budget (20,000
  expansions, 1,500,000-node cap): the same labeller and the same kernel check
  of every stored proof.
* **Split rule:** the test_depth rule, exact distance > ``train_max_dist``.
* **Sample:** exactly the per-distance counts of the E2 test_depth sample
  (``SAMPLE_COUNTS``), ranked within a distance by ``sha256(seed|split|id)`` as
  ``data.stratify`` does. A short stratum is an error, not a smaller sample.

Nothing in ``BuildConfig`` changes: the build uses a config *instance* with
``seed`` and ``out`` replaced. Output files hold no timings, so the same inputs
give byte-identical files; timings are only printed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from ..data.build import (
    FORMAT_VERSION,
    BuildConfig,
    LabelJob,
    _generate,
    _hist,
    _label,
    _seed,
    external_ids,
    problem_id,
)

FRESH_SEED = 1
SPLIT = "test_depth_fresh"
OUT = "data/isobench-fresh"
EXCLUDE_FROM = ("data/isobench-v2", "data/isobench-v2-d6")
E2_IDS = "docs/experiments/E2/ids-test_depth.txt"
IDS_OUT = "docs/experiments/E3/ids-test_depth_fresh.txt"
# Per-distance counts of the E2 test_depth sample (its header); checked against E2_IDS at build time.
SAMPLE_COUNTS = {5: 70, 6: 97, 7: 58, 8: 52, 9: 19, 10: 4}
PARTIAL = "labelled.partial.jsonl"      # resume file while labelling; removed when the build finishes


class FreshError(Exception):
    """The build cannot go on: a check failed or the inputs are not what they should be."""


def fresh_config(workers: int | None = None) -> BuildConfig:
    """The v2 defaults with only the seed and output directory replaced."""
    cfg = replace(BuildConfig(), seed=FRESH_SEED, out=OUT)
    return cfg if workers is None else replace(cfg, workers=workers)


def e2_counts(path: str | Path = E2_IDS) -> dict[int, int]:
    """Sampled problems per distance in an E2 id file's header."""
    from .regen import read_ids

    _, strata = read_ids(path)
    return {int(k): v for k, v in sorted(strata.items(), key=lambda kv: int(kv[0])) if v > 0}


# ---------------------------------------------------------------------------------- generation

def generate_jobs(cfg: BuildConfig, external: frozenset[str]) -> tuple[list[LabelJob], dict]:
    """The eval pool of ``cfg`` (seeded by ``cfg.seed``), avoiding every id in ``external``."""
    jobs: list[LabelJob] = []
    claimed: set[str] = set()
    stats: dict = {"generated": {}, "walks_rejected_outside_size_cap": {}, "replaced_at_generation": {}}
    for name in cfg.train_theories:
        problems, rej, exc = _generate(name, cfg.eval_per_theory, cfg.eval_walk_lengths, cfg,
                                       _seed(cfg.seed, "eval", name), exclude=claimed, external=external)
        stats["generated"][name] = len(problems)
        stats["walks_rejected_outside_size_cap"][name] = rej
        stats["replaced_at_generation"][name] = exc
        for p in problems:
            lhs, rhs = str(p.problem.lhs), str(p.problem.rhs)
            claimed.add(problem_id(name, lhs, rhs))
            jobs.append(LabelJob("eval", name, lhs, rhs, [s.to_json() for s in p.steps], cfg.label_budget,
                                 cfg.label_max_nodes, cfg.pool_max_size, cfg.max_size_slack))
    return jobs, stats


def job_id(job: LabelJob) -> str:
    return problem_id(job.theory, job.lhs, job.rhs)


def label_all(jobs: list[LabelJob], workers: int, partial: Path, progress: bool = True) -> list[dict]:
    """Label every job, one problem per task. Finished records are appended to ``partial`` as
    they arrive, so an interrupted build resumes where it stopped (``_label`` is a pure function
    of the job, so a record from an earlier attempt is the record this attempt would write)."""
    want = {job_id(j): j for j in jobs}
    done: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["id"] not in want:
                    raise FreshError(f"{partial} holds problem {r['id']}, which this build does not generate; "
                                     "it is from another configuration: delete it")
                done[r["id"]] = r
        if progress:
            print(f"  resuming: {len(done)} of {len(want)} problems already labelled", flush=True)
    todo = [j for pid, j in want.items() if pid not in done]
    t0 = time.perf_counter()
    every = max(1, len(todo) // 40)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex, open(partial, "a", encoding="utf-8", newline="\n") as f:
            futures = [ex.submit(_label, j) for j in todo]
            for k, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                rec.pop("_seconds")
                done[rec["id"]] = rec
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                if progress and (k % every == 0 or k == len(todo)):
                    print(f"  labelled {k}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return [done[pid] for pid in sorted(done)]


def keep(rec: dict, cfg: BuildConfig) -> bool:
    """The test_depth rule of ``data.build._split``: exact distance beyond the train range."""
    return rec["dist"] is not None and rec["dist"] > cfg.train_max_dist


# ---------------------------------------------------------------------------------- checks

def collisions(ids: set[str], dirs: tuple[str, ...] = EXCLUDE_FROM, root: str | Path = ".") -> dict[str, int]:
    """{"dir/split.jsonl": number of ``ids`` in that file} for every split of every directory.
    Generation already avoids these ids; this is the independent check on what is written."""
    out: dict[str, int] = {}
    for d in dirs:
        files = sorted((Path(root) / d).glob("*.jsonl"))
        if not files:
            raise FreshError(f"{d}: no *.jsonl splits found; cannot check disjointness")
        for f in files:
            with open(f, encoding="utf-8") as fh:
                theirs = {json.loads(line)["id"] for line in fh if line.strip()}
            out[f"{d}/{f.name}"] = len(ids & theirs)
    return out


# ---------------------------------------------------------------------------------- sample

def rank(seed: int, split: str, pid: str) -> str:
    """The within-stratum order of ``data.stratify``."""
    return hashlib.sha256(f"{seed}|{split}|{pid}".encode()).hexdigest()


def sample(records: list[dict], counts: dict[int, int], seed: int = FRESH_SEED,
           split: str = SPLIT) -> tuple[list[str], dict[int, tuple[int, int]]]:
    """Sorted ids with exactly ``counts[d]`` problems of distance d, and {d: (population, sampled)}
    for every distance in the pool (0 sampled where ``counts`` has none)."""
    groups: dict[int, list[str]] = {}
    for r in records:
        groups.setdefault(r["dist"], []).append(r["id"])
    short = {d: (len(groups.get(d, [])), n) for d, n in counts.items() if len(groups.get(d, [])) < n}
    if short:
        raise FreshError("too few problems for the fixed sample: "
                         + ", ".join(f"distance {d}: {have} available, {n} needed" for d, (have, n) in short.items()))
    ids: list[str] = []
    for d, n in counts.items():
        ids += sorted(groups[d], key=lambda pid: rank(seed, split, pid))[:n]
    per = {d: (len(groups.get(d, [])), counts.get(d, 0)) for d in sorted(set(groups) | set(counts))}
    return sorted(ids), per


def ids_text(ids: list[str], per: dict[int, tuple[int, int]], split_sha256: str, command: str,
             seed: int = FRESH_SEED, split: str = SPLIT) -> str:
    """An id file in the format of ``data.stratify`` (read by ``regen.read_ids``, ``bench --ids``
    and ``seeds.write_sample``)."""
    header = [f"# {command}",
              f"# split: {split}  n: {len(ids)}  strata: dist  seed: {seed}  counts: fixed to {E2_IDS}",
              f"# dataset format: {FORMAT_VERSION}  {split}.jsonl sha256: {split_sha256}",
              "# stratum: population -> sampled"]
    header += [f"#   {d}: {pop} -> {n}" for d, (pop, n) in per.items()]
    return "\n".join(header + ids) + "\n"


# ---------------------------------------------------------------------------------- build

def datasheet(cfg: BuildConfig, stats: dict, labelled: list[dict], kept: list[dict], external: frozenset[str],
              clash: dict[str, int], split_sha256: str, per: dict[int, tuple[int, int]]) -> dict:
    dropped = Counter("dist unknown" if r["dist"] is None else "dist <= train_max_dist"
                      for r in labelled if not keep(r, cfg))
    return {
        "format": FORMAT_VERSION,
        "purpose": ("E3-fresh confirmatory evaluation pool: the v2 eval pool's generator and labeller with a new "
                    "seed; ids disjoint from every split of " + " and ".join(EXCLUDE_FROM)),
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items() if k != "workers"},
        "label_budget": {SPLIT: cfg.label_budget},         # read by `isomer bench` for its circularity guard
        "counts": {SPLIT: len(kept)},
        "by_theory": {SPLIT: dict(sorted(Counter(r["theory"] for r in kept).items()))},
        **stats,
        "labelled": len(labelled),
        "dropped": dict(sorted(dropped.items())),
        "dist_histogram_all_labelled": _hist([r["dist"] for r in labelled]),
        "dist_histogram": {SPLIT: _hist([r["dist"] for r in kept])},
        "label_stop": dict(sorted(Counter(r["label_stop"] for r in labelled).items())),
        "max_nodes_stored": max((r["label_nodes"] for r in labelled), default=0),
        "excluded_ids": {"from": list(EXCLUDE_FROM), "n_ids": len(external),
                         "ids_sha256": hashlib.sha256("\n".join(sorted(external)).encode()).hexdigest()},
        "collisions_with_excluded_splits": clash,
        "split_sha256": {SPLIT: split_sha256},
        "sample": {"ids_file": IDS_OUT, "rank_seed": FRESH_SEED, "counts_from": E2_IDS,
                   "strata": {str(d): {"population": p, "sampled": n} for d, (p, n) in per.items()}},
        "notes": [
            "every stored proof was accepted by the trusted kernel at build time (data.build._label).",
            (f"{SPLIT} keeps the problems with exact dist > {cfg.train_max_dist} (the test_depth rule); problems "
             "with a shorter or unknown distance are counted under dropped and not stored."),
            "no timings are stored, so the same inputs give byte-identical files.",
        ],
    }


def build(root: str | Path = ".", workers: int | None = None, progress: bool = True,
          command: str = "python scripts/e3_build_fresh.py") -> dict:
    """Generate, label, check, then write the pool, its datasheet and the sample id file.
    Every check runs before the first output file is written."""
    root = Path(root)
    cfg = fresh_config(workers)
    out, ids_out = root / cfg.out, root / IDS_OUT
    if (out / f"{SPLIT}.jsonl").exists() or ids_out.exists():
        raise FreshError(f"{(out / f'{SPLIT}.jsonl').as_posix()} or {IDS_OUT} already exists; "
                         "the build never overwrites an evaluation pool")
    counts = e2_counts(root / E2_IDS)
    if counts != SAMPLE_COUNTS:
        raise FreshError(f"{E2_IDS} has counts {counts}, expected {SAMPLE_COUNTS}")

    t0 = time.perf_counter()
    external = frozenset().union(*(external_ids(str(root / d)) for d in EXCLUDE_FROM))
    jobs, stats = generate_jobs(cfg, external)
    if progress:
        print(f"  generated {len(jobs)} problems ({time.perf_counter() - t0:.0f}s); "
              f"{sum(stats['replaced_at_generation'].values())} walks replaced because their id is in "
              f"{' or '.join(EXCLUDE_FROM)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    labelled = label_all(jobs, cfg.workers, out / PARTIAL, progress)
    kept = [r for r in labelled if keep(r, cfg)]           # sorted by id (label_all)

    clash = collisions({r["id"] for r in labelled}, EXCLUDE_FROM, root)
    if any(clash.values()):
        raise FreshError(f"fresh ids collide with existing splits: { {k: v for k, v in clash.items() if v} }")
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept).encode("utf-8")
    split_sha = hashlib.sha256(body).hexdigest()
    ids, per = sample(kept, SAMPLE_COUNTS)

    (out / f"{SPLIT}.jsonl").write_bytes(body)
    sheet = datasheet(cfg, stats, labelled, kept, external, clash, split_sha, per)
    (out / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                        newline="\n")
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    ids_out.write_text(ids_text(ids, per, split_sha, command), encoding="utf-8", newline="\n")
    (out / PARTIAL).unlink()
    if progress:
        print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return sheet
