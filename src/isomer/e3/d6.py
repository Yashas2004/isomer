"""IsoBench v2-d6: v2 with train problems of exact distance up to 6 instead of 4.

Purpose (E3): every seed of the h-v1 configuration saturates just below 5, the
largest distance of a training state. Training on deeper problems tests whether
the training range causes that.

Why not ``isomer data`` with ``train_max_dist=6``: ``build._split`` uses that
field for the eval pool too (``test_depth`` keeps ``dist > train_max_dist``),
so a full build would drop the distance 5-6 problems from test_depth and the
benchmark would no longer be the one E2 and E3 measured. Generation and
labelling do not depend on ``train_max_dist``; only the split rule does. So:

* the **train pool** is regenerated and relabelled exactly as in the v2 build
  (``regen.generate_jobs`` replays the generation order, ``build._label`` is
  the build's labeller) and split with the build's own rule at distance 6;
* **test_depth, test_deep and test_theory** are byte-for-byte copies of v2.

Nothing is written unless three checks pass (``checks`` in datasheet.json):

1. replay: the relabelled pool split at distance 4 is v2's ``train.jsonl`` and
   ``test_id.jsonl``, byte for byte. This is what shows the regenerated pool is
   the v2 pool and not a look-alike.
2. eval identity: the copied files have the hashes recorded in the committed
   E2 id files (an anchor that does not depend on the local v2 directory).
3. no leakage: d6 train shares no id with any test split of v2 or d6, and the
   v2 train and test_id sets are subsets of the d6 ones.

Records carry no timings and files are sorted by id, so a rebuild is
byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from ..data.build import FORMAT_VERSION, BuildConfig, _hist, _label, _split
from .seeds import header_sha256

TRAIN_MAX_DIST = 6
EVAL_SPLITS = ("test_depth", "test_deep", "test_theory")
TRAIN_SPLITS = ("train", "test_id")


class D6Error(Exception):
    """A check failed: the variant would not be what it claims to be."""


def serialise(records: list[dict]) -> bytes:
    """The bytes ``data.build`` writes for a split: sorted by id, one sorted-key JSON object per
    line, in text mode (so the platform's line ending, as in the existing v2 files)."""
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(records, key=lambda r: r["id"]))
    return text.replace("\n", os.linesep).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids(data: bytes) -> set[str]:
    return {json.loads(line)["id"] for line in data.decode("utf-8").splitlines() if line.strip()}


def split_train_pool(records: list[dict], max_dist: int) -> tuple[dict[str, list[dict]], int]:
    """The build's own split rule for train-pool records, at ``max_dist``. Returns the two
    splits and the number of dropped problems (no exact distance, or deeper than ``max_dist``)."""
    cfg = replace(BuildConfig(), train_max_dist=max_dist)
    out: dict[str, list[dict]] = {s: [] for s in TRAIN_SPLITS}
    dropped = 0
    for rec in records:
        dest = _split("train", rec, cfg)
        if dest is None:
            dropped += 1
        else:
            out[dest].append(rec)
    return out, dropped


def check_replay(records: list[dict], v2: Path) -> dict:
    """The relabelled pool, split as v2 was, must be v2's train and test_id files exactly."""
    splits, _ = split_train_pool(records, BuildConfig().train_max_dist)
    out = {}
    for name in TRAIN_SPLITS:
        want, got = _sha((v2 / f"{name}.jsonl").read_bytes()), _sha(serialise(splits[name]))
        if want != got:
            raise D6Error(f"replay check failed: regenerated {name} has sha256 {got}, {v2 / (name + '.jsonl')} "
                          f"has {want}; the train pool is not the one v2 was built from")
        out[name] = {"sha256": got, "problems": len(splits[name])}
    return out


def check_eval(v2: Path, ids_dir: Path) -> dict:
    """The eval splits to be copied are the ones the committed E2 samples were drawn from."""
    out = {}
    for name in EVAL_SPLITS:
        data = (v2 / f"{name}.jsonl").read_bytes()
        got, want = _sha(data), header_sha256(ids_dir / f"ids-{name}.txt")
        if want is None:
            raise D6Error(f"{ids_dir / ('ids-' + name + '.txt')} records no dataset hash")
        if got != want:
            raise D6Error(f"{v2 / (name + '.jsonl')} has sha256 {got}; the E2 id file was sampled from {want}")
        out[name] = {"sha256": got, "problems": len(_ids(data))}
    return out


def check_leakage(splits: dict[str, list[dict]], v2: Path) -> dict:
    train = {r["id"] for r in splits["train"]}
    test_id = {r["id"] for r in splits["test_id"]}
    v2_ids = {name: _ids((v2 / f"{name}.jsonl").read_bytes()) for name in (*TRAIN_SPLITS, *EVAL_SPLITS)}
    overlaps = {"d6 test_id": len(train & test_id), "v2 test_id": len(train & v2_ids["test_id"]),
                **{name: len(train & v2_ids[name]) for name in EVAL_SPLITS}}
    bad = {k: n for k, n in overlaps.items() if n}
    if bad:
        raise D6Error(f"leakage: d6 train shares ids with {bad}")
    if not v2_ids["train"] <= train:
        raise D6Error(f"{len(v2_ids['train'] - train)} v2 train problems are missing from d6 train")
    if not v2_ids["test_id"] <= test_id:
        raise D6Error(f"{len(v2_ids['test_id'] - test_id)} v2 test_id problems are missing from d6 test_id")
    return {"train_overlap_with": overlaps, "v2_train_subset_of_d6_train": True,
            "v2_test_id_subset_of_d6_test_id": True,
            "train_added": len(train - v2_ids["train"]), "test_id_added": len(test_id - v2_ids["test_id"])}


def label_train_pool(workers: int, progress: bool = True) -> list[dict]:
    """Regenerate and relabel v2's train pool (40,000 problems; about 10 minutes)."""
    import time

    from .regen import generate_jobs

    t0 = time.perf_counter()
    jobs = list(generate_jobs(set(), BuildConfig(), keep_pool="train").values())
    if progress:
        print(f"  [train] generated {len(jobs)} problems ({time.perf_counter() - t0:.0f}s)", flush=True)
    records: list[dict] = []
    every = max(1, len(jobs) // 20)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, rec in enumerate(ex.map(_label, jobs, chunksize=16), 1):
            rec.pop("_seconds")
            records.append(rec)
            if progress and (k % every == 0 or k == len(jobs)):
                print(f"  [train] labelled {k}/{len(jobs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return records


def build_d6(records: list[dict], v2: str | Path, out: str | Path, ids_dir: str | Path) -> dict:
    """Check, then write the variant from relabelled train-pool ``records``. Returns the datasheet."""
    v2, out, ids_dir = Path(v2), Path(out), Path(ids_dir)
    if out.exists() and any(out.iterdir()):
        raise D6Error(f"{out} already exists and is not empty")
    if out.resolve() == v2.resolve():
        raise D6Error("refusing to write into the v2 directory")
    splits, dropped = split_train_pool(records, TRAIN_MAX_DIST)
    checks = {"replay_of_v2_train_pool": check_replay(records, v2),
              "eval_splits_match_e2_id_files": check_eval(v2, ids_dir),
              "leakage": check_leakage(splits, v2)}

    files = {name: serialise(splits[name]) for name in TRAIN_SPLITS}
    files.update({name: (v2 / f"{name}.jsonl").read_bytes() for name in EVAL_SPLITS})
    parsed = {name: [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
              for name, data in files.items()}
    v2_sheet = json.loads((v2 / "datasheet.json").read_text(encoding="utf-8"))
    cfg = replace(BuildConfig(), train_max_dist=TRAIN_MAX_DIST, out=out.as_posix())
    sheet = {
        "format": FORMAT_VERSION,
        "variant": "v2-d6",
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items() if k != "workers"},
        "label_budget": v2_sheet["label_budget"],       # read by `isomer bench` for its circularity guard
        "counts": {k: len(v) for k, v in parsed.items()},
        "by_theory": {k: dict(sorted(Counter(r["theory"] for r in v).items())) for k, v in parsed.items()},
        "dist_histogram": {k: _hist([r["dist"] for r in v]) for k, v in parsed.items() if k != "test_deep"},
        "generated": {"train": len(records)},
        "dropped": {"train (dist unknown or > train_max_dist)": dropped},
        "label_stop": {"train": dict(sorted(Counter(r["label_stop"] for r in records).items()))},
        "sha256": {k: _sha(v) for k, v in files.items()},
        "derived_from": {"dataset": v2.as_posix(), "copied_unchanged": list(EVAL_SPLITS),
                         "relabelled": "train pool (train, test_id), same generation seeds and labeller as v2"},
        "checks": checks,
        "notes": [
            (f"IsoBench v2 with train_max_dist = {TRAIN_MAX_DIST} applied to the train pool only. test_depth, "
             "test_deep and test_theory are byte-for-byte the v2 files: in a full build train_max_dist would "
             "also remove the distance 5-6 problems from test_depth."),
            ("test_id is a superset of v2's test_id (it gains held-out problems of distance 5-6). The E2 "
             "samples in docs/experiments/E2 remain valid and held out for models trained on this variant."),
            "all other properties are those of v2; see data/isobench-v2/datasheet.json and ADR-004.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / f"{name}.jsonl").write_bytes(data)
    (out / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True), encoding="utf-8")
    return sheet
