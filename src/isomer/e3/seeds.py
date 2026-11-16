"""E3 seed study: the same diagnostics for h-v1 and its reseeded twins.

The report's findings rest on one checkpoint (section 13, "one model, one
seed"). This module holds the pieces the three ``scripts/e3_*`` drivers share,
so that every model is measured on the same labelled states with the same
commands:

* ``write_sample``: the E2 stratified sample of a split, taken from the built
  dataset by id. No relabelling: the records are the dataset's own lines.
* ``label_status``: is a child-label file complete for a sample? Scoring a
  half-labelled file would silently turn the population into a subset, which
  is exactly the deviation the first E3 session had to report.
* ``score_and_analyze``: ``diagnose score`` then ``analyze base`` for one model.
* ``bench_command``: the ``isomer bench`` command for one model.

torch is imported only inside ``score_and_analyze``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

MODELS = {"h-v1": "runs/h-v1/best.pt", "h-v1-s1": "runs/h-v1-s1/best.pt", "h-v1-s2": "runs/h-v1-s2/best.pt"}
SPLITS = ("test_id", "test_depth")
BUDGETS = (30, 100, 300, 1000)
DATA = "data/isobench-v2"
IDS = "docs/experiments/E2/ids-{split}.txt"
PRECISION = "bf16-fast"


class SeedsError(Exception):
    """A user-facing error: missing input, incomplete labels, a sample that does not match."""


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------------- samples

def header_sha256(ids_path: str | Path) -> str | None:
    """The dataset file hash an id file was sampled from (its ``sha256:`` header line)."""
    for line in Path(ids_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"#.*sha256:\s*([0-9a-f]{64})\s*$", line)
        if m:
            return m.group(1)
    return None


def select_sample(split_path: str | Path, ids: list[str]) -> list[str]:
    """The dataset lines of ``ids``, verbatim, in the order of the id file."""
    by_id: dict[str, str] = {}
    for line in Path(split_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            by_id[json.loads(line)["id"]] = line
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SeedsError(f"{len(missing)} id(s) are not in {split_path}: {', '.join(missing[:5])}")
    if len(set(ids)) != len(ids):
        raise SeedsError("the id file lists an id twice")
    return [by_id[i] for i in ids]


def write_sample(split: str, ids_path: str | Path, data: str | Path, out: str | Path) -> dict:
    """Write the sampled records of ``split`` to ``out`` and check them against the id file:
    same dataset file (hash in the header) and the same count in every stratum."""
    from .regen import read_ids, verify_strata

    split_path = Path(data) / f"{split}.jsonl"
    if not split_path.is_file():
        raise SeedsError(f"{split_path} not found")
    want, got = header_sha256(ids_path), sha256(split_path)
    if want is not None and want != got:
        raise SeedsError(f"{split_path} has sha256 {got}, but {ids_path} was sampled from {want}: "
                         f"this is not the dataset the E2 sample was drawn from")
    ids, strata = read_ids(ids_path)
    lines = select_sample(split_path, ids)
    check = verify_strata([json.loads(line) for line in lines], split, strata)
    if not check["match"]:
        raise SeedsError(f"{split}: strata differ from {ids_path}: {check['mismatch']}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)
    return {"split": split, "problems": len(lines), "dataset_sha256": got, "dataset_hash_checked": want is not None,
            "strata": check["counts"], "out": Path(out).as_posix()}


# ---------------------------------------------------------------------------------- labels

def label_status(sample_path: str | Path, labels_path: str | Path) -> dict:
    """How much of a sample is labelled. A problem with exact distance d is complete when the
    label file holds exactly d rows for it (one per on-path state with dist >= 1)."""
    want = {r["id"]: r["dist"] for r in _jsonl(sample_path) if r["dist"] is not None}
    rows = _jsonl(labels_path) if Path(labels_path).exists() else []
    count: dict[str, int] = {}
    for r in rows:
        count[r["problem_id"]] = count.get(r["problem_id"], 0) + 1
    done = sorted(pid for pid, d in want.items() if count.get(pid) == d)
    kept = [r for r in rows if r["problem_id"] in set(done)]
    return {"problems": len(want), "labelled": len(done), "complete": len(done) == len(want),
            "foreign_problems": len(set(count) - set(want)),
            "states": len(kept), "unknown_states": sum(r["status"] != "ok" for r in kept),
            "problems_with_unknown": len({r["problem_id"] for r in kept if r["status"] != "ok"})}


def require_complete(sample_path: str | Path, labels_path: str | Path) -> dict:
    st = label_status(sample_path, labels_path)
    if not st["complete"] or st["foreign_problems"]:
        raise SeedsError(f"{labels_path}: {st['labelled']}/{st['problems']} problems labelled, "
                         f"{st['foreign_problems']} not in the sample; finish scripts/e3_label.py first")
    return st


# ---------------------------------------------------------------------------------- scoring

def score_paths(out_dir: str | Path, split: str) -> dict[str, Path]:
    d = Path(out_dir)
    return {"scores": d / f"{split}.jsonl", "meta": d / f"{split}.score.json",
            "analysis": d / f"{split}.analysis.json"}


def score_and_analyze(model: str, ckpt: str | Path, split: str, labels: str | Path, out_dir: str | Path,
                      device: str = "cuda", precision: str = PRECISION, redo: bool = False) -> dict:
    """``diagnose score`` then ``analyze base`` for one model on one labelled split.

    The score file is written under a temporary name and renamed when complete, so an
    interrupted run can never leave a truncated file that a later step takes for the
    whole population. Finished scores are reused unless ``redo``."""
    import torch

    from . import analyze, diagnose

    p = score_paths(out_dir, split)
    p["scores"].parent.mkdir(parents=True, exist_ok=True)
    labels_hash, ckpt_hash = sha256(labels), sha256(ckpt)
    meta = json.loads(p["meta"].read_text(encoding="utf-8")) if p["meta"].exists() else None
    stale = meta is None or meta.get("labels_sha256") != labels_hash or meta.get("ckpt_sha256") != ckpt_hash
    if redo or stale or not p["scores"].exists():
        cuda = device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tmp = p["scores"].with_name(p["scores"].name + ".tmp")
        meta = diagnose.score(str(labels), str(tmp), str(ckpt), device, precision)
        meta.update(model=model, split=split, ckpt=Path(ckpt).as_posix(), ckpt_sha256=ckpt_hash,
                    labels=Path(labels).as_posix(), labels_sha256=labels_hash,
                    max_memory_allocated=torch.cuda.max_memory_allocated() if cuda else None,
                    gpu=torch.cuda.get_device_name() if cuda else None, torch=torch.__version__)
        os.replace(tmp, p["scores"])
        p["meta"].write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    res = analyze.base(str(p["scores"]))
    p["analysis"].write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return {"model": model, "split": split, "states": res["overall"]["states"],
            "problems": res["overall"]["problems"], "top1_optimal": res["overall"]["h"]["top1_optimal"],
            "symbol_bag_top1": res["overall"]["symbol_bag"]["top1_optimal"],
            "seq_per_second": meta.get("seq_per_second"), "max_memory_allocated": meta.get("max_memory_allocated")}


# ---------------------------------------------------------------------------------- search

def method_name(ckpt: str, precision: str = PRECISION) -> str:
    return f"neural-w1@{ckpt}#{precision}"


def bench_command(ckpt: str, out: str, split: str = "test_depth", budgets: tuple[int, ...] = BUDGETS,
                  workers: int = 2, python: str = sys.executable) -> list[str]:
    """`isomer bench` for one model: the E2 command with only the checkpoint and output changed."""
    return [python, "-m", "isomer", "bench", "--data", DATA, "--split", split, "--ids", IDS.format(split=split),
            "--methods", method_name(ckpt), "--budgets", ",".join(str(b) for b in budgets),
            "--workers", str(workers), "--out", out]
