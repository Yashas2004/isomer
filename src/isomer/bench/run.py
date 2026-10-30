"""Benchmark runner: every (method, problem) once, kernel-checked, logged.

**One run, many budgets.** Each (method, problem) is run once at the largest
requested budget B. The result at a smaller budget b is derived from it:
``solved@b = solved and expansions <= b``. This is exact, not an approximation,
because every searcher satisfies the *prefix property* (see
``isomer.search.common``): its choices do not depend on the budget, and the
budget is checked only before an expansion, so a run that solves with e
expansions solves identically under any b >= e and fails under any b < e.

**Wall-clock limits break the prefix property**, so they are handled
explicitly. A run stopped by time after e expansions tells us only that the
problem is unsolved for every b <= e. For b > e we do not know, so that cell is
*censored*: excluded from n and counted separately in the summary. A node-cap
stop (``--max-nodes``, ``stop == "memory"``) is treated the same way. Without
``--max-seconds`` and ``--max-nodes`` nothing is ever censored.

**Circularity guard.** bfs-bidir labels the data. If the largest benchmark
budget exceeds a tenth of the split's labelling budget (``datasheet.json``),
bidirectional BFS results risk being circular (E1, ADR-004); the run prints a
WARNING and records ``circularity_warning`` in ``config.json``.

**Solved means kernel-accepted.** The worker rebuilds the certificate from the
dataset record's own goal (not from whatever problem the searcher claims to
have solved) and runs ``kernel.check_certificate``. A proof the kernel
rejects is recorded with status ``"unsound"``; it counts as unsolved, and the
run ends with a loud warning and a non-zero exit code. A searcher that raises
is recorded with status ``"error"``. It also counts as unsolved and makes the
exit code non-zero, but it does not kill the run.

**Neural methods** (``neural-w1@CKPT`` etc., ``search.neural``) run the model
on the GPU if one is available. config.json records every checkpoint's path
and sha256. Each worker process holds its own CUDA context (~0.5 GB) and a
copy of the model, so a run containing a neural method defaults to 2 workers
instead of all cores but one; ``--workers`` overrides that.

``results.jsonl`` is reproducible except for the ``seconds`` and
``model_seconds`` columns, which depend on the machine.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core import kernel
from ..core.proof import Problem, Proof
from ..core.theories import THEORIES
from ..search.common import Budget, SearchResult, get_searcher
from ..search.neural import checkpoint_of, is_neural

# Optional per-row stats, copied from SearchResult.stats when a searcher reports them.
EXTRA_STATS = ("model_calls", "model_seconds", "device", "gpu_peak_bytes")
NEURAL_DEFAULT_WORKERS = 2


class BenchError(Exception):
    """A user-facing error (bad arguments, missing files): reported, not raised."""


@dataclass
class BenchConfig:
    data: str
    split: str
    methods: list[str]
    budgets: list[int]
    out: str
    limit: int | None = None
    workers: int = 0
    max_seconds: float | None = None
    max_nodes: int | None = None
    overwrite: bool = False
    argv: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_problems(data: str, split: str, limit: int | None = None) -> list[dict]:
    path = Path(data) / f"{split}.jsonl"
    if not path.is_file():
        have = sorted(p.stem for p in Path(data).glob("*.jsonl")) if Path(data).is_dir() else []
        raise BenchError(f"no split file {path}; available splits: {', '.join(have) or '(none)'}")
    with open(path, encoding="utf-8") as f:
        records = sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["id"])
    for r in records:
        if r["theory"] not in THEORIES:
            raise BenchError(f"record {r['id']} has unknown theory {r['theory']!r}; "
                             f"valid: {', '.join(sorted(THEORIES))}")
    return records[:limit] if limit is not None else records


def _problem(record: dict) -> Problem:
    return Problem.parse(THEORIES[record["theory"]][0], record["lhs"], record["rhs"])


# --------------------------------------------------------------------------
# One (method, problem): runs in a worker process
# --------------------------------------------------------------------------

def _base_row(record: dict, split: str, method: str, max_budget: int) -> dict:
    return {
        "id": record["id"], "theory": record["theory"], "split": split,
        "dist": record.get("dist"), "dist_lb": record.get("dist_lb"),
        "method": method, "max_budget": max_budget,
        "solved": False, "proof_len": None, "proof": None,
        "expansions": None, "generated": None, "nodes_stored": None, "stop": None, "seconds": None,
    }


def evaluate(record: dict, split: str, method: str, max_budget: int, result: SearchResult) -> dict:
    """Turn a search result into a row. The kernel gate lives here: ``solved``
    is True only if the kernel accepts the steps *for the record's own goal*."""
    row = _base_row(record, split, method, max_budget)
    row.update(expansions=result.expansions, generated=result.generated,
               nodes_stored=result.stats.get("nodes_stored"), stop=result.stats.get("stop"),
               seconds=round(result.seconds, 4), status="ok")
    row.update({k: result.stats[k] for k in EXTRA_STATS if k in result.stats})
    if result.proof is not None:
        proof = Proof(_problem(record), result.proof.steps)
        verdict = kernel.check_certificate(proof.certificate())
        row["proof"] = [s.to_json() for s in proof.steps]
        row["proof_len"] = len(proof.steps)
        if verdict:
            row["solved"] = True
        else:
            row["status"] = "unsound"
            row["reason"] = f"kernel: {verdict.reason} (step {verdict.failed_step})"
    return row


def _run_one(args: tuple[str, dict, str, Budget]) -> dict:
    """Worker entry point. Plain arguments only (picklable on Windows); the
    searcher is looked up by name here. Never raises: a failure becomes an
    ``"error"`` row so one bad problem cannot kill a long run."""
    method, record, split, budget = args
    max_budget = budget.max_expansions
    try:
        result = get_searcher(method)(_problem(record), budget)
        return evaluate(record, split, method, max_budget, result)
    except Exception as exc:  # noqa: BLE001 -- recorded, reported, and fails the run
        row = _base_row(record, split, method, max_budget)
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" | {Path(tb[-1].filename).name}:{tb[-1].lineno} in {tb[-1].name}" if tb else ""
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}{where}")
        return row


# --------------------------------------------------------------------------
# Deriving results at smaller budgets (pure; reused by the report step)
# --------------------------------------------------------------------------

def cell(row: dict, b: int) -> str:
    """Outcome of ``row`` at budget ``b``: "solved", "unsolved" or "censored"."""
    if b > row["max_budget"]:
        return "censored"
    if row["stop"] in ("time", "memory") and b > row["expansions"]:
        return "censored"
    if row["status"] == "ok" and row["solved"] and row["expansions"] <= b:
        return "solved"
    return "unsolved"  # includes "unsound" and "error" rows


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).
    Unlike the normal approximation it behaves at 0/n and n/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Exact endpoints at 0/n and n/n: floating-point rounding otherwise gives
    # lo ~ 5e-17 or hi just below 1, which puts the estimate outside its own interval.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def summarize(rows: list[dict], budgets: list[int]) -> list[dict]:
    """One entry per (method, budget), methods in first-seen order."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out: list[dict] = []
    for m in methods:
        mrows = [r for r in rows if r["method"] == m]
        for b in sorted(budgets):
            cells = [(r, cell(r, b)) for r in mrows]
            solved = [r for r, c in cells if c == "solved"]
            n = sum(c != "censored" for _, c in cells)
            gaps = [r["proof_len"] / r["dist"] for r in solved if r["dist"]]
            lo, hi = wilson(len(solved), n)
            out.append({
                "method": m, "budget": b, "n": n, "solved": len(solved),
                "rate": len(solved) / n if n else None, "ci_lo": lo, "ci_hi": hi,
                "median_expansions": statistics.median(r["expansions"] for r in solved) if solved else None,
                "median_nodes_stored": statistics.median(r["nodes_stored"] for r in solved) if solved else None,
                "opt_gap": sum(gaps) / len(gaps) if gaps else None,
                "censored": len(cells) - n,
            })
    return out


def format_summary(summary: list[dict]) -> str:
    def f(x: object, spec: str) -> str:
        return "" if x is None else format(x, spec)

    head = f"{'method':26s} {'budget':>7s} {'n':>5s} {'solved':>6s} {'rate':>6s} {'95% CI':>15s} " \
           f"{'med_exp':>8s} {'opt_gap':>7s} {'cens':>5s}"
    lines = [head, "-" * len(head)]
    for s in summary:
        ci = f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
        lines.append(f"{s['method']:26s} {s['budget']:7d} {s['n']:5d} {s['solved']:6d} "
                     f"{f(s['rate'], '.3f'):>6s} {ci:>15s} {f(s['median_expansions'], '.0f'):>8s} "
                     f"{f(s['opt_gap'], '.3f'):>7s} {s['censored']:5d}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        cp = subprocess.run(["git", *args], check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return cp.stdout.strip() if cp.returncode == 0 else None


def label_budget_of(data: str, split: str) -> int | None:
    """Expansion budget bfs-bidir had when labelling ``split``. v2 datasheets
    record it per split; v0/v1 had one ``bfs_budget`` for everything."""
    try:
        sheet = json.loads((Path(data) / "datasheet.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    per_split = sheet.get("label_budget")
    if isinstance(per_split, dict) and split in per_split:
        return int(per_split[split])
    legacy = sheet.get("config", {}).get("bfs_budget")
    return int(legacy) if legacy is not None else None


def circularity_warning(max_budget: int, label_budget: int | None) -> bool | None:
    """True if benchmark budgets come within 10x of the labelling budget, where
    bfs-bidir's results stop being independent of how the data was labelled.
    None if the labelling budget is unknown."""
    return None if label_budget is None else max_budget > label_budget / 10


def _config_json(cfg: BenchConfig, budget: Budget, n_problems: int, workers: int) -> dict:
    sheet = Path(cfg.data) / "datasheet.json"
    try:
        fmt = json.loads(sheet.read_text(encoding="utf-8")).get("format", "unknown")
    except (OSError, ValueError):
        fmt = "unknown"
    status = _git("status", "--porcelain")
    return {
        "argv": list(cfg.argv),
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "budget": asdict(budget),
        "budgets": sorted(cfg.budgets),
        "methods": list(cfg.methods),
        "data": cfg.data,
        "split": cfg.split,
        "limit": cfg.limit,
        "workers": workers,
        "n_problems": n_problems,
        "dataset_format": fmt,
        "checkpoints": {m: {"path": checkpoint_of(m),
                            "sha256": hashlib.sha256(Path(checkpoint_of(m)).read_bytes()).hexdigest()}
                        for m in cfg.methods if is_neural(m)},
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(cfg: BenchConfig) -> int:
    """Returns the process exit code: 0 ok, 1 if any row is unsound or errored.
    Raises ``BenchError`` for bad input, before any work is done."""
    # 1. Validate everything up front: a typo should not cost an hour of compute.
    if not cfg.budgets or any(b <= 0 for b in cfg.budgets):
        raise BenchError(f"budgets must be positive integers, got {cfg.budgets}")
    if not cfg.methods:
        raise BenchError("no methods given")
    for m in cfg.methods:
        try:
            get_searcher(m)
        except KeyError as exc:
            raise BenchError(exc.args[0]) from None
        if is_neural(m) and not Path(checkpoint_of(m)).is_file():
            raise BenchError(f"{m}: checkpoint {checkpoint_of(m)} not found")
    records = load_problems(cfg.data, cfg.split, cfg.limit)
    out = Path(cfg.out)
    if out.exists() and not cfg.overwrite:
        raise BenchError(f"{out} already exists; pass --overwrite to replace its results")

    budget = Budget(max_expansions=max(cfg.budgets), max_seconds=cfg.max_seconds, max_nodes=cfg.max_nodes)
    jobs = [(m, r, cfg.split, budget) for m in sorted(cfg.methods) for r in records]
    if cfg.workers:
        workers = cfg.workers
    elif any(is_neural(m) for m in cfg.methods):
        workers = NEURAL_DEFAULT_WORKERS   # one CUDA context per worker
    else:
        workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"bench: {len(cfg.methods)} methods x {len(records)} problems ({cfg.split}), "
          f"max budget {budget.max_expansions}, {workers} worker(s)", flush=True)
    label_budget = label_budget_of(cfg.data, cfg.split)
    warn = circularity_warning(budget.max_expansions, label_budget)
    if warn is None:
        print(f"note: no labelling budget in {cfg.data}/datasheet.json; circularity not checked", flush=True)
    elif warn:
        print(f"WARNING: circularity risk: max budget {budget.max_expansions} > label budget {label_budget} / 10 "
              f"for split {cfg.split} (bfs-bidir is the labeller; see ADR-004)", flush=True)

    # 2. Run. workers=1 stays in-process: deterministic and easy to debug.
    t0 = time.perf_counter()
    every = max(1, math.ceil(len(jobs) / 20))   # ~5% steps
    rows: list[dict] = []

    def tick(k: int) -> None:
        if k % every == 0 or k == len(jobs):
            print(f"  {k}/{len(jobs)} ({100 * k // max(1, len(jobs))}%, {time.perf_counter() - t0:.0f}s)",
                  flush=True)

    if workers == 1:
        for k, job in enumerate(jobs, 1):
            rows.append(_run_one(job))
            tick(k)
    else:
        chunk = max(1, len(jobs) // (workers * 20))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, row in enumerate(ex.map(_run_one, jobs, chunksize=chunk), 1):
                rows.append(row)
                tick(k)

    # 3. Write logs.
    rows.sort(key=lambda r: (r["method"], r["id"]))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "config.json").write_text(
        json.dumps(_config_json(cfg, budget, len(records), workers) | {
            "label_budget": label_budget, "circularity_warning": warn}, indent=2, sort_keys=True),
        encoding="utf-8")

    # 4. Report.
    summary = summarize(rows, cfg.budgets)
    print()
    print(format_summary(summary))
    print(f"censored cells: {sum(s['censored'] for s in summary)}")
    print(f"results: {out / 'results.jsonl'}")

    errors = [r for r in rows if r["status"] == "error"]
    unsound = [r for r in rows if r["status"] == "unsound"]
    if errors:
        print(f"\nERRORS ({len(errors)} rows; counted as unsolved):", file=sys.stderr)
        for r in errors:
            print(f"  {r['method']}  {r['id']}  {r['reason']}", file=sys.stderr)
    if unsound:
        bar = "!" * 72
        print(f"\n{bar}\n!!! UNSOUND PROOFS: {len(unsound)} proof(s) REJECTED by the kernel !!!\n"
              f"!!! A searcher claimed success it did not earn. Counted as unsolved. !!!\n{bar}",
              file=sys.stderr)
        for r in unsound:
            print(f"  {r['method']}  {r['id']}  {r['reason']}", file=sys.stderr)
        print(bar, file=sys.stderr)
    return 1 if (errors or unsound) else 0
