import json

import pytest

from isomer.bench.run import _problem, cell, circularity_warning, evaluate, label_budget_of, summarize, wilson
from isomer.cli import main
from isomer.core.proof import Problem, Proof
from isomer.core.rewrite import Step
from isomer.core.theories import THEORIES
from isomer.data.build import FORMAT_VERSION
from isomer.search.common import SEARCHERS, Budget, SearchResult

SPLIT = "train"
BUDGETS = "30,100,300"
REQUIRED = {"id", "theory", "split", "dist", "dist_lb", "method", "max_budget", "solved", "proof_len",
            "expansions", "generated", "nodes_stored", "stop", "seconds", "status"}


@pytest.fixture(scope="module")
def data(tiny_data):
    return tiny_data


def bench(data, out, methods="bfs-bidir,ucs", *extra):
    return main(["bench", "--data", str(data), "--split", SPLIT, "--methods", methods,
                 "--budgets", BUDGETS, "--workers", "1", "--out", str(out), *extra])


def read_rows(out):
    return [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def records(data):
    return [json.loads(line) for line in (data / f"{SPLIT}.jsonl").read_text(encoding="utf-8").splitlines()]


# Module-level fake searchers (in-process with --workers 1).
def _unsound_searcher(problem, budget):
    return SearchResult(Proof(problem, (Step("inv_l", ()),)), 1, 1, 0.0, stats={"nodes_stored": 1, "stop": "solved"})


def _crashing_searcher(problem, budget):
    raise RuntimeError("boom")


@pytest.fixture(scope="module")
def run_dir(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "two"
    assert bench(data, out) == 0
    return out


def test_run_writes_rows_and_config(data, run_dir):
    rows = read_rows(run_dir)
    n = len(records(data))
    assert len(rows) == 2 * n
    assert [(r["method"], r["id"]) for r in rows] == sorted((r["method"], r["id"]) for r in rows)
    for r in rows:
        assert REQUIRED <= r.keys() and r["status"] == "ok" and r["max_budget"] == 300
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    for key in ("argv", "git_commit", "git_dirty", "python", "platform", "cpu_count", "timestamp_utc",
                "budget", "dataset_format"):
        assert key in cfg
    assert cfg["dataset_format"] == FORMAT_VERSION and cfg["budget"]["max_expansions"] == 300
    assert cfg["budget"]["max_nodes"] is None
    # tiny_data labels train with 300 expansions, so a 300 benchmark is flagged.
    assert cfg["label_budget"] == 300 and cfg["circularity_warning"] is True
    # Re-verify logged proofs from scratch: nothing from the runner is trusted.
    solved = [r for r in rows if r["solved"]]
    assert solved
    recs = {r["id"]: r for r in records(data)}
    for r in solved[:6]:
        rec = recs[r["id"]]
        problem = Problem.parse(THEORIES[rec["theory"]][0], rec["lhs"], rec["rhs"])
        proof = Proof(problem, tuple(Step.from_json(s) for s in r["proof"]))
        assert proof.verify() and len(proof.steps) == r["proof_len"]


def test_derived_budgets_equal_real_reruns(data, run_dir):
    recs = {r["id"]: r for r in records(data)}
    checked = 0
    for r in read_rows(run_dir):
        for b in (5, 30, 100):
            fresh = SEARCHERS[r["method"]](_problem(recs[r["id"]]), Budget(max_expansions=b))
            assert (cell(r, b) == "solved") == fresh.solved, (r["method"], r["id"], b)
            if fresh.solved:
                assert len(fresh.proof.steps) == r["proof_len"]
            checked += 1
    assert checked >= 30


def test_worker_pool_matches_in_process(data, run_dir, tmp_path):
    out = tmp_path / "pool"
    assert main(["bench", "--data", str(data), "--split", SPLIT, "--methods", "bfs-bidir,ucs",
                 "--budgets", BUDGETS, "--workers", "2", "--out", str(out)]) == 0

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "seconds"} for r in rows]

    assert strip(read_rows(out)) == strip(read_rows(run_dir))


def _row(**kw):
    base = {"method": "m", "max_budget": 1000, "status": "ok", "solved": False, "expansions": 10,
            "stop": "budget", "proof_len": None, "dist": None, "nodes_stored": 5}
    base.update(kw)
    return base


def test_censoring():
    timed = _row(stop="time", expansions=50)
    assert cell(timed, 30) == "unsolved" and cell(timed, 50) == "unsolved" and cell(timed, 100) == "censored"
    # A node-cap stop is censored above the expansions it reached, exactly like time.
    capped = _row(stop="memory", expansions=50)
    assert cell(capped, 30) == "unsolved" and cell(capped, 50) == "unsolved" and cell(capped, 100) == "censored"
    s = {x["budget"]: x for x in summarize([capped, _row(stop="budget", expansions=1000)], [50, 100])}
    assert (s[50]["n"], s[50]["censored"]) == (2, 0) and (s[100]["n"], s[100]["censored"]) == (1, 1)
    assert cell(_row(max_budget=100), 300) == "censored"
    assert cell(_row(status="unsound", solved=False, stop="solved"), 100) == "unsolved"
    assert cell(_row(status="error", expansions=None, stop=None), 100) == "unsolved"

    solved = _row(solved=True, expansions=20, stop="solved", proof_len=4, dist=2)
    s = {x["budget"]: x for x in summarize([timed, solved], [30, 100])}
    assert s[30]["n"] == 2 and s[30]["solved"] == 1 and s[30]["censored"] == 0
    assert s[100]["n"] == 1 and s[100]["solved"] == 1 and s[100]["censored"] == 1
    assert s[30]["opt_gap"] == 2.0 and s[30]["median_expansions"] == 20
    assert summarize([_row()], [10])[0]["opt_gap"] is None


def test_wilson():
    for (k, n), (lo, hi) in {(0, 10): (0.0, 0.2775), (10, 10): (0.7225, 1.0), (5, 10): (0.2366, 0.7634)}.items():
        a, b = wilson(k, n)
        assert a == pytest.approx(lo, abs=1e-4) and b == pytest.approx(hi, abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_contains_estimate_exhaustively():
    # Regression: rounding gave lo ~ 5.5e-17 at k=0, i.e. a negative error bar.
    for n in range(1, 301):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)
        assert wilson(0, n)[0] == 0.0 and wilson(n, n)[1] == 1.0


def test_unsound_proof_is_caught(data, tmp_path, monkeypatch, capsys):
    rec = records(data)[0]
    row = evaluate(rec, SPLIT, "fake", 10, _unsound_searcher(_problem(rec), Budget()))
    assert row["status"] == "unsound" and row["solved"] is False and "kernel" in row["reason"]

    monkeypatch.setitem(SEARCHERS, "fake-unsound", _unsound_searcher)
    assert bench(data, tmp_path / "u", "fake-unsound") == 1
    assert "UNSOUND" in capsys.readouterr().err
    assert all(r["status"] == "unsound" and not r["solved"] for r in read_rows(tmp_path / "u"))


def test_worker_error_does_not_kill_run(data, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(SEARCHERS, "fake-crash", _crashing_searcher)
    out = tmp_path / "e"
    assert bench(data, out, "bfs-bidir,fake-crash") == 1
    assert "ERRORS" in capsys.readouterr().err
    rows = read_rows(out)
    crash = [r for r in rows if r["method"] == "fake-crash"]
    good = [r for r in rows if r["method"] == "bfs-bidir"]
    assert len(crash) == len(good) == len(records(data))
    assert all(r["status"] == "error" and not r["solved"] and "RuntimeError: boom" in r["reason"] for r in crash)
    assert all(r["status"] == "ok" for r in good) and any(r["solved"] for r in good)


def test_refuses_to_overwrite(data, tmp_path, capsys):
    out = tmp_path / "o"
    assert bench(data, out, "bfs-bidir") == 0
    assert bench(data, out, "bfs-bidir") == 2
    assert "--overwrite" in capsys.readouterr().err
    assert bench(data, out, "bfs-bidir", "--overwrite") == 0


def test_clear_errors(data, tmp_path, capsys):
    assert bench(data, tmp_path / "x", "no-such-method") == 2
    assert "bfs-bidir" in capsys.readouterr().err  # lists valid names
    assert main(["bench", "--data", str(data), "--split", "nope", "--methods", "ucs",
                 "--budgets", "10", "--out", str(tmp_path / "y")]) == 2
    assert "test_theory" in capsys.readouterr().err  # lists available splits
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_circularity_guard_rule(tmp_path):
    assert circularity_warning(1000, 20000) is False and circularity_warning(2001, 20000) is True
    assert circularity_warning(1000, None) is None
    assert label_budget_of(str(tmp_path), "train") is None          # no datasheet
    (tmp_path / "datasheet.json").write_text(json.dumps({"config": {"bfs_budget": 3000}}), encoding="utf-8")
    assert label_budget_of(str(tmp_path), "test_depth") == 3000     # v0/v1: one budget for all splits
    (tmp_path / "datasheet.json").write_text(
        json.dumps({"label_budget": {"train": 3000, "test_depth": 20000}}), encoding="utf-8")
    assert label_budget_of(str(tmp_path), "test_depth") == 20000


def test_circularity_guard_end_to_end(data, tmp_path, capsys):
    out = tmp_path / "low"
    assert main(["bench", "--data", str(data), "--split", SPLIT, "--methods", "ucs",
                 "--budgets", "10,30", "--workers", "1", "--out", str(out)]) == 0
    assert "WARNING: circularity" not in capsys.readouterr().out
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["circularity_warning"] is False
    assert bench(data, tmp_path / "high", "ucs") == 0
    assert "WARNING: circularity risk" in capsys.readouterr().out


def test_max_nodes_flag_censors(data, tmp_path):
    out = tmp_path / "capped"
    assert bench(data, out, "bfs-fwd", "--max-nodes", "50") == 0
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["budget"]["max_nodes"] == 50
    stopped = [r for r in read_rows(out) if r["stop"] == "memory"]
    assert stopped
    for r in stopped:
        assert not r["solved"]
        assert cell(r, r["expansions"]) == "unsolved" and cell(r, r["expansions"] + 1) == "censored"
