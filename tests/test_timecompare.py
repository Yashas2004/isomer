import importlib.util
import json

import pytest

from isomer.bench.run import BenchError, wilson
from isomer.bench.timecompare import (
    censor_point,
    check_agreement,
    paired_time,
    state_length_table,
    time_cell,
    time_mcnemar,
    time_table,
    timecompare,
)
from isomer.cli import main

NEURAL_FULL = "neural-w1@runs/h-v1/best.pt#bf16-fast"


def row(method, pid, solved, seconds, expansions=10, stop=None, max_budget=1000, **kw):
    r = {"method": method, "id": pid, "theory": "group", "split": "test_x", "dist": 2, "dist_lb": None,
         "max_budget": max_budget, "status": "ok", "solved": solved, "seconds": seconds,
         "expansions": expansions, "proof_len": 2 if solved else None,
         "proof": [["r", pid]] if solved else None, "nodes_stored": 5, "max_state_tokens": 50,
         "stop": stop or ("solved" if solved else "budget")}
    r.update(kw)
    return r


# ---- censoring rules ---------------------------------------------------------

def test_solved_within_t():
    assert time_cell(row("m", "p", True, 2.0), 3) == "solved"
    assert time_cell(row("m", "p", True, 3.0), 3) == "solved"          # boundary: <= T


def test_solved_after_t_is_unsolved():
    assert time_cell(row("m", "p", True, 3.5), 3) == "unsolved"


def test_unsolved_still_running_at_t_is_unsolved():
    assert time_cell(row("m", "p", False, 5.0), 3) == "unsolved"
    assert time_cell(row("m", "p", False, 3.0), 3) == "unsolved"       # boundary: >= T


def test_unsolved_stopped_before_t_is_censored():
    assert time_cell(row("m", "p", False, 2.0), 3) == "censored"
    assert time_cell(row("m", "p", False, 2.0, stop="memory"), 3) == "censored"


def test_exhausted_is_never_censored():
    assert time_cell(row("m", "p", False, 0.1, stop="exhausted"), 30) == "unsolved"


def test_time_table_and_censor_point():
    rows = [row("m", "a", True, 0.5), row("m", "b", True, 5.0), row("m", "c", False, 2.0),
            row("m", "d", False, 40.0)]
    t1, t10 = time_table({"m": rows}, (1.0, 10.0))
    assert (t1["n"], t1["solved"], t1["censored"]) == (4, 1, 0)
    assert (t10["n"], t10["solved"], t10["censored"]) == (3, 2, 1)     # c stopped at 2 s < 10 s
    assert (t10["ci_lo"], t10["ci_hi"]) == wilson(2, 3)
    assert censor_point(rows) == 2.0
    assert censor_point([row("m", "a", True, 1.0)]) is None


# ---- McNemar excludes pairs censored for either method -----------------------------

def test_mcnemar_excludes_censored_pairs():
    neural = [row("n", "p1", True, 1.0), row("n", "p2", True, 1.0), row("n", "p3", False, 2.0),
              row("n", "p4", True, 1.0)]
    base = [row("b", "p1", False, 20.0), row("b", "p2", False, 3.0), row("b", "p3", True, 1.0),
            row("b", "p4", True, 1.0)]
    # At T = 10: p2 censored for the baseline, p3 censored for neural -> only p1, p4 used.
    assert paired_time(neural, base, 10.0) == (1, 0, 2)
    # At T = 1: nothing is censored (all unsolved rows ran >= 1 s).
    assert paired_time(neural, base, 1.0) == (2, 1, 4)
    tests = time_mcnemar({"n": neural, "b": base}, (1.0, 10.0), method="n", references=("b",))
    assert [(t["a"], t["b"], t["pairs"]) for t in tests] == [(2, 1, 4), (1, 0, 2)]
    assert all(t["p_holm"] >= t["p"] for t in tests)


# ---- agreement check -----------------------------------------------------------------

def _pair():
    main_rows = [row("bfs-fwd", "p1", True, 0.2, expansions=50), row("bfs-fwd", "p2", False, 1.0, expansions=1000)]
    time_rows = [row("bfs-fwd", "p1", True, 0.3, expansions=50, max_budget=10000),
                 row("bfs-fwd", "p2", True, 9.0, expansions=4000, max_budget=10000)]
    return main_rows, time_rows


def test_agreement_ok():
    main_rows, time_rows = _pair()
    assert check_agreement(main_rows, time_rows, [30, 100, 1000], ("bfs-fwd",)) == []


@pytest.mark.parametrize("change", [
    {"proof_len": 3}, {"expansions": 51}, {"solved": False, "proof": None, "proof_len": None, "stop": "budget"}])
def test_agreement_detects_changes_on_solved(change):
    main_rows, time_rows = _pair()
    time_rows[0].update(change)
    assert check_agreement(main_rows, time_rows, [30, 100, 1000], ("bfs-fwd",))


def test_agreement_detects_solve_only_in_time_run_within_budget():
    main_rows, time_rows = _pair()
    time_rows[1].update(expansions=900)
    assert check_agreement(main_rows, time_rows, [30, 100, 1000], ("bfs-fwd",))


def test_agreement_detects_missing_id():
    main_rows, time_rows = _pair()
    assert check_agreement(main_rows, time_rows[:1], [1000], ("bfs-fwd",))


# ---- state length ----------------------------------------------------------------------

def test_state_length_table():
    rows = [row(NEURAL_FULL, "a", True, 1, max_state_tokens=126), row(NEURAL_FULL, "b", False, 1, max_state_tokens=126),
            row(NEURAL_FULL, "c", True, 1, max_state_tokens=127),
            row("bestfirst-symbol_bag-w1", "a", True, 1, expansions=1001, max_state_tokens=10)]
    t = {(s["method"], s["group"]): s for s in state_length_table(rows, 1000, 126)}
    assert (t["neural-w1", "<= 126"]["n"], t["neural-w1", "<= 126"]["solved"]) == (2, 1)
    assert (t["neural-w1", "> 126"]["n"], t["neural-w1", "> 126"]["solved"]) == (1, 1)
    assert t["bestfirst-symbol_bag-w1", "<= 126"]["solved"] == 0      # needed 1001 > budget
    assert (t["neural-w1", "<= 126"]["ci_lo"], t["neural-w1", "<= 126"]["ci_hi"]) == wilson(1, 2)


# ---- end to end -----------------------------------------------------------------------

def _write_run(d, rows, budgets, workers, max_nodes=None):
    d.mkdir()
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = {"budgets": budgets, "split": "test_x", "ids_sha256": "abc", "workers": workers, "git_commit": "c0",
           "git_dirty": False, "platform": "Test-OS", "cpu_count": 4, "python": "3.10",
           "budget": {"max_nodes": max_nodes}}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _runs(tmp_path):
    main_rows, time_rows = [], []
    for k in range(4):
        pid = f"p{k}"
        main_rows.append(row(NEURAL_FULL, pid, k < 3, 0.5 + k, device="cuda"))
        for m in ("bfs-fwd", "bestfirst-symbol_bag-w1"):
            main_rows.append(row(m, pid, k == 0, 0.1 if k == 0 else 0.4, expansions=20 if k == 0 else 1000))
            time_rows.append(row(m, pid, k < 2, 0.1 if k == 0 else 2.0, expansions=20 if k == 0 else 4000,
                                 max_budget=10000))
    _write_run(tmp_path / "main", main_rows, [30, 100, 300, 1000], 2)
    _write_run(tmp_path / "time", time_rows, [1000, 10000], 4, 1500000)
    return tmp_path / "main", tmp_path / "time"


def test_cli_end_to_end(tmp_path):
    m, t = _runs(tmp_path)
    out = tmp_path / "report"
    assert main(["timecompare", "--main", str(m), "--time", str(t), "--out", str(out)]) == 0
    md = (out / "timecompare.md").read_text(encoding="utf-8")
    assert "different worker counts" in md and "Test-OS" in md and "censored" in md
    assert "descriptive, not causal" in (out / "state_length.md").read_text(encoding="utf-8").lower()
    res = json.loads((out / "timecompare.json").read_text(encoding="utf-8"))
    assert len(res["mcnemar"]) == 8 and len(res["time_table"]) == 12
    if importlib.util.find_spec("matplotlib") is not None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        assert (out / "cactus_time.png").is_file()


def test_mismatch_aborts_and_writes_nothing(tmp_path):
    m, t = _runs(tmp_path)
    rows = [json.loads(line) for line in (t / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["proof_len"] = 99
    (t / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "report"
    with pytest.raises(BenchError, match="disagrees"):
        timecompare(m, t, out)
    assert not out.exists()
    assert main(["timecompare", "--main", str(m), "--time", str(t), "--out", str(out)]) == 2
