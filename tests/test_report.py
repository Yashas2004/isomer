import subprocess
import sys

import pytest

from isomer.bench.report import (
    fair_gap,
    holm,
    identical_methods,
    mcnemar_exact,
    method_colors,
    paired,
    report,
)
from isomer.cli import main


def row(method, pid, solved, proof_len=None, dist=2, expansions=10, **kw):
    r = {"method": method, "id": pid, "theory": "group", "max_budget": 100, "status": "ok",
         "solved": solved, "proof_len": proof_len, "dist": dist, "expansions": expansions,
         "nodes_stored": 5, "stop": "solved" if solved else "budget"}
    r.update(kw)
    return r


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 2 ** 10)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.6, 0.7]) == [1.0, 1.0]
    assert holm([]) == []
    ps = [0.001, 0.2, 0.015, 0.04, 0.5]
    adj = holm(ps)
    by_p = [adj[i] for i in sorted(range(len(ps)), key=lambda i: ps[i])]
    assert by_p == sorted(by_p) and all(a >= p for a, p in zip(adj, ps)) and max(adj) <= 1.0


def test_fair_gap_uses_only_commonly_solved():
    rows = [
        row("good", "easy", True, proof_len=2), row("lazy", "easy", True, proof_len=3),
        # Only "good" solves the hard one, with a long proof: must not count.
        row("good", "hard", True, proof_len=12, dist=4), row("lazy", "hard", False),
        # Unknown dist never counts.
        row("good", "unk", True, proof_len=5, dist=None), row("lazy", "unk", True, proof_len=5, dist=None),
    ]
    n, gaps = fair_gap(rows)
    assert n == 1
    assert gaps == {"good": 1.0, "lazy": 1.5}
    assert fair_gap([row("a", "p", False), row("b", "p", True, proof_len=2)])[1] == {"a": None, "b": None}


def test_identical_methods_and_paired():
    rows = []
    for m in ("x", "y"):
        rows += [row(m, "p1", True, proof_len=2), row(m, "p2", False)]
    rows += [row("z", "p1", True, proof_len=2), row("z", "p2", True, proof_len=3)]
    assert identical_methods(rows) == [["x", "y"]]
    assert paired(rows, "z", "x", 100) == (1, 0)
    assert paired(rows, "x", "z", 100) == (0, 1)
    assert paired(rows, "z", "x", 5) == (0, 0)  # expansions 10 > 5: nobody solves


def test_method_colors_stable():
    a = method_colors(["ucs", "bfs-fwd", "bfs-bidir"])
    assert a == method_colors(["bfs-bidir", "ucs", "bfs-fwd"])
    assert list(a) == ["bfs-bidir", "bfs-fwd", "ucs"] and len(set(a.values())) == 3


def test_report_does_not_import_matplotlib():
    code = "import sys, isomer.bench.report, isomer.cli; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.fixture(scope="module")
def run_dir(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "r"
    assert main(["bench", "--data", str(tiny_data), "--split", "train", "--methods", "bfs-fwd,ucs,bfs-bidir",
                 "--budgets", "30,100,300", "--workers", "1", "--out", str(out)]) == 0
    return out


def test_markdown_report(run_dir):
    assert report(run_dir, plots=False) == 0
    md = (run_dir / "report" / "results.md").read_text(encoding="utf-8")
    for m in ("bfs-fwd", "ucs", "bfs-bidir"):
        assert f"`{m}`" in md
    for heading in ("## Solve rate", "## Fair optimality gap", "## Paired comparison vs `bfs-fwd`",
                    "## Per theory", "## Notes"):
        assert heading in md
    assert "error rows: 0, unsound rows: 0" in md and "**" in md


def test_full_report_with_plots(run_dir):
    pytest.importorskip("matplotlib")
    assert main(["report", str(run_dir)]) == 0
    for name in ("results.md", "solve_by_dist.png", "cactus.png", "nodes_vs_solved.png"):
        f = run_dir / "report" / name
        assert f.is_file() and f.stat().st_size > 0, name


def test_report_cli_errors(run_dir, tmp_path, capsys):
    assert main(["report", str(run_dir), "--reference", "nope"]) == 2
    assert "bfs-bidir" in capsys.readouterr().err
    assert main(["report", str(tmp_path / "missing")]) == 2
    assert "results.jsonl" in capsys.readouterr().err
