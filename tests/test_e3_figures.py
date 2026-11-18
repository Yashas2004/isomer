"""E3 Figure 1: the plotted numbers are the committed summary's numbers."""

from pathlib import Path

import pytest

from isomer.e3 import figures

ROOT = Path(__file__).resolve().parent.parent


def test_reads_the_committed_fresh_table():
    t = figures.mean_prediction_table((ROOT / figures.SUMMARY).read_text(encoding="utf-8"))
    assert t["control"] == "h-d6m-s0"
    assert [r["k"] for r in t["rows"]] == list(range(1, 10))          # k = 10 has 1 scored state: left out
    k7 = next(r for r in t["rows"] if r["k"] == 7)
    assert k7["states"] == 126
    assert k7["baseline"] == (4.655, 4.651, 4.664) and k7["d6"] == (6.157, 6.025, 6.284) and k7["control"] == 6.032


def test_reads_the_e2_family_table_too():
    t = figures.mean_prediction_table((ROOT / "docs/experiments/E3/d6_family_summary.md").read_text(encoding="utf-8"))
    k1 = t["rows"][0]
    assert k1["d6"] == (-0.118, -0.148, -0.086)                      # negative values parse


def test_refuses_another_table():
    with pytest.raises(figures.FigureError, match="no"):
        figures.mean_prediction_table("# nothing here\n")
    bad = "### Mean prediction\n\n| k | states | h-v1 |\n|---:|---:|---:|\n| 1 | 3 | 0.5 |\n"
    with pytest.raises(figures.FigureError, match="unexpected columns"):
        figures.mean_prediction_table(bad)


def test_plot_writes_the_same_png_twice(tmp_path):
    pytest.importorskip("matplotlib")
    t = figures.mean_prediction_table((ROOT / figures.SUMMARY).read_text(encoding="utf-8"))
    a = figures.plot(t, tmp_path / "a.png", "x").read_bytes()
    b = figures.plot(t, tmp_path / "b.png", "x").read_bytes()
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and a == b
