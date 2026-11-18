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


def test_spread_keeps_labels_apart_and_near_their_lines():
    assert figures.spread([1.0, 5.0, 9.0], 0.5) == [1.0, 5.0, 9.0]                 # far apart: unchanged
    pos = figures.spread([4.6, 6.538, 6.540], 0.5)
    assert pos[0] == 4.6 and pos[2] - pos[1] == pytest.approx(0.5)                # crowded pair split
    assert (pos[1] + pos[2]) / 2 == pytest.approx((6.538 + 6.540) / 2)            # around their mean
    pos = figures.spread([1.0, 1.1, 1.2, 3.0], 0.4)
    assert all(b - a >= 0.4 - 1e-9 for a, b in zip(sorted(pos), sorted(pos)[1:], strict=False))


def test_plot_draws_the_summary_values_and_is_deterministic(tmp_path):
    pytest.importorskip("matplotlib")
    t = figures.mean_prediction_table((ROOT / figures.SUMMARY).read_text(encoding="utf-8"))
    pa, drawn = figures.plot(t, tmp_path / "a.png", "x")
    pb, _ = figures.plot(t, tmp_path / "b.png", "x")
    assert pa.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and pa.read_bytes() == pb.read_bytes()
    assert drawn["d6"] == [r["d6"][0] for r in t["rows"]] and drawn["control"] == [r["control"] for r in t["rows"]]


def test_paper_pdf_is_single_column_size_and_deterministic(tmp_path):
    pytest.importorskip("matplotlib")
    t = figures.mean_prediction_table((ROOT / figures.SUMMARY).read_text(encoding="utf-8"))
    a = figures.plot(t, tmp_path / "a.pdf", "x", paper=True)[0].read_bytes()
    b = figures.plot(t, tmp_path / "b.pdf", "x", paper=True)[0].read_bytes()
    assert a.startswith(b"%PDF") and a == b
    assert b"/MediaBox [ 0 0 468 244.8 ]" in a                                    # 6.5 x 3.4 in
    assert figures.PAPER_SIZE[0] == 6.5
