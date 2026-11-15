"""E3 sample regeneration: id-file parsing and the stratum check that guards a regenerated sample.

Only the pure parts are tested here; ``regen_split`` itself replays the full
generation and is exercised by the E3 runs (report section 3).
"""

from isomer.e3.regen import read_ids, stratum_of, verify_strata

IDS_FILE = """\
# python -m isomer stratify --data data/isobench-v2 --split test_theory --n 6 --strata theory_dist --seed 0
# split: test_theory  n: 6  strata: theory_dist  seed: 0
# dataset format: isobench/2  test_theory.jsonl sha256: 00ff
# stratum: population -> sampled
#   lattice|2: 47 -> 2
#   lattice|None: 24 -> 1
#   ring|3: 55 -> 3
0063e7b00bf66312

00d017f286f04dda
  01dc6aa5c35fa57d
02b84142503d3612
0406adcbd5909c49
052fd980d4b775d4
"""


def _write(tmp_path, text):
    path = tmp_path / "ids.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_read_ids_parses_ids_and_stratum_header(tmp_path):
    ids, strata = read_ids(_write(tmp_path, IDS_FILE))
    # file order is kept (it is the labelling order), blank lines and indentation are ignored
    assert ids == ["0063e7b00bf66312", "00d017f286f04dda", "01dc6aa5c35fa57d", "02b84142503d3612",
                   "0406adcbd5909c49", "052fd980d4b775d4"]
    # the sampled count is the number after "->", not the population
    assert strata == {"lattice|2": 2, "lattice|None": 1, "ring|3": 3}
    assert sum(strata.values()) == len(ids)


def test_read_ids_ignores_arrows_before_the_stratum_header(tmp_path):
    text = "# note: a -> 5\n# stratum: population -> sampled\n#   5: 281 -> 1\nabc\n"
    ids, strata = read_ids(_write(tmp_path, text))
    assert ids == ["abc"]
    assert strata == {"5": 1}


def test_read_ids_survives_crlf(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_bytes(IDS_FILE.replace("\n", "\r\n").encode("utf-8"))   # a Windows checkout with autocrlf
    assert read_ids(path) == read_ids(_write(tmp_path, IDS_FILE))


def test_read_ids_without_header(tmp_path):
    assert read_ids(_write(tmp_path, "aa\nbb\n")) == (["aa", "bb"], {})


def test_stratum_of_uses_the_key_of_each_split():
    rec = {"theory": "ring", "dist": 7, "dist_lb": 7, "dist_ub": 7}
    assert stratum_of(rec, "test_id") == "7"
    assert stratum_of(rec, "test_depth") == "7"
    assert stratum_of(rec, "test_theory") == "ring|7"
    # test_deep has no exact distance: it is stratified by the lower bound
    deep = {"theory": "bool", "dist": None, "dist_lb": 11, "dist_ub": 14}
    assert stratum_of(deep, "test_deep") == "11"
    # unlabelled held-out problems form their own stratum, spelled as in the id file
    assert stratum_of({"theory": "lattice", "dist": None}, "test_theory") == "lattice|None"


def _theory_records():
    return ([{"theory": "lattice", "dist": 2}] * 2 + [{"theory": "lattice", "dist": None}]
            + [{"theory": "ring", "dist": 3}] * 3)


def test_verify_strata_accepts_the_published_counts():
    res = verify_strata(_theory_records(), "test_theory", {"lattice|2": 2, "lattice|None": 1, "ring|3": 3})
    assert res == {"match": True, "mismatch": {},
                   "counts": {"lattice|2": 2, "lattice|None": 1, "ring|3": 3}}


def test_verify_strata_reports_every_kind_of_mismatch():
    records = _theory_records()
    records[0] = {"theory": "lattice", "dist": 4}      # relabelled into a stratum the id file does not have
    expected = {"lattice|2": 2, "lattice|None": 1, "ring|3": 3, "ring|9": 1}   # ring|9 never regenerated
    res = verify_strata(records, "test_theory", expected)
    assert res["match"] is False
    assert res["mismatch"] == {"lattice|2": {"expected": 2, "got": 1},
                               "lattice|4": {"expected": 0, "got": 1},
                               "ring|9": {"expected": 1, "got": 0}}
    assert res["counts"] == {"lattice|2": 1, "lattice|4": 1, "lattice|None": 1, "ring|3": 3}


def test_verify_strata_empty():
    assert verify_strata([], "test_depth", {}) == {"match": True, "mismatch": {}, "counts": {}}
