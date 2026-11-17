"""E3 seed study helpers: sample selection, label completeness, bench command.

No model, no search, no labelling: the long jobs are run by hand from
``scripts/e3_*.py``. What is pinned here is that those jobs cannot quietly
measure a different population or a different command than intended.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from isomer.e3 import seeds

ROOT = Path(__file__).resolve().parent.parent


def _rec(pid, dist, theory="group"):
    return {"id": pid, "theory": theory, "dist": dist, "dist_lb": dist, "lhs": "a", "rhs": "b", "proof": []}


def _dataset(tmp_path, records):
    data = tmp_path / "data"
    data.mkdir()
    # compact separators and a non-sorted key order: the sample must copy lines, not re-serialise them
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    (data / "test_depth.jsonl").write_bytes(text.encode("utf-8"))
    return data, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ids(tmp_path, ids, strata, sha=None):
    lines = ["# python -m isomer stratify ...", "# split: test_depth  n: 3  strata: dist  seed: 0"]
    if sha:
        lines.append(f"# dataset format: isobench/2  test_depth.jsonl sha256: {sha}")
    lines.append("# stratum: population -> sampled")
    lines += [f"#   {k}: 99 -> {n}" for k, n in strata.items()]
    path = tmp_path / "ids.txt"
    path.write_text("\n".join(lines + ids) + "\n", encoding="utf-8")
    return path


RECORDS = [_rec("aa", 5), _rec("bb", 6), _rec("cc", 5), _rec("dd", 7), _rec("ee", 6)]


def test_write_sample_copies_the_dataset_lines_in_id_file_order(tmp_path):
    data, sha = _dataset(tmp_path, RECORDS)
    ids = _ids(tmp_path, ["dd", "aa", "cc"], {"5": 2, "7": 1}, sha)
    out = tmp_path / "out" / "test_depth.jsonl"
    info = seeds.write_sample("test_depth", ids, data, out)
    assert info["problems"] == 3 and info["dataset_hash_checked"] and info["strata"] == {"5": 2, "7": 1}
    source = (data / "test_depth.jsonl").read_bytes().split(b"\n")
    assert out.read_bytes() == b"\n".join([source[3], source[0], source[2]]) + b"\n"
    # same inputs, same bytes
    first = out.read_bytes()
    seeds.write_sample("test_depth", ids, data, out)
    assert out.read_bytes() == first


def test_write_sample_refuses_another_dataset(tmp_path):
    data, _ = _dataset(tmp_path, RECORDS)
    ids = _ids(tmp_path, ["aa"], {"5": 1}, "0" * 64)
    with pytest.raises(seeds.SeedsError, match="not the dataset"):
        seeds.write_sample("test_depth", ids, data, tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_write_sample_refuses_missing_ids_and_wrong_strata(tmp_path):
    data, sha = _dataset(tmp_path, RECORDS)
    with pytest.raises(seeds.SeedsError, match="not in"):
        seeds.write_sample("test_depth", _ids(tmp_path, ["aa", "zz"], {"5": 2}, sha), data, tmp_path / "o.jsonl")
    # "bb" has distance 6 in this dataset, the id file says the sample has two problems at 5
    with pytest.raises(seeds.SeedsError, match="strata differ"):
        seeds.write_sample("test_depth", _ids(tmp_path, ["aa", "bb"], {"5": 2}, sha), data, tmp_path / "o.jsonl")
    with pytest.raises(seeds.SeedsError, match="not found"):
        seeds.write_sample("test_id", _ids(tmp_path, ["aa"], {"5": 1}, sha), data, tmp_path / "o.jsonl")


def test_header_sha256(tmp_path):
    assert seeds.header_sha256(_ids(tmp_path, ["aa"], {"5": 1}, "ab" * 32)) == "ab" * 32
    assert seeds.header_sha256(_ids(tmp_path, ["aa"], {"5": 1})) is None


def test_published_id_files_carry_a_dataset_hash():
    for split in seeds.SPLITS:
        assert seeds.header_sha256(ROOT / seeds.IDS.format(split=split)) is not None


# ---------------------------------------------------------------------------------- labels

def _label_rows(pid, n, unknown=()):
    return [{"problem_id": pid, "i": i, "status": "unknown" if i in unknown else "ok"} for i in range(n)]


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_label_status_counts_only_whole_problems(tmp_path):
    sample = _jsonl(tmp_path / "sample.jsonl", [_rec("aa", 2), _rec("bb", 3), _rec("cc", None)])
    labels = tmp_path / "labels.jsonl"
    st = seeds.label_status(sample, labels)             # nothing labelled yet
    assert (st["problems"], st["labelled"], st["complete"], st["states"]) == (2, 0, False, 0)

    _jsonl(labels, _label_rows("aa", 2) + _label_rows("bb", 2))      # bb was cut short
    st = seeds.label_status(sample, labels)
    assert (st["labelled"], st["complete"], st["states"]) == (1, False, 2)
    with pytest.raises(seeds.SeedsError, match="1/2 problems"):
        seeds.require_complete(sample, labels)

    _jsonl(labels, _label_rows("aa", 2) + _label_rows("bb", 3, unknown={0}))
    st = seeds.require_complete(sample, labels)
    assert st == {"problems": 2, "labelled": 2, "complete": True, "foreign_problems": 0, "states": 5,
                  "unknown_states": 1, "problems_with_unknown": 1}


def test_labels_of_another_sample_are_refused(tmp_path):
    sample = _jsonl(tmp_path / "sample.jsonl", [_rec("aa", 1)])
    labels = _jsonl(tmp_path / "labels.jsonl", _label_rows("aa", 1) + _label_rows("zz", 4))
    assert seeds.label_status(sample, labels)["foreign_problems"] == 1
    with pytest.raises(seeds.SeedsError):
        seeds.require_complete(sample, labels)


# ---------------------------------------------------------------------------------- commands

def test_models_are_the_published_checkpoint_and_the_two_seed_runs():
    assert seeds.MODELS == {"h-v1": "runs/h-v1/best.pt", "h-v1-s1": "runs/h-v1-s1/best.pt",
                            "h-v1-s2": "runs/h-v1-s2/best.pt"}


def test_bench_command_is_the_e2_command_with_another_checkpoint(monkeypatch):
    from isomer import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_bench", lambda ns: seen.append(ns) or 0)
    cmd = seeds.bench_command("runs/h-v1-s1/best.pt", "runs/e3/bench-seeds/h-v1-s1", python="python")
    assert cmd[:4] == ["python", "-m", "isomer", "bench"]
    assert cli.main(cmd[3:]) == 0
    ns = seen[0]
    assert ns.data == "data/isobench-v2" and ns.split == "test_depth"
    assert ns.ids == "docs/experiments/E2/ids-test_depth.txt"
    assert ns.methods == "neural-w1@runs/h-v1-s1/best.pt#bf16-fast"
    assert ns.budgets == "30,100,300,1000"
    assert ns.workers == 2 and ns.out == "runs/e3/bench-seeds/h-v1-s1"
    assert ns.max_seconds is None and ns.max_nodes is None and ns.limit is None    # nothing censored, no subset

    from isomer.search.neural import checkpoint_of, is_neural, precision_of
    assert is_neural(ns.methods) and checkpoint_of(ns.methods) == "runs/h-v1-s1/best.pt"
    assert precision_of(ns.methods) == "bf16-fast"


def test_score_paths():
    p = seeds.score_paths("runs/e3/scores/h-v1", "test_depth")
    assert {k: v.as_posix() for k, v in p.items()} == {
        "scores": "runs/e3/scores/h-v1/test_depth.jsonl",
        "meta": "runs/e3/scores/h-v1/test_depth.score.json",
        "analysis": "runs/e3/scores/h-v1/test_depth.analysis.json"}


# ---------------------------------------------------------------------------------- scripts

def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_script_dry_run_lists_one_command_per_model(capsys):
    assert _script("e3_bench_seeds").main(["--dry-run", "--out", "runs/e3/bench-seeds"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    for line, model in zip(lines, seeds.MODELS, strict=True):
        assert f"neural-w1@runs/{model}/best.pt#bf16-fast" in line
        assert line.endswith(f"--out runs/e3/bench-seeds/{model}")
    assert _script("e3_bench_seeds").main(["--dry-run", "--models", "nope"]) == 2


def test_score_script_refuses_unknown_models_and_incomplete_labels(tmp_path, capsys):
    mod = _script("e3_score_seeds")
    assert mod.main(["--models", "nope"]) == 2
    assert "unknown model" in capsys.readouterr().err
    # an E3 directory with a sample but no labels: refused before any model is loaded
    (tmp_path / "data").mkdir()
    _jsonl(tmp_path / "data" / "test_id.jsonl", [_rec("aa", 2)])
    tiny = "tiny=" + Path(__file__).relative_to(ROOT).as_posix()          # any existing file as checkpoint
    assert mod.main(["--models", tiny, "--splits", "test_id", "--e3", str(tmp_path)]) == 2
    assert "0/1 problems labelled" in capsys.readouterr().err

# ---------------------------------------------------------------------------------- model lists (E3-d6)

def test_parse_models():
    assert seeds.parse_models(None) == seeds.MODELS
    assert seeds.parse_models("h-v1-s1") == {"h-v1-s1": "runs/h-v1-s1/best.pt"}
    got = seeds.parse_models("h-v1, h-d6-s0=runs\\h-d6-s0\\best.pt")
    assert got == {"h-v1": "runs/h-v1/best.pt", "h-d6-s0": "runs/h-d6-s0/best.pt"}
    assert list(got) == ["h-v1", "h-d6-s0"]                 # order is kept: it is the column order
    for bad in ("nope", "a=", "=x.pt", "h-v1,h-v1", "", "a b=x.pt", "h-v1=runs/other/best.pt"):
        with pytest.raises(seeds.SeedsError):
            seeds.parse_models(bad)


def test_norm_path_is_the_same_on_every_platform():
    # pure string handling: no os.sep, no Path, so Linux CI and Windows give the same result
    assert seeds.norm_path("runs\\h-d6-s0\\best.pt") == "runs/h-d6-s0/best.pt"
    assert seeds.norm_path("runs/h-d6-s0\\best.pt") == "runs/h-d6-s0/best.pt"
    assert seeds.norm_path("runs/h-d6-s0/best.pt") == "runs/h-d6-s0/best.pt"
    assert seeds.norm_path(Path("runs") / "h-d6-s0" / "best.pt") == "runs/h-d6-s0/best.pt"


@pytest.mark.parametrize("ckpt", ["runs/h-d6-s0/best.pt", "runs\\h-d6-s0\\best.pt", "runs/h-d6-s0\\best.pt"])
def test_known_name_accepts_its_own_checkpoint_in_any_separator(ckpt):
    assert seeds.parse_models(f"h-d6-s0={ckpt}") == {"h-d6-s0": "runs/h-d6-s0/best.pt"}
    with pytest.raises(seeds.SeedsError, match="pick another name"):
        seeds.parse_models(ckpt.replace("h-d6-s0", "h-d6-s1", 1).join(("h-d6-s0=", "")))


def test_scripts_accept_a_named_checkpoint(capsys):
    assert _script("e3_bench_seeds").main(["--dry-run", "--models", "h-d6-s0=runs/h-d6-s0/best.pt"]) == 0
    line, = capsys.readouterr().out.strip().splitlines()
    assert "neural-w1@runs/h-d6-s0/best.pt#bf16-fast" in line
    assert line.endswith("--out runs/e3/bench-seeds/h-d6-s0")


def test_d6_family_names_resolve():
    got = seeds.parse_models("h-d6-s0,h-d6-s1,h-d6-s2,h-d6m-s0")
    assert got == {"h-d6-s0": "runs/h-d6-s0/best.pt", "h-d6-s1": "runs/h-d6-s1/best.pt",
                   "h-d6-s2": "runs/h-d6-s2/best.pt", "h-d6m-s0": "runs/h-d6m-s0/best.pt"}
    assert seeds.parse_models(None) == seeds.MODELS              # the default stays the three h-v1 seeds
    with pytest.raises(seeds.SeedsError):
        seeds.parse_models("h-d6m-s0=runs/elsewhere/best.pt")    # a known name cannot point somewhere else
