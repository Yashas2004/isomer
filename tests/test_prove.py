"""``isomer prove``: the demo command (neural paths use a tiny random model)."""

import json

import pytest

from isomer.cli import main
from isomer.core import kernel
from isomer.core.proof import Problem
from isomer.core.term import parse
from isomer.core.theories import THEORIES
from isomer.demo import EXAMPLES, RELEASE_SHA256, RELEASE_URL
from isomer.search.common import Budget, get_searcher

GOAL = ["--theory", "group", "--lhs", "(* e (* (inv a) a))", "--rhs", "e"]


def test_neural_method_on_tiny_checkpoint(tiny_ckpt, capsys):
    assert main(["prove", *GOAL, "--ckpt", tiny_ckpt, "--precision", "fp32"]) == 0
    out = capsys.readouterr().out
    assert "neural-w1" in out and "fp32" in out
    assert "model    loaded in" in out and "(one-time)" in out and "search   " in out
    assert "->" in out and "kernel   ACCEPTED" in out


def test_baseline_needs_no_checkpoint(capsys):
    assert main(["prove", *GOAL, "--method", "bfs-bidir", "--ckpt", "does-not-exist.pt"]) == 0
    out = capsys.readouterr().out
    assert "proof    2 steps" in out and "kernel   ACCEPTED" in out


def test_missing_checkpoint_message(capsys):
    assert main(["prove", *GOAL, "--ckpt", "does-not-exist.pt"]) == 2
    err = capsys.readouterr().err
    assert RELEASE_URL in err and RELEASE_SHA256 in err and "bestfirst-symbol_bag-w1" in err


def test_cert_is_accepted_by_kernel(tmp_path, capsys):
    cert_path = tmp_path / "out" / "cert.json"
    assert main(["prove", *GOAL, "--method", "bestfirst-symbol_bag-w1", "--cert", str(cert_path)]) == 0
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    assert kernel.check_certificate(cert)
    assert cert["digest"] == kernel.certificate_digest(cert)
    assert cert["digest"] in capsys.readouterr().out


def test_unsolved_and_usage_errors(capsys):
    # e = a is false in a group: no proof, exit 1
    assert main(["prove", "--theory", "group", "--lhs", "e", "--rhs", "a", "--method", "bfs-fwd",
                 "--budget", "20"]) == 1
    assert main(["prove", *GOAL, "--method", "no-such-method"]) == 2
    assert main(["prove", "--theory", "group", "--lhs", "(* e", "--rhs", "e", "--method", "bfs-fwd"]) == 2
    assert main(["prove", "--theory", "group", "--method", "bfs-fwd"]) == 2


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_examples_parse_and_have_their_dist(name):
    ex = EXAMPLES[name]
    theory, sig = THEORIES[ex.theory]
    problem = Problem.parse(theory, ex.lhs, ex.rhs)
    symbols = set(sig.ops) | set(sig.constants) | set(sig.skolems)
    for s in (ex.lhs, ex.rhs):
        assert set(str(parse(s)).replace("(", " ").replace(")", " ").split()) <= symbols
    # the stated dist is the true shortest length (bfs-bidir is exact)
    r = get_searcher("bfs-bidir")(problem, Budget(max_expansions=20000))
    assert r.solved and len(r.proof.steps) == ex.dist


def test_example_cli_prints_disclaimer(capsys):
    assert main(["prove", "--example", "group-short", "--method", "bfs-bidir"]) == 0
    out = capsys.readouterr().out
    assert "not evidence" in out and "docs/results/E2" in out
