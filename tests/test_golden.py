"""Regression guard: the fp32 inference path must not change by a single bit.

The golden outputs were recorded on 187 real successor batches from dev
searches (``tests/data``) before the ``bf16-fast`` path was added to
model.py. Bit-identity holds only on the same hardware and software:
* CUDA: compared bit for bit when the GPU name and torch version match the
  recorded ones, otherwise skipped (with the reason);
* CPU (tiny model shipped with the golden file): bit for bit when the
  platform and torch version match, otherwise within 1e-6, because BLAS
  backends differ across machines. That still catches any logic change.
"""

import gzip
import hashlib
import json
import platform
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from isomer.core.term import parse
from isomer.learn.infer import LearnedHeuristic
from isomer.search.neural import _load

DATA = Path(__file__).parent / "data"
CK = Path("runs/h-v1/best.pt")


def load(name):
    with gzip.open(DATA / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def outputs(h, golden):
    return [[v.hex() for v in h.scores([parse(s) for s in b["states"]], parse(b["goal"]))]
            for b in golden["batches"]]


def test_fp32_cpu_outputs_unchanged():
    golden = load("golden_fp32_cpu.json.gz")
    h = LearnedHeuristic(str(DATA / "golden_tiny.pt"), device="cpu", precision="fp32", batch_size=4096)
    got = outputs(h, golden)
    same_env = (golden["platform"], golden["machine"], golden["torch"]) == \
               (platform.system(), platform.machine(), torch.__version__)
    if same_env:
        assert got == golden["outputs"]
    else:
        for a, b in zip(got, golden["outputs"], strict=True):
            assert all(abs(float.fromhex(x) - float.fromhex(y)) <= 1e-6 for x, y in zip(a, b, strict=True))


def test_fp32_cuda_outputs_bit_identical():
    if not torch.cuda.is_available() or not CK.is_file():
        pytest.skip("needs CUDA and runs/h-v1/best.pt")
    golden = load("golden_fp32_cuda.json.gz")
    if hashlib.sha256(CK.read_bytes()).hexdigest() != golden["checkpoint_sha256"]:
        pytest.skip("runs/h-v1/best.pt is not the checkpoint the golden outputs were recorded with")
    env = (torch.cuda.get_device_name(0), torch.__version__)
    if env != (golden["gpu"], golden["torch"]):
        pytest.skip(f"recorded on {golden['gpu']} / torch {golden['torch']}, running on {env}")
    assert outputs(_load(str(CK), "cuda", "fp32"), golden) == golden["outputs"]
