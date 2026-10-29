"""isomer, the search code, the data/states builders and the CLI must work
without torch: only isomer.learn.{model,train,infer} may import it."""

import subprocess
import sys

CODE = r"""
import sys
sys.modules["torch"] = None          # any `import torch` now raises ImportError
import isomer, isomer.cli, isomer.core.kernel, isomer.search.bfs, isomer.search.best_first
import isomer.search.heuristics, isomer.bench.run, isomer.data.build
import isomer.learn, isomer.learn.states, isomer.learn.tokenizer, isomer.learn.data, isomer.learn.metrics
from isomer.cli import main
assert main(["demo"]) == 0
try:
    main(["train", "--help"])
except SystemExit as e:
    assert e.code == 0
assert main(["train", "--out", "unused"]) == 2   # clean error, not a traceback
try:
    import isomer.learn.model
except ImportError:
    print("NO_TORCH_OK")
"""


def test_imports_and_cli_without_torch():
    cp = subprocess.run([sys.executable, "-c", CODE], check=False, capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert "NO_TORCH_OK" in cp.stdout
