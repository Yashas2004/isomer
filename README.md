# Isomer

Learned guidance for equational proof search, with a small trusted kernel.

Status: **Phase 1b (IsoBench data) complete** — terms, rewriting, trusted
kernel, certificates, 4 theories, bidirectional BFS with proven lower bounds,
deterministic dataset builder with train / depth / held-out-theory splits.
See [`docs/DESIGN.md`](docs/DESIGN.md) for the research plan.

## Setup (Windows + VS Code + RTX 3050)

```powershell
# 1. Python 3.10-3.12 (64-bit) from python.org; tick "Add to PATH".
cd isomer
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# 2. PyTorch with CUDA. Get the exact command from https://pytorch.org/get-started/locally/
#    (Stable / Windows / Pip / Python / newest CUDA). It looks like:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. This package + dev tools
pip install -e ".[dev]"

# 4. Check everything
python -m isomer doctor     # should print your RTX 3050 and "matmul ok"
pytest
python -m isomer demo
python -m isomer sample --theory group -n 5 -k 6

# 5. Build IsoBench v0 (uses all CPU cores but one; ~10-30 min on a laptop)
python -m isomer data --out data/isobench-v0
```

In VS Code: install the *Python* extension, `Ctrl+Shift+P → Python: Select
Interpreter → .venv`, then the Testing tab picks up pytest automatically.

If `doctor` says CUDA is not available, you installed a CPU wheel:
`pip uninstall torch` and repeat step 2. Update your NVIDIA driver first if needed.

## Layout

```
src/isomer/core/term.py      terms, parser, positions, matching
src/isomer/core/rewrite.py   equations, rules, steps, successor enumeration (untrusted)
src/isomer/core/kernel.py    trusted proof checker + certificates (independent)
src/isomer/core/proof.py     Problem / Proof / certificate bridge
src/isomer/core/theories.py  group, boolean algebra (train); lattice, ring (held out)
src/isomer/core/generate.py  random-walk problem generator
src/isomer/search/bfs.py     bidirectional BFS: exact shortest proofs + lower bounds
src/isomer/data/build.py     IsoBench builder (dedup, labels, splits, datasheet)
docs/DESIGN.md               research questions, architecture, roadmap
docs/adr/                    architecture decision records
```
