# Isomer

Learned guidance for equational proof search, with a small trusted kernel.

Status: **Phase 1 (core) complete** — terms, rewriting, kernel, certificates,
random-walk generator, 25 tests incl. differential + property-based tests.
See [`docs/DESIGN.md`](docs/DESIGN.md) for the research plan.

## Setup (Windows + VS Code + RTX 3050)

```powershell
# 1. Python 3.11 or 3.12 (64-bit) from python.org; tick "Add to PATH".
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
src/isomer/core/theories.py  group, boolean algebra
src/isomer/core/generate.py  random-walk problem generator (v0)
docs/DESIGN.md               research questions, architecture, roadmap
docs/adr/                    architecture decision records
```
