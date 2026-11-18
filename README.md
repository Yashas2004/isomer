# Isomer: Learned Search for Machine-Checked Equational Proofs

Isomer proves equations in algebraic theories by searching over rewrite steps. A small transformer, trained to estimate how many steps remain, guides the search. Every proof is replayed by a tiny trusted kernel that shares no code with the search or the model.

[![ci](https://github.com/Yashas2004/isomer/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashas2004/isomer/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
[![checkpoint](https://img.shields.io/badge/checkpoint-v0.2--e2-informational)](https://github.com/Yashas2004/isomer/releases/tag/v0.2-e2)
[![checkpoints](https://img.shields.io/badge/checkpoints-v0.3--e3-informational)](https://github.com/Yashas2004/isomer/releases/tag/v0.3-e3)

> **Headline (pre-registered, kernel-checked).** On problems whose shortest proofs (5–10 steps) are longer than any proof seen in training (distances up to 4), learned guidance solves **74.7%** within 1,000 expansions. The hand-written symbol_bag heuristic solves **20.0%** and bidirectional BFS **58.3%**. Every problem symbol_bag solved, the learned heuristic also solved, at every budget (Holm p ≤ 1.4e-38).
>
> **What it does not do:** it does not transfer to theories it never saw. On held-out lattice and ring it ties symbol_bag, and at equal wall-clock time it is slower there.

---

## What it is

Given the axioms of a theory (group, boolean algebra, lattice, ring) and two terms `s` and `t`, Isomer searches for a sequence of rewrite steps that turns `s` into `t`. Each step uses one axiom, in either direction, at one position, sometimes with a chosen instantiation. The proof becomes a JSON certificate that the kernel replays independently.

```mermaid
flowchart LR
  G[Generator: random walks] --> I[IsoBench: splits + certified labels]
  I --> S[Searchers: BFS, best-first, neural best-first]
  S --> K[Trusted kernel]
  K --> C[Certificates + SHA-256 digest]
  I -->|train split only| ST[Distance-labelled states]
  ST --> T[Training]
  T --> M["Models: h-v1 (to dist 4), h-d6 (to dist 6)"]
  M --> S
  I --> E["E3 diagnostics: exact child labels, renamings"]
  M --> E
```

- **IsoBench** is a deterministic benchmark built from random walks. Each problem is labelled with its shortest proof length by bidirectional BFS, with certified bounds.
- **Searchers:** forward and bidirectional BFS, best-first search with hand-written heuristics, and best-first search guided by the learned model.
- **The learned model** scores (state, goal) pairs by estimated remaining distance. It is trained only on states from the training split.

## System highlights

- **Learned distance heuristic.** A 6-layer transformer encoder with 4,731,649 trainable parameters (d=256, 8 heads, RoPE, RMSNorm, SwiGLU, segment embeddings). It reads a (state, goal) pair and predicts the remaining proof distance. Training combines a Huber regression loss with a listwise ranking loss over a state's successors, whose soft target spreads over *all* optimal children. Training data: 551,072 distance-labelled states, 20 epochs, peak 3.0 GB of GPU memory on a 6 GB laptop RTX 3050 (bf16, micro-batched).
- **Neural best-first search.** Each expansion scores all of a node's children in one batch. Inference is deterministic, with an fp32 mode and a bf16-fast mode (2.24× faster, pinned by golden fp32 tests).
- **Trusted proof kernel.** `core/kernel.py` is 141 lines and imports nothing from the rewriting, search, data or model code (the de Bruijn criterion, [ADR-001](docs/adr/001-trusted-kernel.md)). Each proof is a JSON certificate with a SHA-256 digest that the kernel replays from the axioms. A proof counts only if the kernel accepts it.
- **Symmetric action space.** Each step uses one axiom, in either direction, at one position, with instantiations drawn from a problem-level pool. Forward and backward search therefore see exactly the same graph, so bidirectional BFS distances are exact ([ADR-003](docs/adr/003-symmetric-action-space.md)).
- **IsoBench v2.** A deterministic benchmark (byte-identical rebuilds) of about 35k problems over group, boolean algebra, lattice and ring. Every label carries certified bounds `dist_lb ≤ dist ≤ dist_ub`. The evaluation splits are labelled with 20,000 expansions, 20× the largest benchmark budget, so the labeller does not grade itself ([ADR-004](docs/adr/004-isobench-v2.md)).
- **Pre-registered evaluation.** Plans, predictions and stratified problem samples are committed before any test search. The analysis uses exact McNemar tests with Holm correction, Wilson intervals, a fair optimality gap, and equal-time comparisons with explicit censoring. Falsified predictions are reported as falsified.

## Architecture

How the modules connect (the pipeline view is the diagram above):

```text
                        +----------------------------------+
                        |  CLI: python -m isomer <command> |
                        |  prove · data · states · train   |
                        |  bench · report · timecompare    |
                        +----------------+-----------------+
                                         |
                +------------------------+-----------------+
                v                                          v
   +--------------------------+               +--------------------------+
+--| IsoBench builder (data/) |   problems    | Search (search/)         |
|  | random walks, 4 theories |-------------->| bfs-fwd, bfs-bidir       |
|  | bidir-BFS labels with    |               | best-first: size_diff,   |
|  |   certified bounds       |        +----->|   symbol_bag             |
|  | train / eval / held-out  |        |      | neural best-first        |
|  +--------------------------+        |      +--------------------------+
|               |                      |                   |
|               | train split only     |  h(state, goal)   | proof steps
|               v                      |                   v
|  +--------------------------+        |      +--------------------------+
|  | Learning (learn/)        |        |      | Trusted kernel (core/)   |
|  | distance-labelled states |        |      | 141 lines, imports no    |
|  | tokenizer + transformer  |--------+      |   search/data/model code |
|  | Huber + listwise ranking | checkpoint    | replays JSON certificate |
|  | bf16, fits a 4 GB GPU    |   h-v1        | -> ACCEPTED / REJECTED   |
|  +--------------------------+               +--------------------------+
|               |                                          |
| eval samples  | checkpoints                              | accepted proofs only
|               v                                          v
|  +--------------------------+               +--------------------------+
+->| E3 analysis (e3/)        |               | Benchmark (bench/)       |
   | exact all-children labels|               | prefix-property runs     |
   | renamings, diagnostics,  |               | McNemar + Holm, Wilson   |
   | summaries                |               | equal-time analysis      |
   +--------------------------+               +--------------------------+
```

## Results

### E3: why the heuristic stops at its training range

Four results, each confirmed by a rule fixed in advance. The full account, with commits, is in [docs/experiments/E3/REPORT.md](docs/experiments/E3/REPORT.md) ("E3 final status").

1. **Saturation.** The baseline's distance estimate flattens just below the largest training distance, in all three training seeds. On test_depth states at true distance 7–8, its mean prediction is 4.685–4.764, while training states reach distance 5. Its drop to the optimal child shrinks to 0.068–0.139 at distance 7.
2. **The d6 intervention.** The same configuration trained on problems up to distance 6 (h-d6, three seeds) no longer saturates there. Its mean prediction at distance 7 is 6.186–6.291, and its all-children top-1 is 0.948–0.964 against 0.645–0.800 for the baseline.
3. **The size-matched control.** h-d6m-s0 has as many training problems and optimizer steps as the baseline, drawn from the d6 data. It is "depth" on all five pre-registered rules, so the gain comes from depth coverage, not data volume. It lies slightly below every d6 seed on most quantities.
4. **Fresh confirmation.** The four hypotheses H1–H4 were tested again on 300 newly generated problems, with ids disjoint from every v2 and d6 split. All four are supported.

![Mean predicted distance vs true distance on the fresh sample: the baseline levels off near 4.7, below its training maximum of 5; the d6 seeds and the control level off near 6.5, below 7](docs/experiments/E3/saturation.png)

*Figure 1.* Mean predicted distance by true distance on the fresh sample. Lines are the mean over seeds and bands span the three seeds. Dashed lines mark each family's training maximum. Drawn by `scripts/e3_figures.py` from [fresh_summary.md](docs/experiments/E3/fresh_summary.md) only. Distance 10 has 1 scored state and is left out.

On the fresh sample (300 problems, 1,939 scored states; ranges over seeds; solved = kernel-accepted, of 300):

| model | trained on | mean prediction at k = 7 | all-children top-1 | solved at 30 | solved at 1,000 |
|---|---|---:|---:|---:|---:|
| baseline h-v1 (3 seeds) | problems up to distance 4 | 4.651–4.664 | 0.613–0.803 | 114–171 | 192–234 |
| d6 (3 seeds) | problems up to distance 6 | 6.025–6.284 | 0.949–0.971 | 261–281 | 290–294 |
| control h-d6m-s0 (1 seed) | distance up to 6, as many problems as the baseline | 6.032 | 0.936 | 248 | 290 |

**Pre-registration.** Each experiment's prediction and decision rules were committed before that experiment's run, and the summaries judge them with those rules:

- d6: [d6_prediction.md](docs/experiments/E3/d6_prediction.md), commit `4821ed0`, amended in `583e232` before the training states were built.
- Size-matched control: [d6_matched_prediction.md](docs/experiments/E3/d6_matched_prediction.md), commit `a2815ff`.
- Fresh confirmation: [fresh_prediction.md](docs/experiments/E3/fresh_prediction.md), rules in `40ebab3` and sample in `9a345d6`, before any model was run on it.

The saturation thresholds of result 1 were chosen after seeing the seed results. They were first tested against a fixed rule on the fresh sample (H1).

Summaries:
- [seeds](docs/experiments/E3/seeds_summary.md)
- [d6](docs/experiments/E3/d6_summary.md)
- [train/test overlap](docs/experiments/E3/overlap_summary.md)
- [control](docs/experiments/E3/d6_matched_summary.md)
- [d6 family](docs/experiments/E3/d6_family_summary.md)
- [fresh](docs/experiments/E3/fresh_summary.md)

Checkpoints: [checkpoints.md](docs/experiments/E3/checkpoints.md).

**E2: learned heuristic vs baselines on IsoBench v2.** Solve rate at 1,000 node expansions; only kernel-accepted proofs count. Plan: [docs/experiments/E2-learned-heuristic.md](docs/experiments/E2-learned-heuristic.md). Full results: [docs/results/E2](docs/results/E2/README.md).

| split | what it tests | neural-w1 (learned) | best-first symbol_bag | bfs-fwd | bfs-bidir |
|---|---|---:|---:|---:|---:|
| test_id (n=300) | dist 1–4, like training | **100.0%** | 93.0% | 91.3% | 100%\* |
| test_depth (n=300) | dist 5–10, longer than training | **74.7%** | 20.0% | 8.7% | 58.3% |
| test_deep (n=100) | certified lower bound 7–11 | 2/100 | 0/100 | 0/100 | 0/100 |
| test_theory (n=300) | held-out lattice + ring | 56.3% | 55.3% | 46.3% | **80.3%** |

\* test_id was labelled by bfs-bidir, so that cell is reported but not interpreted.

The E2 numbers are for h-v1 (seed 0). Across three training seeds, the test_depth solve rate at 1,000 expansions is 65.0–80.0% (mean 73.2%), from [docs/experiments/E3/seeds_summary.md](docs/experiments/E3/seeds_summary.md).

![Solve rate by true proof length on test_depth](docs/results/E2/test_depth/solve_by_dist.png)

| finding (test_depth unless stated) | value |
|---|---|
| Solve rate at 30 expansions, learned vs symbol_bag at 10,000 | 46.3% vs 41.0% |
| Solve rate by true distance 5 / 6 / 7 / 8 / 9 / 10 | 69/70 · 86/97 · 41/58 · 24/52 · 3/19 · 1/4 |
| Fair optimality gap (1.000 = always shortest) | 1.000 learned vs 1.200 symbol_bag |
| Median nodes stored when solved | 374 learned vs 6,887 symbol_bag |
| Equal wall-clock, 1 s per problem | 50.3% learned vs 18.0% symbol_bag |
| Equal wall-clock on test_theory | symbol_bag significantly better at every limit (1–30 s) |

**Pre-registered predictions:** 3 confirmed and 2 falsified. Falsified: (1) on test_id the learned heuristic is significantly *better*, not tied at the ceiling; (2) at 1 s per problem it already beats symbol_bag, where we predicted symbol_bag would be at least as good.

**E1: non-neural baselines on IsoBench v1** ([plan](docs/experiments/E1-baselines.md), [results](docs/results/E1/README.md)). The crude goal-directed heuristic symbol_bag beats forward BFS (582 vs 284 solved on test_depth at 3,000 expansions), which falsified the prediction that no heuristic would be significantly better. It costs proof length (gap 1.168). E1 also exposed a circular labeller, which led to the decoupled labels in IsoBench v2.

## Installation

Requirements: Windows or Linux, Python 3.10–3.12, PyTorch 2.x. An NVIDIA GPU is optional; the searchers that need no model run without PyTorch.

```bash
git clone https://github.com/Yashas2004/isomer.git
cd isomer
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch: pick the right command at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,plots]"

python -m isomer doctor           # checks Python, torch, CUDA, bf16
```

Download the trained checkpoint (release [v0.2-e2](https://github.com/Yashas2004/isomer/releases/tag/v0.2-e2), sha256 `928015f6250a8531e6249e799507d4ca8bd433e50e28b7639aae6fa885376c00`):

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force runs/h-v1
Invoke-WebRequest https://github.com/Yashas2004/isomer/releases/download/v0.2-e2/best.pt -OutFile runs/h-v1/best.pt
certutil -hashfile runs/h-v1/best.pt SHA256
```
```bash
# Linux/macOS
mkdir -p runs/h-v1
curl -L -o runs/h-v1/best.pt https://github.com/Yashas2004/isomer/releases/download/v0.2-e2/best.pt
sha256sum runs/h-v1/best.pt
```

The E3 checkpoints are in release [v0.3-e3](https://github.com/Yashas2004/isomer/releases/tag/v0.3-e3): h-v1-s1, h-v1-s2, h-d6-s0, h-d6-s1, h-d6-s2 and h-d6m-s0, each as `<model>.pt`, with `SHA256SUMS.txt`. Their training data, seeds and sha256 are in [docs/experiments/E3/checkpoints.md](docs/experiments/E3/checkpoints.md). The E3 scripts expect `runs/<model>/best.pt`. For example:

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force runs/h-d6-s0
Invoke-WebRequest https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6-s0.pt -OutFile runs/h-d6-s0/best.pt
certutil -hashfile runs/h-d6-s0/best.pt SHA256
```
```bash
# Linux/macOS
mkdir -p runs/h-d6-s0
curl -L -o runs/h-d6-s0/best.pt https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6-s0.pt
sha256sum runs/h-d6-s0/best.pt
```

## Quickstart

```bash
python -m isomer prove --example bool-deep                                   # learned search, kernel-checked
python -m isomer prove --example bool-deep --method bestfirst-symbol_bag-w1  # same problem, hand-written heuristic
python -m isomer prove --theory group --lhs "(* e (* (inv a) a))" --rhs "e" --cert proof.json
```

`prove` prints each rewrite step, the search statistics and the kernel's verdict. The default method is `neural-w1` with `runs/h-v1/best.pt`, in bf16-fast on CUDA and fp32 on CPU. The other methods (`bestfirst-symbol_bag-w1`, `bfs-bidir`, `bfs-fwd`, ...) need neither a checkpoint nor torch. It exits with 0 if a proof is found and accepted, 1 if there is no proof within budget, and 2 on an error. On an idle laptop GPU, `bool-deep` gives a 6-step proof in 6 expansions (about 0.1 s after a one-time model load of about 4 s); symbol_bag finds no proof within 1,000 expansions.

The built-in examples (`group-short`, `bool-deep`, `lattice`) were hand-picked from the dev set, which is disjoint from the benchmark, to show the learned search working. They are illustrations, not evidence. The evidence is the E2 table above.

## CLI reference

| command | what it does |
|---|---|
| `doctor` | report Python, PyTorch, CUDA and bf16 support |
| `demo` | kernel-check a hand-written 2-step group proof (smoke test, no model needed) |
| `prove` | prove one equation (`--example` or `--theory/--lhs/--rhs`), with any `--method`, and optionally write a `--cert` |
| `sample` | print random-walk problems for a theory |
| `data` | build IsoBench v2 (deterministic; `--workers`) |
| `states` | build distance-labelled training states from the train split only |
| `train` | train the distance heuristic (`--states`, `--out`, `--epochs`, model size flags) |
| `stratify` | draw a fixed stratified sample of problem ids for a pre-registered experiment |
| `bench` | run searchers on a split at several budgets and kernel-check every proof |
| `report` | write results.md, statistics and figures from a bench run (no re-running) |
| `timecompare` | equal-wall-clock analysis from a main run and a time run |

Method names: `bfs-fwd`, `bfs-bidir`, `bestfirst-size_diff-w1`, `bestfirst-symbol_bag-w1`, and `neural-w1@<checkpoint>#<precision>` (for example `neural-w1@runs/h-v1/best.pt#bf16-fast`).

## Reproducing the paper numbers

Measured on one Windows laptop (16 logical CPUs, RTX 3050 6GB Laptop GPU).

| step | command | measured time |
|---|---|---:|
| 1. Build IsoBench v2 | `python -m isomer data --workers N` | 6,283 s |
| 2. Training states | `python -m isomer states --workers N` | 1,050 s |
| 3. Train h-v1 | `python -m isomer train --states data/states-v2 --out runs/h-v1 --epochs 20` | ~9,040 s |
| 4. E2 main run (4 splits) | see below | 8,136 s |
| 5. E2 equal-time run (4 splits) | see below | 5,010 s |

The labelling workers can peak at about 2.3 GiB each (1.5M-node cap), so choose `--workers ≤ free RAM in GiB / 2.3`. The stratified samples are committed in `docs/experiments/E2/`. For each split `S` in `test_id`, `test_depth`, `test_deep` and `test_theory`:

```powershell
python -m isomer bench --data data/isobench-v2 --split S --ids docs/experiments/E2/ids-S.txt `
  --methods "bfs-fwd,bfs-bidir,bestfirst-symbol_bag-w1,neural-w1@runs/h-v1/best.pt#bf16-fast" `
  --budgets 30,100,300,1000 --workers 2 --out runs/E2-S
python -m isomer bench --data data/isobench-v2 --split S --ids docs/experiments/E2/ids-S.txt `
  --methods bfs-fwd,bestfirst-symbol_bag-w1 --budgets 1000,3000,10000 --max-nodes 1500000 `
  --workers 4 --out runs/E2time-S
python -m isomer report runs/E2-S --reference bestfirst-symbol_bag-w1
python -m isomer timecompare --main runs/E2-S --time runs/E2time-S --out runs/E2-S/report
```

Measured E2 wall time per split (elapsed time printed by bench at 100%):

| | test_id | test_depth | test_deep | test_theory | total |
|---|---:|---:|---:|---:|---:|
| main run (2 workers, one GPU) | 109 s | 1,674 s | 2,328 s | 4,025 s | 8,136 s (~2.3 h) |
| equal-time run (4 CPU workers) | 85 s | 2,216 s | 1,014 s | 1,695 s | 5,010 s (~1.4 h) |

`report` and `timecompare` only read logs; each takes seconds. Every run records its git commit and `git_dirty: false` in `config.json`.

## E3 reproduction

The commands below were run in this order, on the same laptop, from the repository root. Each is one line and works in Command Prompt, PowerShell and bash. Every output goes to `runs/` or a new `data/` directory; nothing existing is changed.

Measured times come from the files each step writes:
- training: per-epoch train + validation seconds in `metrics.jsonl`, without model and data loading;
- the states build: `stats.json`;
- search: the elapsed time `bench` prints at 100%;
- scoring: `seconds` in `*.score.json`;
- dataset builds and child labelling: the times these steps printed to the console (they store no timings);
- the size-matched subset: a later rebuild.

| # | step | command | measured time |
|---:|---|---|---:|
| 1 | d6 dataset (train pool relabelled to distance 6) | `python scripts/e3_build_d6.py` | 367 s (generation 75 s, train labelling 292 s; 15 workers) |
| 2 | d6 training states | `python -m isomer states --data data/isobench-v2-d6 --out data/states-v2-d6 --offpath 6 --budget 3000` | 14,075 s |
| 3 | baseline seeds h-v1-s1, h-v1-s2 | `python scripts/train_seeds.py` | 9,166 s, 9,056 s |
| 4 | d6 seeds h-d6-s0, s1, s2 | `python scripts/train_seeds.py --seeds 0,1,2 --states data/states-v2-d6 --prefix h-d6-s` | 13,966 s, 13,803 s, 13,751 s |
| 5 | size-matched subset of the d6 states | `python scripts/e3_build_matched.py` | 68 s (rebuilt to a scratch directory; byte-identical output) |
| 6 | control h-d6m-s0 | `python scripts/train_seeds.py --seeds 0 --states data/states-v2-d6-matched --prefix h-d6m-s` | 11,242 s |
| 7 | label every child on the E2 samples (test_id, test_depth) | `python scripts/e3_label.py --workers 4` | 0.6 min (test_id), 67.0 min (test_depth) |
| 8 | score the seven models | `python scripts/e3_score_seeds.py --models h-v1,h-v1-s1,h-v1-s2,h-d6-s0,h-d6-s1,h-d6-s2,h-d6m-s0` | 16.8–37.2 s (test_depth), 5.1–11.2 s (test_id) per model |
| 9 | neural-w1 search, E2 test_depth sample | `python scripts/e3_bench_seeds.py --models h-v1,h-v1-s1,h-v1-s2,h-d6-s0,h-d6-s1,h-d6-s2,h-d6m-s0` | 637–1,292 s per baseline seed; 57–221 s per d6 seed; 93 s control |
| 10 | seed, d6, control summaries | `python scripts/e3_seeds_summary.py` and `python scripts/e3_seeds_summary.py --family --d6 h-d6-s0,h-d6-s1,h-d6-s2 --matched h-d6m-s0` | about 2 s each |
| 11 | fresh pool and its 300-problem sample | `python scripts/e3_build_fresh.py` | 60.7 min (all cores but one) |
| 12 | label every child on the fresh sample | `python scripts/e3_label.py --splits test_depth_fresh --data data/isobench-fresh --ids docs/experiments/E3/ids-test_depth_fresh.txt --out runs/e3-fresh --workers 5` | 46.8 min |
| 13 | score on the fresh sample | `python scripts/e3_score_seeds.py --models h-v1,h-v1-s1,h-v1-s2,h-d6-s0,h-d6-s1,h-d6-s2,h-d6m-s0 --splits test_depth_fresh --e3 runs/e3-fresh` | 16.7–17.4 s per model |
| 14 | search on the fresh sample | `python scripts/e3_bench_seeds.py --models h-v1,h-v1-s1,h-v1-s2,h-d6-s0,h-d6-s1,h-d6-s2,h-d6m-s0 --data data/isobench-fresh --split test_depth_fresh --ids docs/experiments/E3/ids-test_depth_fresh.txt --out runs/e3-fresh/bench-seeds --baselines bfs-bidir,bestfirst-symbol_bag-w1` | 621–1,207 s per baseline seed; 70–106 s per d6 seed; 146 s control; 247 s bfs-bidir and symbol_bag |
| 15 | fresh summary (H1–H4) | `python scripts/e3_fresh_summary.py` | about 27 s |
| 16 | Figure 1 (`docs/experiments/E3/saturation.png`), from the committed fresh summary | `python scripts/e3_figures.py` | about 2 s |

Notes:
- **Pre-registration timing.** The prediction files were committed before steps 2, 6 and 11 (see [Results](#e3-why-the-heuristic-stops-at-its-training-range)). The fresh sample was committed after step 11 and before step 12.
- **Interruptions.** Steps 7, 11, 12 and 14 resume where they stopped if rerun.
- **Memory.** The labelling steps are sized for 16 GB of RAM (`--workers`).
- **GPU memory.** Training peaked at 3.0 GB, measured with `torch.cuda.max_memory_allocated`.

## Why trust the numbers

- **Kernel gate.** The benchmark runner rebuilds each certificate from the dataset record's own goal and runs the kernel on it. A rejected proof makes the run fail loudly; it is never silently counted.
- **No circular labels.** The evaluation labels come from a 20× larger budget than any benchmark run; the runner warns when a budget comes within 10× of the labelling budget.
- **Frozen before testing.** Questions, predictions, analysis and problem ids were committed before any search on the test splits (E2 plan at commit `cd3d517`).
- **Paired statistics.** Exact McNemar tests with Holm correction; the optimality gap is computed only on problems that every method solved.
- **Determinism tests.** Byte-identical dataset rebuilds; deterministic neural search on CPU and CUDA; the prefix property (one run gives the results for every smaller budget) is tested for every searcher; golden fp32 outputs pin the model. The suite has 383 tests (5 marked slow), and CI runs on Ubuntu and Windows with Python 3.10 and 3.12.

## Limitations

- **Synthetic benchmark.** The problems are random-walk identities in four small theories, with no external benchmark (TPTP UEQ is not evaluated yet).
- **No cross-theory transfer.** The model ties symbol_bag on held-out lattice and ring, and is slower there at equal time.
- **Depth decay.** Success falls with proof length beyond the training range; certified-deep problems are essentially unsolved by every method.
- **Small model, one signal.** 4.7M parameters and a value-style distance head; there is no policy head.
- **Pure Python, one machine.** Absolute speeds are far from optimised provers. The wall-clock comparisons depend on the hardware: the neural searcher ran with 2 workers sharing one laptop GPU, the baselines with 4 CPU workers.
- **Input length untested.** Too few searches reached states longer than the training inputs (126 tokens) to say whether the model handles longer inputs.
- **Not done yet:** e-graph / equality-saturation baselines, expert iteration, ablations and interpretability. See the roadmap in [docs/DESIGN.md](docs/DESIGN.md).

## Documentation

| document | contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | research questions, architecture, roadmap with status |
| [ADR-001](docs/adr/001-trusted-kernel.md) | a tiny trusted kernel, independent of all search and model code |
| [ADR-002](docs/adr/002-action-space-and-labels.md) | the action space (both directions, instantiation) and difficulty labels |
| [ADR-003](docs/adr/003-symmetric-action-space.md) | a symmetric problem-level instantiation pool |
| [ADR-004](docs/adr/004-isobench-v2.md) | IsoBench v2: separate pools, decoupled labels, memory cap, certified bounds |
| [E1 plan](docs/experiments/E1-baselines.md) · [results](docs/results/E1/README.md) | non-neural baselines |
| [E2 plan](docs/experiments/E2-learned-heuristic.md) · [results](docs/results/E2/README.md) | learned heuristic vs baselines |
| [train-h-v1](docs/results/train-h-v1/) | training config, metrics and checkpoint hash |
| [E3 report](docs/experiments/E3/REPORT.md) | diagnostics of the learned heuristic; "E3 final status" at the top |
| [E3 seeds](docs/experiments/E3/seeds_summary.md) | saturation and margin collapse in three baseline seeds |
| [E3 d6 prediction](docs/experiments/E3/d6_prediction.md) · [summary](docs/experiments/E3/d6_summary.md) · [overlap](docs/experiments/E3/overlap_summary.md) | training on deeper problems (distance up to 6) |
| [E3 control prediction](docs/experiments/E3/d6_matched_prediction.md) · [summary](docs/experiments/E3/d6_matched_summary.md) · [family](docs/experiments/E3/d6_family_summary.md) | size-matched control: depth coverage vs data volume |
| [E3 fresh prediction](docs/experiments/E3/fresh_prediction.md) · [summary](docs/experiments/E3/fresh_summary.md) | confirmation on 300 newly generated problems |
| [E3 checkpoints](docs/experiments/E3/checkpoints.md) | the seven E3 models: data, seed, sha256, download |
| [docs/datasets/](docs/datasets/) | datasheets for every IsoBench version |

## Repository layout

```
src/isomer/core/      terms, rewriting (untrusted), kernel (trusted), certificates, theories, generator
src/isomer/search/    BFS (forward, bidirectional), best-first + heuristics, neural best-first
src/isomer/data/      IsoBench builder, stratified sampling
src/isomer/learn/     states, tokenizer, model, training, batched inference
src/isomer/bench/     kernel-checked benchmark runner, report, equal-time analysis
src/isomer/demo.py    `isomer prove`
tests/                pytest suite
docs/                 design, ADRs, pre-registered plans, results, datasheets
```

## Citation

If you use Isomer or IsoBench, please cite:

```bibtex
@software{isomer2026,
  author = {Yashas R},
  title  = {Isomer: Learned Search for Machine-Checked Equational Proofs},
  year   = {2026},
  url    = {https://github.com/Yashas2004/isomer}
}
```

## License

MIT, see [LICENSE](LICENSE).
