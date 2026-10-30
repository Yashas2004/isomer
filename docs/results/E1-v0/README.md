# E1 on IsoBench v0 — superseded (archived)

These are the results of experiment E1 (`docs/experiments/E1-baselines.md`) on
**IsoBench v0** (`isobench/0`). Each run's git commit, machine and settings are
recorded in its config file (`config-<split>.json`). All three runs were made
from commit `985d4dd` with a clean tree (`git_dirty: false`).

**They are superseded.** The v0 labels were computed in an asymmetric action
space: fresh-variable values came from the current term's subterms, while
consumed values were unrestricted. As a result, bidirectional BFS labels were
not always the shortest forward proof. See `docs/adr/003-symmetric-action-space.md`.

What remains valid, and what does not:

- **Valid: solve-rate comparisons between methods** (the solve-rate tables, the
  paired McNemar comparisons, the per-theory breakdown, `cactus.png`,
  `nodes_vs_solved.png`). All methods searched the same action space, and every
  proof counted as solved was accepted by the trusted kernel. Each report lists
  0 error, 0 unsound and 0 censored cells.
- **Not valid: anything that depends on `dist`.** That covers:
  - the optimality gap, including the fair gap
  - solve rate by true distance (`solve_by_dist.png`)
  - which problems belong to `test_depth`, because split membership is decided
    by `dist`.

**The binding E1 results are those on IsoBench v1, in `docs/results/E1/`.**

## Contents

| file | source |
|---|---|
| `test_depth.md`, `test_id.md`, `test_theory.md` | `runs/E1-v0-<split>/report/results.md` |
| `solve_by_dist.png`, `cactus.png`, `nodes_vs_solved.png` | `runs/E1-v0-test_depth/report/` (test_depth only) |
| `config-<split>.json` | `runs/E1-v0-<split>/config.json` |

The per-problem logs (`runs/E1-v0-<split>/results.jsonl`) are not in the
repository (`runs/` is gitignored).
