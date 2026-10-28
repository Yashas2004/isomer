# E1 — Non-neural baselines on IsoBench v0

Status: plan frozen before running (2026-10-28). Results go in docs/results/E1/.

## Questions
1. How much does bidirectional search gain over forward search as proofs get longer?
2. Do hand-written heuristics (size_diff, symbol_bag; w=1 and w=3) beat forward BFS at equal expansion budgets?
3. Does any heuristic's benefit depend on the theory (group vs bool)?

## Setup (fixed)
- Data: IsoBench v0 (datasheet in docs/datasets/), splits test_id (all), test_depth (all), test_theory (all).
- Methods: bfs-bidir, bfs-fwd, bestfirst-size_diff-w1, bestfirst-size_diff-w3, bestfirst-symbol_bag-w1, bestfirst-symbol_bag-w3. (ucs omitted: identical to bfs-fwd, verified in smoke runs.)
- Budgets: 100, 300, 1000, 3000 node expansions; one run at 3000, smaller budgets derived (prefix property). No wall-clock limit.
- Runs are made from a clean, committed tree (config.json git_dirty must be false).

## Analysis (fixed before seeing results)
- Primary metric: solve rate with 95% Wilson intervals.
- Primary comparison: each method vs bfs-fwd (forward-to-forward is the fair comparison for heuristics), exact McNemar, Holm-corrected within each split's table, alpha = 0.05.
- bfs-bidir is reported as an upper reference, not as the heuristics' comparison target.
- Secondary: fair optimality gap, solve rate by true distance, per-theory breakdown, nodes stored (memory).
- Any run with error or unsound rows is invalid and is re-run, not reported.

## Predictions (written before running)
- bfs-bidir > bfs-fwd at every budget on test_depth, with the gap widening with distance.
- No heuristic is significantly better than bfs-fwd overall; at least one is significantly worse.
- Heuristic effects differ by theory.
