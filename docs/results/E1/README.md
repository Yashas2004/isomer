# E1 results — non-neural baselines on IsoBench v1

Plan: docs/experiments/E1-baselines.md (pre-registered 2026-10-28). Commit: `3ecea43`. All runs clean (git_dirty false), 0 errors, 0 unsound proofs, 0 censored cells.

## Deviations from the plan
1. Data version. The plan named IsoBench v0. v0 labels used an asymmetric action space (ADR-003), so E1 was re-run on IsoBench v1. v0 results are archived in docs/results/E1-v0/. Method comparisons were nearly identical on v0 and v1 (e.g. symbol_bag-w1 test_depth 37.4% -> 34.0%, bfs-fwd 17.6% -> 16.6%).
2. Unanticipated confound: bfs-bidir is the labeller. Dataset labels and the test_depth split were produced by bidirectional BFS with a budget of 3000, the same as E1's largest budget. At budget 3000 bfs-bidir therefore solves exactly the labelled problems (test_depth 1299 = 1711 − 412 unlabelled; test_theory 543 = 600 − 57). Its E1 numbers are circular and are reported only as a reference, not as a baseline result. Fix planned for IsoBench v2: label with a budget much larger than any benchmark budget.

## Predictions vs outcomes
| # | Prediction | Outcome |
|---|---|---|
| 1 | bfs-bidir > bfs-fwd at every budget on test_depth, gap widening with distance | Holds at every budget, but confounded (deviation 2); the distance trend is untestable because bfs-bidir is 100% at every labelled distance by construction. |
| 2 | No heuristic significantly better than bfs-fwd overall; at least one significantly worse | First part FALSIFIED: symbol_bag-w1 beats bfs-fwd at every budget on test_depth (582 vs 284 at 3000; Holm p < 1e-50) and test_theory (436 vs 358; Holm p < 1e-11); no heuristic is significantly better on test_id. Second part confirmed: size_diff-w3 is significantly worse at every budget on test_id. |
| 3 | Heuristic effects differ by theory | Confirmed. test_depth at 3000: symbol_bag-w1 vs bfs-fwd is 33.3% vs 6.7% on bool (5x) but 34.9% vs 27.8% on group (1.26x). |

## Other findings
- The benefit of a crude goal-directed heuristic grows with proof depth: none on test_id (dist <= 4), about 2x on test_depth.
- It costs proof quality: fair optimality gap 1.168 for symbol_bag-w1 on the 99 problems all methods solved in test_depth.
- symbol_bag solves problems that the labeller (bidirectional BFS, budget 3000) could not: symbol_bag-w1 4 and symbol_bag-w3 4 on test_depth, symbol_bag-w3 1 on test_theory.
- Target for a learned heuristic: beat symbol_bag-w1 (34.0% on test_depth at 3000) by a significant margin in forward search.
