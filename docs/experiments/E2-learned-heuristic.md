# E2 — Learned distance heuristic vs non-neural baselines on IsoBench v2

Status: plan frozen before any search on v2 test splits (2026-10-30). Results go in docs/results/E2/. (Written at commit b5704cd0b15c5e18bef7e37c38612b8843110558.)

## Questions
1. Does best-first search guided by the learned heuristic (h-v1) solve significantly more problems than the same search guided by symbol_bag, at equal expansion budgets?
2. How does the advantage change with proof depth (test_id dist 1-4, test_depth dist 5-10, test_deep certified dist_lb 7-11)?
3. Does it transfer to held-out theories (test_theory: lattice, ring)?
4. Does the advantage survive at equal wall-clock time?

## Setup (fixed)
- Data: IsoBench v2; stratified samples in docs/experiments/E2/ids-*.txt (test_id 300, test_depth 300, test_deep 100, test_theory 300; seed 0).
- Model: runs/h-v1/best.pt (sha256 928015f6250a8531e6249e799507d4ca8bd433e50e28b7639aae6fa885376c00), trained only on states-v2 from the v2 train split; precision bf16-fast (deterministic; argmin agrees with fp32 on 96.8% of recorded dev batches).
- Main run (equal expansions): bfs-fwd, bfs-bidir, bestfirst-symbol_bag-w1, neural-w1@runs/h-v1/best.pt#bf16-fast; budgets 30, 100, 300, 1000; one run at 1000, smaller budgets derived (prefix property; tested for the neural searcher in fp32 and bf16-fast). Eval labels used 20,000 expansions, so the largest budget is 1/20 of the labelling budget. test_id was labelled by bfs-bidir with 3,000 expansions (circularity warning above 300); bfs-bidir on test_id is reported but not interpreted.
- Equal-time run: bfs-fwd and bestfirst-symbol_bag-w1 on the same samples with budgets up to 10,000 and --max-nodes 1500000 (memory stops are censored, as in ADR-004). Per-problem seconds come from the logged rows; the machine and worker count are recorded.
- Runs from a clean committed tree (git_dirty false), 0 errors and 0 unsound proofs required, otherwise re-run.

## Analysis (fixed before seeing results)
- Primary: solve rate with 95% Wilson intervals per split and budget; primary comparison neural-w1 vs bestfirst-symbol_bag-w1, exact McNemar, Holm-corrected within each split's table, alpha 0.05.
- Secondary: neural-w1 vs bfs-fwd; bfs-bidir as a reference; fair optimality gap; solve rate by dist (test_id, test_depth), by dist_lb (test_deep), by theory (test_theory); solve rate by maximum state length reached (<=126 tokens, as in training, vs longer), to check RoPE length extrapolation.
- Equal time: for each problem, the time each method needed to solve it; cactus plots over seconds; solve rates at matched time limits of 1, 3, 10 and 30 s per problem (derived from logged seconds; machine-dependent, reported with the hardware).
- Caveat stated up front: test_theory shares symbols (*, a, b, c) with training theories under different axioms.

## Predictions (written before running)
- test_depth: neural-w1 beats symbol_bag-w1 significantly at every budget, by a large margin.
- test_id: both high; difference small (ceiling).
- test_deep: all methods near zero at budget 1000; no significant difference.
- test_theory: no significant advantage for neural-w1 over symbol_bag-w1.
- Equal time: on test_depth the neural advantage shrinks but remains significant at 30 s per problem; at 1 s per problem symbol_bag is at least as good.
