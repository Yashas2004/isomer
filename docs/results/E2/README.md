# E2 results — learned distance heuristic vs non-neural baselines (IsoBench v2)

Plan: docs/experiments/E2-learned-heuristic.md (frozen and pushed at commit 39b9aea before any search on v2 test splits). Runs: commit ccc97a5, which differs from 39b9aea only by a .gitignore fix that tracks a test fixture. All 8 runs clean (git_dirty false), 0 errors, 0 unsound proofs. Hardware: Windows 11, 16 logical CPUs, NVIDIA GeForce RTX 3050 6GB Laptop GPU (from `isomer doctor`; not recorded in config.json), Python 3.10.11, torch 2.14.1+cu126. Model: runs/h-v1/best.pt (sha256 928015f6250a8531e6249e799507d4ca8bd433e50e28b7639aae6fa885376c00), precision bf16-fast.

## Headline
On test_depth (shortest proofs of 5–10 steps), best-first search guided by the learned heuristic solves 74.7% of problems within 1,000 expansions, against 20.0% for the same search guided by symbol_bag and 58.3% for bidirectional BFS. At every budget, every problem solved by symbol_bag was also solved by the learned heuristic (b = 0 at all four budgets; Holm p ≤ 1.4e-38). With 30 expansions the learned heuristic solves 46.3%, more than symbol_bag with 10,000 expansions (41.0%). On held-out theories there is no advantage, and at equal wall-clock time the baselines are significantly better there.

## Solve rate (%) at 30 / 100 / 300 / 1000 expansions
| split | neural-w1 | symbol_bag-w1 | bfs-fwd | bfs-bidir |
|---|---|---|---|---|
| test_id (dist 1–4, n=300) | 97.7 / 99.3 / 99.7 / 100.0 | 70.3 / 80.7 / 87.7 / 93.0 | 56.3 / 70.3 / 83.3 / 91.3 | 93.0 / 100 / 100 / 100 * |
| test_depth (dist 5–10, n=300) | 46.3 / 57.7 / 67.3 / 74.7 | 3.0 / 4.3 / 10.7 / 20.0 | 0.0 / 1.7 / 4.3 / 8.7 | 6.0 / 25.7 / 41.0 / 58.3 |
| test_deep (dist_lb 7–11, n=100) | 0 / 1 / 1 / 2 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| test_theory (lattice, ring, n=300) | 31.3 / 41.3 / 49.0 / 56.3 | 27.7 / 43.0 / 49.3 / 55.3 | 21.0 / 31.0 / 39.7 / 46.3 | 50.3 / 62.0 / 72.7 / 80.3 |

* test_id was labelled by bfs-bidir with 3,000 expansions; bfs-bidir on test_id is reported but not interpreted.

## Predictions vs outcomes
| # | Prediction | Outcome |
|---|---|---|
| 1 | test_depth: neural-w1 beats symbol_bag-w1 significantly at every budget, by a large margin | Confirmed. Solved only by neural / only by symbol_bag: 130/0, 160/0, 170/0, 164/0 at 30/100/300/1000; Holm p ≤ 1.4e-38. |
| 2 | test_id: both high; difference small (ceiling) | Falsified. Neural is significantly better at every budget, including 1000 (21/0, Holm p 4.8e-6): 100% vs 93.0%. |
| 3 | test_deep: all near zero; no significant difference | Confirmed. 2/100 vs 0/100 (p = 0.5). Both neural solves are problems the labeller (bidirectional BFS, 20,000 expansions) did not solve; the proofs are kernel-checked. |
| 4 | test_theory: no significant advantage for neural-w1 | Confirmed. No budget is significant (Holm p ≥ 0.46). |
| 5 | Equal time: on test_depth the advantage shrinks but remains significant at 30 s; at 1 s symbol_bag is at least as good | First part: significant at 30 s (51/0 on the 183 problems uncensored for both, Holm p 2.7e-15); "shrinks" is not clearly supported because of censoring. Second part falsified: at 1 s neural solves 50.3% vs 18.0% (103/6, Holm p 2.7e-23). |

## Equal wall-clock time (solve rate % within T seconds per problem; n uncensored in brackets where < 300)
| split | method | 1 s | 3 s | 10 s | 30 s |
|---|---|---|---|---|---|
| test_depth | neural-w1 | 50.3 | 60.7 | 71.0 | 93.7 (238) |
| test_depth | symbol_bag-w1 | 18.0 | 26.3 | 35.4 (294) | 55.2 (221) |
| test_depth | bfs-fwd | 12.7 | 19.0 | 66.0 (103) | 100 (68) |
| test_theory | neural-w1 | 35.3 | 44.3 | 53.4 (298) | 74.6 (224) |
| test_theory | symbol_bag-w1 | 54.7 | 59.3 | 67.9 (296) | 82.7 (248) |
| test_id | neural-w1 | 97.7 | 98.7 | 99.7 | 100 |
| test_id | symbol_bag-w1 | 93.0 | 96.0 | 98.0 | 100 (296) |

Rates at 10–30 s are over different, partly censored subsets and are not comparable across methods; only the paired tests (problems uncensored for both) are. On test_theory, symbol_bag-w1 is significantly better than neural-w1 at every time limit, and bfs-fwd at 1–10 s (not at 30 s: 1/7 on 193 pairs, Holm p 0.070) (vs symbol_bag at 1 s: 6/64, Holm p 2.0e-12): without a search advantage, the neural heuristic's higher per-expansion cost makes it slower. On test_id neural-w1 is better than symbol_bag at 1 s (18/4, Holm p 0.030) and not significantly different at 3 s or more.

## Secondary analyses
- Depth (test_depth, budget 1000), neural-w1 / symbol_bag-w1 solved by true distance: dist 5: 69/70 vs 38/70; 6: 86/97 vs 16/97; 7: 41/58 vs 4/58; 8: 24/52 vs 1/52; 9: 3/19 vs 1/19; 10: 1/4 vs 0/4. The model was trained on distances up to 4 (training states up to 5); performance extrapolates but decays with depth.
- Proof quality: fair optimality gap (problems solved by every method) neural-w1 1.000 on test_depth (15 problems), 1.003 on test_id (260), 1.018 on test_theory (122); symbol_bag-w1 1.200, 1.086, 1.086.
- Memory: median nodes stored when solved on test_depth at budget 1000: neural-w1 374, bfs-bidir 2,421, bfs-fwd 3,123, symbol_bag-w1 6,887.
- State length: only 6 of 300 test_depth problems reached states over 126 tokens (the longest training input), and neither method solved any of them; this data cannot answer whether RoPE extrapolates to longer inputs.
- Exploratory, not pre-registered: on test_theory, neural-w1 vs symbol_bag-w1 is 82/152 vs 71/152 on lattice but 87/148 vs 95/148 on ring. Ring reuses the symbol * from the training theory group under different axioms (the shared-symbol caveat stated in the plan); untested.

## Caveats
- Times are machine-dependent and were measured under different worker counts (neural: 2 workers sharing one GPU; baselines: 4 CPU workers).
- The neural searcher's clock starts before the model loads, so the first problem of each of the 2 workers includes model loading and CUDA start-up (up to 2 rows per split; biases against neural).
- At 10–30 s many runs are censored (their expansion budget ran out first); see timecompare.md per split.
- Small strata: test_depth dist 10 has 4 problems, dist 9 has 19.
