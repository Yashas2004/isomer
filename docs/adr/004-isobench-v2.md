# ADR 004 — IsoBench v2: separate pools, decoupled labels, memory cap

**Status:** accepted · **Date:** 2026-10-28 · **Extends:** ADR-002 decision 3 (labels)

## Context
E1 (`docs/results/E1/`) exposed three problems with IsoBench v1:

1. **Circular labels.** Labels and the test_depth split came from
   bidirectional BFS at 3000 expansions, the same as E1's largest benchmark
   budget. At 3000, bfs-bidir solved exactly the labelled problems
   (test_depth 1299 = 1711 − 412 unlabelled; test_theory 543 = 600 − 57).
   Its numbers say nothing about bfs-bidir as a baseline.
2. **No deep problems.** test_depth had 55 problems with dist ≥ 8 (dist 8: 54,
   dist 9: 1). Depth generalization (RQ3) cannot be studied on that.
3. **Too little training data.** 2,053 train problems is too few for a neural
   model.

## Decision
1. **Three pools, each with its own walks and labeller**:

   | pool | theories | walks k | per theory | labeller (bidirectional BFS) |
   |---|---|---|---|---|
   | train | group, bool | 1..6 | 20,000 | 3,000 expansions |
   | eval | group, bool | 6, 8, 10, 12, 16, 20, 24 | 1,200 | 20,000 expansions, 1.5M stored nodes |
   | held-out | lattice, ring | 2..12 | 300 | same as eval |

   - **Train** keeps only problems with exact dist ≤ 4, split by hash into
     `train` and `test_id` (5%).
   - **Eval** goes to `test_depth` (exact dist ≥ 5) or `test_deep` (no exact
     dist); eval problems with dist ≤ 4 are dropped, since they would
     duplicate test_id.
   - **Held-out** all goes to `test_theory`.
   - The per-theory counts are problems *generated*; the datasheet reports
     what each split kept and what was dropped.
2. **Labels are decoupled from benchmark budgets.** Eval and held-out labels
   use 20,000 expansions, 20× the largest recommended benchmark budget (1,000).
   The benchmark runner reads the per-split labelling budget from
   `datasheet.json` and prints a WARNING, recorded as
   `circularity_warning` in `config.json`, whenever
   `max(budgets) > label_budget / 10`. The guard is per split: test_id was
   labelled with 3,000 expansions, so benchmarking it above 300 is flagged,
   and rightly so, because test_id membership is conditioned on bidirectional
   BFS succeeding within 3,000.
3. **Every record carries bounds** `dist_lb ≤ dist ≤ dist_ub`. `dist` is exact
   or null. `dist_lb` is proven from completed BFS layers. `dist_ub` is the
   length of the best known, kernel-checked proof (the BFS proof, else the
   walk). Records also carry `label_stop` (`solved`, `budget`, `memory`,
   `exhausted`) and `label_nodes`. **test_deep is graded by `dist_lb`**,
   which is certified, never by `dist_ub`.
4. **Walks outside the action space are rejected, not confined.** Walks are
   generated as in v1 (term size up to `max_term_size` = 30). Then every term
   on the walk is checked against the labeller's size cap,
   max(|lhs|, |rhs|) + `max_size_slack`, and a walk that leaves the cap is
   rejected (`build.walk_in_action_space`). An accepted walk is a legal proof
   in the action space, so its length bounds `dist` from above and
   `dist_lb ≤ dist ≤ dist_ub` holds. Without this check, a walk could pass
   through terms the labeller never explores, and `dist_lb > dist_ub` would be
   possible. The datasheet records the number rejected per pool
   (`walks_rejected_outside_size_cap`).

   *Rejected alternative: confining walks while they run* (no term larger
   than `size(start) + slack`). It guarantees the same bound but destroys
   depth: the cap also counts the final term, so confinement is far stricter
   than needed, and confined walks mostly cancel themselves. Measured on 30
   eval-pool walks per theory, labelled with the real settings (20,000
   expansions, 1.5M nodes, 2026-10-28):

   | eval walks, k ∈ {6..24} | dist ≥ 5 or unlabelled | dist ≤ 4 (dropped) | deepest | walks inside the cap |
   |---|---|---|---|---|
   | confined, group | 5 / 30 | 25 | dist 6 | 30 / 30 |
   | confined, bool | 14 / 30 | 16 | dist 8 | 30 / 30 |
   | unconfined (v1-style), group | 23 / 30 | 7 | dist 9; 3 unlabelled, lb up to 10 | 30 / 30 |
   | unconfined (v1-style), bool | 27 / 30 | 3 | dist 8; 12 unlabelled, lb up to 9 | 30 / 30 |

   All 60 unconfined walks stayed inside the cap, so rejection costs almost
   nothing while confinement would have thrown away most of the eval pool.
5. **Test pools win ties.** Held-out and eval problems are generated first.
   Train generation skips every id they claimed (including eval problems
   later dropped), so no test problem appears in training and each id lands
   in at most one split.
6. **Memory cap for every searcher.** `Budget.max_nodes` (default None) caps
   `nodes_stored`. It is checked before each expansion, next to the expansion
   budget, so it can overshoot by one expansion's successors. The stop is
   `"memory"`. BFS lower bounds still count only completed layers, so they
   stay valid. A node stop is deterministic, but the benchmark censors it
   above the expansions reached, exactly like a time stop, because the
   reported budgets are expansion budgets (`isomer bench --max-nodes`).
7. **Order of labelling.** The cheap train pool is labelled first
   (chunksize 16), then eval and held-out together (chunksize 1, so long
   problems do not queue behind each other). Progress and timing are printed
   per pool and recorded in the datasheet.

## Consequences
- **Dataset v2** (`isobench/2`, `data/isobench-v2`) is not comparable with v1:
  the splits, walks and labellers differ.
- **Train selection effect.** train and test_id contain only dist ≤ 4
  problems that bidirectional BFS solved within 3,000 expansions. A dist ≤ 4
  problem whose BFS needs more is dropped, which biases train slightly toward
  low-branching problems. With 3,000 expansions this affects few problems
  (see the datasheet's `dropped` and `label_stop`).
- **Walk rejection barely changes the distribution.** Walks are generated as
  in v1. Only walks that pass through a term larger than the size cap are
  dropped, and in the 60-walk comparison above there were none. The
  datasheet's `walks_rejected_outside_size_cap` gives the exact count for
  the build. Problems that are only reachable by detouring through
  oversized terms are excluded, which is consistent with the action space
  (ADR-002 decision 4): such detours are not proofs the labeller or any
  benchmarked searcher can find.
- **The eval pool still drops shallow walks.** Even unconfined, long walks
  can cancel themselves (7 of 30 group and 3 of 30 bool walks had dist ≤ 4).
  Those problems are dropped, so test_depth plus test_deep has fewer than
  2 × 1,200 problems. The datasheet's `dropped` field gives the count.
- **Measured cost** (2026-10-28, 16 logical CPUs, 15.7 GiB RAM):

  The sample is the first problems of each pool in real build order (seed 0).
  Each was labelled with the real settings in a fresh process, 3 at a time.
  Peak RAM is the per-process peak working set; the interpreter baseline is
  23 MiB.

  | pool | theory | n | s mean / median / max | max nodes | peak RAM | solved / budget / memory |
  |---|---|---|---|---|---|---|
  | eval | group | 20 | 6.3 / 0.4 / 22.3 | 456,654 | 725 MiB | 17 / 3 / 0 |
  | eval | bool | 20 | 12.8 / 1.6 / 62.6 | 728,527 | 916 MiB | 14 / 6 / 0 |
  | held-out | lattice | 10 | 13.4 / 0.0 / 104.1 | 1,500,141 | 1,909 MiB | 9 / 0 / 1 |
  | held-out | ring | 10 | 0.9 / 0.0 / 7.2 | 68,459 | 103 MiB | 10 / 0 / 0 |
  | train | group | 100 | 0.00 / 0.00 / 0.1 | 3,423 | 28 MiB | 100 / 0 / 0 |
  | train | bool | 100 | 0.07 / 0.00 / 2.4 | 68,695 | 102 MiB | 100 / 0 / 0 |

  - **Memory per node:** 1.17–1.54 KiB per stored node over all problems
    above 200k nodes. So the 1.5M cap means about 1.9–2.3 GiB per worker in
    the worst case.
  - **The decoupling is real.** No eval problem hit the node cap: all 9 eval
    stops used the full 20,000 expansions, with 205k–729k nodes. The one
    memory stop (lattice) came at 17,571 expansions, still 17× the largest
    recommended benchmark budget.
  - **Projected full build:** about 28,600 CPU-seconds:
    - generation, single process: 130 s
    - train: about 1,500 s
    - eval: about 22,900 s
    - held-out: about 4,300 s, dominated by rare lattice problems that hit
      the cap.

    That is roughly 2.7 h with 3 workers, 1.4 h with 6 and 1 h with 8. The
    held-out estimate is the least certain (10 problems per theory).
  - **Yield, from the sample:** 79–92% of train walks keep dist ≤ 4, giving
    about 34k train + test_id problems. About 80% of eval walks reach
    dist ≥ 5 or stay unlabelled, giving about 1.9k test_depth + test_deep
    problems.

- **The kernel is unchanged.**
