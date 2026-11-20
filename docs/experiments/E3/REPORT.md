# Isomer E3 — What the learned heuristic extrapolates, and what it does not

## E3 final status (2026-11-18; result 5 added 2026-11-20)

This section was added after the report. The report below (2026-11-10) is unchanged. Every summary file is in `docs/experiments/E3/`.

| # | result | prediction committed | result committed | summary file |
|---|---|---|---|---|
| 1 | Saturation | none (exploratory) | `87ff411` | `seeds_summary.md` |
| 2 | d6 intervention | `4821ed0`, amended `583e232` before the states were built | `6ba5b23` (seed 0), `25450f4` (seeds 0-2) | `d6_summary.md`, `d6_family_summary.md`, `overlap_summary.md` |
| 3 | Size-matched control | `a2815ff` | `1db28a7` (1 d6 seed), `25450f4` (3 d6 seeds) | `d6_matched_summary.md`, `d6_family_summary.md` |
| 4 | Fresh confirmation | `40ebab3` (rules), `9a345d6` (sample, before any model ran) | `b68f2dc` | `fresh_prediction.md`, `fresh_summary.md` |
| 5 | Training range d5 (does the ceiling track the range?) | `cef3f69` | `3c6a6b5` | `d5_prediction.md`, `d5_summary.md` |

1. **Saturation of the baseline is not a seed effect.** h-v1 and two reseeded runs (h-v1-s1, h-v1-s2) all show the saturation and the margin collapse of section 1. Their mean prediction at true distance 7-8 is 4.69-4.76, below the training maximum of 5. What varies between seeds is the level: top-1 is 0.645-0.800 and 65-80% of problems are solved at 1,000 expansions. The verdict thresholds were chosen after seeing the results; result 4 tests the saturation again under a rule fixed in advance.
2. **Training on deeper problems removes the saturation (d6).** The d6 models have the same configuration, with training problems up to distance 6. They meet every pre-registered quantity of `d6_prediction.md`, all three seeds:
   - mean prediction 6.19-6.29 at k = 7 and 6.58-6.65 at k = 8;
   - drop to the optimal child 0.57-0.58 at k = 7.

   All-children top-1 rises to 0.948-0.964 and 298/300 problems are solved at 1,000 expansions (seed 0). Exact train/test (state, goal) overlap is 5 of 1,935 scored states, so it does not explain the gain.
3. **The d6 gain comes from depth coverage, not data volume.** The control h-d6m-s0 has as many training problems and optimizer steps as the baseline, drawn from the d6 data. It is "depth" on all five rules of `d6_matched_prediction.md`. Two caveats:
   - It lies just below every d6 seed on four of the five quantities (E2 sample): top-1 is 0.940 against 0.948-0.964.
   - It is one seed and one subset. The pre-registration says such a result is to be reported this way.
4. **Fresh confirmation: H1-H4 all supported** on 300 newly generated problems, with the decision rules committed before the sample existed. The sample uses the v2 eval-pool generator with seed 1; its ids are disjoint from v2 and d6, with the E2 per-distance counts. 1,939 states are scored; 26 unknown states are excluded.
   - H1, saturation: baseline mean prediction is 4.651-4.724 at k = 7-8, and d6 is 6.025-6.565.
   - H2, move choice: d6 top-1 is 0.949-0.971, against 0.803 for the best baseline seed.
   - H3, search: d6 solves 261-281 problems against 171 at 30 expansions, and 290-294 against 234 at 1,000. All 18 Holm-adjusted McNemar tests give p < 1e-15.
   - H4, control: "depth" on all five rules. As on the E2 sample, the control is below every d6 seed on some quantities (three of five here). For example, its top-1 is 0.936 against 0.949-0.971.
5. **Training range d5: not supported as predicted, but the ceiling follows the range** (one seed, h-d5-s0, trained on problems up to distance 5; prediction `cef3f69`, committed before any d5 state existed). H-d5c fails: all-children top-1 is 0.793 against the required > 0.800. That value lies inside the baseline seed range (0.645-0.800), and the baseline seeds alone span 0.155. H-d5a and H-d5b hold: the mean prediction is 5.493 at k = 7 and 5.683 at k = 8, between the baseline and d6, and its highest mean prediction at k >= 6 is 5.794. The plateaus are 4.703-4.769 (baseline, problems up to 4), 5.794 (d5, up to 5) and 6.671-6.841 (d6, up to 6), so the ceiling rises about one step per step of training range. This is one seed between two three-seed ranges.

**Still open.**
- **No objective or encoding intervention was run.** These are the interventional part of the original hypothesis in section 1.
- **The symbol-token findings (section 1, point 4) were not retested.**
- **Limited generality:** all results come from one generator, two theories and one action space.

---

Date: 2026-11-10. Companion data: `ISOMER_E3_RESULTS.json` (every number below is taken from it or from the repository's committed E2 documents; experiment ids in brackets refer to its `experiments[]`).

Labels used throughout: **Established** (published in the repository, not rerun), **Reproduced** (rerun here, matches), **Newly measured** (E3), **Proposed** (not run), **Not verified**.

---

## 1. Executive summary

This session ran on a 2-core CPU container with no GPU. That was enough to audit the repository, reproduce part of E2, and run the frozen-checkpoint diagnostics and the invariance tests. It was not enough to train models, so the ablations (Phase D) produced no result.

The main findings, all on the frozen E2 checkpoint h-v1:

1. **The distance estimate saturates at the training maximum; the move ordering does not collapse.** On states of test_depth problems, the mean prediction is 4.17, 4.51, 4.70, 4.78, 4.78 for true distances 5, 6, 7, 8, 9 (training states go up to 5). The signed bias grows from −0.83 to −4.23. Yet over *all* legal children, the model still ranks an optimal child first at 64–90% of states with distance 5–9, against 21–52% for symbol_bag.
2. **The ranking is done with a vanishing margin at depth.** The predicted drop from a state to its optimal child is 1.57 at distance 1 and 0.07 at distance 7 and 8. This explains the depth decay of E2: beyond about five steps from the goal the heuristic is nearly flat, and search degrades towards uninformed search.
3. **Move accuracy along the proof predicts search success.** On 135 problems, problems where every on-path state is ranked correctly were solved 31/36 at 30 expansions; with one error 17/32, two errors 5/23, three or more 4/44 (AUC of the per-problem accuracy: 0.83).
4. **The knowledge is tied to symbol tokens, not to the rewrite graph.** Writing group theory with boolean symbols (a kernel-certified isomorphic problem) drops top-1 from 0.99 to 0.63; boolean algebra written with ring symbols drops from 0.80 to 0.38. Permuting skolems changes nothing. On real held-out theories the model's top-1 is 0.30, the same as symbol_bag's 0.27.
5. **Even in distribution the model is not a calibrated distance.** 67% of parent–child edges on test_id break |h(u) − h(v)| ≤ 1, because the ranking loss pushes non-optimal children far up and optimal ones down (the goal itself is predicted below zero).
6. **Reproduction.** Regenerated E2 samples match the published strata exactly; non-neural baselines on test_depth match E2 at all four budgets; neural search at 30 expansions solves 139/300 = 46.3%, the published figure.

The primary hypothesis is supported in its descriptive part (partial structure, limited extrapolation, constrained by representation and distribution shift). Its interventional part (H4, H5: that a different objective or encoding fixes it) is untested.

## 2. Repository and environment audit

**Repository.** Commit `77ae48c9fad4ae150ab72fe2debb20e58d20e36b` on `main`, clean tree: unchanged from the previously reviewed commit. E3 work is on a local branch `e3` (two commits, head `d00724d`), not pushed.

**Checkpoint.** `best.pt` from release v0.2-e2, SHA-256 `928015f6…376c00`: matches the published hash. Configuration from `docs/results/train-h-v1/config.json`: 6 layers, d=256, 4,731,649 parameters, Huber + listwise ranking (λ=1, τ=1, target `optimal`), 20 epochs, seed 0, 551,072 training states.

**Contracts confirmed in code.**

| distinction | what the code does |
|---|---|
| Problem distance vs state distance | Train problems have exact dist ≤ 4. Off-path successors of the start state can be one step further, so training *states* reach distance 5 (the training log has a validation MAE row for distance 5). |
| Exact vs bounded labels | `dist` is exact (bidirectional BFS met) or null; `dist_lb ≤ dist ≤ dist_ub` always present. test_deep has no exact distance and was not used for any calibration here. |
| Concrete rewrite graph | Every distance is relative to the problem-level pool (ADR-003) and size cap max(|lhs|,|rhs|)+6. This is not unconstrained equational provability. |
| Accuracy vs move choice | Training validation ranks *sampled* sibling groups (on-path child plus ≤ 6 random off-path children). Search ranks all children. E3 measures the latter. |
| Expansions vs time | Separate analyses in E2 (`timecompare`). E3 reports expansions only; this machine's times are not comparable. |

**Reproducibility issues found.**

- `isomer states` defaulted to `--offpath 3`, but the published states-v2 (`docs/datasets/states-v2.stats.json`) used 6. **Fixed**: the default is now 6.
- E2 committed no per-problem logs (`results.jsonl`), so per-problem agreement with E2 cannot be checked by anyone. Recommend committing them.
- The project's `addopts = -q` plus a user `-q` suppresses pytest's summary line; use `--junitxml` or no extra `-q` when recording counts.

**Environment.** Linux container, Intel Xeon 2.1 GHz, 2 vCPU, 7.8 GiB RAM, no GPU, Python 3.12.3, torch 2.15.0 (CUDA build from PyPI, run on CPU; the PyTorch CPU index was blocked). Model throughput: 22 sequences/s in fp32, 62 in bf16-fast.

**Smoke tests** [A2]. 248 non-slow tests, 0 failures, 3 skipped (need CUDA); slow tests not run. `demo`: kernel accepts, tampered proof rejected. `prove --example bool-deep`: 6-step proof in 6 expansions, kernel accepted. With symbol_bag: no proof in 1,000 expansions (expected).

## 3. Verified existing results

**Established** (docs/results/E2, 1,000 expansions): the table in the brief matches the repository. test_id 100.0 / 93.0 / 100.0\*, test_depth 74.7 / 20.0 / 58.3, test_deep 2/100 / 0/100 / 0/100, test_theory 56.3 / 55.3 / 80.3 (neural / symbol_bag / bfs-bidir; \* labeller, not interpreted).

**Reproduced** here:

| check | result |
|---|---|
| Regenerate the three 300-problem E2 samples from the id files [A1] | all 900 ids found; every stratum count equals the id file's header (test_depth 70/97/58/52/19/4; test_id 52/77/86/85; test_theory all 18 theory×distance strata) |
| Baselines on test_depth, 30/100/300/1,000 [A3] | symbol_bag 3.0/4.3/10.7/20.0, bfs-fwd 0.0/1.7/4.3/8.7, bfs-bidir 6.0/25.7/41.0/58.3: identical to E2. symbol_bag by distance also identical (38/70, 16/97, 4/58, 1/52, 1/19, 0/4) |
| neural-w1 on test_depth at 30 expansions, all 300 problems | 139/300 = 46.3% (Wilson 95% 40.8–52.0). Published: 46.3% |

**Not reproduced:** neural results at 100/300/1,000 expansions, test_id and test_theory search, test_deep, all wall-clock results, and training. The neural reproduction used CPU bf16 rather than CUDA bf16; the count agrees, per-problem agreement is unverifiable.

## 4. Literature review and novelty comparison

Searched November 2026. Numbers from these papers are on unrelated benchmarks and are not compared with Isomer's.

| work | setting | what it already establishes |
|---|---|---|
| Piepenbrock, Heskes, Janota, Urban, *Guiding an ATP with Neural Rewriting* (IJCAR 2022) | Neural policy over concrete rewrite actions on AIM loop problems, trained by 3SIL (imitating shortest found solutions); its intermediate equalities are given to Prover9 as lemmas | Learned guidance of equational rewrite steps is not new. It is a policy with no distance function. Variables get random vectors resampled per episode, a symbol-identity remedy. Closest prior proving setting. |
| Wu, Jiang, Ba, Grosse, *INT* (ICLR 2021) | Synthetic inequality proving with generator knobs for six kinds of generalization | Controlled synthetic generalization studies in proving are not new. |
| *ENIGMA Anonymous* (Jakubův et al., 2020); Olšák, Kaliszyk, Urban, *Property Invariant Embedding* (2020) | Clause selection in E; features and GNNs that ignore symbol names | Symbol-independent encodings as the answer to cross-problem transfer are established. |
| Chvalovský, Suda, Urban, *Teaching Vampire New Tricks* (arXiv 2609.16228) | Neural clause selection in Vampire on four corpora | Guidance helps in-distribution and usually does not transfer across corpora; the authors leave the role of symbol encodings to future work. |
| Axelrod, Johansson, Smallbone, *Twitch* (IJCAR 2026) | Twee on TPTP UEQ with abstractions learned from proofs | Learning for real UEQ problems is active; Twee is the reference prover for any external comparison. |
| Petruzzellis, Testolin, Sperduti, *Learning neuro-symbolic convergent term rewriting systems* (arXiv 2507.19372) | Neural normalization (no search), trained on nesting 1–3, tested on 4–6 | Depth extrapolation of neural rewriting to normal form; a different task from proving s = t. |
| *egg* (POPL 2021); Koehler et al., *Guided Equality Saturation* (POPL 2024); Heinimann et al., *ML Guided EqSat* (EGRAPHS 2025) | Equality saturation, mostly program optimization; human or learned guides | An e-graph decides equality without producing a rewrite sequence under Isomer's pool and size cap: a different search problem. |
| Chrestien, Edelkamp, Pevný, Komenda, *Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal* (NeurIPS 2023) | A*/GBFS in planning domains | Ranking losses beat regression for search, with a theorem that a perfect ranking gives optimally efficient search. Finding 1 here is consistent with it and must cite it. |
| Sahay, *Learning Empirically Admissible Neural Heuristics* (arXiv 2606.04860) | Puzzles; underestimating Bellman operator | Bellman-style objectives for learned heuristics exist. |
| *LeanProgress* (arXiv 2502.17925) | Predict remaining steps in Lean, use in search | Steps-to-go prediction is established at LLM scale; its error also grows with proof length. |

**Unoriginal to claim:** a learned heuristic beating hand-written ones; ranking mattering more than regression; symbol-independent encodings helping transfer; a consistency or Bellman penalty as a new loss.

**What Isomer can add.** Most proving benchmarks cannot say which next step is optimal. Isomer can, exactly, for every legal child. That turns three things the literature states qualitatively into measurements: how the numeric estimate and the ordering separate under length extrapolation, how small the margin becomes, and how much of a symbol-identity model's skill survives a certified isomorphic renaming.

**Practical baselines.** size_diff, symbol_bag and both BFS variants are implemented. Twee needs a TPTP translation and a binary. egg/egglog would be an equality-decision baseline only.

## 5. Hypotheses and protocol

Primary hypothesis and H1–H5 as given in the brief. Nothing in E3 is prospectively pre-registered: the E2 outcomes were known, the E2 test samples were reused, and analysis choices were made during the session. All E3 results are therefore **exploratory**.

Fixed before scoring:

- **Populations.** The E2 stratified samples (ids committed at `cd3d517`). States are those on the stored shortest proof of each problem. Problems with exact distance only.
- **Primary metric.** `top1_optimal` over all distinct legal children within the size cap, ties split (the repository's convention). A child is optimal when its exact distance is one less than the state's.
- **Labels.** Exact, by a complete backward BFS ball around the goal joined with forward BFS layers that carry child bitmasks (`e3/children.py`). A state whose search exceeds 300,000 nodes is recorded as unknown and excluded. A test checks every label against an independent per-child BFS.
- **Uncertainty.** Wilson intervals for proportions; cluster bootstrap over problems for state-level means; exact McNemar for paired search outcomes.

Deviation: test_depth child labelling was stopped for time at 135 of 300 problems (first 104 in id-hash order, the rest by seeded round-robin over theory × distance). It is a subset, and deeper strata are over-represented relative to the 300-problem sample (32/38/27/22/13/3 against 70/97/58/52/19/4).

## 6. Experiments executed

| id | what | size | status |
|---|---|---|---|
| A1 | Regenerate and relabel E2 samples | 3 × 300 problems | done |
| A2 | Tests and smoke commands | 248 tests | done |
| A3 | Baselines on test_depth | 300 × 4 methods | done |
| B (test_id) | All-children labels and scores | 300 problems, 804 states, 18,907 children | done |
| B (test_depth) | same | 135 problems, 884 states (16 unknown), 27,409 children | subset |
| B (test_theory) | same, distance 2–4 | 100 problems, 314 states | subset |
| B (search) | neural-w1 at 30 expansions | 300 problems | done |
| B (search variants) | neural-greedy, neural-w3 at 30 | 135 problems | done |
| C (scores) | Six renamings, paired | 300 test_id states, 7,143 children each | done |
| C (search) | neural-w1 on renamed problems at 30 | 135 (skolem), 57 (dual) | done |
| D | Objective ablation pilots | 1 partial run | stopped, inconclusive |
| E | Twee/E, equality saturation, multi-seed | — | not run |

Search runs record `git_dirty: true` because analysis code was still uncommitted while they ran; no searcher, kernel or model code differs from `77ae48c`.

## 7. Numerical results and interpretation

### 7.1 Calibration and move choice by distance (Newly measured)

States of test_depth problems (135 problems), grouped by the state's own exact distance k. "Mean h" is the model's prediction for the state. "Drop" is h(state) − h(optimal child), which is 1 for a true distance.

| k | states | children/state | chance | h-v1 top-1 (95% CI) | symbol_bag top-1 | mean h | bias | drop | margin |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 135 | 46.3 | 0.025 | 0.830 (0.76–0.89) | 0.145 | 0.45 | −0.55 | 1.57 | −1.16 |
| 2 | 135 | 42.0 | 0.046 | 0.741 (0.67–0.82) | 0.167 | 1.66 | −0.34 | 1.11 | −0.72 |
| 3 | 135 | 36.8 | 0.077 | 0.748 (0.67–0.82) | 0.158 | 2.76 | −0.24 | 1.02 | −0.59 |
| 4 | 135 | 30.2 | 0.116 | 0.637 (0.56–0.72) | 0.168 | 3.58 | −0.42 | 0.77 | −0.38 |
| 5 | 135 | 23.5 | 0.229 | 0.681 (0.60–0.76) | 0.250 | 4.17 | −0.83 | 0.58 | −0.35 |
| 6 | 103 | 20.5 | 0.241 | 0.641 (0.54–0.73) | 0.207 | 4.51 | −1.49 | 0.30 | −0.20 |
| 7 | 61 | 13.7 | 0.342 | 0.656 (0.53–0.77) | 0.326 | 4.70 | −2.30 | 0.07 | −0.15 |
| 8 | 33 | 8.4 | 0.477 | 0.788 (0.64–0.91) | 0.432 | 4.78 | −3.22 | 0.07 | −0.31 |
| 9 | 10 | 3.5 | 0.641 | 0.900 (0.70–1.00) | 0.520 | 4.78 | −4.23 | 0.02 | −0.18 |

"Chance" is the mean fraction of children that are optimal. "Margin" is min h over optimal children minus min h over non-optimal children (negative means the right move is preferred). Overall: h-v1 0.717 (0.676–0.758), symbol_bag 0.207, size_diff 0.124. Spearman correlation between prediction and true distance over all on-path states is 0.855, so the order of distances is largely kept while the scale is lost.

Reading:

- **Saturation.** Predictions stop rising at about 4.8, just under the largest training distance. Beyond k = 5 the number carries almost no information about how far the goal is.
- **Ordering survives, weakly.** Top-1 stays well above both chance and symbol_bag at every k. Corrected for chance, (top-1 − chance)/(1 − chance) falls from 0.83 at k = 1 to 0.59 at k = 4–5 and 0.48 at k = 7. The higher raw values at k = 8–9 are mostly chance: those states have very few children.
- **The margin is the casualty.** The preference for the right child shrinks from 1.16 to about 0.2. Best-first search with priority g + h needs h to fall by about 1 per step to keep following a path; with a drop of 0.07 the g term dominates and shallow nodes are expanded first.

In distribution (test_id, 804 states): top-1 0.908 (0.883–0.930) against symbol_bag 0.299 and size_diff 0.168; by distance 0.973, 0.903, 0.871, 0.765 for k = 1–4. The validation figure in the training log (0.952) is over sampled sibling groups of at most 7; over all children (23.5 on average) it is 0.908.

### 7.2 What goes wrong when the choice is wrong (Newly measured)

Share of states by where the lowest-scored child leads:

| split | heuristic | optimal | sideways (same distance) | away (+1) | non-optimal, distance not resolved |
|---|---|---:|---:|---:|---:|
| test_id | h-v1 | 0.908 | 0.005 | 0.020 | 0.067 |
| test_id | symbol_bag | 0.299 | 0.039 | 0.218 | 0.443 |
| test_depth | h-v1 | 0.717 | 0.051 | 0.071 | 0.161 |
| test_depth | symbol_bag | 0.207 | 0.040 | 0.370 | 0.383 |

Errors grow with the number of children (test_depth: 0.789 below 25 children, 0.711 for 25–49, 0.556 for 50–99) and with term size (0.776 below size 10, 0.538 at 20 or more, n = 26). Bool and group are similar on test_depth (0.701 and 0.728); on test_id bool is clearly harder (0.842 against 0.986).

### 7.3 Edge consistency of the frozen model (Newly measured)

True distances across one rewrite differ by at most 1. Share of parent–child edges where the model's predictions differ by more than 1:

| split | k = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_id | 0.84 | 0.62 | 0.45 | 0.37 | – | – | – | – | 0.665 |
| test_depth | 0.74 | 0.55 | 0.42 | 0.31 | 0.25 | 0.16 | 0.09 | 0.04 | 0.447 |

Near the goal the model over-separates: optimal children are pushed down (the goal is predicted at about −1.1) and non-optimal children pushed up. That is what a softmax ranking loss rewards, and it is why the "distance" is not a distance even in distribution. Far from the goal the opposite holds: everything is predicted near 4.8 and the edges are trivially consistent.

This matters for Ablation 4. The proposed penalty relu(|h(u) − h(v)| − 1)² acts only where differences exceed 1, which is the near-goal regime where ranking already works. It does nothing in the deep regime, where the failure is differences that are too small.

### 7.4 Diagnostics against search outcomes (Newly measured)

neural-w1 at 30 expansions on the 135 labelled problems: 57 solved, against 5 for symbol_bag (52 solved only by neural, 0 only by symbol_bag; exact McNemar p = 4.4e-16).

| on-path states ranked wrong | 0 | 1 | 2 | 3 or more |
|---|---:|---:|---:|---:|
| solved / problems | 31/36 | 17/32 | 5/23 | 4/44 |

AUC for predicting "solved" from the per-problem mean top-1: 0.83. All-correct against not: 31/36 and 26/99 (Fisher p = 3.8e-10). Within distance strata the pattern holds (distance 6: 11/13 against 8/25; distance 7: 2/2 against 4/25). Distance alone also predicts success, so this is association, not an isolated causal effect.

Solve rate by distance at 30 expansions on all 300 problems: 61/70, 54/97, 19/58, 4/52, 1/19, 0/4.

**Search interaction (exploratory).** Same model, same 135 problems, 30 expansions: g + h solves 57, h alone (greedy) 57, g + 3h solves 64 (16 gained, 9 lost, p = 0.23). At distance 8 the counts are 2/22, 7/22 and 8/22. The direction fits the saturation account (down-weighting g helps where h is flat), but nothing here is significant.

### 7.5 Structural invariance (Newly measured)

Each renaming is applied to theory and problem together, and a test confirms that the renamed proof is accepted by the kernel and that distance and pool size are unchanged. 300 test_id states (169 bool, 131 group), all children, same states before and after.

| renaming | theory | mean \|Δh\| | correlation | top-1 before → after | difference (95% CI) | top choice changed |
|---|---|---:|---:|---|---|---:|
| Permute skolems | group | 0.05 | 0.998 | 0.992 → 0.992 | 0.000 | 1.5% |
| Permute skolems | bool | 0.12 | 0.979 | 0.799 → 0.828 | +0.030 (−0.011, +0.067) | 11.2% |
| Skolems → another trained theory's | group | 0.26 | 0.947 | 0.992 → 0.992 | 0.000 (−0.023, +0.023) | 9.9% |
| Skolems → another trained theory's | bool | 0.22 | 0.942 | 0.799 → 0.775 | −0.024 (−0.069, +0.018) | 15.4% |
| Skolems → never-trained tokens (x, y, z) | group | 0.37 | 0.856 | 0.992 → 0.947 | −0.046 (−0.088, −0.008) | 19.8% |
| Skolems → never-trained tokens | bool | 0.38 | 0.827 | 0.799 → 0.657 | −0.142 (−0.219, −0.067) | 35.5% |
| Dual (& ↔ \|, T ↔ F) | bool | 0.43 | 0.823 | 0.799 → 0.864 | +0.065 (0.000, +0.131) | 34.9% |
| Group written in bool symbols | group | 0.82 | 0.591 | 0.992 → 0.626 | −0.366 (−0.446, −0.288) | 51.9% |
| Bool written in ring symbols | bool | 0.85 | 0.367 | 0.799 → 0.379 | −0.420 (−0.506, −0.333) | 72.8% |

Search on renamed test_depth problems at 30 expansions: skolem permutation 57/135 (original 57/135; 3 gained, 3 lost); dual 21/57 (original 21/57; 6 gained, 6 lost). symbol_bag's outcomes did not change on any problem.

Reading:

- Skolem permutation is a symmetry the model has learned, exactly in group and approximately in bool.
- Duality is a symmetry of boolean algebra that the training data contains, but the model's outputs are not invariant to it: a third of top choices change and 12 of 57 search outcomes flip, with no net loss.
- The last two rows keep the rewrite graph identical and change only which tokens label it. Losing 37 and 42 points shows the model's skill is attached to the tokens. The model has no input that says which axioms hold, so this is a limit of the representation as much as of the weights.
- Never-trained tokens cost 5–14 points. Their embeddings are at initialization, so this is also a representational limit and not a failure the model could have avoided.

### 7.6 Held-out theories (Newly measured)

100 test_theory problems with distance 2–4, all children labelled (314 states).

| theory | children/state | chance | h-v1 top-1 (95% CI) | symbol_bag (95% CI) | Spearman(h, distance) |
|---|---:|---:|---|---|---:|
| lattice | 68.4 | 0.056 | 0.322 (0.237–0.407) | 0.188 (0.152–0.225) | 0.45 |
| ring | 20.6 | 0.156 | 0.273 (0.199–0.349) | 0.338 (0.294–0.386) | 0.08 |

Overall 0.296 against 0.267; difference +0.029 (−0.026, +0.086). The model's move accuracy falls from 0.91 in distribution to the level of a symbol-counting heuristic, which is exactly the E2 tie. In ring, where `*` is reused under different axioms and every other symbol is untrained, the prediction is almost unrelated to distance (mean h is 2.36, 2.34, 2.42, 2.75 for true distances 1–4). The lattice-ahead, ring-behind pattern that E2 flagged as exploratory appears again in move accuracy.

## 8. Statistical analysis

- All intervals are 95%. State-level means use a cluster bootstrap over problems (2,000 resamples, seed 0), since states of one problem are dependent.
- Paired search comparisons use the exact McNemar test. The two renamed-search tests are one family; both raw p-values are 1.0, so Holm correction changes nothing.
- No correction is applied across the diagnostic tables. They are descriptive and exploratory, and should be read by effect size and interval, not as a set of confirmed tests.
- Censoring: no time or memory limits were used in E3 runs, so no search result is censored. 16 of 900 test_depth states hit the labelling node cap and are excluded as unknown; they are likely the highest-branching states, which biases test_depth top-1 slightly upward.
- No run, problem or seed was discarded. The one stopped training run is reported in section 10.

## 9. Strengths and failure mechanisms

| candidate cause from the brief | verdict | evidence |
|---|---|---|
| Numerical underestimation or saturation | **Yes, dominant at depth** | mean h ≈ 4.8 for all k ≥ 7; bias to −4.2 |
| Incorrect local rankings | **Partly** | top-1 0.64–0.68 at k = 4–7 against 0.91 in distribution; errors compound over longer proofs and predict failure (7.4) |
| Insufficient coverage of long-distance states | **Consistent, not tested** | saturation sits at the training maximum; needs a model trained on deeper states |
| Expression length or tokenization shift | **Minor here** | top-1 falls with size, but few states are large; E2 found only 6/300 searches beyond the training length |
| Dependence on symbol identities | **Yes, dominant for transfer** | re-signing: −37 and −42 points; held-out theories at symbol_bag level |
| High branching | **Yes, moderate** | 0.79 → 0.56 from under 25 to 50–99 children |
| Search algorithm interaction | **Suggestive** | g + 3h and greedy solve 8/22 and 7/22 at distance 8 against 2/22 for g + h; not significant |

Strength: over the full action set, including states beyond the training range, the model's first choice is optimal about three times as often as symbol_bag's, and the optimal child's mean rank is 1.9 of 31.

## 10. Ablation findings

**No ablation result.** A CPU pilot of the original objective (595,585 parameters, 2,861 problems from a faithful subset of the train split, seed 0) reached validation top-1 of 0.38, 0.30 and 0.34 after epochs 1–3, below symbol_bag's 0.50 on the same groups. A pilot that does not reach the hand-written baseline cannot rank objectives, so it was stopped and the regression-only, ranking-only and consistency runs were not started. Ablations 1–4 remain **Proposed**; commands are in section 15.

What the frozen-model evidence says about the choice of intervention (7.3): the consistency penalty as written targets the regime that already works. Two alternatives follow more directly from the measurements:

- **For depth:** make the deep regime non-flat. Either label deeper training states, or bootstrap with a Bellman target h(u) ≈ 1 + min over children h(v) on unlabelled deeper states, which needs no exact labels.
- **For transfer:** remove the token binding. Train with random arity-preserving symbol renaming (the re-signing transformation of 7.5 as augmentation), or condition the model on the axioms.

## 11. Methodological improvements implemented

Source changes on branch `e3`, with tests:

| change | file | test |
|---|---|---|
| Regenerate any published stratified sample without the 1.7-hour full build | `e3/regen.py` | strata check against the id file |
| Exact optimal labels for every legal child | `e3/children.py`, `e3/label_children.py` (resumable, stratified order) | `tests/test_e3_children.py` (12 cases against per-child BFS) |
| Kernel-certified renamings | `e3/rename.py` | `tests/test_e3_rename.py` (12 cases) |
| Calibration, all-children ranking, choice classes, edge consistency, paired invariance | `e3/diagnose.py`, `e3/analyze.py` | exercised by the runs |
| Faithful train subset for pilots | `e3/pilot_data.py` | drop rates match the full build (19% by distance, 5% to test_id) |
| `--lambda-huber`, `--lambda-consistency` | `learn/train.py`, `learn/data.py`, `cli.py` | `tests/test_train_e3.py`: defaults give the h-v1 gradient exactly |
| `states --offpath` default 3 → 6 | `cli.py` | existing tests pass |

The kernel, rewrite semantics, pool, splits and existing results are untouched.

## 12. Contributions supported by evidence

1. An exact, all-children measurement of a learned proof-distance heuristic under length extrapolation, showing that the estimate saturates at the training maximum while the ordering degrades slowly and the margin collapses.
2. A per-problem link between on-path move accuracy and search success.
3. A certified-isomorphism test showing that the heuristic's skill is bound to symbol tokens, with a matching measurement on real held-out theories.
4. The observation that a ranking-trained "distance" violates the unit-step property on two thirds of in-distribution edges.
5. A reproduction path for the benchmark samples that needs no full rebuild, and a partial independent reproduction of E2.

Each rests on one checkpoint, one seed and one benchmark.

## 13. Weaknesses and threats to validity

- **One model, one seed.** Every finding describes h-v1. Whether saturation and token binding are properties of the method or of this run is unknown.
- **On-path states only.** Diagnostics use states on one stored shortest proof. Search visits other states, where accuracy may differ.
- **Subset.** test_depth diagnostics cover 135 of 300 problems, with the deviation described in section 5.
- **Depth is confounded with branching.** A problem gets an exact deep label only if bidirectional BFS could solve it, which favours low-branching problems (3.5 children per state at k = 9). "Deeper" in test_depth also means "narrower".
- **Exploratory analysis** on test samples already seen in E2. A confirmatory test needs freshly generated problems.
- **Short budget for the linkage.** 30 expansions only.
- **Synthetic benchmark, no external prover**, as in E2.
- **Re-signing mixes two effects:** unfamiliar token–dynamics pairing and, for ring symbols, untrained embeddings. The group-in-bool-symbols case uses only trained tokens and still loses 37 points.

## 14. Publication readiness

**Not ready for a journal.** What exists now is a careful benchmark, one trained model, and a diagnostic study with clear effects. Reviewers would ask for:

1. Three or more seeds of the base model, with the section 7 diagnostics repeated.
2. At least one intervention that tests the diagnosis: a model that fixes saturation or token binding, with the predicted gain in search.
3. Confirmatory evaluation on new, independently generated problems with a pre-registered plan.
4. A stronger baseline: Twee or E on translated problems, with the difference in search semantics stated.
5. The objective ablations (regression-only, ranking-only).

With items 1–3 this is a credible workshop or short conference paper on generalization of learned proof heuristics. A journal submission needs all five.

## 15. Ranked next experiments

Times are for the reference RTX 3050 laptop unless stated.

1. **Seeds for the base model** (about 2.5 GPU-hours each, 3 GB VRAM).
   `python -m isomer train --states data/states-v2 --out runs/h-v1-s1 --epochs 20 --seed 1` (and `--seed 2`)
2. **Diagnostics for each seed, full test_depth sample** (labelling about 5 CPU-hours on 2 cores, less with more; scoring minutes on GPU).
   `python -m isomer.e3.regen test_depth docs/experiments/E2/ids-test_depth.txt runs/e3/data/test_depth.jsonl 4`
   `python -m isomer.e3.label_children runs/e3/data/test_depth.jsonl runs/e3/labels/test_depth.jsonl 4`
   `python -m isomer.e3.diagnose score runs/e3/labels/test_depth.jsonl OUT.jsonl --ckpt runs/h-v1-s1/best.pt --device cuda --precision bf16-fast`
   `python -m isomer.e3.analyze base OUT.jsonl`
3. **Objective ablations, 3 seeds each** (about 22 GPU-hours).
   `python -m isomer train --states data/states-v2 --out runs/abl-reg-s0 --epochs 20 --seed 0 --lambda-rank 0`
   `python -m isomer train --states data/states-v2 --out runs/abl-rank-s0 --epochs 20 --seed 0 --lambda-huber 0`
   `python -m isomer train --states data/states-v2 --out runs/abl-cons-s0 --epochs 20 --seed 0 --lambda-consistency 1.0`
   The ranking-only model has no anchored scale; evaluate it with `neural-greedy`.
4. **Deeper training states** (build about 2 CPU-hours plus training): raise `train_max_dist` to 6 in a new dataset version and retrain. Tests the coverage explanation of saturation directly. Needs a small `BuildConfig` flag in the CLI.
5. **Renaming augmentation** (needs about a day of code: apply `e3/rename.py` mappings inside `learn/data.collate`): tests H2 and H5 by intervention, evaluated on test_theory.
6. **Weighted search at full budget** (about 2.5 GPU-hours, no training): `neural-w3` and `neural-greedy` at 30/100/300/1,000 on test_depth, to settle the exploratory result of 7.4.
   `python -m isomer bench --data data/isobench-v2 --split test_depth --ids docs/experiments/E2/ids-test_depth.txt --methods "neural-w1@runs/h-v1/best.pt#bf16-fast,neural-w3@runs/h-v1/best.pt#bf16-fast,neural-greedy@runs/h-v1/best.pt#bf16-fast" --budgets 30,100,300,1000 --workers 2 --out runs/E3-weights`
7. **Twee baseline:** export problems to TPTP UEQ with the theory's axioms, run Twee with a time limit, and report it as provability under unrestricted equational reasoning, not as a like-for-like search comparison.

## 16. Paper outline and title

**Suggested title:** *Ranking Outlives Calibration: What a Learned Distance Heuristic Extrapolates in Equational Proof Search*

1. Introduction: learned heuristics extrapolate in proof length, but what exactly extrapolates?
2. Setting: symmetric rewrite graph, certified distances, trusted kernel.
3. Exact all-children labelling.
4. E2 recap: search results and their reproduction.
5. Length extrapolation: saturation of the estimate, survival of the ordering, collapse of the margin; link to search success.
6. Symbol binding: certified renamings and held-out theories.
7. Interventions: deeper or bootstrapped targets; renaming augmentation (to be run).
8. Related work (section 4).
9. Limitations.
