# ADR 003 — Symmetric, problem-level action space

**Status:** accepted · **Date:** 2026-10-28 · **Supersedes:** ADR-002 decision 4

## Context
ADR-002 drew fresh-variable values from the 8 smallest subterms of the *current*
term. The values a step *consumes* were unrestricted. So the action space was
not symmetric: some forward steps have no legal inverse, and the backward half
of bidirectional BFS cannot replay them. IsoBench v0's `dist` labels come from
bidirectional BFS, so they are not always the shortest *forward* proof.

Concrete case (bool): `(~ F) = T`.
- Forward BFS proves it in 3 steps: `or_id~` at `[]`, `or_comm` at `[]`, and
  `or_compl` at `[]`. The last step consumes `?x = F`.
- Backward from `T`, the inverse `or_compl~` may instantiate `?x` only from the
  subterms of `T`, so `F` is not available. Bidirectional BFS, and therefore
  the v0 label, says 5.

We found this by cross-checking `bfs-fwd` proof lengths against the v0 labels
in the E1 runs: one problem in `test_depth` had a forward proof shorter than
its "exact" label.

## Decision
1. **One pool per problem.** `problem_pool` contains:
   - the theory constants (always)
   - the skolems that **occur** in lhs or rhs
   - every distinct subterm of lhs or rhs with size <= `pool_max_size` (default 3).

   It is sorted by (size, str). It depends on the problem only, never on the
   current term.

   Restricting to mentioned skolems is lossless. Skolems carry no axioms, so in
   any proof that introduces an unmentioned skolem c, replacing every c by a
   mentioned atom of size 1 (every problem has one: a theory constant or a
   skolem, since terms are ground) gives a valid proof of the same length with
   no term growing, so the size cap and pool membership still hold (pool
   subterms never contain c). Hence shortest distances are unchanged, and
   branching drops modestly (7–15% fewer successors per term, measured on
   identical problems).
2. **Legality.** A step is legal iff every value bound to a one-sided variable
   is in the pool. One-sided means *introduced* (in the rule's rhs only) or
   *consumed* (in its lhs only).
   - `successors` enumerates introduced values from the whole pool, using the
     full product when a rule introduces several.
   - It skips steps whose consumed values are outside the pool.
3. **Symmetry follows.** The inverse of a legal step introduces exactly what
   the forward step consumed and consumes exactly what it introduced, so it is
   legal under the same pool. Bidirectional BFS distances are exact again
   (within the size cap). A property test checks this on all four theories.
4. **The signature is part of the theory.** `Theory` carries its `Signature`,
   so a `Problem` alone determines its action space: in worker processes, in
   every searcher, and for ad-hoc problems in tests. A lookup through the
   `THEORIES` registry would couple `rewrite`/`proof` to the registry by name.
5. **Generator.** A random walk uses the pool of its *start* term only
   (constants, the skolems it mentions, its small subterms). The
   generated problem's lhs is that start term, so its pool contains the walk's
   pool, and every walk step is legal in the problem's action space.
6. `Budget.inst_limit` is replaced by `Budget.pool_max_size`.

## Consequences
- **Dataset v1** (`isobench/1`, `data/isobench-v1`). The datasheet records
  `pool_max_size`. v0 labels are not comparable with v1 labels.
- **E1 is re-run on v1.** The v0 E1 results are archived and reported only as
  a documented deviation from the pre-registered plan, not as E1 results.
- **New restriction:** a large invented term can no longer be introduced and
  later cancelled unless it is in the pool. Some proofs that needed such a
  detour are now longer, or impossible within the action space. This is the
  price of symmetry, and the paper must state the action space.
- **Mentioned skolems only:** shortest distances are the same as with every
  skolem of the signature in the pool (argument in decision 1; a slow test
  checks it empirically on small group and lattice problems). The saving is
  modest. Measured on identical problems (2026-10-28):
  - successors per term fall by 7–15% across the four theories
  - for bfs-fwd on 20 lattice problems at 300 expansions: generated per
    expansion 154 → 150, total expansions 3061 → 2474, with the same solves
    and the same proof lengths.

  Most of lattice's branching comes from `absorb_mj~` introducing *any* pool
  value at *every* position, whatever skolems are in the pool.
- **Pool size:** the pool no longer adapts to the current term. Terms built
  during a proof are not available for instantiation unless they are small
  subterms of the problem.
- **The kernel is unchanged.** Soundness does not depend on the pool; the pool
  only shapes the search space.
