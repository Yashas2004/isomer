# ADR 002 — Action space, inverse rules, and difficulty labels

**Status:** accepted · **Date:** 2026-10-27

## Context
Phase 1b needs an honest difficulty label for each problem. Random-walk
length is only an upper bound: in a 200-problem smoke build, the walk was
longer than the true shortest proof for ~55% of labelled problems, by up to
10 steps. Training on walk length would teach the model wrong distances and
make "depth extrapolation" results meaningless.

## Decisions
1. **Both orientations of every equation are rules**, including those whose
   lhs is a bare variable (`?x -> (* e ?x)`). Phase 1 dropped these; that
   made some true theorems unprovable and made steps non-invertible.
2. **Every step has an inverse** (`rewrite.invert_step`). This is what lets
   bidirectional BFS stitch the backward half into a forward proof.
3. **Labels come from bidirectional BFS.** `dist` = exact shortest length
   within the action space; if the budget runs out, `dist = null` and
   `dist_lb` is a *proven* lower bound (completed layers on both sides).
4. **The action space is defined, not implied:** fresh variables are
   instantiated from the 8 smallest distinct subterms of the current term,
   and terms larger than max(|lhs|, |rhs|) + 6 are not explored. All
   distances are "within this action space"; the paper must say so.

## Consequences
- Bool and lattice branch 50–120 ways per term, so BFS labelling is
  expensive and some problems stay unlabelled. They go to `test_depth`
  (hard problems), which is the right place for them.
- An oracle test checks bidirectional BFS against plain BFS on small cases.
