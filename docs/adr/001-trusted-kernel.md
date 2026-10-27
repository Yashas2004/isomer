# ADR 001 — A tiny, independent trusted kernel

**Status:** accepted · **Date:** 2026-10-27

## Context
Search and data generation use a rewrite engine (`core/rewrite.py`) that will
grow optimisations (indexing, caching, batching for the GPU). Any bug there
could produce "proofs" of false statements, and every benchmark number would be
silently wrong.

## Decision
`core/kernel.py` re-implements matching, substitution and replay from scratch.
It imports only the `Var`/`App` dataclasses and `parse` from `core/term.py`,
never `rewrite.py`. Proofs are checked from their serialized JSON certificate,
so the kernel sees exactly what an outside reviewer would.

## Consequences
* A soundness bug must appear identically in two independent implementations
  to go unnoticed. `tests/test_kernel.py` runs them against each other on
  random walks (differential testing) and checks that mutated proofs are
  rejected.
* Some duplicated code (~60 lines). Accepted: duplication is the point.
* The kernel must stay small enough to read in one sitting (<200 lines).
