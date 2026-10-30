"""Best-first search guided by a learned distance model h(state, goal).

Method names carry the checkpoint, ``neural-w1@runs/h-v1/best.pt``:

    neural-w1, neural-w3   priority g + w * h
    neural-greedy          priority h

Otherwise this is ``best_first.best_first``, with the same choices
(duplicates detected on generation, no reopening, goal test on generation, and
FIFO tie-breaking by an insertion counter). There is one difference:
**scoring is batched.** When a term is expanded, all of its new successors
(not seen before, within the size cap) are scored in ONE model call, then
pushed in generation order. Calling h one term at a time would make the GPU
the bottleneck.

**Determinism** is needed for the prefix property (``common``) and for
reproducible benchmarks.
* Inference is fp32 without autocast.
* Deterministic cuBLAS/cuDNN settings and ``use_deterministic_algorithms``
  are set when the model loads, before this process does any CUDA work.
* There is **no score cache across problems.** A value computed inside
  another problem's batch can differ in the last bit, which would make results
  depend on how jobs are spread over workers. Within one search every state is
  scored exactly once anyway.
A given search therefore issues the same batches in the same order and gets
identical values. The tests check this.

**Workers.** Each process loads the checkpoint once (``_MODELS``) and uses
CUDA if available. Every process holds its own CUDA context (~0.5 GB) plus
the model, so a 6 GB card runs 2-3 workers usefully.

torch is imported only when a model is loaded: ``isomer.search`` and the CLI
work without it.
"""

from __future__ import annotations

import heapq
import os
import time
from functools import partial
from typing import Any

from ..core.proof import Problem, Proof
from ..core.rewrite import successors
from ..core.term import Term, size
from .common import (
    PARAMETRIC,
    Budget,
    Parents,
    SearchResult,
    out_of_memory,
    out_of_time,
    size_cap,
    steps_from_parents,
)

_MODELS: dict[tuple[str, str | None], Any] = {}


def _load(ckpt: str, device: str | None = None) -> Any:
    """One ``LearnedHeuristic`` per (checkpoint, device) per process."""
    key = (ckpt, device)
    if key not in _MODELS:
        # Must precede CUDA initialization in this process to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        from ..learn.infer import LearnedHeuristic

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS[key] = LearnedHeuristic(ckpt, device=dev, precision="fp32", batch_size=4096)
    return _MODELS[key]


def neural_best_first(problem: Problem, budget: Budget, ckpt: str, w: float = 1.0, greedy: bool = False,
                      device: str | None = None) -> SearchResult:
    t0 = time.perf_counter()
    h = _load(ckpt, device)
    rule_list = list(problem.theory.rules().values())
    pool = problem.pool(budget.pool_max_size)
    max_size = size_cap(problem, budget.max_size_slack)
    goal = problem.rhs
    parents: Parents = {problem.lhs: None}
    expansions = generated = calls = 0
    model_seconds = 0.0

    def score(terms: list[Term]) -> list[float]:
        nonlocal calls, model_seconds
        ts = time.perf_counter()
        out = h.scores(terms, goal)   # returns host floats: the GPU has synchronized
        model_seconds += time.perf_counter() - ts
        calls += 1
        return out

    def result(stop: str, proof: Proof | None = None) -> SearchResult:
        stats = {"nodes_stored": len(parents), "stop": stop, "model_calls": calls,
                 "model_seconds": round(model_seconds, 4), "device": h.device.type, "gpu_peak_bytes": None}
        if h.device.type == "cuda":
            import torch
            stats["gpu_peak_bytes"] = torch.cuda.max_memory_allocated(h.device)
        return SearchResult(proof, expansions, generated, time.perf_counter() - t0,
                            exhausted=stop == "exhausted", lower_bound=len(proof.steps) if proof else 0,
                            stats=stats)

    def priority(g: int, hv: float) -> float:
        return hv if greedy else g + w * hv

    if problem.lhs == goal:
        return result("solved", Proof(problem, ()))

    counter = 0
    heap: list[tuple[float, int, int, Term]] = [(priority(0, score([problem.lhs])[0]), counter, 0, problem.lhs)]
    while heap:
        if expansions >= budget.max_expansions:
            return result("budget")
        if out_of_memory(len(parents), budget.max_nodes):
            return result("memory")
        if out_of_time(t0, budget.max_seconds):
            return result("time")
        _, _, g, t = heapq.heappop(heap)
        expansions += 1
        fresh: list[Term] = []
        for step, u in successors(t, rule_list, pool):
            generated += 1
            if u in parents or size(u) > max_size:
                continue
            parents[u] = (t, step)
            if u == goal:
                return result("solved", Proof(problem, tuple(steps_from_parents(parents, u))))
            fresh.append(u)
        if fresh:
            for u, hv in zip(fresh, score(fresh), strict=True):
                counter += 1
                heapq.heappush(heap, (priority(g + 1, hv), counter, g + 1, u))
    return result("exhausted")


def _make(ckpt: str, w: float, greedy: bool) -> partial:
    """Picklable searcher for one checkpoint. Loads nothing: also used to validate names."""
    return partial(neural_best_first, ckpt=ckpt, w=w, greedy=greedy)


PARAMETRIC["neural-w1"] = partial(_make, w=1.0, greedy=False)
PARAMETRIC["neural-w3"] = partial(_make, w=3.0, greedy=False)
PARAMETRIC["neural-greedy"] = partial(_make, w=1.0, greedy=True)


def is_neural(method: str) -> bool:
    prefix, sep, _ = method.partition("@")
    return bool(sep) and prefix.startswith("neural-") and prefix in PARAMETRIC


def checkpoint_of(method: str) -> str:
    return method.partition("@")[2]
