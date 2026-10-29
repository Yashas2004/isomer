"""Batched inference for a trained distance model. Not used by search yet.

``LearnedHeuristic`` has the ``Heuristic`` call signature ``h(t, goal)`` and
a batched ``h_batch``. Terms are hashable and immutable, so an LRU cache keyed
by ``(state, goal)`` is safe. Search revisits states, and the cache saves
repeated forward passes.

Limit: a pair longer than the model's ``max_len`` raises ``ValueError``
(``tokenizer.encode``). It is never truncated, because truncation would
change the term.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from ..core.term import Term
from .model import load_checkpoint
from .tokenizer import PAD_ID, encode


class LearnedHeuristic:
    def __init__(self, ckpt_path: str, device: str = "cuda", batch_size: int = 512,
                 cache_size: int = 100_000) -> None:
        self.device = torch.device(device)
        self.model, self.meta = load_checkpoint(ckpt_path, self.device)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[Term, Term], float] = OrderedDict()
        self.hits = self.misses = 0

    @torch.inference_mode()
    def _forward(self, states: list[Term], goal: Term) -> list[float]:
        out: list[float] = []
        max_len = self.model.cfg.max_len
        for i in range(0, len(states), self.batch_size):
            seqs = [encode(s, goal, max_len=max_len) for s in states[i:i + self.batch_size]]
            ids = torch.full((len(seqs), max(map(len, seqs))), PAD_ID, dtype=torch.long)
            for row, s in enumerate(seqs):
                ids[row, :len(s)] = torch.tensor(s, dtype=torch.long)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                out.extend(self.model(ids.to(self.device)).float().cpu().tolist())
        return out

    def h_batch(self, states: list[Term], goal: Term) -> list[float]:
        """Predicted distances of ``states`` to ``goal``; only cache misses are computed."""
        result: list[float | None] = []
        todo: dict[Term, None] = {}
        for s in states:
            v = self._cache.get((s, goal))
            if v is None:
                todo[s] = None
            else:
                self._cache.move_to_end((s, goal))
                self.hits += 1
            result.append(v)
        if todo:
            self.misses += len(todo)
            fresh = dict(zip(todo, self._forward(list(todo), goal), strict=True))
            for s, v in fresh.items():
                self._cache[(s, goal)] = v
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            result = [fresh[s] if v is None else v for s, v in zip(states, result, strict=True)]
        return [float(v) for v in result]  # type: ignore[arg-type]

    def __call__(self, t: Term, goal: Term) -> float:
        return self.h_batch([t], goal)[0]
