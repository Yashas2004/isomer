"""Batched inference for a trained distance model.

``LearnedHeuristic`` has the ``Heuristic`` call signature ``h(t, goal)`` and
a batched ``h_batch``. Terms are hashable and immutable, so an LRU cache keyed
by ``(state, goal)`` is safe. Search revisits states, and the cache saves
repeated forward passes.

Limit: a pair longer than the model's ``max_len`` raises ``ValueError``
(``tokenizer.encode``). It is never truncated, because truncation would
change the term.

**Precision modes** (``PRECISIONS``):
* ``fp32``: fp32, TF32 off. The reference.
* ``tf32``: fp32 storage, TF32 tensor-core matmuls. The flag is set around each
  forward pass and restored, so modes in one process never leak into each other.
* ``bf16``: bf16 autocast, on every device.
* ``bf16-fast``: a bf16 copy of the weights run by ``DistanceModel.forward_fast``,
  with no fp32 round-trips inside the blocks and no autocast. The fastest mode.
* ``*-bucket``: the batch is stably sorted by encoded length and split into
  buckets of lengths rounded up to a multiple of ``BUCKET`` tokens. Each bucket
  is padded only to its own longest, so less padding is computed. Results are
  scattered back to input order. The split depends only on the lengths, so a
  given batch always runs the same shapes.
Every mode is deterministic for a given batch when the deterministic CUDA
settings of ``search.neural`` are in force. Search needs that, because scores
must be reproducible bit for bit. ``scores`` bypasses the cache, because a
cached value computed inside another batch can differ in the last bit from a
fresh one.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from ..core.term import Term
from .model import load_checkpoint, make_fast_copy
from .tokenizer import PAD_ID, encode

PRECISIONS = ("fp32", "tf32", "bf16", "fp32-bucket", "bf16-bucket", "bf16-fast")
BUCKET = 16


def length_buckets(lengths: list[int], quantum: int = BUCKET) -> list[list[int]]:
    """Indices grouped by ceil(length / quantum): shortest bucket first, input
    order inside a bucket. A pure function of the lengths."""
    groups: dict[int, list[int]] = {}
    for i, n in sorted(enumerate(lengths), key=lambda x: (x[1], x[0])):
        groups.setdefault(-(-n // quantum), []).append(i)
    return [groups[k] for k in sorted(groups)]


class LearnedHeuristic:
    def __init__(self, ckpt_path: str, device: str = "cuda", batch_size: int = 512,
                 cache_size: int = 100_000, precision: str = "bf16") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, got {precision!r}")
        self.device = torch.device(device)
        self.precision = precision
        self.bf16 = precision in ("bf16", "bf16-bucket")
        self.tf32 = precision == "tf32"
        self.bucket = precision.endswith("-bucket")
        self.model, self.meta = load_checkpoint(ckpt_path, self.device)
        # bf16-fast: separate bf16 copy run by forward_fast (model.py); no autocast
        self.fast = make_fast_copy(self.model) if precision == "bf16-fast" else None
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[Term, Term], float] = OrderedDict()
        self.hits = self.misses = 0

    def encode_batch(self, states: list[Term], goal: Term) -> list[list[int]]:
        return [encode(s, goal, max_len=self.model.cfg.max_len) for s in states]

    @torch.inference_mode()
    def forward_encoded(self, seqs: list[list[int]]) -> list[float]:
        """Model outputs for already-encoded sequences, in input order."""
        n = len(seqs)
        groups = length_buckets([len(s) for s in seqs]) if self.bucket else [list(range(n))]
        out = [0.0] * n
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            for g in groups:
                for i in range(0, len(g), self.batch_size):
                    idx = g[i:i + self.batch_size]
                    ids = np.full((len(idx), max(len(seqs[j]) for j in idx)), PAD_ID, dtype=np.int64)
                    for row, j in enumerate(idx):
                        ids[row, :len(seqs[j])] = seqs[j]
                    t = torch.from_numpy(ids).to(self.device)
                    if self.fast is not None:
                        vals = self.fast.forward_fast(t).float().cpu().tolist()
                    else:
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
                            vals = self.model(t).float().cpu().tolist()
                    for j, v in zip(idx, vals, strict=True):
                        out[j] = v
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return out

    def _forward(self, states: list[Term], goal: Term) -> list[float]:
        return self.forward_encoded(self.encode_batch(states, goal))

    def scores(self, states: list[Term], goal: Term) -> list[float]:
        """Uncached predictions, in order: one forward pass per ``batch_size`` states."""
        return self._forward(states, goal) if states else []

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
