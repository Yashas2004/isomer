"""Transformer encoder predicting the distance h(state, goal) from tokens.

Input is ``[CLS] state [SEP] goal [SEP]`` (``tokenizer.encode``). Design choices:

* **Rotary positions (RoPE), no position table.** Attention scores depend
  only on relative offsets, and there are no per-position parameters, so the
  model accepts any length up to ``max_len`` without untrained rows. Test
  problems are larger than training ones. ``max_len`` is only a guard for
  ``encode``. It is 192 because no state within any IsoBench v2 problem's
  size cap can encode longer than 179 tokens.
* **Segment embedding** (state = 0, goal = 1), derived from the ids: the
  model always knows which side a token is on.
* **Pre-norm blocks** with RMSNorm, ``scaled_dot_product_attention`` with a
  key-padding mask, and a SwiGLU feed-forward. Padding at the end never
  changes a prediction; the tests check this.
* **Unclamped scalar head** on [CLS], computed in fp32. Deeper test
  problems need predictions above the training maximum. Keeping the head
  fp32 avoids spurious ties from bf16 rounding when siblings are ranked.

**Fast inference path** (``forward_fast``, ``make_fast_copy``; used only by
``LearnedHeuristic(precision="bf16-fast")``). The same weights, cast to bf16,
run with no fp32 round-trips inside the blocks. RoPE tables are precomputed
once (non-persistent buffers, so checkpoints are unchanged) and the rotation
is done in bf16. RMSNorm uses ``F.rms_norm`` where available. Only the final
norm and the head compute in fp32. Training and ``forward`` are untouched: the
fp32 outputs are pinned bit for bit by ``tests/test_golden.py``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer import PAD_ID, SEP_ID, VOCAB

CHECKPOINT_FORMAT = "isomer-heuristic/1"


def default_ffn_hidden(d_model: int) -> int:
    """SwiGLU width 8/3 d keeps the FFN parameter count equal to a 4d GELU MLP."""
    return math.ceil(8 * d_model / 3)


@dataclass
class ModelConfig:
    vocab_size: int = len(VOCAB)
    max_len: int = 192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_hidden: int = 683
    rope_base: float = 10_000.0


class RMSNorm(nn.Module):
    """Own implementation: ``torch.nn.RMSNorm`` needs torch >= 2.4 and the
    ``learn`` extra allows 2.2. Normalizes in fp32 for bf16 stability."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_tables(length: int, head_dim: int, base: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [length, head_dim] in fp32 (rotate-half layout)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    ang = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate [B, H, L, Dh] by position; computed in fp32, returned in x's dtype."""
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rot * sin).to(x.dtype)


def apply_rope_fast(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rope`` in x's own dtype (no fp32 round-trip); cos/sin already in that dtype."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def rms_norm_fast(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in x's dtype: the fused ``F.rms_norm`` (torch >= 2.4) when available."""
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and (cfg.d_model // cfg.n_heads) % 2 == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, keep: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))

    def forward_fast(self, x: torch.Tensor, keep: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope_fast(q, cos, sin), apply_rope_fast(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x: torch.Tensor, keep: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), keep, cos, sin)
        return x + self.ffn(self.norm2(x))

    def forward_fast(self, x: torch.Tensor, keep: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_fast(rms_norm_fast(x, self.norm1.weight, self.norm1.eps), keep, cos, sin)
        return x + self.ffn(rms_norm_fast(x, self.norm2.weight, self.norm2.eps))


def segment_ids(ids: torch.Tensor) -> torch.Tensor:
    """0 up to and including the first [SEP], 1 after it."""
    sep = (ids == SEP_ID).long()
    return ((sep.cumsum(1) - sep) >= 1).long()


class DistanceModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.seg_emb = nn.Embedding(2, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)
        for blk in self.blocks:  # residual-branch outputs scaled down with depth (GPT-2 style)
            for w in (blk.attn.out.weight, blk.ffn.w2.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layers))
        # Fast path only: precomputed RoPE tables (fp32, not saved in checkpoints).
        cos, sin = rope_tables(cfg.max_len, cfg.d_model // cfg.n_heads, cfg.rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._rope_cast: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, L] (PAD = 0) -> predicted distance [B], fp32."""
        if ids.shape[1] > self.cfg.max_len:
            raise ValueError(f"sequence length {ids.shape[1]} > max_len {self.cfg.max_len}")
        keep = ids != PAD_ID
        x = self.tok_emb(ids) + self.seg_emb(segment_ids(ids))
        cos, sin = rope_tables(ids.shape[1], self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_base, ids.device)
        for blk in self.blocks:
            x = blk(x, keep, cos, sin)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            return self.head(self.norm(x[:, 0].float())).squeeze(-1)

    def forward_fast(self, ids: torch.Tensor) -> torch.Tensor:
        """Inference only, on a ``make_fast_copy`` model: blocks in the
        embeddings' dtype (bf16), final norm and head in fp32. Run without autocast."""
        if ids.shape[1] > self.cfg.max_len:
            raise ValueError(f"sequence length {ids.shape[1]} > max_len {self.cfg.max_len}")
        keep = ids != PAD_ID
        dtype = self.tok_emb.weight.dtype
        key = (str(self.rope_cos.device), dtype)
        if key not in self._rope_cast:
            self._rope_cast[key] = (self.rope_cos.to(dtype), self.rope_sin.to(dtype))
        cos, sin = (t[:ids.shape[1]] for t in self._rope_cast[key])
        x = self.tok_emb(ids) + self.seg_emb(segment_ids(ids))
        for blk in self.blocks:
            x = blk.forward_fast(x, keep, cos, sin)
        return self.head(self.norm(x[:, 0].float())).squeeze(-1)


def make_fast_copy(model: DistanceModel, dtype: torch.dtype = torch.bfloat16) -> DistanceModel:
    """A copy for ``forward_fast``: embeddings and blocks in ``dtype``, final
    norm and head (and the RoPE buffers) left in fp32. The original is untouched."""
    fast = copy.deepcopy(model)
    fast._rope_cast = {}
    for m in (fast.tok_emb, fast.seg_emb, fast.blocks):
        m.to(dtype)
    return fast.eval()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path: str | Path, model: DistanceModel, extra: dict | None = None) -> None:
    """state_dict + config + vocab; only plain types, so ``weights_only`` loading works."""
    torch.save({"format": CHECKPOINT_FORMAT, "config": asdict(model.cfg), "vocab": list(VOCAB),
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[DistanceModel, dict]:
    """Rebuild the model in eval mode. Refuses a different checkpoint format or
    vocabulary: embedding rows would silently mean different symbols."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {ck.get('format')!r} != {CHECKPOINT_FORMAT!r}")
    if list(ck.get("vocab", [])) != list(VOCAB):
        raise ValueError(f"{path}: checkpoint vocabulary differs from the current tokenizer's; "
                         "the embedding rows would mean different symbols")
    model = DistanceModel(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("extra", {})
