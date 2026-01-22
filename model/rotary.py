"""Rotary Position Embedding (RoPE) implementation."""

import torch
import torch.nn as nn
import math


class Rotary(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, base: float = 10000.0, rope_ratio: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_ratio = rope_ratio
        self.inv_freq = None

    def _compute_inv_freq(self, device: torch.device, dtype: torch.dtype):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        return inv_freq

    def forward(
        self, x: torch.Tensor, seq_len_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        device = x.device
        dtype = x.dtype

        if self.inv_freq is None or self.inv_freq.device != device:
            self.inv_freq = self._compute_inv_freq(device, dtype)

        t = torch.arange(seq_len_offset, seq_len_offset + seq_len, device=device, dtype=torch.float32)
        if self.rope_ratio != 1.0:
            t = t / self.rope_ratio

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, T, H, D) or (B, T, D)
    # cos, sin: (T, D)
    if x.ndim == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(0)  # (1, T, D)
        sin = sin.unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)
