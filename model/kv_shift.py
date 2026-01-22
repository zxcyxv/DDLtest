"""KV Shift Linear layer implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftLinear(nn.Module):
    """Linear layer with optional temporal shift for KV."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(
        self, x: torch.Tensor, past: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with optional shift."""
        # Simple implementation without actual shifting for now
        return self.linear(x)
