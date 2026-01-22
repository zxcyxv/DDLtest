"""Activation function utilities."""

import torch
import torch.nn.functional as F
from typing import Literal

ActivationName = Literal["relu", "gelu", "silu", "tanh", "sigmoid"] | None


def validate_activation_name(name: str | None, field_name: str = "activation") -> ActivationName:
    """Validate activation name."""
    if name is None:
        return None
    valid_names = {"relu", "gelu", "silu", "tanh", "sigmoid"}
    if name not in valid_names:
        raise ValueError(f"{field_name} must be one of {valid_names}, got {name}")
    return name


def apply_activation(x: torch.Tensor, activation: ActivationName) -> torch.Tensor:
    """Apply activation function to tensor."""
    if activation is None:
        return x
    elif activation == "relu":
        return F.relu(x)
    elif activation == "gelu":
        return F.gelu(x)
    elif activation == "silu":
        return F.silu(x)
    elif activation == "tanh":
        return torch.tanh(x)
    elif activation == "sigmoid":
        return torch.sigmoid(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")
