"""Weight initialization utilities."""

import torch
import torch.nn as nn
import math


def init_gpt_weights(model: nn.Module, config) -> None:
    """Initialize GPT model weights."""
    embedding_std = getattr(config, "embedding_init_std", 0.02)
    hidden_factor = getattr(config, "hidden_init_std_factor", 0.5)
    hidden_size = getattr(config, "hidden_size", 768)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue

        # Embedding layers
        if "wte" in name or "lm_head" in name:
            nn.init.normal_(param, mean=0.0, std=embedding_std)
        # Other layers (skip if already initialized)
        elif "c_proj" not in name and "beta" not in name and "v_proj" not in name:
            std = hidden_factor / math.sqrt(hidden_size)
            if param.std() > std * 2:  # Only init if not already initialized
                nn.init.normal_(param, mean=0.0, std=std)
