"""Short convolution implementation for state compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseShortConv1d(nn.Module):
    """Depthwise 1D convolution with causal padding."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            y: (B, T, C)
        """
        # Transpose to (B, C, T) for conv1d
        x = x.transpose(1, 2)

        # Causal padding (pad left only)
        x = F.pad(x, (self.padding, 0))

        # Apply convolution
        y = self.conv(x)

        # Transpose back to (B, T, C)
        return y.transpose(1, 2)
