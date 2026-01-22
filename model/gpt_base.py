"""GPT base utilities."""

import torch
from typing import Tuple

# Type alias for past key-value cache
PastKeyValue = Tuple[torch.Tensor, torch.Tensor]
