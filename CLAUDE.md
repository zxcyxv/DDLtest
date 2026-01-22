# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the reference implementation for **Deep Delta Learning (DDL)**, a neural network architecture that generalizes residual connections with learnable geometric transformations. The paper is available at arXiv:2601.00417.

## Architecture

DDL replaces the standard additive residual connection `X_{l+1} = X_l + F(X_l)` with a rank-1 Householder-style update:

```
X_{l+1} = (I - β_l * k_l * k_l^T) * X_l + β_l * k_l * v_l^T
```

This can be rewritten as the Delta Rule over depth:
```
X_{l+1} = X_l + β_l * k_l * (v_l^T - k_l^T * X_l)
```

Key components:
- **k (direction vector)**: Normalized vector determining which subspace to modify
- **v (value vector)**: New information to write into the state
- **β (gate scalar)**: Controls interpolation between identity (β→0), projection (β→1), and reflection (β→2)

## Code Structure

Three DDL implementations exist in `model/`:

| File | Description | Hidden State |
|------|-------------|--------------|
| `DDL.py` | Original expanded state version | `X ∈ R^{d × d_v}` |
| `DDL-vdim1.py` | Scalar value limit version | `x ∈ R^d` |
| `DDL_corrected.py` | Theoretically corrected version | `X ∈ R^{d × d_v}` |

### DDL_corrected.py - Key Improvements

The corrected implementation fixes theoretical and stability issues:

| Component | Original (Flawed) | Corrected |
|-----------|-------------------|-----------|
| **v (content)** | `W_proj * h_in` (simple linear) | `W_v * Backbone(h_norm)` (deep computation) |
| **k (direction)** | `Backbone(h_in)` (expensive) | `MLP_light(h_norm)` (lightweight) |
| **k, β input** | `h_in` (un-normalized) | `h_norm` (RMSNorm normalized) |
| **k MLP init** | Standard init | ReZero-style (near-zero output layer) |

**Stability fixes:**
1. Using `h_norm` prevents variance explosion in deep layers
2. ReZero initialization ensures identity-like behavior at start
3. Clear separation: `h_norm` → direction/gate, `h_out` → content

## Training

Run experiments comparing original vs corrected DDL on TinyShakespeare:

```bash
# Train both models and generate comparison plots
python train.py --model both --max_iters 3000

# Train only corrected model
python train.py --model corrected --max_iters 5000

# Custom configuration
python train.py --model both \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 4 \
    --batch_size 64 \
    --block_size 256 \
    --learning_rate 3e-4
```

Outputs saved to `outputs/`:
- `history_original.json`, `history_corrected.json`: Training logs
- `loss_comparison.png`: Loss curve comparison plot

## Dependencies

Core dependencies (install via pip):
- `torch`
- `transformers`
- `matplotlib`

Utility modules included in `model/`:
- `rmsnorm.py`: RMSNorm implementation
- `rotary.py`: Rotary Position Embedding (RoPE)
- `activations.py`: Activation function utilities
- `shortconv.py`: Depthwise causal convolution for state compression
- `kv_shift.py`: KV shift linear layer
- `init_utils.py`: Weight initialization
- `pydantic_config.py`: Config validation
- `gpt_base.py`: Base types (PastKeyValue)

## Key Configuration Options (GPTConfig)

Common DDL parameters:
- `ddl_value_channels`: Expansion factor d_v (default: 4)
- `ddl_beta_init`: Initial β value in [0, 2] (default: 1.0)
- `ddl_k_eps`: Epsilon for k normalization (default: 1e-5)
- `ddl_beta_hidden_size`: Hidden size for β MLP (default: 128)

Corrected DDL specific:
- `ddl_k_mlp_hidden_size`: Hidden size for k MLP (default: hidden_size // 4)
- `ddl_k_rezero_scale`: ReZero init scale for k MLP (default: 0.01)
