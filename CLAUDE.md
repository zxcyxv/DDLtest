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

Two implementations exist in `model/`:

- **DDL.py**: Expanded state version with matrix hidden states `X ∈ R^{d × d_v}` where `d_v > 1` (default 4). Uses `ResidualShortConvCompressor` to map expanded state back to `R^d` for sublayer input.

- **DDL-vdim1.py**: Scalar value limit version with vector hidden states `x ∈ R^d` (equivalent to `d_v = 1`). Simpler architecture without state compression.

Both implementations are built on a GPT backbone with Multi-Head Attention + RoPE and SwiGLU MLP.

## Dependencies

The code depends on external modules not included in this repo:
- `.activations`, `.gpt_base`, `.rmsnorm`, `.kv_shift`, `.init_utils`, `.pydantic_config`, `.rotary`, `.shortconv`

These are expected to be available in the parent package when using this as a submodule.

## Key Configuration Options (GPTConfig)

DDL-specific parameters:
- `ddl_value_channels`: Expansion factor d_v for matrix states (DDL.py only, default: 4)
- `ddl_beta_init`: Initial β value in [0, 2] (default: 1.0)
- `ddl_k_eps`: Epsilon for k normalization stability
- `ddl_beta_single_linear`: Use single linear layer for β (vs 2-layer MLP)
- `ddl_v_sigmoid`: Apply sigmoid activation to v
- `ddl_v_sigmoid_scale`: Scale factor when using sigmoid on v
