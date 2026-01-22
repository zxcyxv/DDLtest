"""
Deep Delta Learning (DDL) - Theoretically Corrected Implementation

This implementation corrects the data flow according to the theoretical formulation:

Original (Flawed):
    - v: W_proj * h_in  (simple linear projection of input - lacks semantic richness)
    - k: Backbone(h_in)  (expensive backbone computation for direction - inefficient)

Corrected:
    - v: W_v * Backbone(RMSNorm(h_in)) + b_v  (deep computation result = semantic content)
    - k: normalize(MLP_light(h_norm))  (lightweight geometric direction from NORMALIZED state)
    - β: 2 * σ(MLP(h_norm))  (dynamic gate from NORMALIZED state)

Critical Stability Fixes:
    1. k and β use h_norm (normalized), NOT h_in (un-normalized residual stream)
       - Residual stream variance grows with depth; normalization ensures stable training
    2. Clear data flow: h_norm → direction/gate, h_out → content
    3. ReZero-style initialization for k MLP output layer (near-zero init for stable start)

The Delta update rule:
    X_{l+1} = X_l + β(X_l) * k(X_l) * (v(X_l)^T - k(X_l)^T * X_l)

Where:
    - v(X_l): Content to write (from backbone's deep computation)
    - k(X_l): Direction to write (from lightweight analysis of NORMALIZED state)
    - β(X_l): Dynamic step size in [0, 2] (from NORMALIZED state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Any
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .activations import ActivationName, apply_activation, validate_activation_name
from .gpt_base import PastKeyValue
from .rmsnorm import RMSNorm
from .kv_shift import ShiftLinear
from .init_utils import init_gpt_weights
from .pydantic_config import validate_pretrained_config_kwargs
from .rotary import Rotary, apply_rotary_emb
from .shortconv import DepthwiseShortConv1d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p) - math.log(1.0 - p)


class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with RoPE."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.q_activation = validate_activation_name(getattr(config, "q_activation", None), field_name="q_activation")
        self.k_activation = validate_activation_name(getattr(config, "k_activation", None), field_name="k_activation")
        self.v_activation = validate_activation_name(getattr(config, "v_activation", None), field_name="v_activation")
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
        self.use_output_gate = getattr(config, "use_output_gate", False)

        self.c_q = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_k_shift:
            self.c_k = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_k = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_v = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std = factor / math.sqrt(config.hidden_size) / math.sqrt(config.num_hidden_layers)
            self.c_proj.weight.normal_(mean=0.0, std=std)

        rope_ratio = float(getattr(config, "rope_ratio", 1.0))
        self.rotary = Rotary(self.head_dim, base=getattr(config, "rope_base", 10000.0), rope_ratio=rope_ratio)
        self.using_groupnorm = config.using_groupnorm

        self.use_qk_rmsnorm = getattr(config, "use_qk_rmsnorm", True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
            if not self.using_groupnorm:
                self.o_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-5), elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_past(x)
        return y

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        past_len = 0
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must have at least 2 tensors: (key, value).")
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_len = int(past_k.shape[-2])

        q = apply_activation(q, self.q_activation)
        k = apply_activation(k, self.k_activation)
        v = apply_activation(v, self.v_activation)

        cos, sin = self.rotary(q, seq_len_offset=past_len)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        if past_k is not None and past_v is not None:
            k_t = torch.cat([past_k, k_t], dim=-2)
            v_t = torch.cat([past_v, v_t], dim=-2)

        total_len = int(k_t.shape[-2])
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}")
            if int(attention_mask.shape[0]) != B:
                raise ValueError(f"attention_mask batch mismatch: expected {B}, got {int(attention_mask.shape[0])}")
            if int(attention_mask.shape[1]) != total_len:
                raise ValueError(
                    f"attention_mask sequence mismatch: expected {total_len}, got {int(attention_mask.shape[1])}"
                )
            attn_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]

        use_is_causal = past_len == 0 and attn_mask is None
        if not use_is_causal:
            if T > 1:
                key_positions = torch.arange(total_len, device=x.device)
                query_positions = past_len + torch.arange(T, device=x.device)
                causal_mask = key_positions <= query_positions[:, None]
            else:
                causal_mask = torch.ones((T, total_len), dtype=torch.bool, device=x.device)
            if attn_mask is not None:
                attn_mask = attn_mask & causal_mask[None, None, :, :]
            else:
                attn_mask = causal_mask[None, None, :, :]

        y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=use_is_causal)

        if self.using_groupnorm:
            y = self.subln(y)
        elif self.use_output_gate:
            y = self.o_norm(y)

        if self.use_output_gate:
            gate = self.g_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            y = y * F.silu(gate)

        y = y.transpose(1, 2).contiguous().reshape(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        present: PastKeyValue | None = None
        if use_cache:
            present = (k_t, v_t)
        return y, present


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * config.hidden_size)
        self.c_fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.hidden_size, bias=False)

        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std = factor / math.sqrt(config.hidden_size) / math.sqrt(config.num_hidden_layers)
            self.c_proj.weight.normal_(mean=0.0, std=std)

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return x


class ResidualShortConvCompressor(nn.Module):
    """Compress expanded state X ∈ R^{d × d_v} to h_in ∈ R^d."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.value_channels = int(getattr(config, "ddl_value_channels", 4))
        if self.value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")
        self.residual_size = self.hidden_size * self.value_channels

        kernel_size = int(getattr(config, "ddl_state_shortconv_kernel_size", 4))
        self.shortconv = DepthwiseShortConv1d(self.residual_size, kernel_size=kernel_size)

        read_init_raw = getattr(config, "ddl_state_read_init", None)
        if read_init_raw is None:
            read_init = 1.0 / float(self.value_channels)
        else:
            read_init = float(read_init_raw)
        self.read = nn.Parameter(torch.full((self.value_channels,), read_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d, dv = x.shape
        if d != self.hidden_size:
            raise ValueError(f"Expected residual d={self.hidden_size}, got {d}.")
        if dv != self.value_channels:
            raise ValueError(f"Expected residual d_v={self.value_channels}, got {dv}.")

        x_flat = x.reshape(B, T, self.residual_size)
        x_conv = self.shortconv(x_flat).reshape(B, T, d, dv)
        return torch.sum(x_conv * self.read, dim=-1)


class LightweightDirectionMLP(nn.Module):
    """
    Lightweight MLP φ_k for computing geometric direction k from NORMALIZED state h_norm.

    [STABILITY FIX #1]: Input must be h_norm (RMSNorm normalized), NOT h_in.
    - Residual stream variance grows with depth
    - Normalized input ensures stable k across all layers

    [STABILITY FIX #3]: ReZero-style initialization for fc2 (output layer).
    - Initialize fc2 weights to near-zero
    - Initial behavior ≈ identity connection (stable training start)
    - Gradually learns meaningful directions during training

    k̃ = φ_k(h_norm)
    k = k̃ / ||k̃||_2
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        k_hidden_size_raw = getattr(config, "ddl_k_mlp_hidden_size", None)
        k_hidden_size = int(k_hidden_size_raw) if k_hidden_size_raw is not None else hidden_size // 4

        self.fc1 = nn.Linear(hidden_size, k_hidden_size, bias=False)
        self.fc2 = nn.Linear(k_hidden_size, hidden_size, bias=False)

        # [STABILITY FIX #3] ReZero-style initialization
        # fc1: Standard initialization for diverse feature extraction
        # fc2: Near-zero initialization for identity-like initial behavior
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std_fc1 = factor / math.sqrt(hidden_size)
            self.fc1.weight.normal_(mean=0.0, std=std_fc1)

            # fc2: Very small initialization (ReZero style)
            # This ensures initial k is small, making β*k*(v - k^T X) ≈ 0
            # i.e., X_{l+1} ≈ X_l (identity) at initialization
            rezero_scale = float(getattr(config, "ddl_k_rezero_scale", 0.01))
            std_fc2 = rezero_scale / math.sqrt(k_hidden_size)
            self.fc2.weight.normal_(mean=0.0, std=std_fc2)

        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-5))

    def forward(self, h_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_norm: NORMALIZED input state (B, T, d) - must be RMSNorm output!
        Returns:
            k: Normalized direction vector (B, T, d)
        """
        # φ_k(h_norm) with SiLU activation
        k_tilde = self.fc2(F.silu(self.fc1(h_norm)))

        # L2 normalize: k = k̃ / ||k̃||_2
        k_dim = k_tilde.size(-1)
        eps_rms = (self.k_eps * self.k_eps) / float(k_dim)
        k = F.rms_norm(k_tilde, [k_dim], eps=eps_rms)

        return k


class DeepDeltaResidualCorrected(nn.Module):
    """
    Corrected Deep Delta Residual module implementing:

    X_{l+1} = X_l + β(X_l) · k(X_l) · (v(X_l)^T - k(X_l)^T · X_l)

    [STABILITY FIX #1 & #2]: Clear input separation
    - h_norm (normalized): Used for k (direction) and β (gate)
      → Ensures stable computation across all layers
      → "Unit sphere" geometric interpretation as per paper
    - h_out (backbone output): Used for v (content)
      → Rich semantic information from deep computation

    Key corrections from original:
    - v (content): h_out from backbone (Attention/MLP output)
    - k (direction): Lightweight MLP on h_norm (NORMALIZED state)
    - β (gate): MLP on h_norm (NORMALIZED state)
    """

    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        value_channels = int(getattr(config, "ddl_value_channels", 4))
        if value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")
        self.value_channels = value_channels
        self.hidden_size = hidden_size

        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-5))

        # Lightweight MLP for direction k
        # [FIX] Takes h_norm (normalized), not h_in
        self.k_mlp = LightweightDirectionMLP(config)

        # β gate: 2-layer MLP
        # β(X_l) = 2 · σ(W_β_out · tanh(W_β_in · h_norm))
        # [FIX] Takes h_norm (normalized), not h_in
        beta_hidden_size = int(getattr(config, "ddl_beta_hidden_size", 128))
        self.beta_in = nn.Linear(hidden_size, beta_hidden_size, bias=False)
        self.beta_out = nn.Linear(beta_hidden_size, 1, bias=True)

        # v projection: maps backbone output h_out to value vector v ∈ R^{d_v}
        # v(X_l) = W_v · h_out + b_v
        self.v_proj = nn.Linear(hidden_size, self.value_channels, bias=True)

        # Optional: sigmoid scaling for v
        self.v_sigmoid = bool(getattr(config, "ddl_v_sigmoid", False))
        self.v_sigmoid_scale = float(getattr(config, "ddl_v_sigmoid_scale", 4.0))

        # Initialize β bias for desired initial value
        beta_init = float(getattr(config, "ddl_beta_init", 1.0))
        beta_init = min(max(beta_init, 0.0), 2.0)
        beta_init_p = beta_init / 2.0
        with torch.no_grad():
            self.beta_out.bias.fill_(_logit(beta_init_p))

    def forward(
        self,
        x: torch.Tensor,
        *,
        h_norm: torch.Tensor,
        h_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Expanded hidden state (B, T, d, d_v)
            h_norm: NORMALIZED compressed state (B, T, d) - for k and β
                    [CRITICAL] Must be RMSNorm output for training stability!
            h_out: Backbone output (B, T, d) - for v (semantic content)

        Returns:
            Updated state X_{l+1} (B, T, d, d_v)
        """
        # === k(X_l): Direction from lightweight MLP on NORMALIZED state ===
        k = self.k_mlp(h_norm)  # (B, T, d), normalized
        k_scale = 1.0 / math.sqrt(self.hidden_size)

        # === β(X_l): Gate from NORMALIZED state ===
        # β = 2 · σ(W_β_out · tanh(W_β_in · h_norm)) ∈ [0, 2]
        beta_logits = self.beta_out(torch.tanh(self.beta_in(h_norm))).float()
        beta = 2.0 * torch.sigmoid(beta_logits)  # (B, T, 1), fp32 for stability

        # === v(X_l): Content from backbone output ===
        v = self.v_proj(h_out)  # (B, T, d_v)
        if self.v_sigmoid:
            v = torch.sigmoid(v) * self.v_sigmoid_scale

        # === Validate shapes ===
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, T, d, d_v), got {tuple(x.shape)}")
        if int(x.size(-2)) != self.hidden_size:
            raise ValueError(f"Expected x feature dim {self.hidden_size}, got {int(x.size(-2))}.")
        if int(x.size(-1)) != self.value_channels:
            raise ValueError(f"Expected x value channels {self.value_channels}, got {int(x.size(-1))}.")

        # === Delta Update Rule ===
        # X_{l+1} = X_l + β · k · (v^T - k^T · X_l)

        # k^T · X_l: projection onto direction k (B, T, d_v)
        proj = torch.sum(k.unsqueeze(-1) * x, dim=-2, dtype=torch.float32) * k_scale

        # Innovation signal: (v^T - k^T · X_l)
        innovation = v - proj  # (B, T, d_v)

        # Scaled update
        delta_row = (beta * innovation) * k_scale  # (B, T, d_v), fp32

        # Rank-1 update: k · delta_row^T
        update = k.unsqueeze(-1) * delta_row.to(dtype=x.dtype).unsqueeze(-2)  # (B, T, d, d_v)

        return x + update


class Block(nn.Module):
    """
    Corrected DDL Block with proper data flow:

    [STABILITY FIX #2]: Clear separation of inputs to DDL module

    For each sublayer (Attention, MLP):
    1. Compress: h_in = Compress(X_l)
    2. Normalize: h_norm = RMSNorm(h_in)
    3. Backbone: h_out = F(h_norm)  -- F is Attention or MLP
    4. Delta Update with CORRECT inputs:
       - h_norm → k (direction) and β (gate)  [NORMALIZED for stability]
       - h_out → v (content)  [Rich semantic from backbone]
       - X_{l+1} = X_l + β · k · (v^T - k^T · X_l)

    Data flow diagram:
                                    ┌──────────────────┐
        X_l ──► Compress ──► h_in ──► RMSNorm ──► h_norm ──┬──► k_mlp ──► k
                                                          │
                                                          ├──► β_mlp ──► β
                                                          │
                                                          └──► Backbone ──► h_out ──► v_proj ──► v
                                                                (Attn/MLP)

        X_{l+1} = X_l + β · k · (v^T - k^T · X_l)
    """

    def __init__(self, config):
        super().__init__()
        self.compress = ResidualShortConvCompressor(config)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

        self.ddl_attn = DeepDeltaResidualCorrected(config)
        self.ddl_mlp = DeepDeltaResidualCorrected(config)

        self.ln_1 = RMSNorm(config.hidden_size)
        self.ln_2 = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Attention sublayer ===
        h_in = self.compress(x)           # (B, T, d) - compressed state
        h_norm = self.ln_1(h_in)          # (B, T, d) - NORMALIZED for k, β
        h_out = self.attn(h_norm)         # (B, T, d) - backbone output for v

        # [FIX] Pass h_norm (not h_in) for k and β computation
        x = self.ddl_attn(x, h_norm=h_norm, h_out=h_out)

        # === MLP sublayer ===
        h_in = self.compress(x)           # (B, T, d)
        h_norm = self.ln_2(h_in)          # (B, T, d) - NORMALIZED
        h_out = self.mlp(h_norm)          # (B, T, d) - backbone output

        # [FIX] Pass h_norm (not h_in) for k and β computation
        x = self.ddl_mlp(x, h_norm=h_norm, h_out=h_out)

        return x


# -----------------------------------------------------------------------------
# GPT Configuration and Model


@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "ddl-corrected"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_size: int = 768
    head_dim: int = 128
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    scale_attn_by_inverse_layer_idx: bool = False
    using_groupnorm: bool = False
    use_output_gate: bool = False
    use_qk_rmsnorm: bool = True
    use_k_shift: bool = False
    use_v_shift: bool = False

    q_activation: ActivationName | None = None
    k_activation: ActivationName | None = None
    v_activation: ActivationName | None = None

    rope_ratio: float = 1.0
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5

    # DDL expanded-state parameters
    ddl_value_channels: int = 4
    ddl_state_shortconv_kernel_size: int = 4
    ddl_state_read_init: float | None = None

    # DDL corrected architecture parameters
    ddl_k_eps: float = 1e-5
    ddl_k_mlp_hidden_size: int | None = None  # Default: hidden_size // 4
    ddl_k_rezero_scale: float = 0.01  # ReZero-style init scale for k MLP output
    ddl_beta_hidden_size: int = 128
    ddl_beta_init: float = 1.0
    ddl_v_sigmoid: bool = False
    ddl_v_sigmoid_scale: float = 4.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**validate_pretrained_config_kwargs(type(self), kwargs))


class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "ddl-corrected"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        if not isinstance(self, PreTrainedModel):
            super().__init__()
        else:
            super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)]),
            )
        )
        self.readout = ResidualShortConvCompressor(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.ln_f = RMSNorm(config.hidden_size)
        init_gpt_weights(self, config)

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):
        x_emb = self.transformer.wte(idx)
        value_channels = int(getattr(self.config, "ddl_value_channels", 4))
        x = x_emb.unsqueeze(-1).repeat(1, 1, 1, value_channels)

        for block in self.transformer.h:
            x = block(x)

        x_out = self.readout(x)
        x_out = self.ln_f(x_out)

        logits_scale = 1.0
        if getattr(self.config, "mup", False):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        if targets is not None:
            logits = self.lm_head(x_out)
            logits = logits.float()
            logits = logits * logits_scale
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        elif output_all_seq:
            logits = self.lm_head(x_out[:, :, :])
            logits = logits * logits_scale
            loss = None
        else:
            logits = self.lm_head(x_out[:, [-1], :])
            logits = logits.float()
            logits = logits * logits_scale
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def crop_block_size(self, block_size):
        block_size_int = int(block_size)
        if block_size_int <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size_int}.")

        current = getattr(self.config, "block_size", None)
        if isinstance(current, int):
            current_int = int(current)
            if block_size_int > current_int:
                raise ValueError(f"block_size must be <= {current_int} to crop, got {block_size_int}.")

        setattr(self.config, "block_size", block_size_int)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model
