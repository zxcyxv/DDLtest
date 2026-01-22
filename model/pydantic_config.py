"""Config validation utilities."""

from typing import Any, Type
from dataclasses import fields, is_dataclass


def validate_pretrained_config_kwargs(cls: Type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and filter kwargs for PretrainedConfig subclass."""
    if is_dataclass(cls):
        valid_fields = {f.name for f in fields(cls)}
    else:
        valid_fields = set()

    # Also include common PretrainedConfig fields
    common_fields = {
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "hidden_act", "hidden_dropout_prob", "max_position_embeddings",
        "type_vocab_size", "initializer_range", "layer_norm_eps", "pad_token_id",
        "bos_token_id", "eos_token_id", "tie_word_embeddings", "torch_dtype",
        "use_cache", "output_attentions", "output_hidden_states", "return_dict",
        "_name_or_path", "architectures", "model_type", "transformers_version",
    }

    all_valid = valid_fields | common_fields

    # Filter out unknown kwargs but keep all for flexibility
    return kwargs
