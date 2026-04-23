"""
Architecture inspection.

psplice targets a specific set of decoder-only model families.  This module
inspects a loaded model to extract the architecture facts needed by
interventions (layer count, head count, how to reach o_proj, etc.) and
reports clearly when a model family is not supported.

Supported families
------------------
* llama  — LlamaForCausalLM (Llama-2, Llama-3.x, Code Llama, …)
* qwen2  — Qwen2ForCausalLM (Qwen2.5-7B-Instruct, etc.)
* mistral — MistralForCausalLM (Mistral-7B, Mixtral, …)
* gemma  — GemmaForCausalLM / Gemma2ForCausalLM (best-effort)

Unknown families are reported as "other" and most surgery operations are
refused with a clear message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Known model_type strings from HF config
_KNOWN_FAMILIES: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2",
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
    "phi": "other",
    "phi3": "other",
    "falcon": "other",
    "mpt": "other",
    "gpt_neox": "other",
    "bloom": "other",
}

# Families that expose layers at model.model.layers (the standard HF pattern)
_STANDARD_LAYERS_FAMILIES = {"llama", "qwen2", "mistral", "gemma"}


@dataclass
class ArchitectureInfo:
    """Architecture facts extracted from a loaded model + its config."""

    family: str                     # e.g. "llama", "qwen2", "mistral", "other"
    model_type: str                 # raw config.model_type
    model_class: str                # class name of the model object
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int        # GQA — may differ from num_attention_heads
    head_dim: int
    max_position_embeddings: int
    vocab_size: int
    # Architecture support flags
    has_standard_layers: bool       # model.model.layers accessible
    supports_head_masking: bool     # o_proj accessible for head surgery
    supports_steering: bool         # can hook decoder layer outputs
    supports_layer_skip: bool       # can bypass decoder layers
    # Used by head-masking to reach o_proj
    attn_module_attr: str = "self_attn"    # attribute name on each layer
    o_proj_attr: str = "o_proj"            # attribute name on attn module
    attention_class: Optional[str] = None  # class name of attention module


def inspect_model(model) -> ArchitectureInfo:
    """
    Inspect *model* and return an ArchitectureInfo dataclass.

    The model should already be fully loaded (weights on device).
    """
    cfg = model.config
    model_type: str = getattr(cfg, "model_type", "unknown")
    family = _KNOWN_FAMILIES.get(model_type, "other")

    # Layer / head counts — every supported family exposes these on config
    num_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or _count_layers(model)
    )
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
        or 0
    )
    num_attention_heads = getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    head_dim = hidden_size // num_attention_heads if num_attention_heads else 0
    max_pos = getattr(cfg, "max_position_embeddings", 0) or getattr(cfg, "n_positions", 0)
    vocab_size = getattr(cfg, "vocab_size", 0)

    has_standard = family in _STANDARD_LAYERS_FAMILIES and _has_model_layers(model)

    # Detect attention module and o_proj
    attn_attr = "self_attn"
    o_proj_attr = "o_proj"
    attention_class = None

    if has_standard:
        try:
            first_layer = model.model.layers[0]
            attn = getattr(first_layer, attn_attr, None)
            if attn is not None:
                attention_class = type(attn).__name__
                # Verify o_proj exists (it should for all supported families)
                if not hasattr(attn, o_proj_attr):
                    o_proj_attr = _find_o_proj(attn)
        except Exception:
            pass

    supports_head = has_standard and o_proj_attr is not None
    supports_steering = has_standard
    supports_skip = has_standard

    return ArchitectureInfo(
        family=family,
        model_type=model_type,
        model_class=type(model).__name__,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        vocab_size=vocab_size,
        has_standard_layers=has_standard,
        supports_head_masking=supports_head,
        supports_steering=supports_steering,
        supports_layer_skip=supports_skip,
        attn_module_attr=attn_attr,
        o_proj_attr=o_proj_attr,
        attention_class=attention_class,
    )


def get_decoder_layers(model) -> list:
    """Return the list of decoder layer modules (model.model.layers)."""
    return list(model.model.layers)


def get_attention_module(model, layer_idx: int):
    """Return the attention sub-module for the given layer index."""
    arch = inspect_model(model)
    layer = model.model.layers[layer_idx]
    return getattr(layer, arch.attn_module_attr)


def get_o_proj(model, layer_idx: int):
    """Return the output projection module for the given layer."""
    arch = inspect_model(model)
    attn = get_attention_module(model, layer_idx)
    return getattr(attn, arch.o_proj_attr)


def detect_attention_implementation(model) -> str:
    """
    Return a string describing the active attention implementation.

    Checks the config attribute used by HuggingFace Transformers >= 4.36.
    """
    cfg = model.config
    impl = getattr(cfg, "_attn_implementation", None)
    if impl:
        return impl
    # Older models — infer from class name
    try:
        first_layer = model.model.layers[0]
        attn = first_layer.self_attn
        cls = type(attn).__name__.lower()
        if "flash" in cls:
            return "flash_attention_2"
        if "sdpa" in cls:
            return "sdpa"
        return "eager"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_model_layers(model) -> bool:
    try:
        layers = model.model.layers
        return len(layers) > 0
    except Exception:
        return False


def _count_layers(model) -> int:
    try:
        return len(model.model.layers)
    except Exception:
        return 0


def _find_o_proj(attn_module) -> Optional[str]:
    """Try common attribute names for the output projection."""
    for name in ("o_proj", "out_proj", "dense", "c_proj"):
        if hasattr(attn_module, name):
            return name
    return None
