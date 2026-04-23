"""
Head masking intervention.

Disables selected attention heads at inference time by zeroing their
contributions to the attention output before the o_proj linear layer.

Requirements
------------
* The model must be loaded with --eager-attn (attn_implementation="eager").
  sdpa and flash_attention_2 fuse the attention kernel in a way that prevents
  reliable per-head intervention after the fact.
* The model must belong to a supported family (llama, qwen2, mistral, gemma)
  so that model.model.layers[i].self_attn.o_proj is accessible.

Implementation
--------------
For each (layer, head) pair a pre-hook is registered on the o_proj module of
that layer's attention.  The hook receives the input tensor to o_proj, which
has shape [batch, seq_len, num_heads * head_dim], and zeros out the columns
that correspond to the masked heads before the projection is applied.

This is a surgical, reversible modification: it does not change any model
weights and is fully removed when the intervention is cleared.

Limitations
-----------
* Only tested on standard MHA architectures (Llama, Qwen2, Mistral).
* GQA (grouped-query attention) models with num_key_value_heads !=
  num_attention_heads are masked at the query-head granularity only, which is
  the correct conceptual unit for output contribution.
* Mixed precision (bfloat16/float16) is handled transparently.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Intervention, InterventionError

logger = logging.getLogger(__name__)


class HeadMaskIntervention(Intervention):
    """Zero out selected attention heads via o_proj pre-hooks."""

    intervention_type = "head_mask"

    def __init__(self, name: str, layer_heads: dict[int, list[int]]) -> None:
        """
        Parameters
        ----------
        name:
            Unique intervention name.
        layer_heads:
            Mapping from layer index to list of head indices to mask.
            Example: {3: [0, 2], 7: [4]}
        """
        super().__init__(name)
        # Normalise: ensure head lists are sorted
        self.layer_heads: dict[int, list[int]] = {
            int(k): sorted(int(h) for h in v) for k, v in layer_heads.items()
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, arch) -> None:
        if not arch.supports_head_masking:
            raise InterventionError(
                f"Model family '{arch.family}' does not expose o_proj in a supported "
                f"location.  Head masking is not available for this model."
            )

        attn_impl = _detect_attn_impl(arch)
        if attn_impl in ("sdpa", "flash_attention_2"):
            raise InterventionError(
                f"Head masking requires eager attention, but this model is using "
                f"'{attn_impl}'.  Reload the model with `psplice load <model> --eager-attn` "
                f"to enable per-head surgery."
            )

        # Check layer/head ranges
        for layer_idx, heads in self.layer_heads.items():
            if layer_idx < 0 or layer_idx >= arch.num_layers:
                raise InterventionError(
                    f"Layer {layer_idx} is out of range (model has {arch.num_layers} layers)."
                )
            bad_heads = [h for h in heads if h < 0 or h >= arch.num_attention_heads]
            if bad_heads:
                raise InterventionError(
                    f"Layer {layer_idx}: head indices {bad_heads} out of range "
                    f"(model has {arch.num_attention_heads} attention heads)."
                )

    # ------------------------------------------------------------------
    # Apply / Remove
    # ------------------------------------------------------------------

    def apply(self, model, hook_manager, arch) -> None:
        self.validate(arch)

        head_dim = arch.head_dim
        num_heads = arch.num_attention_heads

        for layer_idx, heads in self.layer_heads.items():
            attn = getattr(model.model.layers[layer_idx], arch.attn_module_attr)
            o_proj = getattr(attn, arch.o_proj_attr)

            hook_fn = _make_head_mask_hook(heads, num_heads, head_dim)
            hook_manager.register(
                key=self.name,
                module=o_proj,
                hook_fn=hook_fn,
                hook_type="pre",
            )
            logger.debug(
                "Head mask hook registered: layer=%d heads=%s name=%s",
                layer_idx,
                heads,
                self.name,
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.intervention_type,
            "name": self.name,
            "layer_heads": {str(k): v for k, v in self.layer_heads.items()},
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "name": self.name,
            "layer_heads": {str(k): v for k, v in self.layer_heads.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "HeadMaskIntervention":
        layer_heads = {int(k): v for k, v in data["layer_heads"].items()}
        return cls(name=data["name"], layer_heads=layer_heads)


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def _make_head_mask_hook(heads: list[int], num_heads: int, head_dim: int):
    """
    Build a forward pre-hook for o_proj that zeros selected head columns.

    The input to o_proj has shape [batch, seq_len, num_heads * head_dim].
    Head i occupies columns [i*head_dim : (i+1)*head_dim].

    The hook clones the tensor, zeroes the relevant columns in-place, and
    returns the modified args tuple so the projection sees the masked input.
    """
    def hook(module, args):
        # args is a tuple; args[0] is the input tensor
        x = args[0].clone()
        for h in heads:
            col_start = h * head_dim
            col_end = (h + 1) * head_dim
            x[..., col_start:col_end] = 0.0
        return (x,)

    return hook


def _detect_attn_impl(arch) -> str:
    """Extract the attention implementation tag from the arch info."""
    # arch is ArchitectureInfo from inspector.py; we re-check via model_type
    # The caller passes arch after loading, so we use the class embedded there.
    # We do a best-effort parse of the attention_class name.
    attn_cls = (arch.attention_class or "").lower()
    if "flash" in attn_cls:
        return "flash_attention_2"
    if "sdpa" in attn_cls:
        return "sdpa"
    # Also read from model_type hints — if arch doesn't embed the impl tag,
    # we conservatively assume eager if the class name doesn't scream sdpa/flash.
    return "eager"
