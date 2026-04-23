"""
Activation steering intervention.

Adds a steering vector to the residual stream at selected decoder layers
during the forward pass.  The vector is loaded from a .pt file.

Supported tensor formats
------------------------
1. A single 1D tensor of shape [hidden_size].
   The same vector is broadcast to every specified layer.

2. A Python dict mapping int layer indices to 1D tensors of shape [hidden_size].
   Each layer gets its own vector.  The requested layer_indices must be a subset
   of the dict keys.

Any other format raises a clear validation error.

Implementation
--------------
A post-hook is registered on each target decoder layer (model.model.layers[i]).
The hook adds  scale * vector  to the first element of the layer's output
tuple (the residual-stream hidden states).  Hooks are stored under the
intervention name and removed cleanly via HookManager.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .base import Intervention, InterventionError

logger = logging.getLogger(__name__)


class SteeringIntervention(Intervention):
    """Activation steering via residual-stream additive vectors."""

    intervention_type = "steering"

    def __init__(
        self,
        name: str,
        vector_path: str,
        layer_indices: list[int],
        scale: float = 1.0,
    ) -> None:
        super().__init__(name)
        self.vector_path = str(Path(vector_path).resolve())
        self.layer_indices = sorted(layer_indices)
        self.scale = scale
        # Loaded tensors are set during apply()
        self._layer_vectors: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, arch) -> None:
        if not arch.supports_steering:
            raise InterventionError(
                f"Model family '{arch.family}' does not support activation steering "
                f"through psplice's hook-based approach (model.model.layers not accessible)."
            )
        if not arch.has_standard_layers:
            raise InterventionError(
                "Activation steering requires model.model.layers to be accessible."
            )
        # Validate layer indices against actual layer count
        bad = [i for i in self.layer_indices if i >= arch.num_layers or i < 0]
        if bad:
            raise InterventionError(
                f"Layer indices out of range for {arch.num_layers}-layer model: {bad}"
            )
        # Validate vector file
        path = Path(self.vector_path)
        if not path.exists():
            raise InterventionError(f"Steering vector file not found: {path}")

    # ------------------------------------------------------------------
    # Apply / Remove
    # ------------------------------------------------------------------

    def apply(self, model, hook_manager, arch) -> None:
        self.validate(arch)

        # Load tensor
        raw = torch.load(self.vector_path, map_location="cpu", weights_only=True)
        self._layer_vectors = self._parse_vector(raw, arch.hidden_size)

        # Register one hook per target layer
        for layer_idx in self.layer_indices:
            vec = self._layer_vectors[layer_idx]
            layer = model.model.layers[layer_idx]

            hook_fn = _make_steer_hook(vec, self.scale)
            hook_manager.register(
                key=self.name,
                module=layer,
                hook_fn=hook_fn,
                hook_type="post",
            )
            logger.debug("Steering hook registered: layer=%d name=%s scale=%.3f", layer_idx, self.name, self.scale)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.intervention_type,
            "name": self.name,
            "vector_path": self.vector_path,
            "layers": self.layer_indices,
            "scale": self.scale,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "name": self.name,
            "vector_path": self.vector_path,
            "layer_indices": self.layer_indices,
            "scale": self.scale,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "SteeringIntervention":
        return cls(
            name=data["name"],
            vector_path=data["vector_path"],
            layer_indices=data["layer_indices"],
            scale=data.get("scale", 1.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_vector(self, raw, hidden_size: int) -> dict[int, torch.Tensor]:
        """
        Parse the loaded tensor into a dict of {layer_idx: 1D_tensor}.

        Accepts exactly two formats:
          - 1D tensor [hidden_size]: broadcast to all requested layers
          - dict {int: 1D tensor [hidden_size]}: per-layer assignment
        """
        if isinstance(raw, torch.Tensor):
            if raw.dim() != 1:
                raise InterventionError(
                    f"Steering vector must be 1D, got shape {list(raw.shape)}. "
                    f"Expected a tensor of shape [{hidden_size}]."
                )
            if raw.shape[0] != hidden_size:
                raise InterventionError(
                    f"Steering vector size {raw.shape[0]} does not match "
                    f"model hidden_size {hidden_size}."
                )
            # Broadcast
            return {idx: raw.float() for idx in self.layer_indices}

        elif isinstance(raw, dict):
            result: dict[int, torch.Tensor] = {}
            for idx in self.layer_indices:
                if idx not in raw:
                    raise InterventionError(
                        f"Layer index {idx} not found in per-layer steering dict. "
                        f"Available keys: {sorted(raw.keys())}"
                    )
                vec = raw[idx]
                if not isinstance(vec, torch.Tensor) or vec.dim() != 1:
                    raise InterventionError(
                        f"Layer {idx} vector must be a 1D tensor, got {type(vec)}."
                    )
                if vec.shape[0] != hidden_size:
                    raise InterventionError(
                        f"Layer {idx} vector size {vec.shape[0]} != hidden_size {hidden_size}."
                    )
                result[idx] = vec.float()
            return result

        else:
            raise InterventionError(
                f"Unsupported steering vector format: {type(raw).__name__}. "
                f"Expected a 1D torch.Tensor of shape [hidden_size] or a dict "
                f"mapping layer indices to 1D tensors."
            )


def _make_steer_hook(vector: torch.Tensor, scale: float):
    """
    Build a forward hook that adds scale * vector to the residual stream.

    The hook captures *vector* and *scale* by closure.  The tensor is cast to
    match the hidden-state dtype on the fly so bfloat16/float16 models work
    without separate vectors.
    """
    def hook(module, inputs, output):
        # Decoder layers return a tuple; first element is hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            v = vector.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden + scale * v
            return (patched,) + output[1:]
        else:
            v = vector.to(device=output.device, dtype=output.dtype)
            return output + scale * v

    return hook
