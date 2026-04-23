"""
Layer skip (early-exit approximation) intervention.

Bypasses decoder layers starting from *skip_from* by replacing each layer's
output hidden states with its input hidden states.  The remaining layers in
the forward pass still execute (PyTorch hooks cannot short-circuit the
computation graph) but their transformations are discarded.

This approximation is honest: the model still runs all layers, but layers
skip_from..N-1 do not contribute to the final representation.  The result
is equivalent to stopping the residual stream update at layer skip_from.

Use cases
---------
* Examine representations formed by early layers.
* Test how much later layers contribute to a given output.
* Trade quality for speed in a controlled, observable way (though the forward
  pass still runs all layers — true compute savings require custom kernels).

Limitations
-----------
* Not a true early exit: compute cost is unchanged.
* The LM head is always applied to the full-depth representation shape; only
  the content of that representation is truncated.
* Positional encodings and causal masks are still computed for all layers.
* Only tested on Llama/Qwen2/Mistral families.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Intervention, InterventionError

logger = logging.getLogger(__name__)


class LayerSkipIntervention(Intervention):
    """Bypass decoder layers from skip_from onward."""

    intervention_type = "layer_skip"

    def __init__(self, name: str, skip_from: int) -> None:
        super().__init__(name)
        self.skip_from = skip_from

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, arch) -> None:
        if not arch.supports_layer_skip:
            raise InterventionError(
                f"Model family '{arch.family}' does not support layer skipping via "
                f"psplice's hook mechanism."
            )
        if self.skip_from < 1:
            raise InterventionError("skip_from must be >= 1 (cannot skip all layers).")
        if self.skip_from >= arch.num_layers:
            raise InterventionError(
                f"skip_from={self.skip_from} is >= num_layers={arch.num_layers}. "
                f"No layers would be skipped."
            )

    # ------------------------------------------------------------------
    # Apply / Remove
    # ------------------------------------------------------------------

    def apply(self, model, hook_manager, arch) -> None:
        self.validate(arch)

        # For each layer that should be skipped, register a pair of hooks:
        # 1. pre-hook: capture the input hidden states
        # 2. post-hook: replace the output hidden states with the saved input
        for layer_idx in range(self.skip_from, arch.num_layers):
            layer = model.model.layers[layer_idx]
            skipper = _LayerSkipper()
            hook_manager.register(
                key=self.name,
                module=layer,
                hook_fn=skipper.pre_hook,
                hook_type="pre",
            )
            hook_manager.register(
                key=self.name,
                module=layer,
                hook_fn=skipper.post_hook,
                hook_type="post",
            )

        logger.debug(
            "Layer skip hooks registered: layers %d..%d name=%s",
            self.skip_from,
            arch.num_layers - 1,
            self.name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.intervention_type,
            "name": self.name,
            "skip_from": self.skip_from,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "name": self.name,
            "skip_from": self.skip_from,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LayerSkipIntervention":
        return cls(name=data["name"], skip_from=data["skip_from"])


class _LayerSkipper:
    """
    Stateful hook pair that captures layer input and replaces layer output.

    One instance per layer ensures there is no cross-layer state contamination.
    """

    def __init__(self) -> None:
        self._saved_hidden: Any = None

    def pre_hook(self, module, args):
        """Save the hidden_states (first positional arg) before forward."""
        if args:
            self._saved_hidden = args[0]
        return args  # pass through unmodified

    def post_hook(self, module, args, output):
        """Replace output hidden states with the saved input."""
        if self._saved_hidden is None:
            return output
        h = self._saved_hidden
        self._saved_hidden = None  # clear for next token
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
