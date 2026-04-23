"""
LoRA hot-injection intervention.

Loads a PEFT adapter onto the resident model without restarting the daemon.
One adapter can be active at a time (v1 constraint).

Implementation notes
--------------------
PEFT's PeftModel.from_pretrained() wraps the base model in-place, adding
low-rank adapter matrices alongside the original weight tensors.  Original
weights are not modified — LoRA adds delta_W = A @ B (scaled) during the
forward pass.

To "unload" the adapter cleanly, psplice stores a reference to the original
base model before wrapping and restores it when the user calls
`psplice lora unload`.  This avoids the complication of trying to unwrap a
PeftModel in place.

The daemon's runtime state is updated to point to the PEFT-wrapped model when
active, and back to the base model when unloaded.

Limitation: if the adapter directory is deleted or moved after loading,
unloading will still work (we only need the base model reference) but
re-loading from a preset will fail with a clear error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import Intervention, InterventionError

logger = logging.getLogger(__name__)


class LoraIntervention(Intervention):
    """Hot-inject a PEFT LoRA adapter onto the resident model."""

    intervention_type = "lora"

    def __init__(self, name: str, adapter_path: str) -> None:
        super().__init__(name)
        self.adapter_path = str(Path(adapter_path).resolve())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, arch) -> None:
        path = Path(self.adapter_path)
        if not path.exists():
            raise InterventionError(f"Adapter path does not exist: {path}")
        # Look for standard PEFT adapter files
        has_config = (path / "adapter_config.json").exists()
        has_weights = (path / "adapter_model.safetensors").exists() or (
            path / "adapter_model.bin"
        ).exists()
        if not has_config:
            raise InterventionError(
                f"adapter_config.json not found in {path}.  "
                f"Make sure this is a valid PEFT adapter directory."
            )
        if not has_weights:
            raise InterventionError(
                f"No adapter weights found in {path} "
                f"(expected adapter_model.safetensors or adapter_model.bin)."
            )

    # ------------------------------------------------------------------
    # Apply — LoRA requires special handling at the runtime state level
    # ------------------------------------------------------------------

    def apply(self, model, hook_manager, arch) -> None:
        """
        LoRA loading is handled by the daemon's runtime state manager, not
        via forward hooks.  This method is intentionally left as a no-op here;
        the actual PEFT wrapping happens in daemon/server.py's lora_load route,
        which has access to the full runtime state.

        This intervention object is used for preset serialization only.
        """
        # Validation still runs to catch bad paths early
        self.validate(arch)
        logger.debug("LoraIntervention.apply() called; actual wrapping done by runtime.")

    def remove(self, hook_manager) -> None:
        """LoRA removal is also handled at the runtime level, not via hooks."""
        pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.intervention_type,
            "name": self.name,
            "adapter_path": self.adapter_path,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "name": self.name,
            "adapter_path": self.adapter_path,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "LoraIntervention":
        return cls(name=data["name"], adapter_path=data["adapter_path"])
