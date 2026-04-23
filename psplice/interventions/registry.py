"""
Intervention registry.

The registry tracks all active interventions in the daemon and provides a
unified API for adding, removing, applying, serializing and deserializing them.

Each intervention is identified by a user-chosen name (a string).  Names must
be unique; adding an intervention with a name that already exists raises an
error.

The registry does NOT manage hook lifecycle directly — it delegates that to
the HookManager.  It holds the Intervention objects and coordinates with the
HookManager to apply or remove their hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from .base import Intervention, InterventionError
from .steering import SteeringIntervention
from .heads import HeadMaskIntervention
from .layers import LayerSkipIntervention
from .lora_adapter import LoraIntervention

if TYPE_CHECKING:
    from psplice.runtime.hooks import HookManager
    from psplice.modeling.inspector import ArchitectureInfo

logger = logging.getLogger(__name__)

# Registry of known intervention types for deserialization
INTERVENTION_TYPES: dict[str, type[Intervention]] = {
    "steering": SteeringIntervention,
    "head_mask": HeadMaskIntervention,
    "layer_skip": LayerSkipIntervention,
    "lora": LoraIntervention,
}


class InterventionRegistry:
    """Tracks active interventions and coordinates hook management."""

    def __init__(self) -> None:
        self._interventions: dict[str, Intervention] = {}

    # ------------------------------------------------------------------
    # Add / Remove
    # ------------------------------------------------------------------

    def add(
        self,
        intervention: Intervention,
        model: Any,
        hook_manager: "HookManager",
        arch: "ArchitectureInfo",
    ) -> None:
        """
        Validate and apply an intervention, then register it.

        Raises InterventionError if:
          - the name is already in use, or
          - the intervention is incompatible with the model.
        """
        if intervention.name in self._interventions:
            raise InterventionError(
                f"An intervention named '{intervention.name}' is already active. "
                f"Remove it first with `psplice {intervention.intervention_type} remove {intervention.name}`."
            )
        intervention.validate(arch)
        intervention.apply(model, hook_manager, arch)
        self._interventions[intervention.name] = intervention
        logger.info("Applied intervention: %r", intervention)

    def remove(self, name: str, hook_manager: "HookManager") -> bool:
        """
        Remove an active intervention by name.

        Returns True if removed, False if not found.
        """
        iv = self._interventions.pop(name, None)
        if iv is None:
            return False
        iv.remove(hook_manager)
        logger.info("Removed intervention: %r", iv)
        return True

    def clear(self, hook_manager: "HookManager") -> None:
        """Remove every active intervention."""
        for name in list(self._interventions.keys()):
            self.remove(name, hook_manager)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Intervention]:
        return self._interventions.get(name)

    def all(self) -> list[Intervention]:
        return list(self._interventions.values())

    def by_type(self, itype: str) -> list[Intervention]:
        return [iv for iv in self._interventions.values() if iv.intervention_type == itype]

    def names(self) -> list[str]:
        return list(self._interventions.keys())

    def is_empty(self) -> bool:
        return len(self._interventions) == 0

    # ------------------------------------------------------------------
    # Serialization (for presets)
    # ------------------------------------------------------------------

    def serialize_all(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable list of all active interventions."""
        return [iv.serialize() for iv in self._interventions.values()]

    def describe_all(self) -> list[dict[str, Any]]:
        """Return human-readable descriptions of all active interventions."""
        return [iv.describe() for iv in self._interventions.values()]

    def restore_from_serialized(
        self,
        data: list[dict[str, Any]],
        model: Any,
        hook_manager: "HookManager",
        arch: "ArchitectureInfo",
    ) -> list[str]:
        """
        Reconstruct and re-apply interventions from serialized data.

        Returns a list of error messages for any interventions that failed to
        restore (e.g. missing vector files).  Other interventions are still
        applied.
        """
        errors: list[str] = []
        for item in data:
            itype = item.get("intervention_type", "")
            cls = INTERVENTION_TYPES.get(itype)
            if cls is None:
                errors.append(f"Unknown intervention type: {itype!r}")
                continue
            try:
                iv = cls.deserialize(item)
                self.add(iv, model, hook_manager, arch)
            except (InterventionError, Exception) as exc:
                errors.append(f"{item.get('name', '?')}: {exc}")
        return errors
