"""
Base intervention interface.

Every intervention in psplice implements this ABC.  The common interface lets
the intervention registry apply, remove, describe and serialize any
intervention without knowing its concrete type.

Implementing a new intervention
--------------------------------
1. Subclass Intervention.
2. Override all abstract methods.
3. Register the subclass in INTERVENTION_TYPES at the bottom of registry.py so
   that deserialization works.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from psplice.runtime.hooks import HookManager
    from psplice.modeling.inspector import ArchitectureInfo


class InterventionError(Exception):
    """Raised when an intervention cannot be applied to the current model."""


class Intervention(ABC):
    """Abstract base for all runtime interventions."""

    # Subclasses must set these class-level attributes
    intervention_type: str = "base"

    def __init__(self, name: str) -> None:
        self.name = name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def validate(self, arch: "ArchitectureInfo") -> None:
        """
        Raise InterventionError if this intervention is incompatible with the
        model described by *arch*.  Called before apply().
        """

    @abstractmethod
    def apply(self, model: Any, hook_manager: "HookManager", arch: "ArchitectureInfo") -> None:
        """
        Register PyTorch hooks that implement the intervention.

        All hooks MUST be filed under self.name in *hook_manager* so they can
        be cleanly removed later.
        """

    def remove(self, hook_manager: "HookManager") -> None:
        """
        Remove all hooks registered by this intervention.

        The default implementation removes by self.name, which is correct for
        all interventions that use name as their hook key.
        """
        hook_manager.remove(self.name)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """
        Return a human-readable summary dict (used by `psplice status`).
        """

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict that can be round-tripped through
        deserialize().  Must include an "intervention_type" key.
        """

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict[str, Any]) -> "Intervention":
        """
        Reconstruct an intervention from the output of serialize().
        Called by the registry when loading a preset.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
