"""
Hook manager.

All PyTorch forward hooks registered by psplice interventions are tracked
here.  Every hook is stored under a caller-chosen key so it can be cleanly
removed later — preventing hook leakage across sessions or after unloading
an intervention.

Usage::

    mgr = HookManager()

    # Register one or more hooks under a single key
    mgr.register("steer:honesty", module, hook_fn, hook_type="post")
    mgr.register("steer:honesty", another_module, hook_fn2, hook_type="pre")

    # Remove all hooks for that key
    mgr.remove("steer:honesty")

    # Remove every hook registered through this manager
    mgr.clear()
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import torch.nn as nn

logger = logging.getLogger(__name__)

HookType = Literal["pre", "post", "pre_full"]


class HookManager:
    """Tracks and removes PyTorch forward hooks by logical key."""

    def __init__(self) -> None:
        # key -> list of RemovableHook objects
        self._handles: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        module: nn.Module,
        hook_fn: Callable,
        hook_type: HookType = "post",
    ) -> None:
        """
        Register a hook on *module* and file it under *key*.

        hook_type:
          "post"     — register_forward_hook (most common; sees input + output)
          "pre"      — register_forward_pre_hook (sees input before forward)
          "pre_full" — register_forward_pre_hook with with_kwargs=True
        """
        if hook_type == "post":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "pre":
            handle = module.register_forward_pre_hook(hook_fn)
        elif hook_type == "pre_full":
            handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
        else:
            raise ValueError(f"Unknown hook_type: {hook_type!r}")

        self._handles.setdefault(key, []).append(handle)
        logger.debug("Registered %s hook for key=%r on %s", hook_type, key, type(module).__name__)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, key: str) -> bool:
        """
        Remove all hooks filed under *key*.

        Returns True if any hooks were removed, False if the key was not found.
        """
        handles = self._handles.pop(key, None)
        if handles is None:
            return False
        for h in handles:
            h.remove()
        logger.debug("Removed %d hook(s) for key=%r", len(handles), key)
        return True

    def clear(self) -> None:
        """Remove every hook tracked by this manager."""
        for key in list(self._handles.keys()):
            self.remove(key)
        logger.debug("HookManager cleared")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_keys(self) -> list[str]:
        """Return the sorted list of active hook keys."""
        return sorted(self._handles.keys())

    def hook_count(self) -> int:
        """Total number of individual hook handles currently registered."""
        return sum(len(v) for v in self._handles.values())

    def has_key(self, key: str) -> bool:
        return key in self._handles

    def __repr__(self) -> str:  # noqa: D105
        return f"HookManager(keys={self.active_keys()}, total_hooks={self.hook_count()})"
