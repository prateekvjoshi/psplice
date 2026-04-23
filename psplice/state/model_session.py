"""
ModelSession — single source of truth for all runtime model state.

Every state mutation in the daemon (loading LoRA, applying an intervention,
running a compare pass, saving/restoring a preset) goes through this class.
Route handlers in daemon/server.py delegate here; they never touch
hook_manager or intervention_registry directly.

This centralisation means:
  - Invariants can be checked in one place (e.g. no duplicate LoRA loads)
  - compare() atomically suspends and restores hooks without registry drift
  - Preset restore is a single method that clears, re-applies, and reports errors
  - Tests can exercise full state transitions without an HTTP layer

Thread/concurrency note
-----------------------
The daemon runs on a single uvicorn worker. All async handlers yield to the
event loop only when awaiting executor tasks (model.generate). State mutations
in the route handlers themselves are effectively serialised, so no locking is
needed for v1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from psplice.interventions.base import Intervention, InterventionError
from psplice.interventions.registry import InterventionRegistry
from psplice.runtime.generation import DecodeSettings
from psplice.runtime.hooks import HookManager

logger = logging.getLogger(__name__)


class ModelSessionError(Exception):
    """Raised when a state transition is invalid."""


class ModelSession:
    """
    Owns the loaded model and all associated runtime state.

    Create one instance when the daemon starts (after model loading) and
    store it at ``app.state.session``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        arch: Any,              # ArchitectureInfo
        model_id: str,
        device: str,
        dtype: str,
        eager_attn: bool,
        param_count: int,
    ) -> None:
        self.model = model
        self._base_model = model        # preserved for LoRA restore
        self.tokenizer = tokenizer
        self.arch = arch
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.eager_attn = eager_attn
        self.param_count = param_count

        self.hook_manager = HookManager()
        self.intervention_registry = InterventionRegistry()
        self.decode_settings = DecodeSettings()

        self.active_lora_path: Optional[str] = None
        self.active_preset: Optional[str] = None

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    def apply_intervention(self, iv: Intervention) -> None:
        """Validate and register an intervention. Raises ModelSessionError on failure."""
        try:
            self.intervention_registry.add(
                iv, self.model, self.hook_manager, self.arch
            )
        except InterventionError as exc:
            raise ModelSessionError(str(exc)) from exc
        # Applying a manual intervention breaks out of a named preset
        self.active_preset = None

    def remove_intervention(self, name: str) -> bool:
        """
        Remove a named intervention. Returns True if it existed.
        Clears the active preset name since config no longer matches.
        """
        removed = self.intervention_registry.remove(name, self.hook_manager)
        if removed:
            self.active_preset = None
        return removed

    def clear_interventions(self) -> None:
        """Remove every active intervention and reset the preset name."""
        self.intervention_registry.clear(self.hook_manager)
        self.active_preset = None

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def load_lora(self, adapter_path: str) -> None:
        """
        Hot-inject a PEFT LoRA adapter.

        Raises ModelSessionError if an adapter is already loaded or if
        the adapter directory is invalid.
        """
        if self.active_lora_path is not None:
            raise ModelSessionError(
                f"A LoRA adapter is already loaded from {self.active_lora_path}. "
                f"Call unload_lora() first."
            )

        resolved = str(Path(adapter_path).resolve())
        _validate_adapter_path(resolved)

        from peft import PeftModel

        try:
            peft_model = PeftModel.from_pretrained(self._base_model, resolved)
            peft_model.eval()
        except Exception as exc:
            raise ModelSessionError(f"Failed to load adapter: {exc}") from exc

        self.model = peft_model
        self.active_lora_path = resolved
        logger.info("LoRA adapter loaded: %s", resolved)

    def unload_lora(self) -> None:
        """Restore the base model, discarding the active LoRA adapter."""
        if self.active_lora_path is None:
            raise ModelSessionError("No LoRA adapter is currently loaded.")
        self.model = self._base_model
        self.active_lora_path = None
        logger.info("LoRA adapter unloaded.")

    # ------------------------------------------------------------------
    # Compare (base vs modified)
    # ------------------------------------------------------------------

    def compare(
        self,
        prompt: str,
        settings: DecodeSettings,
    ) -> tuple:
        """
        Generate the same prompt twice and return (base_result, modified_result).

        "Base" means the model with zero active interventions and no LoRA — the
        true model baseline.  "Modified" means the model with all active hooks
        and LoRA restored.

        Both generations use greedy decoding (do_sample=False) regardless of the
        current decode settings, so the only difference between the two outputs
        is the active interventions.  This makes compare trustworthy: if the two
        outputs differ, it is because of the interventions, not sampling noise.

        Sequence:
          1. Build a deterministic (greedy) copy of the decode settings
          2. Suspend all hooks and swap to the base model (no LoRA)
          3. Run base generation
          4. Restore hooks and LoRA
          5. Run modified generation
          6. Return (base, modified) — registry unchanged throughout
        """
        import copy
        import torch
        from psplice.runtime.generation import generate, DecodeSettings as DS

        # Greedy settings: preserve max_new_tokens, drop sampling params
        greedy_settings = DS(max_new_tokens=settings.max_new_tokens)

        # 1+2: Suspend hooks and drop to base model (no LoRA)
        self.hook_manager.clear()
        active_model = self.model          # may be PeftModel
        self.model = self._base_model      # bare weights, no adapter

        # Fixed seed so both runs start from identical RNG state (belt-and-suspenders)
        torch.manual_seed(0)
        base = generate(self.model, self.tokenizer, prompt, greedy_settings)
        base.label = "base"

        # 3: Restore model (LoRA if it was active) and hooks
        self.model = active_model
        self._restore_hooks()

        # 4: Run modified (hooks + LoRA now active)
        torch.manual_seed(0)
        modified = generate(self.model, self.tokenizer, prompt, greedy_settings)
        modified.label = "modified"

        return base, modified

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def save_preset(self, name: str) -> None:
        from psplice.state.presets import save_preset

        data = {
            "interventions": self.intervention_registry.serialize_all(),
            "decode_settings": self.decode_settings.serialize(),
            "active_lora": self.active_lora_path,
        }
        save_preset(name, data)
        self.active_preset = name
        logger.info("Preset saved: %s", name)

    def load_preset(self, name: str) -> list[str]:
        """
        Load a named preset.

        Returns a list of error strings for any interventions that could not
        be restored (e.g. missing vector files).  The remaining interventions
        and decode settings are applied regardless.

        The active preset name is set only if the load completes without errors.
        If there are partial errors, the name is still set (the user asked for
        it) but the errors are surfaced.
        """
        from psplice.state.presets import load_preset

        data = load_preset(name)
        if data is None:
            raise ModelSessionError(f"Preset '{name}' not found.")

        # Tear down current state fully before restoring
        self.clear_interventions()
        if self.active_lora_path:
            try:
                self.unload_lora()
            except ModelSessionError:
                pass  # already gone

        # Restore decode settings
        if "decode_settings" in data:
            self.decode_settings = DecodeSettings.deserialize(data["decode_settings"])

        # Restore LoRA (best-effort)
        errors: list[str] = []
        if data.get("active_lora"):
            lora_path = data["active_lora"]
            try:
                self.load_lora(lora_path)
            except ModelSessionError as exc:
                errors.append(f"LoRA restore failed: {exc}")

        # Restore interventions
        restore_errors = self.intervention_registry.restore_from_serialized(
            data.get("interventions", []),
            self.model,
            self.hook_manager,
            self.arch,
        )
        errors.extend(restore_errors)

        self.active_preset = name
        logger.info("Preset loaded: %s (%d errors)", name, len(errors))
        return errors

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def status_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of current session state."""
        from psplice.modeling.inspector import detect_attention_implementation

        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "eager_attn": self.eager_attn,
            "param_count": self.param_count,
            "arch_family": self.arch.family,
            "arch_class": self.arch.model_class,
            "num_layers": self.arch.num_layers,
            "num_attention_heads": self.arch.num_attention_heads,
            "hidden_size": self.arch.hidden_size,
            "max_position_embeddings": self.arch.max_position_embeddings,
            "attention_impl": detect_attention_implementation(self.model),
            "active_preset": self.active_preset,
            "active_lora": self.active_lora_path,
            "interventions": self.intervention_registry.describe_all(),
            "decode_settings": self.decode_settings.serialize(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_hooks(self) -> None:
        """
        Re-register hooks for every intervention currently in the registry.

        Called after hook_manager.clear() (e.g. during compare).  If re-applying
        an intervention fails, it is removed from the registry so the registry
        and hook_manager stay in sync.

        Hook keys are intervention names, so re-applying under the same name
        is safe even when the key already existed (hooks were cleared).
        """
        failed: list[str] = []
        for iv in self.intervention_registry.all():
            try:
                iv.apply(self.model, self.hook_manager, self.arch)
            except Exception as exc:
                logger.error(
                    "Failed to restore hook for intervention '%s': %s", iv.name, exc
                )
                failed.append(iv.name)

        # Remove interventions whose hooks could not be restored
        for name in failed:
            # Pop directly from registry dict — hook_manager is already clear
            self.intervention_registry._interventions.pop(name, None)
            logger.warning("Intervention '%s' removed from registry (hook restore failed).", name)


def _validate_adapter_path(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise ModelSessionError(f"Adapter path does not exist: {path}")
    if not (p / "adapter_config.json").exists():
        raise ModelSessionError(
            f"adapter_config.json not found in {path}. "
            f"Ensure this is a valid PEFT adapter directory."
        )
    has_weights = (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    if not has_weights:
        raise ModelSessionError(
            f"No adapter weights found in {path} "
            f"(expected adapter_model.safetensors or adapter_model.bin)."
        )
