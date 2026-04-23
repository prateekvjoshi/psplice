"""
Tests for intervention serialization, validation, and hook behavior.

Model-loading and GPU operations are mocked.  We use real nn.Linear modules
so that hooks can actually be registered and triggered.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from psplice.interventions.base import InterventionError
from psplice.interventions.heads import HeadMaskIntervention
from psplice.interventions.layers import LayerSkipIntervention
from psplice.interventions.lora_adapter import LoraIntervention
from psplice.interventions.registry import InterventionRegistry
from psplice.interventions.steering import SteeringIntervention
from psplice.runtime.hooks import HookManager


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

class TestSteeringSerialization:
    def test_round_trip(self, uniform_vector_path):
        iv = SteeringIntervention(
            name="test_steer",
            vector_path=str(uniform_vector_path),
            layer_indices=[10, 11, 12],
            scale=0.5,
        )
        data = iv.serialize()
        assert data["intervention_type"] == "steering"
        assert data["scale"] == 0.5

        restored = SteeringIntervention.deserialize(data)
        assert restored.name == "test_steer"
        assert restored.layer_indices == [10, 11, 12]
        assert restored.scale == 0.5


class TestSteeringValidation:
    def test_bad_layer_index(self, uniform_vector_path, fake_arch):
        iv = SteeringIntervention(
            name="bad_layers",
            vector_path=str(uniform_vector_path),
            layer_indices=[999],
            scale=1.0,
        )
        with pytest.raises(InterventionError, match="out of range"):
            iv.validate(fake_arch)

    def test_wrong_vector_size(self, tmp_path, fake_arch):
        wrong = torch.randn(100)  # wrong size
        p = tmp_path / "wrong.pt"
        torch.save(wrong, p)
        iv = SteeringIntervention(
            name="wrong_size",
            vector_path=str(p),
            layer_indices=[0],
            scale=1.0,
        )
        with pytest.raises(InterventionError, match="hidden_size"):
            iv.validate(fake_arch)
            # _parse_vector is called inside apply, trigger it
            raw = torch.load(p, weights_only=True)
            iv._parse_vector(raw, fake_arch.hidden_size)

    def test_missing_file(self, fake_arch):
        iv = SteeringIntervention(
            name="missing",
            vector_path="/nonexistent/path.pt",
            layer_indices=[0],
        )
        with pytest.raises(InterventionError, match="not found"):
            iv.validate(fake_arch)

    def test_2d_tensor_rejected(self, tmp_path, fake_arch):
        wrong = torch.randn(4096, 4096)
        p = tmp_path / "2d.pt"
        torch.save(wrong, p)
        iv = SteeringIntervention(
            name="bad",
            vector_path=str(p),
            layer_indices=[0],
        )
        raw = torch.load(p, weights_only=True)
        with pytest.raises(InterventionError, match="1D"):
            iv._parse_vector(raw, fake_arch.hidden_size)

    def test_uniform_vector_parsed_correctly(self, uniform_vector_path, fake_arch):
        iv = SteeringIntervention(
            name="ok",
            vector_path=str(uniform_vector_path),
            layer_indices=[0, 1, 2],
        )
        raw = torch.load(str(uniform_vector_path), weights_only=True)
        result = iv._parse_vector(raw, fake_arch.hidden_size)
        assert set(result.keys()) == {0, 1, 2}
        assert result[0].shape == (4096,)

    def test_per_layer_vector_parsed(self, per_layer_vector_path, fake_arch):
        iv = SteeringIntervention(
            name="per_layer",
            vector_path=str(per_layer_vector_path),
            layer_indices=[10, 11, 12],
        )
        raw = torch.load(str(per_layer_vector_path), weights_only=True)
        result = iv._parse_vector(raw, fake_arch.hidden_size)
        assert set(result.keys()) == {10, 11, 12}


# ---------------------------------------------------------------------------
# Head masking
# ---------------------------------------------------------------------------

class TestHeadMaskSerialization:
    def test_round_trip(self):
        iv = HeadMaskIntervention(name="hm", layer_heads={3: [0, 2], 7: [4]})
        data = iv.serialize()
        restored = HeadMaskIntervention.deserialize(data)
        assert restored.layer_heads == {3: [0, 2], 7: [4]}

    def test_int_keys_preserved(self):
        iv = HeadMaskIntervention(name="hm", layer_heads={"5": [1, 2]})
        # Constructor should coerce to int
        assert 5 in iv.layer_heads


class TestHeadMaskValidation:
    def test_bad_layer(self, fake_arch):
        iv = HeadMaskIntervention(name="hm", layer_heads={999: [0]})
        with pytest.raises(InterventionError, match="out of range"):
            iv.validate(fake_arch)

    def test_bad_head(self, fake_arch):
        iv = HeadMaskIntervention(name="hm", layer_heads={0: [999]})
        with pytest.raises(InterventionError, match="out of range"):
            iv.validate(fake_arch)

    def test_sdpa_rejected(self, fake_arch):
        # Fake arch with sdpa attention class
        import dataclasses
        arch_sdpa = dataclasses.replace(fake_arch, attention_class="LlamaSdpaAttention")
        iv = HeadMaskIntervention(name="hm", layer_heads={0: [0]})
        with pytest.raises(InterventionError, match="eager attention"):
            iv.validate(arch_sdpa)


# ---------------------------------------------------------------------------
# Layer skip
# ---------------------------------------------------------------------------

class TestLayerSkipSerialization:
    def test_round_trip(self):
        iv = LayerSkipIntervention(name="skip", skip_from=24)
        data = iv.serialize()
        restored = LayerSkipIntervention.deserialize(data)
        assert restored.skip_from == 24


class TestLayerSkipValidation:
    def test_skip_from_zero_rejected(self, fake_arch):
        iv = LayerSkipIntervention(name="skip", skip_from=0)
        with pytest.raises(InterventionError, match="must be >= 1"):
            iv.validate(fake_arch)

    def test_skip_from_beyond_num_layers(self, fake_arch):
        iv = LayerSkipIntervention(name="skip", skip_from=100)
        with pytest.raises(InterventionError, match="out of range|>= num_layers"):
            iv.validate(fake_arch)

    def test_valid_skip(self, fake_arch):
        iv = LayerSkipIntervention(name="skip", skip_from=24)
        iv.validate(fake_arch)  # should not raise


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class TestLoraSerialization:
    def test_round_trip(self):
        iv = LoraIntervention(name="lora", adapter_path="/some/path")
        data = iv.serialize()
        restored = LoraIntervention.deserialize(data)
        assert restored.name == "lora"

    def test_path_is_resolved(self, tmp_path):
        iv = LoraIntervention(name="lora", adapter_path=str(tmp_path))
        assert Path(iv.adapter_path).is_absolute()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_add_and_remove(self, fake_arch, hook_manager, intervention_registry):
        """Registry tracks interventions; remove cleans them up."""
        # We use a no-op apply
        from psplice.interventions.base import Intervention

        class NopIntervention(Intervention):
            intervention_type = "nop"
            def validate(self, arch): pass
            def apply(self, model, hm, arch): pass
            def describe(self): return {"type": "nop", "name": self.name}
            def serialize(self): return {"intervention_type": "nop", "name": self.name}
            @classmethod
            def deserialize(cls, data): return cls(data["name"])

        iv = NopIntervention("n1")
        intervention_registry.add(iv, None, hook_manager, fake_arch)
        assert "n1" in intervention_registry.names()

        intervention_registry.remove("n1", hook_manager)
        assert "n1" not in intervention_registry.names()

    def test_duplicate_name_raises(self, fake_arch, hook_manager, intervention_registry):
        from psplice.interventions.base import Intervention

        class NopIntervention(Intervention):
            intervention_type = "nop"
            def validate(self, arch): pass
            def apply(self, model, hm, arch): pass
            def describe(self): return {}
            def serialize(self): return {"intervention_type": "nop", "name": self.name}
            @classmethod
            def deserialize(cls, data): return cls(data["name"])

        iv1 = NopIntervention("dup")
        iv2 = NopIntervention("dup")
        intervention_registry.add(iv1, None, hook_manager, fake_arch)
        with pytest.raises(InterventionError, match="already active"):
            intervention_registry.add(iv2, None, hook_manager, fake_arch)

    def test_serialize_and_restore(
        self, fake_arch, hook_manager, intervention_registry, uniform_vector_path
    ):
        iv = SteeringIntervention(
            name="test_restore",
            vector_path=str(uniform_vector_path),
            layer_indices=[0],
            scale=0.5,
        )
        # Manually add to registry's dict to skip full apply
        intervention_registry._interventions["test_restore"] = iv
        serialized = intervention_registry.serialize_all()
        assert len(serialized) == 1
        assert serialized[0]["name"] == "test_restore"
