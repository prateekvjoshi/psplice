"""
Integration tests for daemon state transitions.

These tests exercise the FastAPI route handlers against a real ModelSession
backed by a mock model.  No actual GPU operations happen — model.generate
and the tokenizer are stubbed — but the full intervention/hook/registry
lifecycle runs for real, covering the paths most likely to have bugs:

  - Intervention add / remove via HTTP
  - compare() hook suspend-and-restore sequence (registry sync invariant)
  - LoRA load / unload lifecycle (model reference tracking)
  - Preset save → clear → restore round-trip
  - Error responses for bad inputs and missing state
  - Preset clear deactivates interventions without deleting disk presets
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

import psplice.daemon.server as server_module
from psplice.daemon.server import app
from psplice.modeling.inspector import ArchitectureInfo
from psplice.runtime.generation import DecodeSettings, GenerationResult
from psplice.state.model_session import ModelSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_data(tmp_data_dir):
    """Run with an isolated preset/session directory."""
    pass


@pytest.fixture
def arch() -> ArchitectureInfo:
    return ArchitectureInfo(
        family="llama",
        model_type="llama",
        model_class="LlamaForCausalLM",
        num_layers=4,           # small for test speed
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=2048,
        vocab_size=1000,
        has_standard_layers=True,
        supports_head_masking=True,
        supports_steering=True,
        supports_layer_skip=True,
        attn_module_attr="self_attn",
        o_proj_attr="o_proj",
        attention_class="LlamaAttention",   # "eager" implied
    )


@pytest.fixture
def real_layers(arch):
    """
    Build real nn.Linear modules arranged like decoder layers so that
    hook registration actually works (MagicMock doesn't support hooks).
    """
    layers = []
    for _ in range(arch.num_layers):
        o_proj = nn.Linear(arch.hidden_size, arch.hidden_size, bias=False)
        attn = MagicMock()
        attn.o_proj = o_proj

        layer = nn.Module()
        layer.self_attn = attn
        layers.append(layer)
    return layers


@pytest.fixture
def fake_model(arch, real_layers):
    model = MagicMock()
    model.model.layers = real_layers
    model.config.model_type = arch.model_type
    model.config.num_hidden_layers = arch.num_layers
    model.config.hidden_size = arch.hidden_size
    model.config.num_attention_heads = arch.num_attention_heads
    model.config.num_key_value_heads = arch.num_key_value_heads
    model.config.max_position_embeddings = arch.max_position_embeddings
    model.config.vocab_size = arch.vocab_size
    # Satisfy _model_device lookup
    mock_param = torch.zeros(1)
    model.parameters.return_value = iter([mock_param])
    return model


@pytest.fixture
def fake_tokenizer():
    tok = MagicMock()
    tok.chat_template = None
    tok.eos_token_id = 2
    # tokenizer(text, return_tensors="pt") → {"input_ids": tensor}
    tok.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    tok.decode.return_value = "mock response"
    return tok


@pytest.fixture
def session(fake_model, fake_tokenizer, arch) -> ModelSession:
    return ModelSession(
        model=fake_model,
        tokenizer=fake_tokenizer,
        arch=arch,
        model_id="test/model",
        device="cpu",
        dtype="float32",
        eager_attn=True,
        param_count=1_000_000,
    )


@pytest.fixture
def client(session) -> TestClient:
    """TestClient with a pre-injected ModelSession."""
    server_module._session = session
    yield TestClient(app)
    server_module._session = None  # reset after test


# ---------------------------------------------------------------------------
# Helper: stub generate() so tests don't run real inference
# ---------------------------------------------------------------------------

def _stub_generate(label: str = "output") -> GenerationResult:
    return GenerationResult(text=f"[{label}]", tokens_generated=3, time_seconds=0.01, label=label)


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

class TestHealthStatus:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_id"] == "test/model"

    def test_health_no_session(self):
        server_module._session = None
        c = TestClient(app)
        r = c.get("/health")
        assert r.status_code == 503

    def test_status_contains_expected_keys(self, client):
        with patch(
            "psplice.modeling.inspector.detect_attention_implementation",
            return_value="eager",
        ):
            r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["model_id"] == "test/model"
        assert data["arch_family"] == "llama"
        assert "interventions" in data
        assert "decode_settings" in data


# ---------------------------------------------------------------------------
# Steering intervention lifecycle
# ---------------------------------------------------------------------------

class TestSteeringLifecycle:
    def test_add_and_list(self, client, session, tmp_path):
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        r = client.post("/steer/add", json={
            "name": "test_steer",
            "vector_path": str(p),
            "layer_indices": [0, 1],
            "scale": 0.5,
        })
        assert r.status_code == 200, r.text

        # Registry should have it
        assert "test_steer" in session.intervention_registry.names()
        # Hook manager should have hooks
        assert session.hook_manager.has_key("test_steer")

        r = client.get("/steer")
        assert r.status_code == 200
        names = [iv["name"] for iv in r.json()]
        assert "test_steer" in names

    def test_remove(self, client, session, tmp_path):
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        client.post("/steer/add", json={
            "name": "to_remove",
            "vector_path": str(p),
            "layer_indices": [0],
            "scale": 1.0,
        })
        assert "to_remove" in session.intervention_registry.names()

        r = client.delete("/steer/to_remove")
        assert r.status_code == 200
        assert "to_remove" not in session.intervention_registry.names()
        assert not session.hook_manager.has_key("to_remove")

    def test_remove_missing_returns_404(self, client):
        r = client.delete("/steer/does_not_exist")
        assert r.status_code == 404

    def test_duplicate_name_returns_400(self, client, tmp_path):
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        payload = {"name": "dup", "vector_path": str(p), "layer_indices": [0], "scale": 1.0}
        client.post("/steer/add", json=payload)
        r = client.post("/steer/add", json=payload)
        assert r.status_code == 400
        assert "already active" in r.json()["detail"]

    def test_bad_vector_path_returns_400(self, client):
        r = client.post("/steer/add", json={
            "name": "bad",
            "vector_path": "/nonexistent/path.pt",
            "layer_indices": [0],
            "scale": 1.0,
        })
        assert r.status_code == 400

    def test_out_of_range_layer_returns_400(self, client, tmp_path):
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        r = client.post("/steer/add", json={
            "name": "bad_layer",
            "vector_path": str(p),
            "layer_indices": [999],
            "scale": 1.0,
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Head masking (requires eager attention)
# ---------------------------------------------------------------------------

class TestHeadMaskLifecycle:
    def test_mask_and_clear(self, client, session):
        r = client.post("/heads/mask", json={"layer_heads": {"0": [0, 1], "1": [2]}})
        assert r.status_code == 200, r.text
        assert "_head_mask" in session.intervention_registry.names()
        assert session.hook_manager.has_key("_head_mask")

        r = client.delete("/heads")
        assert r.status_code == 200
        assert "_head_mask" not in session.intervention_registry.names()
        assert not session.hook_manager.has_key("_head_mask")

    def test_replace_semantics(self, client, session):
        """Second heads/mask replaces the first without duplication."""
        client.post("/heads/mask", json={"layer_heads": {"0": [0]}})
        r = client.post("/heads/mask", json={"layer_heads": {"1": [1]}})
        assert r.status_code == 200
        masks = session.intervention_registry.by_type("head_mask")
        assert len(masks) == 1

    def test_sdpa_model_returns_400(self, client, session):
        """Attempt to mask heads on a non-eager model should fail."""
        import dataclasses
        sdpa_arch = dataclasses.replace(session.arch, attention_class="LlamaSdpaAttention")
        session.arch = sdpa_arch

        r = client.post("/heads/mask", json={"layer_heads": {"0": [0]}})
        assert r.status_code == 400
        assert "eager" in r.json()["detail"].lower()

        # Restore
        session.arch = session.arch


# ---------------------------------------------------------------------------
# Layer skip lifecycle
# ---------------------------------------------------------------------------

class TestLayerSkipLifecycle:
    def test_skip_and_clear(self, client, session):
        r = client.post("/layers/skip", json={"skip_from": 2})
        assert r.status_code == 200, r.text
        assert "_layer_skip" in session.intervention_registry.names()

        r = client.delete("/layers")
        assert r.status_code == 200
        assert "_layer_skip" not in session.intervention_registry.names()

    def test_replace_semantics(self, client, session):
        """Second layers/skip replaces the first."""
        client.post("/layers/skip", json={"skip_from": 2})
        r = client.post("/layers/skip", json={"skip_from": 3})
        assert r.status_code == 200
        skips = session.intervention_registry.by_type("layer_skip")
        assert len(skips) == 1
        assert skips[0].skip_from == 3

    def test_skip_from_zero_rejected(self, client):
        r = client.post("/layers/skip", json={"skip_from": 0})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Compare — registry/hook sync invariant
# ---------------------------------------------------------------------------

class TestCompareSync:
    """
    The most critical invariant: after compare(), the registry and hook_manager
    must be in sync with the state they were in before compare() was called.
    """

    def test_compare_registry_unchanged_after(self, client, session, tmp_path):
        # Add a steering intervention
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)
        client.post("/steer/add", json={
            "name": "steer1",
            "vector_path": str(p),
            "layer_indices": [0],
            "scale": 1.0,
        })

        initial_names = set(session.intervention_registry.names())
        initial_hooks = set(session.hook_manager.active_keys())

        with patch("psplice.runtime.generation.generate", side_effect=[
            _stub_generate("base"),
            _stub_generate("modified"),
        ]):
            r = client.post("/compare", json={"prompt": "test"})

        assert r.status_code == 200, r.text
        # Registry and hooks must be identical after compare
        assert set(session.intervention_registry.names()) == initial_names
        assert set(session.hook_manager.active_keys()) == initial_hooks

    def test_compare_base_has_no_hooks_modified_has_hooks(self, session, tmp_path):
        """
        Verify the actual generate order: base runs with hooks suspended,
        modified runs with hooks active.
        """
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        from psplice.interventions.steering import SteeringIntervention
        iv = SteeringIntervention("s", str(p), [0], scale=1.0)
        session.apply_intervention(iv)

        hook_counts_during_generate: list[int] = []

        def capture_generate(model, tokenizer, prompt, settings, device=None):
            hook_counts_during_generate.append(session.hook_manager.hook_count())
            return _stub_generate("run")

        with patch("psplice.runtime.generation.generate", side_effect=capture_generate):
            base, modified = session.compare("hello", session.decode_settings)

        # First call = base (hooks suspended → 0 hooks)
        assert hook_counts_during_generate[0] == 0, "Base should run with hooks suspended"
        # Second call = modified (hooks restored)
        assert hook_counts_during_generate[1] > 0, "Modified should run with hooks active"

    def test_compare_empty_interventions_works(self, client):
        """compare() with no interventions should not crash."""
        with patch("psplice.runtime.generation.generate", return_value=_stub_generate()):
            r = client.post("/compare", json={"prompt": "hello"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# LoRA lifecycle
# ---------------------------------------------------------------------------

class TestLoraLifecycle:
    def test_load_and_unload(self, client, session, tmp_path):
        # Create a fake adapter directory
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        fake_peft = MagicMock()
        fake_peft.eval.return_value = fake_peft

        with patch("peft.PeftModel") as MockPeft:
            MockPeft.from_pretrained.return_value = fake_peft
            r = client.post("/lora/load", json={"adapter_path": str(adapter_dir)})

        assert r.status_code == 200, r.text
        assert session.active_lora_path is not None
        assert session.model is fake_peft

        r = client.delete("/lora")
        assert r.status_code == 200
        assert session.active_lora_path is None
        # Model should be restored to base
        assert session.model is session._base_model

    def test_double_load_returns_conflict(self, client, session, tmp_path):
        """Loading a second adapter without unloading the first must fail."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        # Simulate first adapter already loaded
        session.active_lora_path = "/fake/existing"

        r = client.post("/lora/load", json={"adapter_path": str(adapter_dir)})
        assert r.status_code == 400
        assert "already loaded" in r.json()["detail"]

    def test_unload_when_none_returns_404(self, client):
        r = client.delete("/lora")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Preset round-trip
# ---------------------------------------------------------------------------

class TestPresetRoundTrip:
    def test_save_clear_load(self, client, session, tmp_path):
        # Add an intervention
        vec = torch.randn(64)
        p = tmp_path / "v.pt"
        torch.save(vec, p)

        client.post("/steer/add", json={
            "name": "steer1",
            "vector_path": str(p),
            "layer_indices": [0],
            "scale": 0.5,
        })
        assert "steer1" in session.intervention_registry.names()

        # Save preset
        r = client.post("/preset/save", json={"name": "my_preset"})
        assert r.status_code == 200
        assert session.active_preset == "my_preset"

        # Clear interventions
        r = client.post("/preset/clear", json={})
        assert r.status_code == 200
        assert session.intervention_registry.is_empty()
        assert session.hook_manager.hook_count() == 0
        assert session.active_preset is None

        # Load preset back
        r = client.post("/preset/load", json={"name": "my_preset"})
        assert r.status_code == 200, r.json()
        assert r.json()["errors"] == []
        assert "steer1" in session.intervention_registry.names()
        assert session.hook_manager.has_key("steer1")
        assert session.active_preset == "my_preset"

    def test_load_missing_preset_returns_404(self, client):
        r = client.post("/preset/load", json={"name": "nonexistent"})
        assert r.status_code == 404

    def test_preset_clear_does_not_delete_disk_preset(self, client, session, tmp_path):
        """
        Preset clear only deactivates interventions.  It must NOT delete the
        preset file from disk (that would be a destructive mis-feature).
        """
        from psplice.state.presets import save_preset, load_preset

        save_preset("on_disk", {"interventions": [], "decode_settings": {}, "active_lora": None})

        client.post("/preset/clear", json={})

        assert load_preset("on_disk") is not None, "Preset file must still exist after clear"

    def test_preset_list(self, client):
        from psplice.state.presets import save_preset
        save_preset("alpha", {})
        save_preset("beta", {})

        r = client.get("/preset/list")
        assert r.status_code == 200
        names = r.json()
        assert "alpha" in names
        assert "beta" in names


# ---------------------------------------------------------------------------
# Decode settings
# ---------------------------------------------------------------------------

class TestDecodeSettings:
    def test_set_and_show(self, client, session):
        r = client.post("/decode/set", json={"temperature": 0.7, "top_p": 0.9})
        assert r.status_code == 200
        assert session.decode_settings.temperature == 0.7
        assert session.decode_settings.top_p == 0.9

        r = client.get("/decode")
        assert r.json()["temperature"] == 0.7

    def test_reset(self, client, session):
        client.post("/decode/set", json={"temperature": 0.5})
        client.delete("/decode")
        assert session.decode_settings.temperature is None

    def test_partial_update_preserves_others(self, client, session):
        client.post("/decode/set", json={"temperature": 0.8, "top_k": 40})
        client.post("/decode/set", json={"temperature": 0.3})
        assert session.decode_settings.top_k == 40
        assert session.decode_settings.temperature == 0.3
