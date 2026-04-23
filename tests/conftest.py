"""
Shared pytest fixtures for psplice tests.

Heavyweight model-loading flows are mocked throughout.  These tests focus on
configuration handling, serialization, hook lifecycle, and client/daemon
contract logic — not on actual GPU inference.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Temporary data directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch):
    """Redirect platformdirs user_data_dir to a temp directory."""
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda app_name, *a, **kw: str(tmp_path / app_name),
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Fake model objects
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_arch():
    """A minimal ArchitectureInfo-like object for testing."""
    from psplice.modeling.inspector import ArchitectureInfo

    return ArchitectureInfo(
        family="llama",
        model_type="llama",
        model_class="LlamaForCausalLM",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
        vocab_size=128256,
        has_standard_layers=True,
        supports_head_masking=True,
        supports_steering=True,
        supports_layer_skip=True,
        attn_module_attr="self_attn",
        o_proj_attr="o_proj",
        attention_class="LlamaAttention",
    )


@pytest.fixture
def fake_model(fake_arch):
    """
    A minimal mock model with model.model.layers and self_attn.o_proj structure,
    suitable for hook registration tests.
    """
    import torch.nn as nn

    # Build real nn.Linear modules so hooks can actually be registered
    o_proj = nn.Linear(4096, 4096, bias=False)
    attn = MagicMock()
    attn.o_proj = o_proj

    layer = MagicMock()
    layer.self_attn = attn

    layers = nn.ModuleList([layer for _ in range(32)])

    inner = MagicMock()
    inner.layers = layers

    model = MagicMock()
    model.model = inner
    model.config.model_type = "llama"
    model.config.num_hidden_layers = 32
    model.config.hidden_size = 4096
    model.config.num_attention_heads = 32
    model.config.num_key_value_heads = 8
    model.config.max_position_embeddings = 131072
    model.config.vocab_size = 128256

    return model


@pytest.fixture
def hook_manager():
    from psplice.runtime.hooks import HookManager
    return HookManager()


@pytest.fixture
def intervention_registry():
    from psplice.interventions.registry import InterventionRegistry
    return InterventionRegistry()


# ---------------------------------------------------------------------------
# Steering vector fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_vector_path(tmp_path: Path) -> Path:
    """A 1D tensor of shape [4096] saved as a .pt file."""
    vec = torch.randn(4096)
    p = tmp_path / "vec_uniform.pt"
    torch.save(vec, p)
    return p


@pytest.fixture
def per_layer_vector_path(tmp_path: Path) -> Path:
    """A dict of {layer_idx: tensor[4096]} saved as a .pt file."""
    data = {
        10: torch.randn(4096),
        11: torch.randn(4096),
        12: torch.randn(4096),
    }
    p = tmp_path / "vec_per_layer.pt"
    torch.save(data, p)
    return p
