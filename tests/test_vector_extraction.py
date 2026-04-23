"""
Tests for contrastive vector extraction.

The extraction logic is tested with real nn.Linear-based layers (so hooks
actually fire) but with a tiny hidden_size so everything runs on CPU in
milliseconds.  Model.generate is not called — only model(**inputs) for
the forward pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from psplice.runtime.vector_extraction import (
    TokenAggregation,
    extract_contrastive_vector,
    save_vector,
)

HIDDEN = 16
NUM_LAYERS = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeLayer(nn.Module):
    """A real nn.Module (so hooks attach) that returns a fixed hidden state."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self._h = hidden

    def forward(self, hidden_states, **kwargs):
        # Return a tuple like a real decoder layer
        return (hidden_states,)


@pytest.fixture
def fake_model():
    layers = nn.ModuleList([_FakeLayer(HIDDEN) for _ in range(NUM_LAYERS)])

    model = MagicMock()
    model.model.layers = layers

    # model(**inputs) just calls forward — we mock it to fire layer hooks
    def _forward(**inputs):
        # Simulate running through all layers; hooks fire via register_forward_hook
        x = inputs["input_ids"].float().unsqueeze(-1).expand(-1, -1, HIDDEN)
        for layer in layers:
            x, = layer(x)
        return MagicMock()

    model.side_effect = _forward
    model.__call__ = model.side_effect

    # parameters() → a cpu zero tensor so _resolve_device works
    model.parameters.return_value = iter([torch.zeros(1)])

    return model


@pytest.fixture
def fake_tokenizer():
    tok = MagicMock()
    # Always returns a small 1×5 input_ids tensor
    tok.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
    return tok


# ---------------------------------------------------------------------------
# Core extraction tests
# ---------------------------------------------------------------------------

class TestExtractContrastiveVector:
    def test_returns_dict_per_layer(self, fake_model, fake_tokenizer):
        vectors = extract_contrastive_vector(
            fake_model, fake_tokenizer,
            positive_prompts=["Be brief."],
            negative_prompts=["Elaborate fully."],
            layer_indices=[0, 1, 2],
        )
        assert set(vectors.keys()) == {0, 1, 2}
        for v in vectors.values():
            assert v.shape == (HIDDEN,)
            assert v.dtype == torch.float32

    def test_multiple_prompts_averaged(self, fake_model, fake_tokenizer):
        # With deterministic layers, vectors from different prompts should still
        # produce a valid (non-NaN, non-zero) mean
        vectors = extract_contrastive_vector(
            fake_model, fake_tokenizer,
            positive_prompts=["A.", "B.", "C."],
            negative_prompts=["X.", "Y.", "Z."],
            layer_indices=[0],
        )
        v = vectors[0]
        assert not torch.isnan(v).any()

    def test_mean_and_last_aggregation_both_work(self, fake_model, fake_tokenizer):
        for agg in ("mean", "last"):
            vectors = extract_contrastive_vector(
                fake_model, fake_tokenizer,
                positive_prompts=["positive"],
                negative_prompts=["negative"],
                layer_indices=[0],
                token_aggregation=agg,
            )
            assert 0 in vectors
            assert vectors[0].shape == (HIDDEN,)

    def test_empty_positive_raises(self, fake_model, fake_tokenizer):
        with pytest.raises(ValueError, match="positive"):
            extract_contrastive_vector(
                fake_model, fake_tokenizer,
                positive_prompts=[],
                negative_prompts=["x"],
                layer_indices=[0],
            )

    def test_empty_negative_raises(self, fake_model, fake_tokenizer):
        with pytest.raises(ValueError, match="negative"):
            extract_contrastive_vector(
                fake_model, fake_tokenizer,
                positive_prompts=["x"],
                negative_prompts=[],
                layer_indices=[0],
            )

    def test_hooks_removed_after_extraction(self, fake_model, fake_tokenizer):
        """Hooks must not persist after extraction completes."""
        # Collect hook counts before
        before = sum(
            len(list(layer._forward_hooks.values()))
            for layer in fake_model.model.layers
        )
        extract_contrastive_vector(
            fake_model, fake_tokenizer,
            positive_prompts=["a"], negative_prompts=["b"],
            layer_indices=[0, 1],
        )
        after = sum(
            len(list(layer._forward_hooks.values()))
            for layer in fake_model.model.layers
        )
        assert after == before, "Hooks were not removed after extraction"

    def test_hooks_removed_on_exception(self, fake_model, fake_tokenizer):
        """Even if extraction raises, hooks must be cleaned up."""
        # Make the tokenizer raise after the first call
        call_count = [0]
        original = fake_tokenizer.side_effect

        def _fail_on_second(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("tokenizer exploded")
            return {"input_ids": torch.ones(1, 5, dtype=torch.long)}

        fake_tokenizer.side_effect = _fail_on_second

        with pytest.raises(RuntimeError):
            extract_contrastive_vector(
                fake_model, fake_tokenizer,
                positive_prompts=["a", "b"],
                negative_prompts=["x"],
                layer_indices=[0],
            )

        after = sum(
            len(list(layer._forward_hooks.values()))
            for layer in fake_model.model.layers
        )
        assert after == 0, "Hooks leaked after exception"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveVector:
    def test_single_layer_saves_as_1d_tensor(self, tmp_path):
        vectors = {5: torch.randn(HIDDEN)}
        path = str(tmp_path / "vec.pt")
        save_vector(vectors, path)

        loaded = torch.load(path, weights_only=True)
        assert isinstance(loaded, torch.Tensor)
        assert loaded.shape == (HIDDEN,)

    def test_multi_layer_saves_as_dict(self, tmp_path):
        vectors = {0: torch.randn(HIDDEN), 1: torch.randn(HIDDEN)}
        path = str(tmp_path / "vec.pt")
        save_vector(vectors, path)

        loaded = torch.load(path, weights_only=True)
        assert isinstance(loaded, dict)
        assert set(loaded.keys()) == {0, 1}

    def test_creates_parent_dirs(self, tmp_path):
        vectors = {0: torch.randn(HIDDEN)}
        path = str(tmp_path / "nested" / "dir" / "vec.pt")
        save_vector(vectors, path)
        assert Path(path).exists()

    def test_single_layer_loadable_by_steering(self, tmp_path):
        """A single-layer vector saved by save_vector must pass SteeringIntervention parsing."""
        from psplice.interventions.steering import SteeringIntervention

        vectors = {3: torch.randn(HIDDEN)}
        path = str(tmp_path / "v.pt")
        save_vector(vectors, path)

        iv = SteeringIntervention("test", path, layer_indices=[3])
        raw = torch.load(path, weights_only=True)
        # Single-layer → saved as 1D tensor → should parse as uniform broadcast
        result = iv._parse_vector(raw, HIDDEN)
        assert 3 in result
        assert result[3].shape == (HIDDEN,)

    def test_multi_layer_loadable_by_steering(self, tmp_path):
        """A multi-layer vector saved by save_vector must pass SteeringIntervention parsing."""
        from psplice.interventions.steering import SteeringIntervention

        vectors = {0: torch.randn(HIDDEN), 1: torch.randn(HIDDEN)}
        path = str(tmp_path / "v.pt")
        save_vector(vectors, path)

        iv = SteeringIntervention("test", path, layer_indices=[0, 1])
        raw = torch.load(path, weights_only=True)
        result = iv._parse_vector(raw, HIDDEN)
        assert set(result.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Daemon endpoint integration
# ---------------------------------------------------------------------------

class TestVectorExtractEndpoint:
    """Test the /vectors/extract route with a real TestClient."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        pass

    @pytest.fixture
    def arch(self):
        from psplice.modeling.inspector import ArchitectureInfo
        return ArchitectureInfo(
            family="llama", model_type="llama", model_class="LlamaForCausalLM",
            num_layers=NUM_LAYERS, hidden_size=HIDDEN,
            num_attention_heads=4, num_key_value_heads=4, head_dim=4,
            max_position_embeddings=2048, vocab_size=100,
            has_standard_layers=True, supports_head_masking=True,
            supports_steering=True, supports_layer_skip=True,
            attn_module_attr="self_attn", o_proj_attr="o_proj",
            attention_class="LlamaAttention",
        )

    @pytest.fixture
    def client(self, fake_model, fake_tokenizer, arch):
        import psplice.daemon.server as server_module
        from psplice.daemon.server import app
        from psplice.state.model_session import ModelSession
        from fastapi.testclient import TestClient

        session = ModelSession(
            model=fake_model, tokenizer=fake_tokenizer, arch=arch,
            model_id="test/model", device="cpu", dtype="float32",
            eager_attn=False, param_count=1000,
        )
        server_module._session = session
        yield TestClient(app)
        server_module._session = None

    def test_extract_creates_file(self, client, tmp_path):
        out = str(tmp_path / "v.pt")
        r = client.post("/vectors/extract", json={
            "positive_prompts": ["Be concise."],
            "negative_prompts": ["Elaborate please."],
            "layer_indices": [0, 1],
            "output_path": out,
        })
        assert r.status_code == 200, r.text
        assert Path(out).exists()

        data = r.json()
        assert data["hidden_size"] == HIDDEN
        assert set(data["layer_indices"]) == {0, 1}
        assert data["format"] == "per_layer"

    def test_extract_bad_layer_returns_400(self, client, tmp_path):
        out = str(tmp_path / "v.pt")
        r = client.post("/vectors/extract", json={
            "positive_prompts": ["a"],
            "negative_prompts": ["b"],
            "layer_indices": [999],
            "output_path": out,
        })
        assert r.status_code == 400
        assert "out of range" in r.json()["detail"]

    def test_extract_bad_aggregation_returns_400(self, client, tmp_path):
        out = str(tmp_path / "v.pt")
        r = client.post("/vectors/extract", json={
            "positive_prompts": ["a"],
            "negative_prompts": ["b"],
            "layer_indices": [0],
            "output_path": out,
            "token_aggregation": "bogus",
        })
        assert r.status_code == 400
