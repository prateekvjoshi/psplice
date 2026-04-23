"""Tests for preset save/load/list/delete."""

from __future__ import annotations

import pytest

from psplice.state.presets import (
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_data_dir):
    pass


SAMPLE_DATA = {
    "interventions": [
        {
            "intervention_type": "steering",
            "name": "honesty",
            "vector_path": "/tmp/honesty.pt",
            "layer_indices": [10, 11, 12],
            "scale": 0.6,
        }
    ],
    "decode_settings": {
        "temperature": 0.7,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "max_new_tokens": 256,
    },
    "active_lora": None,
}


class TestSaveLoad:
    def test_round_trip(self):
        save_preset("my_preset", SAMPLE_DATA)
        loaded = load_preset("my_preset")
        assert loaded is not None
        assert loaded["interventions"][0]["name"] == "honesty"
        assert loaded["decode_settings"]["temperature"] == 0.7

    def test_load_missing_returns_none(self):
        assert load_preset("does_not_exist") is None

    def test_overwrite(self):
        save_preset("p", {"v": 1})
        save_preset("p", {"v": 2})
        assert load_preset("p")["v"] == 2


class TestList:
    def test_empty(self):
        assert list_presets() == []

    def test_multiple(self):
        save_preset("alpha", {})
        save_preset("beta", {})
        save_preset("gamma", {})
        assert list_presets() == ["alpha", "beta", "gamma"]


class TestDelete:
    def test_delete_existing(self):
        save_preset("to_delete", {})
        assert delete_preset("to_delete") is True
        assert load_preset("to_delete") is None

    def test_delete_missing(self):
        assert delete_preset("nope") is False
