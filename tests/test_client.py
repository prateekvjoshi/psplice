"""
Tests for the daemon client layer.

All HTTP calls are mocked — these tests verify argument passing,
error handling, and the 'no daemon' flow.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from psplice.client.daemon_client import DaemonClient, DaemonUnavailableError
from psplice.state.session import SessionMetadata


def make_meta():
    return SessionMetadata(
        pid=os.getpid(),
        port=54321,
        model_id="test/model",
        device="cpu",
        dtype="float32",
        eager_attn=False,
    )


def make_client():
    return DaemonClient(make_meta())


class TestGetNone:
    def test_get_returns_none_when_no_session(self, tmp_data_dir):
        client = DaemonClient.get()
        assert client is None


class TestHttpMethods:
    """Verify that client methods call the correct endpoints."""

    def _mock_get(self, return_value):
        return patch(
            "psplice.client.daemon_client.httpx.get",
            return_value=MagicMock(status_code=200, json=lambda: return_value),
        )

    def _mock_post(self, return_value):
        return patch(
            "psplice.client.daemon_client.httpx.post",
            return_value=MagicMock(status_code=200, json=lambda: return_value),
        )

    def _mock_delete(self, return_value):
        return patch(
            "psplice.client.daemon_client.httpx.delete",
            return_value=MagicMock(status_code=200, json=lambda: return_value),
        )

    def test_status_calls_get(self):
        client = make_client()
        with self._mock_get({"model_id": "test"}) as mock_get:
            result = client.status()
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert "/status" in url

    def test_generate_calls_post(self):
        client = make_client()
        with self._mock_post({"text": "hi", "tokens_generated": 5, "time_seconds": 0.1}) as mock_post:
            result = client.generate("hello")
        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert "/generate" in url

    def test_steer_add_calls_post(self):
        client = make_client()
        with self._mock_post({"ok": True, "name": "test"}) as mock_post:
            client.steer_add("test", "/path/vec.pt", [10, 11], scale=0.5)
        url = mock_post.call_args[0][0]
        assert "/steer/add" in url

    def test_steer_remove_calls_delete(self):
        client = make_client()
        with self._mock_delete({"ok": True}) as mock_del:
            client.steer_remove("test")
        url = mock_del.call_args[0][0]
        assert "/steer/test" in url

    def test_heads_mask_calls_post(self):
        client = make_client()
        with self._mock_post({"ok": True}) as mock_post:
            client.heads_mask({3: [0, 1]})
        url = mock_post.call_args[0][0]
        assert "/heads/mask" in url

    def test_lora_load_calls_post(self):
        client = make_client()
        with self._mock_post({"ok": True, "adapter_path": "/p"}) as mock_post:
            client.lora_load("/p")
        url = mock_post.call_args[0][0]
        assert "/lora/load" in url

    def test_lora_unload_calls_delete(self):
        client = make_client()
        with self._mock_delete({"ok": True}) as mock_del:
            client.lora_unload()
        url = mock_del.call_args[0][0]
        assert "/lora" in url

    def test_decode_set_calls_post(self):
        client = make_client()
        with self._mock_post({"ok": True, "settings": {}}) as mock_post:
            client.decode_set(temperature=0.7)
        url = mock_post.call_args[0][0]
        assert "/decode/set" in url

    def test_preset_save_calls_post(self):
        client = make_client()
        with self._mock_post({"ok": True, "name": "my_preset"}) as mock_post:
            client.preset_save("my_preset")
        url = mock_post.call_args[0][0]
        assert "/preset/save" in url


class TestErrorHandling:
    def test_connection_refused_raises_daemon_unavailable(self):
        client = make_client()
        with patch(
            "psplice.client.daemon_client.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(DaemonUnavailableError):
                client.status()

    def test_4xx_response_raises_runtime_error(self):
        client = make_client()
        with patch(
            "psplice.client.daemon_client.httpx.get",
            return_value=MagicMock(
                status_code=404,
                json=lambda: {"detail": "not found"},
                text="not found",
            ),
        ):
            with pytest.raises(RuntimeError, match="404"):
                client.status()

    def test_streaming_yields_tokens(self):
        """Verify streaming generator parses SSE lines correctly."""
        client = make_client()

        # Build a mock streaming response with SSE lines
        lines = [
            'data: {"token": "Hello"}',
            'data: {"token": " world"}',
            "data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines = lambda: iter(lines)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("psplice.client.daemon_client.httpx.stream", return_value=mock_response):
            tokens = list(client.generate_streaming("hello"))

        assert tokens == ["Hello", " world"]
