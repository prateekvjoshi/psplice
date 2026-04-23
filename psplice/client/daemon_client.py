"""
Daemon HTTP client.

Provides a clean Python API over the daemon's REST endpoints.  All CLI
commands import DaemonClient rather than calling httpx directly.

Usage::

    client = DaemonClient.require()  # exits with friendly error if no daemon
    result = client.generate("hello world")
"""

from __future__ import annotations

import json
import sys
from typing import Generator, Iterator, Optional

import httpx

from psplice.state.session import SessionMetadata, daemon_url, get_active_session

# Default per-request timeout (seconds).  Generation can be slow on large models.
DEFAULT_TIMEOUT = 300.0


class DaemonUnavailableError(Exception):
    """Raised when the daemon is not running or not reachable."""


class DaemonClient:
    """Thin client for the psplice daemon REST API."""

    def __init__(self, meta: SessionMetadata) -> None:
        self._meta = meta
        self._base = daemon_url(meta)

    @classmethod
    def get(cls) -> Optional["DaemonClient"]:
        """Return a client for the active session, or None if none is running."""
        meta = get_active_session()
        if meta is None:
            return None
        return cls(meta)

    @classmethod
    def require(cls) -> "DaemonClient":
        """
        Return a client or exit with a user-friendly error.

        Use this in all CLI commands that require a running daemon.
        """
        client = cls.get()
        if client is None:
            _die_no_daemon()
        assert client is not None
        # Quick health check
        try:
            client._get("/health")
        except DaemonUnavailableError:
            _die_no_daemon()
        return client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        return self._post(
            "/generate",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "conversation_history": conversation_history,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "streaming": False,
            },
        )

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Yield text chunks from the daemon's streaming endpoint.
        Uses SSE (Server-Sent Events) format: data: {"token": "..."}\n\n
        """
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "conversation_history": conversation_history,
            "max_new_tokens": max_new_tokens,
            "streaming": True,
        }
        url = self._base + "/generate"
        try:
            with httpx.stream(
                "POST",
                url,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            ) as response:
                _check_response(response)
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(raw)
                        yield obj.get("token", "")
                    except json.JSONDecodeError:
                        pass
        except httpx.ConnectError:
            raise DaemonUnavailableError(_conn_error_msg(self._meta))

    def extract_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer_indices: list[int],
        output_path: str,
        token_aggregation: str = "mean",
    ) -> dict:
        return self._post(
            "/vectors/extract",
            {
                "positive_prompts": positive_prompts,
                "negative_prompts": negative_prompts,
                "layer_indices": layer_indices,
                "output_path": output_path,
                "token_aggregation": token_aggregation,
            },
        )

    def compare(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        return self._post(
            "/compare",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return self._get("/status")

    def health(self) -> dict:
        return self._get("/health")

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    def steer_add(
        self,
        name: str,
        vector_path: str,
        layer_indices: list[int],
        scale: float = 1.0,
    ) -> dict:
        return self._post(
            "/steer/add",
            {
                "name": name,
                "vector_path": vector_path,
                "layer_indices": layer_indices,
                "scale": scale,
            },
        )

    def steer_remove(self, name: str) -> dict:
        return self._delete(f"/steer/{name}")

    def steer_list(self) -> list:
        return self._get("/steer")

    # ------------------------------------------------------------------
    # Head masking
    # ------------------------------------------------------------------

    def heads_mask(self, layer_heads: dict[int, list[int]]) -> dict:
        return self._post(
            "/heads/mask",
            {"layer_heads": {str(k): v for k, v in layer_heads.items()}},
        )

    def heads_clear(self) -> dict:
        return self._delete("/heads")

    def heads_list(self) -> list:
        return self._get("/heads")

    # ------------------------------------------------------------------
    # Layer skip
    # ------------------------------------------------------------------

    def layers_skip(self, skip_from: int) -> dict:
        return self._post("/layers/skip", {"skip_from": skip_from})

    def layers_clear(self) -> dict:
        return self._delete("/layers")

    def layers_info(self) -> list:
        return self._get("/layers")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def lora_load(self, adapter_path: str) -> dict:
        return self._post("/lora/load", {"adapter_path": adapter_path})

    def lora_unload(self) -> dict:
        return self._delete("/lora")

    def lora_info(self) -> dict:
        return self._get("/lora")

    # ------------------------------------------------------------------
    # Decode settings
    # ------------------------------------------------------------------

    def decode_set(self, **kwargs) -> dict:
        return self._post("/decode/set", kwargs)

    def decode_show(self) -> dict:
        return self._get("/decode")

    def decode_reset(self) -> dict:
        return self._delete("/decode")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def preset_save(self, name: str) -> dict:
        return self._post("/preset/save", {"name": name})

    def preset_load(self, name: str) -> dict:
        return self._post("/preset/load", {"name": name})

    def preset_list(self) -> list[str]:
        return self._get("/preset/list")

    def preset_clear(self) -> dict:
        return self._post("/preset/clear", {})

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> dict:
        try:
            return self._post("/stop", {})
        except Exception:
            return {"ok": True}

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = self._base + path
        try:
            resp = httpx.get(url, timeout=DEFAULT_TIMEOUT)
            _check_response(resp)
            return resp.json()
        except httpx.ConnectError:
            raise DaemonUnavailableError(_conn_error_msg(self._meta))

    def _post(self, path: str, payload: dict) -> dict:
        url = self._base + path
        try:
            resp = httpx.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            _check_response(resp)
            return resp.json()
        except httpx.ConnectError:
            raise DaemonUnavailableError(_conn_error_msg(self._meta))

    def _delete(self, path: str) -> dict:
        url = self._base + path
        try:
            resp = httpx.delete(url, timeout=DEFAULT_TIMEOUT)
            _check_response(resp)
            return resp.json()
        except httpx.ConnectError:
            raise DaemonUnavailableError(_conn_error_msg(self._meta))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_response(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Daemon error {resp.status_code}: {detail}")


def _conn_error_msg(meta: SessionMetadata) -> str:
    return (
        f"Cannot reach daemon at {daemon_url(meta)}. "
        f"The process (PID {meta.pid}) may have crashed. "
        f"Run `psplice stop` to clean up, then `psplice load <model>` to restart."
    )


def _die_no_daemon() -> None:
    from rich.console import Console
    from rich.panel import Panel

    Console().print(
        Panel(
            "[bold red]No psplice daemon is running.[/bold red]\n\n"
            "Start one with:\n"
            "  [cyan]psplice load <model-id>[/cyan]\n\n"
            "Example:\n"
            "  [cyan]psplice load Qwen/Qwen2.5-7B-Instruct[/cyan]",
            title="[red]Daemon not found[/red]",
            border_style="red",
        )
    )
    sys.exit(1)
