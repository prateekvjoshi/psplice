"""
Session state management.

The session file is a JSON document stored in the user's data directory that
lets CLI commands discover the currently running daemon. It stores the daemon's
PID, the port it is listening on, and metadata about the loaded model.

Only one daemon can be active at a time. psplice load refuses to start a second
daemon if the session file already exists and the daemon is still running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import platformdirs
from pydantic import BaseModel, Field


def get_data_dir() -> Path:
    """Return the psplice user data directory, creating it if necessary."""
    d = Path(platformdirs.user_data_dir("psplice"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_session_path() -> Path:
    return get_data_dir() / "session.json"


class SessionMetadata(BaseModel):
    """Snapshot of daemon identity and loaded model metadata."""

    pid: int
    port: int
    model_id: str
    device: str
    dtype: str
    eager_attn: bool = False
    max_new_tokens_default: int = 512
    active_preset: Optional[str] = None
    started_at: str = Field(default_factory=lambda: _now_iso())


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_session(meta: SessionMetadata) -> None:
    """Persist session metadata to disk."""
    get_session_path().write_text(meta.model_dump_json(indent=2))


def read_session() -> Optional[SessionMetadata]:
    """
    Read the current session from disk.

    Returns None if no session file exists or if the file is malformed.
    Does NOT check whether the daemon process is still alive — call
    is_daemon_alive() for that.
    """
    path = get_session_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return SessionMetadata(**data)
    except Exception:
        return None


def remove_session() -> None:
    """Delete the session file if it exists."""
    path = get_session_path()
    if path.exists():
        path.unlink()


def is_daemon_alive(meta: SessionMetadata) -> bool:
    """Return True if the daemon process identified by meta.pid is still running."""
    try:
        import psutil
        return psutil.pid_exists(meta.pid)
    except Exception:
        # Fallback: try sending signal 0
        try:
            os.kill(meta.pid, 0)
            return True
        except OSError:
            return False


def get_active_session() -> Optional[SessionMetadata]:
    """
    Return the current active session, or None.

    Cleans up a stale session file if the process is gone.
    """
    meta = read_session()
    if meta is None:
        return None
    if not is_daemon_alive(meta):
        # Stale — process died without cleanup
        remove_session()
        return None
    return meta


def daemon_url(meta: SessionMetadata) -> str:
    """Build the base URL for the daemon."""
    return f"http://127.0.0.1:{meta.port}"
