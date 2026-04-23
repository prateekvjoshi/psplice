"""
Daemon lifecycle management.

Responsible for:
  - Spawning the daemon subprocess (psplice load)
  - Polling until the daemon is healthy
  - Sending a stop signal (psplice stop)
  - Detecting stale or duplicate daemons

The daemon is launched as a detached subprocess running
`python -m psplice.daemon.server --model-id ...`.  The CLI process exits
immediately after confirming the daemon is healthy; the daemon lives on.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from psplice.state.session import (
    SessionMetadata,
    daemon_url,
    get_active_session,
    read_session,
    remove_session,
    write_session,
)

# How long to wait for the daemon to start (seconds)
STARTUP_TIMEOUT = 600
# Polling interval during startup
POLL_INTERVAL = 2.0


class DaemonAlreadyRunningError(Exception):
    """Raised when a daemon is already active and --force was not set."""


class DaemonStartupError(Exception):
    """Raised when the daemon fails to start within STARTUP_TIMEOUT."""


def start(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    eager_attn: bool = False,
    trust_remote_code: bool = False,
    force: bool = False,
) -> SessionMetadata:
    """
    Launch the daemon subprocess and wait for it to become healthy.

    If a daemon is already running and *force* is False, raises
    DaemonAlreadyRunningError.  If *force* is True, stops the old daemon
    first.

    Returns the SessionMetadata once the daemon is healthy.
    """
    existing = get_active_session()
    if existing is not None:
        if not force:
            raise DaemonAlreadyRunningError(
                f"A daemon is already running (PID {existing.pid}, model: {existing.model_id}).\n"
                f"Run `psplice stop` first, or use `psplice load --force` to replace it."
            )
        stop()

    cmd = [
        sys.executable,
        "-m",
        "psplice.daemon.server",
        "--model-id",
        model_id,
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if eager_attn:
        cmd.append("--eager-attn")
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    # Detach from parent process group so daemon survives CLI exit
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Poll for the session file to appear and the health endpoint to respond
    deadline = time.time() + STARTUP_TIMEOUT
    meta: Optional[SessionMetadata] = None

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)

        # The daemon writes the session file as soon as it knows its port
        if meta is None:
            meta = read_session()

        if meta is None:
            # Session file not yet written
            if proc.poll() is not None:
                # Daemon exited early
                raise DaemonStartupError(
                    f"Daemon process exited early (code {proc.returncode}). "
                    f"Check that the model path is valid and dependencies are installed."
                )
            continue

        # Session file exists; check health
        url = daemon_url(meta) + "/health"
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return meta
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        if proc.poll() is not None:
            remove_session()
            raise DaemonStartupError(
                f"Daemon process exited early (code {proc.returncode})."
            )

    remove_session()
    raise DaemonStartupError(
        f"Daemon did not become healthy within {STARTUP_TIMEOUT}s. "
        f"This may indicate an OOM condition or a model loading error. "
        f"Try running the daemon manually to see detailed output:\n\n"
        f"  python -m psplice.daemon.server --model-id {model_id}"
    )


def stop() -> bool:
    """
    Stop the running daemon.

    Sends POST /stop to the daemon and removes the session file.
    Returns True if a daemon was stopped, False if none was running.
    """
    meta = get_active_session()
    if meta is None:
        return False

    url = daemon_url(meta) + "/stop"
    try:
        httpx.post(url, timeout=10.0)
    except Exception:
        # Daemon may already be dead; clean up session file regardless
        pass

    # Give the process a moment to exit, then check
    time.sleep(1.0)
    remove_session()
    return True


def fetch_status() -> Optional[dict]:
    """Return the daemon's /status JSON, or None if daemon is not reachable."""
    meta = get_active_session()
    if meta is None:
        return None
    url = daemon_url(meta) + "/status"
    try:
        resp = httpx.get(url, timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None
