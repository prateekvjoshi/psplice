"""Tests for session state management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from psplice.state.session import (
    SessionMetadata,
    daemon_url,
    get_active_session,
    get_session_path,
    is_daemon_alive,
    read_session,
    remove_session,
    write_session,
)


@pytest.fixture(autouse=True)
def _isolated_session(tmp_data_dir):
    """All session tests run with an isolated data directory."""
    pass


def make_meta(pid: int = 99999, port: int = 54321) -> SessionMetadata:
    return SessionMetadata(
        pid=pid,
        port=port,
        model_id="test/model",
        device="cpu",
        dtype="float32",
        eager_attn=False,
    )


class TestWriteReadRemove:
    def test_round_trip(self):
        meta = make_meta()
        write_session(meta)
        loaded = read_session()
        assert loaded is not None
        assert loaded.pid == meta.pid
        assert loaded.port == meta.port
        assert loaded.model_id == meta.model_id

    def test_remove_clears_file(self):
        write_session(make_meta())
        assert read_session() is not None
        remove_session()
        assert read_session() is None

    def test_read_returns_none_when_missing(self):
        assert read_session() is None

    def test_read_returns_none_on_corrupt_file(self):
        get_session_path().write_text("not valid json{{{")
        assert read_session() is None


class TestIsAlive:
    def test_current_process_is_alive(self):
        meta = make_meta(pid=os.getpid())
        assert is_daemon_alive(meta) is True

    def test_nonexistent_pid_is_dead(self):
        # PID 1 is always init; we pick an unlikely dead PID
        # Use a definitely-dead PID by using a value that wraps around
        meta = make_meta(pid=999_999_999)
        assert is_daemon_alive(meta) is False


class TestGetActiveSession:
    def test_returns_none_when_no_file(self):
        assert get_active_session() is None

    def test_cleans_up_stale_session(self):
        # Write a session with a dead PID
        meta = make_meta(pid=999_999_999)
        write_session(meta)
        result = get_active_session()
        assert result is None
        # File should have been removed
        assert read_session() is None

    def test_returns_session_for_live_pid(self):
        meta = make_meta(pid=os.getpid())
        write_session(meta)
        result = get_active_session()
        assert result is not None
        assert result.pid == os.getpid()


class TestDaemonUrl:
    def test_format(self):
        meta = make_meta(port=7341)
        assert daemon_url(meta) == "http://127.0.0.1:7341"
