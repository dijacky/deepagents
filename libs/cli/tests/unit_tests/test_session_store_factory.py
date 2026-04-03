"""Tests for session store construction."""

from __future__ import annotations

from deepagents_cli._remote_session_store import RemoteSessionStore
from deepagents_cli._session_store import LocalSessionStore, build_session_store
from deepagents_cli.remote_server_settings import RemoteServerSettings


def test_build_session_store_local_without_remote() -> None:
    """No remote URL -> SQLite-backed local store."""
    store = build_session_store(None)
    assert isinstance(store, LocalSessionStore)


def test_build_session_store_remote_when_url_configured() -> None:
    """Remote settings -> HTTP-backed store."""
    settings = RemoteServerSettings(url="http://localhost:2026", graph_name="agent")
    store = build_session_store(settings)
    assert isinstance(store, RemoteSessionStore)
    assert store.checkpoint_backend == "remote"
