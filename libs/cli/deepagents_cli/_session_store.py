"""Thread listing and checkpoint access — local SQLite or remote HTTP API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from deepagents_cli.remote_server_settings import RemoteServerSettings
    from deepagents_cli.sessions import ThreadInfo

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL: LocalSessionStore | None = None


@runtime_checkable
class SessionStore(Protocol):
    """Abstract thread persistence for the TUI and ``threads`` subcommands."""

    @property
    def checkpoint_backend(self) -> str:
        """``\"sqlite\"`` for local checkpointer files, ``\"remote\"`` for HTTP state."""

    async def list_threads(
        self,
        agent_name: str | None = None,
        limit: int = 20,
        include_message_count: bool = False,
        sort_by: str = "updated",
        branch: str | None = None,
    ) -> list[ThreadInfo]:
        """List recent threads (same contract as ``sessions.list_threads``)."""

    async def populate_thread_message_counts(
        self,
        threads: list[ThreadInfo],
    ) -> list[ThreadInfo]:
        """Populate message counts for rows."""

    async def populate_thread_checkpoint_details(
        self,
        threads: list[ThreadInfo],
        *,
        include_message_count: bool = True,
        include_initial_prompt: bool = True,
    ) -> list[ThreadInfo]:
        """Populate checkpoint-derived fields."""

    async def prewarm_thread_message_counts(self, limit: int | None = None) -> None:
        """Warm listing cache for faster ``/threads``."""

    def get_cached_threads(
        self,
        agent_name: str | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo] | None:
        """Return cached rows when available."""

    async def get_most_recent(self, agent_name: str | None) -> str | None:
        """Most recently updated thread id, optionally filtered by agent."""

    async def get_thread_agent(self, thread_id: str) -> str | None:
        """Agent name stored for the thread, if any."""

    async def thread_exists(self, thread_id: str) -> bool:
        """Whether the thread has persistent state."""

    async def find_similar_threads(self, thread_id: str) -> list[str]:
        """Return thread ids with the same prefix (fuzzy match hint)."""

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete thread state; return ``True`` if something was removed."""

    async def read_checkpoint_channel_values(self, thread_id: str) -> dict[str, Any]:
        """Latest checkpoint channel values (or state ``values`` for remote)."""


class LocalSessionStore:
    """Delegates to ``deepagents_cli.sessions`` (SQLite + local checkpointer)."""

    @property
    def checkpoint_backend(self) -> str:
        return "sqlite"

    async def list_threads(
        self,
        agent_name: str | None = None,
        limit: int = 20,
        include_message_count: bool = False,
        sort_by: str = "updated",
        branch: str | None = None,
    ) -> list[ThreadInfo]:
        from deepagents_cli import sessions

        # Omit unset optional filters so call sites (and tests that mock
        # ``sessions.list_threads``) match the historical keyword-only pattern.
        kw: dict[str, Any] = {
            "limit": limit,
            "include_message_count": include_message_count,
            "sort_by": sort_by,
        }
        if agent_name is not None:
            kw["agent_name"] = agent_name
        if branch is not None:
            kw["branch"] = branch
        return await sessions.list_threads(**kw)

    async def populate_thread_message_counts(
        self,
        threads: list[ThreadInfo],
    ) -> list[ThreadInfo]:
        from deepagents_cli import sessions

        return await sessions.populate_thread_message_counts(threads)

    async def populate_thread_checkpoint_details(
        self,
        threads: list[ThreadInfo],
        *,
        include_message_count: bool = True,
        include_initial_prompt: bool = True,
    ) -> list[ThreadInfo]:
        from deepagents_cli import sessions

        return await sessions.populate_thread_checkpoint_details(
            threads,
            include_message_count=include_message_count,
            include_initial_prompt=include_initial_prompt,
        )

    async def prewarm_thread_message_counts(self, limit: int | None = None) -> None:
        from deepagents_cli import sessions

        await sessions.prewarm_thread_message_counts(limit=limit)

    def get_cached_threads(
        self,
        agent_name: str | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo] | None:
        from deepagents_cli import sessions

        return sessions.get_cached_threads(agent_name=agent_name, limit=limit)

    async def get_most_recent(self, agent_name: str | None) -> str | None:
        from deepagents_cli import sessions

        return await sessions.get_most_recent(agent_name)

    async def get_thread_agent(self, thread_id: str) -> str | None:
        from deepagents_cli import sessions

        return await sessions.get_thread_agent(thread_id)

    async def thread_exists(self, thread_id: str) -> bool:
        from deepagents_cli import sessions

        return await sessions.thread_exists(thread_id)

    async def find_similar_threads(self, thread_id: str) -> list[str]:
        from deepagents_cli import sessions

        return await sessions.find_similar_threads(thread_id)

    async def delete_thread(self, thread_id: str) -> bool:
        from deepagents_cli import sessions

        return await sessions.delete_thread(thread_id)

    async def read_checkpoint_channel_values(self, thread_id: str) -> dict[str, Any]:
        from deepagents_cli import sessions

        return await sessions.read_sqlite_checkpoint_channel_values(thread_id)


def get_default_session_store() -> LocalSessionStore:
    """Return a process-wide default local store (minimal allocation)."""
    global _DEFAULT_LOCAL  # noqa: PLW0603
    if _DEFAULT_LOCAL is None:
        _DEFAULT_LOCAL = LocalSessionStore()
    return _DEFAULT_LOCAL


def build_session_store(
    remote: RemoteServerSettings | None,
) -> SessionStore:
    """Construct the session store for the current run mode."""
    if remote is not None:
        from deepagents_cli._remote_session_store import RemoteSessionStore

        return RemoteSessionStore(remote)
    return get_default_session_store()
