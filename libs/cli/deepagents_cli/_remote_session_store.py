"""Thread listing backed by a remote LangGraph-compatible server (e.g. Aegra)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from deepagents_cli.remote_server_settings import RemoteServerSettings

if TYPE_CHECKING:
    from langgraph_sdk.client import LangGraphClient

    from deepagents_cli.sessions import ThreadInfo

logger = logging.getLogger(__name__)

_MAX_THREADS_PER_SEARCH = 100
"""Aegra caps ``ThreadSearchRequest.limit`` at 100."""

_STATE_CONCURRENCY = 8


def _coerce_thread_row(row: object) -> dict[str, Any]:
    """Coerce SDK thread objects to plain dicts."""
    if isinstance(row, dict):
        return row
    model_dump = getattr(row, "model_dump", None)
    if callable(model_dump):
        return model_dump()  # type: ignore[no-any-return]
    return dict(row)  # type: ignore[arg-type]


def _iso_timestamp(value: object | None) -> str | None:
    """Normalize API timestamps to ISO strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    md = row.get("metadata")
    return md if isinstance(md, dict) else {}


def _summarize_messages(values: dict[str, Any] | None) -> tuple[int, str | None]:
    """Return message count and first human text from graph state values."""
    if not values or not isinstance(values, dict):
        return 0, None
    raw = values.get("messages")
    if not isinstance(raw, list):
        return 0, None
    count = len(raw)
    first_prompt: str | None = None
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = msg.get("type") or msg.get("role")
        if role not in {"human", "user"}:
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            first_prompt = content.strip()
            break
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            merged = "".join(parts).strip()
            if merged:
                first_prompt = merged
                break
    return count, first_prompt


class RemoteSessionStore:
    """Uses ``langgraph_sdk`` thread APIs against a remote deployment."""

    def __init__(self, settings: RemoteServerSettings) -> None:
        self._settings = settings
        self._recent_cache: dict[tuple[str | None, int], list[ThreadInfo]] = {}
        self._sem = asyncio.Semaphore(_STATE_CONCURRENCY)
        self._client: LangGraphClient | None = None

    @property
    def checkpoint_backend(self) -> str:
        return "remote"

    def _get_client(self) -> LangGraphClient:
        if self._client is None:
            from langgraph_sdk import get_client

            kwargs: dict[str, Any] = {"url": self._settings.url}
            if self._settings.api_key is not None:
                kwargs["api_key"] = self._settings.api_key
            if self._settings.headers:
                kwargs["headers"] = self._settings.headers
            self._client = get_client(**kwargs)
        return self._client

    def _record_to_thread_info(self, row: dict[str, Any]) -> ThreadInfo:
        from deepagents_cli.sessions import ThreadInfo

        md = _row_metadata(row)
        thread_id = str(row.get("thread_id", ""))
        return ThreadInfo(
            thread_id=thread_id,
            agent_name=md.get("agent_name") if isinstance(md.get("agent_name"), str) else None,
            updated_at=_iso_timestamp(row.get("updated_at")),
            created_at=_iso_timestamp(row.get("created_at")),
            git_branch=md.get("git_branch") if isinstance(md.get("git_branch"), str) else None,
            cwd=md.get("cwd") if isinstance(md.get("cwd"), str) else None,
            latest_checkpoint_id=None,
        )

    async def list_threads(
        self,
        agent_name: str | None = None,
        limit: int = 20,
        include_message_count: bool = False,
        sort_by: str = "updated",
        branch: str | None = None,
    ) -> list[ThreadInfo]:
        client = self._get_client()
        metadata: dict[str, str] = {}
        if agent_name:
            metadata["agent_name"] = agent_name
        if branch:
            metadata["git_branch"] = branch

        fetch_limit = min(max(limit, 1), _MAX_THREADS_PER_SEARCH)
        try:
            rows = await client.threads.search(
                metadata=metadata or None,
                limit=fetch_limit,
                offset=0,
            )
        except Exception:
            logger.warning("Remote thread search failed", exc_info=True)
            return []

        if not isinstance(rows, list):
            return []

        threads = [self._record_to_thread_info(_coerce_thread_row(r)) for r in rows]
        key = "updated_at" if sort_by == "updated" else "created_at"

        def _sort_key(info: ThreadInfo) -> str:
            return (info.get(key) or info.get("updated_at") or info.get("created_at") or "")

        threads.sort(key=_sort_key, reverse=True)
        threads = threads[:limit]

        if include_message_count:
            await self.populate_thread_message_counts(threads)
        return threads

    async def populate_thread_message_counts(
        self,
        threads: list[ThreadInfo],
    ) -> list[ThreadInfo]:
        return await self.populate_thread_checkpoint_details(
            threads,
            include_message_count=True,
            include_initial_prompt=False,
        )

    async def populate_thread_checkpoint_details(
        self,
        threads: list[ThreadInfo],
        *,
        include_message_count: bool = True,
        include_initial_prompt: bool = True,
    ) -> list[ThreadInfo]:
        if not threads or (not include_message_count and not include_initial_prompt):
            return threads

        client = self._get_client()
        from langgraph_sdk.errors import NotFoundError

        async def _one(tid: str) -> tuple[str, int, str | None]:
            async with self._sem:
                try:
                    state = await client.threads.get_state(tid)
                except NotFoundError:
                    return tid, 0, None
                except Exception:
                    logger.debug("get_state failed for %s", tid, exc_info=True)
                    return tid, 0, None
            values: dict[str, Any] | None
            if isinstance(state, dict):
                raw_values = state.get("values")
                values = raw_values if isinstance(raw_values, dict) else None
            else:
                raw_values = getattr(state, "values", None)
                values = raw_values if isinstance(raw_values, dict) else None
            count, first = _summarize_messages(values)
            return tid, count, first

        ids = [t["thread_id"] for t in threads]
        results = await asyncio.gather(*(_one(tid) for tid in ids))
        by_id = {tid: (c, fp) for tid, c, fp in results}

        for row in threads:
            tid = row["thread_id"]
            count, first = by_id.get(tid, (0, None))
            if include_message_count and "message_count" not in row:
                row["message_count"] = count
            if include_initial_prompt and "initial_prompt" not in row:
                row["initial_prompt"] = first
        return threads

    async def prewarm_thread_message_counts(self, limit: int | None = None) -> None:
        from deepagents_cli.model_config import load_thread_config
        from deepagents_cli.sessions import ThreadInfo, get_thread_limit

        thread_limit = limit if limit is not None else get_thread_limit()
        if thread_limit < 1:
            return
        try:
            threads = await self.list_threads(
                limit=thread_limit,
                include_message_count=False,
                sort_by="updated",
            )
            if not threads:
                self._recent_cache.clear()
                return
            cfg = load_thread_config()
            await self.populate_thread_checkpoint_details(
                threads,
                include_message_count=cfg.columns.get("messages", False),
                include_initial_prompt=cfg.columns.get("initial_prompt", False),
            )
            key = (None, max(1, thread_limit))
            self._recent_cache[key] = [ThreadInfo(**dict(t)) for t in threads]
        except Exception:
            logger.warning("Remote thread prewarm failed", exc_info=True)

    def get_cached_threads(
        self,
        agent_name: str | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo] | None:
        from deepagents_cli.sessions import ThreadInfo, get_thread_limit

        thread_limit = limit if limit is not None else get_thread_limit()
        if thread_limit < 1:
            return None
        exact = self._recent_cache.get((agent_name, thread_limit))
        if exact is not None:
            return [ThreadInfo(**dict(t)) for t in exact]
        return None

    async def get_most_recent(self, agent_name: str | None) -> str | None:
        threads = await self.list_threads(
            agent_name=agent_name,
            limit=_MAX_THREADS_PER_SEARCH,
            include_message_count=False,
            sort_by="updated",
        )
        return threads[0]["thread_id"] if threads else None

    async def get_thread_agent(self, thread_id: str) -> str | None:
        from langgraph_sdk.errors import NotFoundError

        try:
            row = await self._get_client().threads.get(thread_id)
        except NotFoundError:
            return None
        except Exception:
            logger.debug("get_thread_agent failed", exc_info=True)
            return None
        md = _row_metadata(_coerce_thread_row(row))
        name = md.get("agent_name")
        return str(name) if isinstance(name, str) and name else None

    async def thread_exists(self, thread_id: str) -> bool:
        from langgraph_sdk.errors import NotFoundError

        try:
            await self._get_client().threads.get(thread_id)
        except NotFoundError:
            return False
        except Exception:
            logger.warning("thread_exists remote check failed", exc_info=True)
            return False
        else:
            return True

    async def find_similar_threads(self, thread_id: str) -> list[str]:
        rows = await self.list_threads(
            agent_name=None,
            limit=_MAX_THREADS_PER_SEARCH,
            include_message_count=False,
            sort_by="updated",
        )
        prefix = thread_id[:8]
        return [r["thread_id"] for r in rows if r["thread_id"].startswith(prefix)]

    async def delete_thread(self, thread_id: str) -> bool:
        from langgraph_sdk.errors import NotFoundError

        try:
            await self._get_client().threads.delete(thread_id)
        except NotFoundError:
            return False
        except Exception:
            logger.warning("Remote delete_thread failed", exc_info=True)
            raise
        else:
            for key in list(self._recent_cache):
                self._recent_cache[key] = [
                    t for t in self._recent_cache[key] if t["thread_id"] != thread_id
                ]
            return True

    async def read_checkpoint_channel_values(self, thread_id: str) -> dict[str, Any]:
        from langgraph_sdk.errors import NotFoundError

        try:
            state = await self._get_client().threads.get_state(thread_id)
        except NotFoundError:
            return {}
        except Exception:
            logger.warning("Remote read_checkpoint_channel_values failed", exc_info=True)
            return {}
        if isinstance(state, dict):
            raw = state.get("values")
            return dict(raw) if isinstance(raw, dict) else {}
        raw = getattr(state, "values", None)
        return dict(raw) if isinstance(raw, dict) else {}
