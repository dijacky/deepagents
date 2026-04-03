"""Resolve optional remote LangGraph-compatible server settings (e.g. Aegra)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents_cli._env_vars import REMOTE_API_KEY, REMOTE_GRAPH_NAME, REMOTE_URL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteServerSettings:
    """CLI configuration for connecting to a remote graph server."""

    url: str
    api_key: str | None = None
    graph_name: str = "agent"
    headers: dict[str, str] | None = None


def _read_server_table_from_config_toml() -> dict[str, Any]:
    """Load the optional ``[server]`` table from ``~/.deepagents/config.toml``."""
    import tomllib

    path = Path.home() / ".deepagents" / "config.toml"
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read config.toml for [server]", exc_info=True)
        return {}
    raw = data.get("server")
    return raw if isinstance(raw, dict) else {}


def resolve_remote_server_settings(
    *,
    cli_remote_url: str | None = None,
) -> RemoteServerSettings | None:
    """Merge remote URL from CLI flag, env vars, and ``[server]`` in config.toml.

    Precedence: ``--remote-url`` > ``DEEPAGENTS_CLI_REMOTE_URL`` > ``[server].remote_url``.

    Args:
        cli_remote_url: Value from ``--remote-url`` when set.

    Returns:
        Settings when a URL is configured, otherwise ``None``.
    """
    toml_server = _read_server_table_from_config_toml()

    url = (cli_remote_url or "").strip() or os.environ.get(REMOTE_URL, "").strip()
    if not url and isinstance(toml_server.get("remote_url"), str):
        url = str(toml_server["remote_url"]).strip()
    if not url:
        return None

    api_key = os.environ.get(REMOTE_API_KEY, "").strip() or None
    if api_key is None and isinstance(toml_server.get("api_key"), str):
        raw_key = str(toml_server["api_key"]).strip()
        api_key = raw_key or None

    graph_name = os.environ.get(REMOTE_GRAPH_NAME, "").strip() or "agent"
    if isinstance(toml_server.get("graph_name"), str) and toml_server["graph_name"].strip():
        graph_name = str(toml_server["graph_name"]).strip()

    headers: dict[str, str] | None = None
    raw_headers = toml_server.get("headers")
    if isinstance(raw_headers, dict):
        headers = {str(k): str(v) for k, v in raw_headers.items()}

    return RemoteServerSettings(
        url=url.rstrip("/"),
        api_key=api_key,
        graph_name=graph_name,
        headers=headers,
    )
