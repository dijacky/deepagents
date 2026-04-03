#!/usr/bin/env python3
"""Optional smoke check: RemoteGraph against a running Aegra (or LangGraph) server.

Set ``AEGRA_URL`` (or ``LANGGRAPH_BASE_URL``) to the server base URL, e.g.
``http://127.0.0.1:2026``. If unset, the script exits 0 without doing anything.

This script is not run in CI; it exists for manual verification after deploying
a graph named ``agent`` (or set ``REMOTE_GRAPH_ID``).
"""

from __future__ import annotations

import asyncio
import os
import sys


async def _main() -> int:
    base = os.environ.get("AEGRA_URL") or os.environ.get("LANGGRAPH_BASE_URL")
    if not base:
        print("Skip: set AEGRA_URL or LANGGRAPH_BASE_URL to run this check.", file=sys.stderr)
        return 0

    graph_id = os.environ.get("REMOTE_GRAPH_ID", "agent")
    thread_id = os.environ.get("VERIFY_THREAD_ID", "deepagents-verify-thread")

    from langgraph.pregel.remote import RemoteGraph

    graph = RemoteGraph(graph_id, url=base.rstrip("/"))
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(cfg)
    except Exception as exc:
        print(f"RemoteGraph.aget_state failed: {exc}", file=sys.stderr)
        return 1

    print("RemoteGraph.aget_state ok:", "None" if state is None else type(state).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
