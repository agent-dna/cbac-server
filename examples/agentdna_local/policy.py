"""Build an agent's v2 policy document and upload it to the CBAC service.

The document is the one ``cbac_service.parsers.split_policy_document`` splits:

    ---
    Policy Instructions
    <the policy, in prose>
    ---
    Tool Schema
    <JSON list of MCP tools>
    ---

The instructions come from a Markdown file the example keeps beside its agent;
the tool schemas are listed from the example's own FastMCP server, in process,
so the catalog always matches the tool names the middleware will report.

The policy is stored under the agent's DID, because that is the ``agent_id``
agentdna's middleware sends: :func:`actor_did` derives it from the agent's
name the same way the agent's own ``AgentDNA`` does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .provenance import did_from_public_key, load_or_create_key, raw_public_key
from .store import LocalStore, default_dir

DEFAULT_CBAC_URL = "http://127.0.0.1:8767"


def policy_document(instructions: str, tools: list[dict[str, Any]]) -> str:
    """The two-section Markdown document ``/cbac/v2/policies`` takes.

    A line that is exactly ``---`` separates sections, so the instructions
    must not contain one (use ``***`` for a Markdown rule)."""
    if any(line.strip() == "---" for line in instructions.splitlines()):
        raise ValueError("policy instructions must not contain a '---' line")
    return (
        f"---\nPolicy Instructions\n{instructions.strip()}\n"
        f"---\nTool Schema\n{json.dumps(tools, indent=2)}\n---\n"
    )


async def server_tools(server: Any) -> list[dict[str, Any]]:
    """Every tool a FastMCP server lists, as ``{name, description, inputSchema}``.

    Lists through FastMCP's in-memory client, so nothing has to be running.
    agentdna's middleware governs ``tools/call`` only; listing passes through.
    """
    from fastmcp import Client

    async with Client(server) as client:
        tools = await client.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema,
        }
        for t in tools
    ]


def actor_did(name: str, directory: str | os.PathLike[str] | None = None) -> str:
    """The DID the actor called ``name`` has (or will have) in the local store.

    Same key file, same derivation as ``LocalProvenance``, so uploading before
    the agent first runs is fine: it will come up with this DID.
    """
    local_dir = directory or default_dir()
    key = load_or_create_key(local_dir, name)
    public_key = raw_public_key(key)
    did = did_from_public_key(public_key)
    LocalStore(local_dir).put_actor(did, name, public_key)
    return did


def upload_policy(
    agent_id: str,
    document: str,
    trusted_data: dict[str, Any] | None = None,
    review_required: bool = False,
    cbac_url: str | None = None,
    policy_key: str | None = None,
    timeout: float = 900,
) -> dict[str, Any]:
    """POST the document to ``/cbac/v2/policies``; returns the response body.

    ``policy_key`` defaults to ``CBAC_POLICY_KEY``, ``cbac_url`` to ``CBAC_URL``
    or ``http://127.0.0.1:8767``. The timeout is long because the policy parser
    is an LLM pass over every schema.
    """
    key = policy_key or os.environ.get("CBAC_POLICY_KEY", "")
    if not key:
        raise ValueError("CBAC_POLICY_KEY is required to upload a policy")
    url = (cbac_url or os.environ.get("CBAC_URL") or DEFAULT_CBAC_URL).rstrip("/")
    response = requests.post(
        f"{url}/cbac/v2/policies",
        json={
            "agent_id": agent_id,
            "policy": document,
            "trusted_data": trusted_data or {},
            "review_required": review_required,
        },
        headers={"X-CBAC-Key": key},
        timeout=timeout,
    )
    try:
        return response.json()
    except ValueError:
        return {
            "success": False,
            "message": response.text,
            "status": response.status_code,
        }


async def upload_agent_policy(
    agent_name: str,
    instructions: str | os.PathLike[str],
    server: Any,
    trusted_data: dict[str, Any] | None = None,
    review_required: bool = False,
    cbac_url: str | None = None,
) -> dict[str, Any]:
    """Build and upload one agent's policy: its instructions file, the server's
    tools, its trusted data, under its local DID. Prints and returns the reply."""
    did = actor_did(agent_name)
    document = policy_document(
        Path(instructions).read_text(encoding="utf-8"), await server_tools(server)
    )
    print(f"[policy] {agent_name} -> {did}")
    reply = upload_policy(
        did,
        document,
        trusted_data=trusted_data,
        review_required=review_required,
        cbac_url=cbac_url,
    )
    print(json.dumps(reply, indent=2))
    return reply
