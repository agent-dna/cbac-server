"""The example's CBAC v2 wiring, with no service running and no network."""

import asyncio

import mcp_server
from agents.sqlite_agent.agent import SQLITE_AGENT
from config import settings
from upload_policy import POLICY, TRUSTED_DATA

import agentdna_local
from agentdna_local import actor_did, policy_document, server_tools


def test_policy_document_lists_every_server_tool() -> None:
    tools = asyncio.run(server_tools(mcp_server.mcp))
    assert {tool["name"] for tool in tools} == {
        "sqlite_list_tables",
        "sqlite_describe_table",
        "sqlite_query",
    }
    document = policy_document(POLICY.read_text(encoding="utf-8"), tools)
    assert "\n---\nTool Schema\n" in document


def test_trusted_data_is_the_analytical_tables() -> None:
    assert TRUSTED_DATA == {"tables": sorted(mcp_server.TABLES)}


def test_policy_is_uploaded_under_the_agents_did() -> None:
    assert SQLITE_AGENT.get_actor_id() == actor_did(settings.sqlite_agent_name)


def test_mcp_server_asks_the_v2_gate(monkeypatch) -> None:
    calls = []

    async def authorize_call(agent_id, intent_id, user_query, tool, args, **kwargs):
        calls.append((intent_id, tool, args, kwargs["cbac_url"]))
        return "deny", 4103, "h"

    monkeypatch.setattr(agentdna_local.cbac_v2, "authorize_call", authorize_call)
    verdict = asyncio.run(
        mcp_server.cbac_fn("bafyA", "bafyT", "sqlite_query", {"query": "SELECT 1"}, "q", None, "QmW")
    )
    assert verdict == ("deny", 4103, "h")
    assert calls == [("QmW:bafyA", "sqlite_query", {"query": "SELECT 1"}, settings.cbac_url)]
