"""The example's CBAC v2 wiring, with no service running and no network."""

import asyncio

import mcp_server
from agents.github_agent.agent import GITHUB_AGENT
from config import settings
from upload_policy import POLICY

import agentdna_local
from agentdna_local import actor_did, policy_document, server_tools

TOOLS = {
    "repository_metadata",
    "directory_listing",
    "file_contents",
    "commits",
    "issues",
    "pull_requests",
}


def test_policy_document_lists_every_server_tool() -> None:
    tools = asyncio.run(server_tools(mcp_server.mcp))
    assert {tool["name"] for tool in tools} == TOOLS
    document = policy_document(POLICY.read_text(encoding="utf-8"), tools)
    assert document.startswith("---\nPolicy Instructions\n")
    assert "\n---\nTool Schema\n" in document


def test_policy_is_uploaded_under_the_agents_did() -> None:
    assert GITHUB_AGENT.get_actor_id() == actor_did(settings.github_agent_name)


def test_mcp_server_asks_the_v2_gate(monkeypatch) -> None:
    calls = []

    async def authorize_call(agent_id, intent_id, user_query, tool, args, **kwargs):
        calls.append((agent_id, intent_id, tool, kwargs["cbac_url"]))
        return "allow", 4301, "h"

    monkeypatch.setattr(agentdna_local.cbac_v2, "authorize_call", authorize_call)
    verdict = asyncio.run(
        mcp_server.cbac_fn("bafyA", "bafyT", "issues", {}, "List issues", None, "QmW")
    )
    assert verdict == ("allow", 4301, "h")
    assert calls == [("bafyA", "QmW:bafyA", "issues", settings.cbac_url)]
