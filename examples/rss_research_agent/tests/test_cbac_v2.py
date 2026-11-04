"""The example's CBAC v2 wiring, with no service running and no network."""

import asyncio

import mcp_server
from agents.rss_security_agent.agent import RSS_SECURITY_AGENT
from agents.rss_technology_agent.agent import RSS_TECHNOLOGY_AGENT
from config import settings
from upload_policy import AGENTS, POLICIES

import agentdna_local
from agentdna_local import actor_did, policy_document, server_tools

TOOLS = {"rss_list_feeds", "rss_get_latest_articles", "rss_search_articles", "rss_get_article"}


def test_each_tool_calling_agent_has_a_policy_document() -> None:
    tools = asyncio.run(server_tools(mcp_server.mcp))
    assert {tool["name"] for tool in tools} == TOOLS
    for policy in AGENTS:
        document = policy_document((POLICIES / f"{policy}.md").read_text(encoding="utf-8"), tools)
        assert "\n---\nTool Schema\n" in document


def test_policies_are_uploaded_under_the_agents_dids() -> None:
    assert AGENTS["rss_security_agent"] == settings.security_agent_name
    assert AGENTS["rss_technology_agent"] == settings.technology_agent_name
    assert RSS_SECURITY_AGENT.get_actor_id() == actor_did(settings.security_agent_name)
    assert RSS_TECHNOLOGY_AGENT.get_actor_id() == actor_did(settings.technology_agent_name)


def test_trusted_data_lists_the_configured_feeds() -> None:
    feeds = settings.trusted_data["feeds"]
    assert [feed["id"] for feed in feeds] == [feed["id"] for feed in settings.feeds]


def test_two_agents_on_one_workflow_get_separate_intents(monkeypatch) -> None:
    intents = []

    async def authorize_call(agent_id, intent_id, user_query, tool, args, **kwargs):
        intents.append(intent_id)
        return "allow", 4302, "h"

    monkeypatch.setattr(agentdna_local.cbac_v2, "authorize_call", authorize_call)
    for agent in (RSS_SECURITY_AGENT, RSS_TECHNOLOGY_AGENT):
        verdict = asyncio.run(
            mcp_server.cbac_fn(
                agent.get_actor_id(), "bafyT", "rss_list_feeds", {}, "Brief me", None, "QmW"
            )
        )
        assert verdict[0] == "allow"
    assert len(set(intents)) == 2
