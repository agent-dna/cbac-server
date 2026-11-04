"""The example's CBAC v2 wiring, with no service running and no network."""

import asyncio

import mcp_server
import swarm
from upload_policy import MEMBERS, POLICIES

import agentdna_local
from agentdna_local import actor_did, policy_document, server_tools


def test_every_tool_calling_member_has_a_policy() -> None:
    assert set(MEMBERS) == {member for member, tools in swarm.TOOLS.items() if tools}
    tools = asyncio.run(server_tools(mcp_server.mcp))
    served = {tool["name"] for tool in tools}
    for member in MEMBERS:
        assert set(swarm.TOOLS[member]) <= served
        text = (POLICIES / f"{member}.md").read_text(encoding="utf-8")
        # The policy names the member's own tools.
        assert all(f"`{tool}`" in text for tool in swarm.TOOLS[member])
        assert "\n---\nTool Schema\n" in policy_document(text, tools)


def test_members_get_separate_intents_on_one_workflow(monkeypatch) -> None:
    intents = []

    async def authorize_call(agent_id, intent_id, user_query, tool, args, **kwargs):
        intents.append(intent_id)
        return "allow", 4301, "h"

    monkeypatch.setattr(agentdna_local.cbac_v2, "authorize_call", authorize_call)
    for name in MEMBERS.values():
        did = actor_did(name)
        asyncio.run(mcp_server.cbac_fn(did, "bafyT", "package_info", {"name": "httpx"}, "q", None, "QmW"))
    assert len(set(intents)) == len(MEMBERS)
