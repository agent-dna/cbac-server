"""
Wiring check for the CBAC gateway demo. No LLM, no model in the loop.

``agent.py`` is the demo; this is the proof it works. Every path an agent can
take is driven directly and asserted on, so a broken gateway fails here loudly
rather than in a transcript nobody reads.

Run (with the MCP server, the gateway and the CBAC service already up):

    python agent_check.py

See README.md for the prerequisites.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cbac" / "src"))

from agent import (
    CBAC_AGENT_ID,
    CBAC_URL,
    GATEWAY_URL,
    MCP_URL,
    THIRDPARTY_REPO,
    _mcp_client,
    draft_summary,
)

from cbac import cbac_context
from cbac.mcp import cbac_headers


def _tool(tools: dict, name: str):
    """Look a tool up by name, prefixed or not.

    The gateway fronts more than one upstream, so FastMCP prefixes each
    server's tools with its name (``github_close_issue``). A client pointed
    straight at a single MCP server sees the bare name.
    """
    for key, tool in tools.items():
        if key == name or key.endswith(f"_{name}"):
            return tool
    raise KeyError(f"{name} is not served here: {sorted(tools)}")


def _was_blocked(result: str) -> bool:
    """True when CBAC stopped the call.

    ``deny`` and ``advise`` both stop it: the gateway forwards only on
    ``allow``, so a gray-zone verdict is as much a block as a flat refusal —
    only the reason differs. Asserting on the word "denied" alone would call a
    correctly-blocked gray-zone call a failure.
    """
    return '"status": "denied"' in result or '"status": "error"' in result


async def _raw_client(url: str = GATEWAY_URL) -> Client:
    """A plain MCP client carrying the governance context on the connection.

    ``cbac_propagate`` hooks *tool* calls, because that is the only per-call
    hook a LangChain MCP client exposes. Resources and prompts have none, so
    the same headers ride on the connection instead — same encoder, same
    values, and the gateway cannot tell the difference.
    """
    return Client(StreamableHttpTransport(url, headers=cbac_headers()))


async def _read_collaborators(owner: str, name: str) -> str:
    """resources/read through the gateway. Returns the denial if there is one."""
    async with await _raw_client() as client:
        template = next(
            str(t.uriTemplate)
            for t in await client.list_resource_templates()
            if str(t.uriTemplate).endswith("/collaborators")
        )
        uri = template.replace("{owner}", owner).replace("{name}", name)
        try:
            return str((await client.read_resource(uri))[0])
        except Exception as exc:
            return str(exc)


async def _get_review_checklist(repo: str) -> str:
    """prompts/get through the gateway. Ungated, so this should come back."""
    async with await _raw_client() as client:
        name = next(
            prompt.name
            for prompt in await client.list_prompts()
            if prompt.name.endswith("review_checklist")
        )
        try:
            result = await client.get_prompt(name, {"repo": repo})
            return str(result.messages[0].content)
        except Exception as exc:
            return str(exc)


async def check() -> None:
    """Drive every path directly, no LLM. Verifies the wiring and the bypasses.

    Three MCP probes, all of them the same forbidden call, each trying a
    different way out:

    1. straight through the gateway — denied by the policy.
    2. from a client with **no** ``cbac_propagate``, the developer who
       "forgot" the CBAC wiring — still denied: the call arrives with no
       user intent to check drift against, so the policy decides it alone.
    3. is not a call. It lists the MCP server's tools **directly**, to show
       that the server answers anyone who can reach it: it carries no check
       of its own, and is not supposed to. Keeping it unroutable is the
       deployment's job — the gateway holds the only public address. Actually
       *calling* ``close_issue`` there would succeed and hit GitHub for real,
       which is exactly the point, and exactly why this only lists.

    Then two probes against a **real** third-party server — deepwiki, which
    nobody here owns, added to the gateway by URL alone:

    4. ``read_wiki_structure`` — permitted; the gateway relays it upstream.
    5. ``ask_question``        — forbidden; deepwiki never sees the call.

    Then the other two MCP primitives:

    6. ``collaborators``    — a *resource*; forbidden, so the read is refused.
    7. ``review_checklist`` — a *prompt*; not gated at all, so it comes back.

    Then ``draft_summary``, to show the boundary: it is in-process, so no
    gateway sees it and no decision is made about it.

    No probe touches GitHub — 1 and 2 die at the gateway, 3 only lists, 6 dies
    at the gateway before the GitHub request, 7 only formats text, and
    ``draft_summary`` only rewraps text. Probe 4 is the one real network call,
    and it is read-only.
    """
    # One correlation id for the whole run: every decision below lands in the
    # audit log under it, so one check run is one queryable trace.
    intent_id = uuid.uuid4().hex
    tools = {tool.name: tool for tool in await _mcp_client().get_tools()}
    unlabelled = {
        tool.name: tool for tool in await _mcp_client(labelled=False).get_tools()
    }
    # Not a call — just a listing, to show what the MCP server answers to
    # anyone who can reach it. Calling `close_issue` here would succeed and hit
    # GitHub for real: the server has no check of its own, by design.
    direct = [tool.name for tool in await _mcp_client(url=MCP_URL).get_tools()]

    print(f"[cbac] service={CBAC_URL} gateway={GATEWAY_URL} agent_id={CBAC_AGENT_ID}\n")
    close = {"repo": "owner/repo", "number": 3}
    with cbac_context(
        agent_id=CBAC_AGENT_ID,
        user_intent="Close issue #3 in owner/repo",
        intent_id=intent_id,
    ):
        denied = await _tool(tools, "close_issue").ainvoke(close)
        unlabelled_result = await _tool(unlabelled, "close_issue").ainvoke(close)
        local = await draft_summary(repo="owner/repo", notes="fixed  the   parser")

    # Same gateway, same policy, a server nobody here owns or can decorate.
    with cbac_context(
        agent_id=CBAC_AGENT_ID,
        user_intent="What docs exist for " + THIRDPARTY_REPO,
        intent_id=intent_id,
    ):
        third_party = await _tool(tools, "read_wiki_structure").ainvoke(
            {"repoName": THIRDPARTY_REPO}
        )
        third_party_denied = await _tool(tools, "ask_question").ainvoke(
            {"repoName": THIRDPARTY_REPO, "question": "How does authentication work?"}
        )

    # The other two MCP primitives. resources/read is gated: a gate wired to
    # tools/call alone is not a gate, because the same data is usually
    # reachable as a resource. prompts/get is not gated — see the note in
    # gateway.py — so it passes through and is here to show that.
    with cbac_context(
        agent_id=CBAC_AGENT_ID,
        user_intent="Who can write to owner/repo?",
        intent_id=intent_id,
    ):
        resource = await _read_collaborators("owner", "repo")
    with cbac_context(
        agent_id=CBAC_AGENT_ID,
        user_intent="Give me a review checklist for owner/repo",
        intent_id=intent_id,
    ):
        prompt = await _get_review_checklist("owner/repo")

    print(f"[gateway      close_issue        ] {denied}")
    print(f"[gateway      close_issue        ] {unlabelled_result}")
    print("                                   ^ client skipped cbac_propagate")
    print(f"[direct :8765 tools/list         ] {direct}")
    print("                                   ^ no check of its own — keeping this")
    print(
        "                                     port unroutable is the deployment's job"
    )
    print(f"[in-process   draft_summary      ] {local}")
    print("                                   ^ never became MCP traffic — ungoverned")
    print(f"[gateway      read_wiki_structure] {str(third_party)[:90]}…")
    print(f"[gateway      ask_question       ] {third_party_denied}")
    print("                                   ^ real third-party server, same policy")
    print(f"[gateway      collaborators      ] {resource[:90]}")
    print(f"[gateway      review_checklist   ] {prompt[:90]}…")
    print("                                   ^ a resource, gated; a prompt, not")

    assert "denied" in str(denied), f"forbidden tool was not blocked: {denied}"
    assert "denied" in str(unlabelled_result), (
        f"unlabelled call ran: {unlabelled_result}"
    )
    assert "close_issue" in direct, f"the MCP server did not answer at all: {direct}"
    assert "denied" not in str(third_party), (
        f"permitted third-party call was blocked: {third_party}"
    )
    assert "denied" in str(third_party_denied), (
        f"forbidden third-party call ran: {third_party_denied}"
    )
    assert _was_blocked(resource), f"forbidden resource was read: {resource}"
    assert "Review the open pull requests" in prompt, f"prompt did not run: {prompt}"
    print("\nOK — permitted calls relayed, forbidden ones blocked at the gateway:")
    print("     by policy, labelled or not. Tools and resources alike, and the")
    print("     third-party server on the same policy as the local one.")


if __name__ == "__main__":
    asyncio.run(check())
