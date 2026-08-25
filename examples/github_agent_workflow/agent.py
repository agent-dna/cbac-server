"""
A GitHub agent whose MCP tool calls are authorized by the CBAC decision
service — from a gateway the agent developer does not control.

  close_issue     served over MCP, and forbidden by the policy. The gateway
                  in front of the MCP server blocks it; the call never
                  reaches the server, let alone GitHub.

  deepwiki_*      served by a real third-party MCP server, added to the
                  gateway by URL alone. Same policy, same enforcement point:
                  read_wiki_structure is relayed, ask_question is blocked.

  draft_summary   an ordinary in-process function. No guard, no gateway — a
                  gateway governs MCP traffic, and this never becomes MCP
                  traffic. It is here to keep that boundary visible.

This process only *labels* its outgoing MCP calls (``cbac_propagate``) with
the governance context — who is acting and what the user asked for. It makes
no decision and holds no policy. Drop the label and the gateway denies every
call. Pointing the client straight at the MCP server is not a way out either,
but nothing in this file stops it: the gateway is the only service with a
public address, and every upstream sits on a private network behind it. See
README.md, "Pointing straight at the MCP server".

The label comes from the ambient ``cbac_context``, opened once around the
agent invocation. It is never a tool argument, so the LLM can neither see nor
forge it.

The local MCP server is deliberately CBAC-unaware — it stands in for a
third-party server you don't own and can't decorate. DeepWiki is not a
stand-in: it is a real public MCP server whose operator has never heard of
this repo, and the same policy decides its calls.

Run (three terminals):

    python agent.py serve      # the local MCP server      :8765
    python gateway.py          # the CBAC gateway (PEP)    :8766
    python agent.py "Close issue #3 in owner/repo"
    python agent.py "What docs exist for modelcontextprotocol/python-sdk?"

Or without an LLM, to check the wiring and both bypass attempts:

    python agent_check.py

See README.md for the CBAC service prerequisites.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# The cbac package lives in this repo's checkout (it ships no wheel), two
# directories up from examples/github_agent_workflow/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cbac import cbac_context, configure
from cbac.mcp import cbac_propagate

load_dotenv()

GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
MCP_HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_SERVER_PORT", "8765"))
# Only agent_check.py uses this, to show what an unrouted server would answer.
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp/"
# The agent talks to the gateway and nothing else.
GATEWAY_URL = (
    f"http://{os.environ.get('CBAC_GATEWAY_HOST', '127.0.0.1')}"
    f":{os.environ.get('CBAC_GATEWAY_PORT', '8766')}/mcp/"
)

# The repo the third-party (deepwiki) probes read about. Any public repo works;
# nothing is written, and the forbidden probe never leaves the gateway.
THIRDPARTY_REPO = os.environ.get(
    "MCP_THIRDPARTY_REPO", "modelcontextprotocol/python-sdk"
)

CBAC_URL = os.environ.get("CBAC_URL", "http://127.0.0.1:8767")
# Whose policy is checked. Matches policies/<CBAC_AGENT_ID>.md when the service
# runs with CBAC_LOCAL_POLICY_DIR; an on-chain actor id otherwise.
CBAC_AGENT_ID = os.environ.get("CBAC_AGENT_ID", "github-worker")
# A cold service loads three transformer models on the first request.
CBAC_TIMEOUT = float(os.environ.get("CBAC_TIMEOUT", "600"))


# ── MCP server half: plain GitHub tools, no CBAC awareness ───────────────────

mcp = FastMCP("github-mcp")


# Forbidden by the policy. The server exposes it happily; CBAC is what stops it.
@mcp.tool()
async def close_issue(repo: str, number: int) -> dict:
    """Close an existing GitHub issue.

    Args:
        repo: Repository in owner/name format.
        number: Issue number to close.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.patch(
            f"{GITHUB_API_URL}/repos/{repo}/issues/{number}",
            headers=headers,
            json={"state": "closed"},
        )
    if response.status_code >= 300:
        return {
            "status": "failed",
            "http_status": response.status_code,
            "error": response.text,
        }
    return {"status": "ok", **response.json()}


# Forbidden by the policy — a *resource*, not a tool. Same server, same
# gateway, different MCP primitive: if the gate only covered tools/call, the
# collaborator list would walk straight out through resources/read.
@mcp.resource("github://{owner}/{name}/collaborators")
async def collaborators(owner: str, name: str) -> dict:
    """Read the people with write access to a repository."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{GITHUB_API_URL}/repos/{owner}/{name}/collaborators", headers=headers
        )
    if response.status_code >= 300:
        return {"status": "failed", "http_status": response.status_code}
    return {"status": "ok", "collaborators": response.json()}


# A *prompt*, the third MCP primitive — and the one the gateway does not gate.
# Prompts are user-controlled in MCP's model: the person picks one, the model
# does not reach for it. See the note in gateway.py.
@mcp.prompt()
async def review_checklist(repo: str) -> str:
    """Produce a checklist for reviewing a pull request."""
    return (
        f"Review the open pull requests in {repo}. For each one check: "
        "tests cover the change, the public API did not change silently, "
        "and the description matches the diff."
    )


# ── Agent half: a plain in-process tool, outside CBAC's reach ────────────────


# No guard, and no gateway in front of it — a gateway governs MCP traffic, and
# this call never becomes MCP traffic. It is here to make that boundary visible:
# moving a capability in-process moves it out of scope. Anything that matters
# belongs behind the MCP boundary, where the gateway can see it.
async def draft_summary(repo: str, notes: str) -> dict:
    """Condense raw notes into a short change summary for a repo.

    Args:
        repo: Repository in owner/name format.
        notes: Raw notes to condense.
    """
    return {"status": "ok", "repo": repo, "summary": " ".join(notes.split())[:280]}


SYSTEM = "You are a GitHub agent. Call exactly one tool to satisfy the request."


# ── LLM ──────────────────────────────────────────────────────────────────────


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is not set (required by LLM_PROVIDER)")
    return value


def make_llm(temperature: float = 0.0):
    """Build the chat model named by LLM_PROVIDER. Must support tool calling."""
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_require("GEMINI_MODEL"),
            google_api_key=_require("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=_require("OLLAMA_MODEL"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": _require("OPENAI_MODEL"),
            "api_key": _require("OPENAI_API_KEY"),
            "temperature": temperature,
        }
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return ChatOpenAI(**kwargs)

    raise SystemExit(
        f"unknown LLM_PROVIDER={provider!r}; expected gemini, ollama or openai"
    )


# ── Run ──────────────────────────────────────────────────────────────────────


def _cbac_decision(message) -> tuple[str, str] | None:
    """Read CBAC's verdict off a tool-result message, or None if not one.

    A blocked call carries ``{"status": "denied"|"error", "error": reason}``,
    raised by the gateway and relayed as the MCP text block. Any other tool
    result means the gateway authorized the call and the tool's own result
    came back from the MCP server.
    """
    if getattr(message, "type", "") != "tool":
        return None

    content = getattr(message, "content", "")
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
    else:
        texts = []

    for text in texts:
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("status") in ("denied", "error"):
            return str(payload["status"]).upper(), str(payload.get("error", ""))
    return "ALLOW", ""


def _print_transcript(messages: list) -> None:
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[call] {call['name']}({call['args']})")

        verdict = _cbac_decision(message)
        if verdict is not None:
            decision, reason = verdict
            print(
                f"[cbac] {decision} — {reason or 'authorized by policy, tool executed'}"
            )

        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            print(f"[{getattr(message, 'type', '?')}] {content}")


def _mcp_client(labelled: bool = True, url: str = GATEWAY_URL) -> MultiServerMCPClient:
    """A client pointed at the CBAC gateway, not at the MCP server.

    ``labelled=False`` drops ``cbac_propagate`` — the "developer skipped the
    CBAC wiring" case. It is not a bypass: the calls still go through the
    gateway, they just arrive with nothing to authorize, and get denied.

    ``url=MCP_URL`` is the other thing a developer might try — routing around
    the gateway entirely. Also not a bypass: the MCP server rejects callers
    without the gateway's credential.

    One entry, and it is the *gateway* — not one entry per upstream. The
    gateway is an MCP server in its own right: it merges the tool lists of
    every upstream it fronts and serves the union, so ``get_tools()`` here
    returns ``github_*`` and ``deepwiki_*`` alike. The agent never learns how
    many upstreams there are and cannot address one individually; adding a
    third is a gateway-side config change this file never notices.
    """
    # Give the MCP client more headroom than CBAC itself: a tool call blocks on
    # the gateway's authorization round-trip, so the client must not give up
    # first.
    return MultiServerMCPClient(
        {
            # Names the connection, nothing else. The tool prefixes come from
            # the gateway's own upstream names, not from this key.
            "gateway": {
                "url": url,
                "transport": "streamable_http",
                "timeout": CBAC_TIMEOUT + 60,
                "sse_read_timeout": CBAC_TIMEOUT + 60,
            }
        },
        tool_interceptors=[cbac_propagate] if labelled else [],
    )


async def run(request: str) -> None:
    configure(cbac_url=CBAC_URL, cbac_timeout=CBAC_TIMEOUT)

    tools = [
        *await _mcp_client().get_tools(),  # decided at the gateway
        StructuredTool.from_function(coroutine=draft_summary),  # in-process, not
    ]
    agent = create_agent(make_llm(), tools, system_prompt=SYSTEM)

    print(f"\n[cbac] service={CBAC_URL} gateway={GATEWAY_URL} agent_id={CBAC_AGENT_ID}")
    print(f"[user] {request}\n")

    # The root request is the governance context: each guarded call is checked
    # against the agent's policy AND for drift away from what the user asked.
    with cbac_context(agent_id=CBAC_AGENT_ID, user_intent=request):
        result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})

    _print_transcript(result.get("messages", []))


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
    elif len(sys.argv) == 2:
        asyncio.run(run(sys.argv[1]))
    else:
        raise SystemExit(f"usage: {sys.argv[0]} serve|'<request>'")
