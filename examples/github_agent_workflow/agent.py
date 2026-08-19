"""
A GitHub agent whose every tool call is authorized by the CBAC decision service.

Two tools, one per enforcement path:

  draft_summary   in-process, guarded by @cbac_guard. The policy permits it,
                  so the guard authorizes and the function runs.

  close_issue     served over MCP, guarded by cbac_intercept on the client.
                  The policy forbids it, so the call is blocked before it
                  leaves this process. The MCP server is deliberately
                  CBAC-unaware — it stands in for a third-party server you
                  don't own and can't decorate.

Both read the same ambient ``cbac_context``, opened once around the agent
invocation. It is never a tool argument, so the LLM can neither see nor forge
it.

Run (two terminals):

    python agent.py serve
    python agent.py "Draft a summary for owner/repo from notes: rewrote the parser"
    python agent.py "Close issue #3 in owner/repo"

Or without an LLM, to check the guard wiring alone:

    python agent.py check

See README.md for the CBAC service prerequisites.
"""

from __future__ import annotations

import asyncio
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

from cbac import cbac_context, cbac_guard, configure
from cbac.mcp import cbac_intercept

load_dotenv()

GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
MCP_HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_SERVER_PORT", "8765"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp/"

CBAC_URL = os.environ.get("CBAC_URL", "http://127.0.0.1:8767")
# Whose policy is checked. Matches policies/<CBAC_AGENT_ID>.md when the service
# runs with CBAC_LOCAL_POLICY_DIR; an on-chain actor id otherwise.
CBAC_AGENT_ID = os.environ.get("CBAC_AGENT_ID", "github-worker")
# A cold service loads three transformer models on the first request.
CBAC_TIMEOUT = float(os.environ.get("CBAC_TIMEOUT", "600"))


# ── MCP server half: a plain GitHub tool, no CBAC awareness ──────────────────

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


# ── Agent half: an in-process guarded tool ───────────────────────────────────


# Permitted by the policy. Runs locally, so it needs no GitHub credentials.
@cbac_guard()
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


def _print_transcript(messages: list) -> None:
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[call] {call['name']}({call['args']})")
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            print(f"[{getattr(message, 'type', '?')}] {content}")


def _mcp_client() -> MultiServerMCPClient:
    # Give the MCP client more headroom than CBAC itself: a tool call blocks on
    # the authorization round-trip, so the client must not give up first.
    return MultiServerMCPClient(
        {
            "github": {
                "url": MCP_URL,
                "transport": "streamable_http",
                "timeout": CBAC_TIMEOUT + 60,
                "sse_read_timeout": CBAC_TIMEOUT + 60,
            }
        },
        tool_interceptors=[cbac_intercept],
    )


async def check() -> None:
    """Drive both guard paths directly, no LLM. Verifies the demo's wiring.

    Neither call touches GitHub: ``draft_summary`` is local, and the forbidden
    ``close_issue`` is blocked before the request leaves this process.
    """
    configure(cbac_url=CBAC_URL, cbac_timeout=CBAC_TIMEOUT)
    tools = {tool.name: tool for tool in await _mcp_client().get_tools()}

    print(f"[cbac] service={CBAC_URL} agent_id={CBAC_AGENT_ID}\n")
    with cbac_context(
        agent_id=CBAC_AGENT_ID, user_intent="Summarise my changes for owner/repo"
    ):
        allowed = await draft_summary(repo="owner/repo", notes="fixed  the   parser")
        denied = await tools["close_issue"].ainvoke({"repo": "owner/repo", "number": 3})

    print(f"[@cbac_guard    draft_summary] {allowed}")
    print(f"[cbac_intercept close_issue  ] {denied}")

    assert allowed.get("status") == "ok", f"allowed tool was blocked: {allowed}"
    assert "denied" in str(denied), f"forbidden tool was not blocked: {denied}"
    print("\nOK — permitted tool ran, forbidden tool blocked before the GitHub call.")


async def run(request: str) -> None:
    configure(cbac_url=CBAC_URL, cbac_timeout=CBAC_TIMEOUT)

    tools = [
        StructuredTool.from_function(coroutine=draft_summary),  # @cbac_guard
        *await _mcp_client().get_tools(),  # cbac_intercept
    ]
    agent = create_agent(make_llm(), tools, system_prompt=SYSTEM)

    print(f"\n[cbac] service={CBAC_URL} agent_id={CBAC_AGENT_ID}")
    print(f"[user] {request}\n")

    # The root request is the governance context: each guarded call is checked
    # against the agent's policy AND for drift away from what the user asked.
    with cbac_context(agent_id=CBAC_AGENT_ID, user_intent=request):
        result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})

    _print_transcript(result.get("messages", []))


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
    elif sys.argv[1:] == ["check"]:
        asyncio.run(check())
    elif len(sys.argv) == 2:
        asyncio.run(run(sys.argv[1]))
    else:
        raise SystemExit(f"usage: {sys.argv[0]} serve|check|'<request>'")
