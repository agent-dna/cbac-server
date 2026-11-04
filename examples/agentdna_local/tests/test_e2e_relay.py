"""End to end on loopback: user → agent → Provenance Relay → FastMCP server
behind AgentDNAMCPMiddleware → the v2 cbac_fn → a stub of /cbac/v2/authorize.

Nothing is faked inside agentdna: CoCA verification, the whitelist check and
signing all run against the local store. Needs fastmcp and uvicorn, so it is
skipped in the root venv and runs in an example's venv.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

fastmcp = pytest.importorskip("fastmcp")
uvicorn = pytest.importorskip("uvicorn")

from agentdna.core import AgentDNA
from agentdna.mcp.relay import ProvenanceRelay
from agentdna.mcp.server.fastmcp import AgentDNAMCPMiddleware
from agentdna_local import make_cbac_fn


class _Gate(BaseHTTPRequestHandler):
    """Allows ``add``; refuses everything else with 4103."""

    seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, **body})
        code = 4301 if body["tool"] == "add" else 4103
        decision = "allow" if code == 4301 else "deny"
        payload = json.dumps(
            {
                "success": True,
                "message": "",
                "data": {"decision": decision, "status_code": code, "hash": "h"},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args) -> None:
        pass


def _serve_mcp(app) -> tuple[str, object]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("MCP server did not start")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}/mcp", server


@pytest.fixture
def gate() -> Iterator[tuple[str, type[_Gate]]]:
    handler = type("Gate", (_Gate,), {"seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", handler
    server.shutdown()
    server.server_close()


def test_a_relayed_tool_call_is_decided_by_the_v2_gate(
    local_dir, tmp_path, gate
) -> None:
    gate_url, handler = gate
    policy = tmp_path / "SKILLS.md"
    policy.write_text("# Skills\n\n- Add numbers.\n", encoding="utf-8")

    user = AgentDNA(name="user@example.local", type="user")
    agent = AgentDNA(name="adder-agent", type="agent", agent_policy_file=policy)
    tool = AgentDNA(name="maths-mcp", type="tool")

    server = fastmcp.FastMCP("maths")
    server.add_middleware(
        AgentDNAMCPMiddleware(tool, cbac_fn=make_cbac_fn(gate_url, 5))
    )

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def shout(text: str) -> str:
        """Upper-case some text."""
        return text.upper()

    url, mcp_server = _serve_mcp(server.http_app(path="/mcp"))
    relay = ProvenanceRelay({"maths": url})
    request = user.build(payload="Add 1 and 2")

    async def run():
        async with (
            relay.session(agent, request) as session,
            fastmcp.Client(session.url("maths")) as client,
        ):
            added = await client.call_tool("add", {"a": 1, "b": 2})
        return added, session

    async def refused():
        async with (
            relay.session(agent, user.build(payload="Add 1 and 2")) as session,
            fastmcp.Client(session.url("maths")) as client,
        ):
            await client.call_tool("shout", {"text": "hi"})

    try:
        added, session = asyncio.run(run())
        with pytest.raises(Exception):  # noqa: B017  the refusal ends the run's chain
            asyncio.run(refused())
    finally:
        relay.close()
        mcp_server.should_exit = True

    assert added.data == 3
    allowed, denied = handler.seen
    assert allowed["path"] == "/cbac/v2/authorize"
    assert allowed["agent_id"] == agent.get_actor_id()
    assert allowed["mcp_did"] == tool.get_actor_id()
    assert allowed["user_query"] == "Add 1 and 2"
    assert allowed["intent_id"] == f"{request.id}:{agent.get_actor_id()}"
    assert allowed["arguments"] == {"a": 1, "b": 2}
    assert denied["tool"] == "shout"

    # The chain the agent brings back verifies and records locally.
    [tip] = session.workflows
    user.record(tip)
