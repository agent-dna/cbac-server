"""The v2 cbac_fn adapter, and the policy helper's document and upload.

The HTTP tests stand up a stub service on 127.0.0.1, so they exercise the real
``cbac.authorize_call`` and ``requests`` paths without leaving the machine.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest
from agentdna_local import (
    actor_did,
    make_cbac_fn,
    policy_document,
    scoped_intent_id,
    upload_policy,
)
from agentdna_local import cbac_v2 as adapter


def _call(cbac_fn, **overrides):
    args = {
        "agent_id": "bafyagent",
        "mcp_did": "bafytool",
        "tool": "issues",
        "args": {"repository": "octo/repo", "limit": 5},
        "user_intent": "List the open issues",
        "description": "Return open issues",
        "intent_id": "QmWorkflow",
    }
    args.update(overrides)
    return asyncio.run(cbac_fn(*args.values()))


@pytest.fixture
def fake_gate(monkeypatch):
    """Replaces authorize_call; records what it was given."""
    calls: list[dict] = []
    answer = {"value": ("allow", 4301, "h-1")}

    async def authorize_call(agent_id, intent_id, user_query, tool, args, **kwargs):
        calls.append(
            {
                "agent_id": agent_id,
                "intent_id": intent_id,
                "user_query": user_query,
                "tool": tool,
                "args": args,
                **kwargs,
            }
        )
        return answer["value"]

    monkeypatch.setattr(adapter, "authorize_call", authorize_call)
    return calls, answer


def test_adapter_maps_agentdna_arguments_onto_authorize_call(
    fake_gate, monkeypatch
) -> None:
    monkeypatch.delenv("CBAC_URL", raising=False)
    calls, _ = fake_gate
    assert _call(make_cbac_fn()) == ("allow", 4301, "h-1")
    assert calls == [
        {
            "agent_id": "bafyagent",
            "intent_id": "QmWorkflow:bafyagent",
            "user_query": "List the open issues",
            "tool": "issues",
            "args": {"repository": "octo/repo", "limit": 5},
            "mcp_did": "bafytool",
            "cbac_url": "http://127.0.0.1:8767",
            "cbac_timeout": 600.0,
        }
    ]


def test_url_and_timeout_come_from_the_environment(fake_gate, monkeypatch) -> None:
    calls, _ = fake_gate
    monkeypatch.setenv("CBAC_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("CBAC_TIMEOUT", "12")
    _call(make_cbac_fn())
    assert (calls[0]["cbac_url"], calls[0]["cbac_timeout"]) == (
        "http://127.0.0.1:9999",
        12.0,
    )
    _call(make_cbac_fn("http://127.0.0.1:1234", 3))
    assert (calls[1]["cbac_url"], calls[1]["cbac_timeout"]) == (
        "http://127.0.0.1:1234",
        3,
    )


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        (("allow", 4301, "h"), "allow"),
        (("allow", 4302, "h"), "allow"),
        # Only the two allow codes allow, whatever the decision word says.
        (("allow", 4103, "h"), "deny"),
        (("deny", 4103, "h"), "deny"),
        (("deny", 4202, "h"), "deny"),
        (("error", 9103, ""), "error"),
    ],
)
def test_only_4301_and_4302_allow(fake_gate, gate, expected) -> None:
    _, answer = fake_gate
    answer["value"] = gate
    decision, status, interaction_hash = _call(make_cbac_fn())
    assert (decision, status, interaction_hash) == (expected, gate[1], gate[2])


def test_each_agent_gets_its_own_intent() -> None:
    # One workflow id, two agents: two intents, so v2 never answers 4002.
    assert scoped_intent_id("QmW", "bafyA") != scoped_intent_id("QmW", "bafyB")
    assert scoped_intent_id("", "bafyA") == ""
    assert scoped_intent_id(None, "bafyA") == ""


# ── against a stub service on loopback ───────────────────────────────────────
class _Stub(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict, dict]]] = []
    reply: ClassVar[dict] = {}

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, dict(self.headers), body))
        payload = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args) -> None:
        pass


@pytest.fixture
def stub() -> Iterator[tuple[str, type[_Stub]]]:
    handler = type("Handler", (_Stub,), {"requests": [], "reply": {}})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", handler
    server.shutdown()
    server.server_close()


def test_adapter_posts_to_the_v2_route(stub) -> None:
    url, handler = stub
    handler.reply = {
        "success": True,
        "message": "ok",
        "data": {"decision": "allow", "status_code": 4301, "hash": "abc"},
    }
    assert _call(make_cbac_fn(url, 5)) == ("allow", 4301, "abc")
    path, _, body = handler.requests[0]
    assert path == "/cbac/v2/authorize"
    assert body["agent_id"] == "bafyagent"
    assert body["intent_id"] == "QmWorkflow:bafyagent"
    assert body["user_query"] == "List the open issues"
    assert body["arguments"] == {"repository": "octo/repo", "limit": 5}
    assert body["mcp_did"] == "bafytool"


def test_unreachable_service_fails_closed(monkeypatch) -> None:
    # Nothing listens on port 9 of loopback: no verdict, so no allow.
    decision, status, _ = _call(make_cbac_fn("http://127.0.0.1:9", 2))
    assert decision != "allow" and status not in adapter.ALLOW_CODES


def test_policy_document_splits_back_into_its_parts() -> None:
    parsers = pytest.importorskip("cbac_service.parsers")
    tools = [{"name": "issues", "description": "d", "inputSchema": {"type": "object"}}]
    instructions, parsed = parsers.split_policy_document(
        policy_document("# Policy\n\nOnly the configured repository.", tools)
    )
    assert instructions == "# Policy\n\nOnly the configured repository."
    assert parsed == tools


def test_policy_document_refuses_a_separator_in_the_instructions() -> None:
    with pytest.raises(ValueError, match="---"):
        policy_document("intro\n---\nmore", [])


def test_upload_policy_posts_the_document_with_the_key(stub) -> None:
    url, handler = stub
    handler.reply = {"success": True, "message": "ok", "data": {"version_id": 1}}
    reply = upload_policy(
        "bafyagent",
        "DOC",
        trusted_data={"repository": "o/r"},
        cbac_url=url,
        policy_key="k",
    )
    assert reply["data"] == {"version_id": 1}
    path, headers, body = handler.requests[0]
    assert path == "/cbac/v2/policies"
    assert headers["X-CBAC-Key"] == "k"
    assert body == {
        "agent_id": "bafyagent",
        "policy": "DOC",
        "trusted_data": {"repository": "o/r"},
        "review_required": False,
    }


def test_upload_policy_needs_the_key(monkeypatch) -> None:
    monkeypatch.delenv("CBAC_POLICY_KEY", raising=False)
    with pytest.raises(ValueError, match="CBAC_POLICY_KEY"):
        upload_policy("bafyagent", "DOC")


def test_actor_did_is_the_did_the_agent_will_have(local_dir, tmp_path) -> None:
    from agentdna.core import AgentDNA

    did = actor_did("policy-agent")
    policy = tmp_path / "SKILLS.md"
    policy.write_text("# Skills\n", encoding="utf-8")
    agent = AgentDNA(name="policy-agent", type="agent", agent_policy_file=policy)
    assert agent.get_actor_id() == did


def test_server_tools_lists_a_fastmcp_server() -> None:
    fastmcp = pytest.importorskip("fastmcp")
    from agentdna_local import server_tools

    server = fastmcp.FastMCP("t")

    @server.tool()
    def issues(repository: str, limit: int = 20) -> list:
        """Return open issues."""
        return []

    [tool] = asyncio.run(server_tools(server))
    assert tool["name"] == "issues"
    assert tool["description"] == "Return open issues."
    assert set(tool["inputSchema"]["properties"]) == {"repository", "limit"}
