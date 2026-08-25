"""The ext_authz adapter: what an enterprise gateway sends, and what it gets.

The gateway forwards the request it is about to proxy and enforces the status
code. These pin that contract — the CBAC pipeline itself is faked, since what
is under test is the wire handling, not the decision.
"""

import asyncio
import json

import pytest

pytest.importorskip("sentence_transformers")

from cbac_service import main
from cbac_service.cbac import CBACResult
from cbac_service.tests.test_main_endpoints import authorizing, install_cbac


def gateway_request(body, headers=None):
    """What the gateway forwards: the raw body plus the headers it set."""

    async def _body():
        return body if isinstance(body, bytes) else json.dumps(body).encode()

    class _Request:
        def __init__(self):
            self.headers = {k.lower(): v for k, v in (headers or {}).items()}

        body = staticmethod(_body)

    return _Request()


CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "close_issue", "arguments": {"repo": "owner/repo", "number": 3}},
}
HEADERS = {
    "X-CBAC-Agent-Id": "github-worker",
    "X-CBAC-User-Intent": "Close%20issue%20%233",
}


def call(body, headers=None):
    return asyncio.run(main.ext_authz(gateway_request(body, headers)))


def test_allow_is_2xx_and_deny_is_403(monkeypatch):
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    response = call(CALL, HEADERS)
    assert response.status_code == 200
    assert response.headers["X-CBAC-Decision"] == "allow"

    # The tool name and arguments are read off the wire and rendered as prose,
    # exactly as the guard would have phrased them.
    assert calls[0]["intended_action"] == (
        "The agent wants to close issue, with repo = owner/repo, number = 3."
    )
    assert calls[0]["user_intent"] == "Close issue #3"  # header was unquoted
    assert calls[0]["callee_name"] == "close_issue"

    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="deny", reason="forbidden")),
    )
    response = call(CALL, HEADERS)
    assert response.status_code == 403
    assert response.headers["X-CBAC-Decision"] == "deny"


def test_tool_description_header_sharpens_the_prose(monkeypatch):
    """Without it the de-snaked name is the verb; with it, the tool's own words."""
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    call(CALL, {**HEADERS, "X-CBAC-Tool-Description": "Close an existing GitHub issue"})
    assert calls[0]["intended_action"].startswith(
        "The agent wants to close an existing GitHub issue"
    )


def test_non_tool_calls_pass_through(monkeypatch):
    """initialize, tools/list and empty session traffic are not tool calls."""
    install_cbac(monkeypatch, verify_cbac=authorizing(None))
    for body in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        b"",
    ):
        assert call(body).status_code == 200


def test_refuses_what_it_cannot_read(monkeypatch):
    """Fail-closed on anything that might be a tool call in disguise."""
    install_cbac(monkeypatch, verify_cbac=authorizing(None))
    assert call(b"not json at all").status_code == 403
    assert call([CALL, CALL]).status_code == 403  # batched JSON-RPC


def test_missing_identity_or_context_is_refused(monkeypatch):
    install_cbac(monkeypatch, verify_cbac=authorizing(None))

    no_id = {"X-CBAC-User-Intent": "Close%20issue%20%233"}
    assert call(CALL, no_id).status_code == 403

    no_intent = {"X-CBAC-Agent-Id": "github-worker"}
    assert call(CALL, no_intent).status_code == 403


def test_context_can_be_made_optional(monkeypatch):
    """CBAC_REQUIRE_CONTEXT=false degrades to a policy-only check."""
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    monkeypatch.setattr(main, "REQUIRE_CONTEXT", False)
    assert call(CALL, {"X-CBAC-Agent-Id": "github-worker"}).status_code == 200
    assert calls[0]["user_intent"] is None
