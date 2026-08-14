import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("sentence_transformers")

from cbac_service import main
from cbac_service.cbac import CBACResult


def stub_request(body):
    async def _json():
        return body

    return SimpleNamespace(json=_json)


def install_cbac(monkeypatch, **attrs):
    cbac = SimpleNamespace(**attrs)
    monkeypatch.setattr(main, "_get_cbac", lambda: cbac)

    # The authorize/precompute endpoints open get_session(); yield a dummy
    # session so they run without a real DB (the stubbed cbac ignores it).
    @asynccontextmanager
    async def _fake_get_session():
        yield SimpleNamespace()

    monkeypatch.setattr(main, "get_session", _fake_get_session)
    return cbac


def authorizing(result, calls=None):
    async def verify_cbac(session, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return result

    return verify_cbac


AUTHORIZE_BODY = {
    "agent_id": "did:agent",
    "intended_action": "read pull requests",
    "user_intent": "show me the PRs",
    "callee_name": "github_tool",
    "callee_type": "tool",
}


def test_authorize_returns_decision_and_reason_only(monkeypatch):
    """The component scores stay server-side now — they are folded into trust
    while deciding, so nothing rides back on the wire but the verdict."""
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(
            CBACResult(
                decision="allow",
                reason="Tier 1 allow",
                intent_score=0.9,
                policy_score=0.8,
                hallucination_score=0.95,
                trust=0.87,
            )
        ),
    )
    response = asyncio.run(main.authorize_cbac(stub_request(AUTHORIZE_BODY)))

    assert response.headers["X-CBAC-Decision"] == "allow"
    assert response.body == b"Tier 1 allow"
    assert not any(
        h.startswith("x-cbac-") and h != "x-cbac-decision" for h in response.headers
    )


def test_authorize_forwards_the_callee_edge(monkeypatch):
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    asyncio.run(main.authorize_cbac(stub_request(AUTHORIZE_BODY)))

    assert calls == [
        {
            "agent_id": "did:agent",
            "intended_action": "read pull requests",
            "user_intent": "show me the PRs",
            "callee_name": "github_tool",
            "callee_type": "tool",
        }
    ]


def test_authorize_defaults_the_callee_edge(monkeypatch):
    """An older client that sends neither field still authorizes; the empty
    callee_name is what tells verify_cbac to skip the trust update."""
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    asyncio.run(main.authorize_cbac(stub_request({"agent_id": "did:agent"})))

    assert calls[0]["callee_name"] == ""
    assert calls[0]["callee_type"] == "tool"


def test_authorize_failure_fails_closed(monkeypatch):
    async def boom(session, **kwargs):
        raise RuntimeError("policy lookup failed")

    install_cbac(monkeypatch, verify_cbac=boom)
    response = asyncio.run(main.authorize_cbac(stub_request(AUTHORIZE_BODY)))

    assert response.headers["X-CBAC-Decision"] == "error"
    assert b"policy lookup failed" in response.body


def test_compute_lhi_endpoint_is_gone():
    """Trust is folded in by /authorize-cbac; there is no second call."""
    assert not hasattr(main, "compute_lhi")
    assert "/compute-lhi" not in {route.path for route in main.app.routes}


def _recording_precompute(calls, result=7):
    async def precompute_policy(session, agent_id, policy):
        calls.append({"agent_id": agent_id, "policy": policy})
        return result

    return precompute_policy


def test_precompute_forwards_supplied_policy(monkeypatch):
    calls = []
    install_cbac(monkeypatch, index_policy=_recording_precompute(calls))
    response = asyncio.run(
        main.precompute_policy(
            stub_request({"agent_id": "did:agent", "policy": "allowed-actions: read"})
        )
    )

    assert calls == [{"agent_id": "did:agent", "policy": "allowed-actions: read"}]
    assert json.loads(response.body) == {"agent_id": "did:agent", "chunks_stored": 7}


def test_precompute_without_policy_passes_none(monkeypatch):
    # None is the signal to read the agent's card from the Provenance Layer.
    calls = []
    install_cbac(monkeypatch, index_policy=_recording_precompute(calls))
    asyncio.run(main.precompute_policy(stub_request({"agent_id": "did:agent"})))

    assert calls == [{"agent_id": "did:agent", "policy": None}]


def test_precompute_rejects_non_string_policy(monkeypatch):
    async def never(session, agent_id, policy):
        raise AssertionError("should not reach the engine")

    install_cbac(monkeypatch, precompute_policy=never)
    response = asyncio.run(
        main.precompute_policy(
            stub_request({"agent_id": "did:agent", "policy": {"a": 1}})
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "policy must be a string"
