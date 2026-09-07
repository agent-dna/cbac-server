import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("sentence_transformers")

from fastapi.testclient import TestClient

from cbac_service import error_codes as ec
from cbac_service import main
from cbac_service.cbac import CBACResult
from cbac_service.entity import (
    MAX_LHI_AGENTS,
    AuthorizeRequest,
    LHIScoresRequest,
    PrecomputePolicyRequest,
)


def stub_request(body):
    """The authorize body as the model FastAPI would have parsed it."""
    return AuthorizeRequest(**body)


def envelope(response):
    """The {success, message, data} envelope every endpoint answers with."""
    return json.loads(response.body)


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
    assert envelope(response) == {
        "success": True,
        "message": "Tier 1 allow",
        "data": {"decision": "allow", "status_code": None, "hash": None},
    }
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
            "intent_id": None,
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


def test_authorize_renders_the_intent_from_the_call_facts(monkeypatch):
    """An enforcement point sends facts; the phrasing happens here.

    This is the whole point of rendering server-side — every PEP, whatever
    protocol or language it speaks, gets the same sentence for the same call,
    so none of them can word its way to a different verdict.
    """
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    asyncio.run(
        main.authorize_cbac(
            stub_request(
                {
                    "agent_id": "did:agent",
                    "callee_name": "github_close_issue",
                    "callee_description": "Close an existing GitHub issue.",
                    "arguments": {"repo": "owner/repo", "number": "3"},
                    "user_intent": "close issue 3",
                }
            )
        )
    )

    assert calls[0]["intended_action"] == (
        "The agent wants to close an existing GitHub issue, "
        "with repo = owner/repo, number = 3."
    )


def test_authorize_falls_back_to_the_callee_name_as_the_verb(monkeypatch):
    """No description: the de-snaked name carries the verb instead."""
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    asyncio.run(
        main.authorize_cbac(
            stub_request({"agent_id": "did:agent", "callee_name": "close_issue"})
        )
    )

    assert calls[0]["intended_action"] == "The agent wants to close issue."


def test_authorize_keeps_a_pre_rendered_intended_action(monkeypatch):
    """A caller with phrasing of its own keeps it, rendering skipped entirely
    even when the fields to render from are all present."""
    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    asyncio.run(
        main.authorize_cbac(
            stub_request(
                {
                    "agent_id": "did:agent",
                    "intended_action": "wire $5000 to account 12345",
                    "callee_name": "transfer_funds",
                    "callee_description": "Transfer money between accounts.",
                }
            )
        )
    )

    assert calls[0]["intended_action"] == "wire $5000 to account 12345"


def test_guard_payload_round_trips_into_the_rendered_intent(monkeypatch):
    """The guard's body and the endpoint's reader, pinned to each other.

    They are the two halves of one contract and live in separate packages, so
    a renamed key fails *silently*: the endpoint just stops seeing a
    description and quietly scores the de-snaked name instead. Building the
    body with the real guard and feeding it to the real endpoint is what
    catches that.
    """
    from cbac.authorize import _payload

    calls = []
    install_cbac(
        monkeypatch,
        verify_cbac=authorizing(CBACResult(decision="allow", reason="ok"), calls),
    )
    body = _payload(
        "did:agent",
        "github_close_issue",
        # A value json.dumps cannot encode: the guard renders arguments to text
        # before they hit the wire, so this must survive rather than fail closed.
        {"repo": "owner/repo", "when": datetime(2026, 8, 17, tzinfo=timezone.utc)},
        "close issue 3",
        "Close an existing GitHub issue.",
        "mcp",
    )
    json.dumps(body)  # the guard POSTs this; unserializable would fail closed
    asyncio.run(main.authorize_cbac(stub_request(body)))

    assert calls[0]["intended_action"] == (
        "The agent wants to close an existing GitHub issue, "
        "with repo = owner/repo, when = 2026-08-17 00:00:00+00:00."
    )
    assert calls[0]["callee_name"] == "github_close_issue"
    assert calls[0]["callee_type"] == "mcp"


def test_guard_endpoint_matches_the_route_the_service_serves(monkeypatch):
    """The other half of the same contract: the path the guard POSTs to has to
    be a route this app actually registers, prefix and all."""
    from cbac.authorize import _authorize

    monkeypatch.delenv("CBAC_PATH", raising=False)
    posted = []

    class _Response:
        text = ""

        @staticmethod
        def json():
            return {"data": {"decision": "allow"}, "message": ""}

    monkeypatch.setattr(
        "requests.post", lambda url, **kw: (posted.append(url), _Response())[1]
    )

    asyncio.run(_authorize({}, cbac_url="http://svc:8767"))
    assert posted[-1] == "http://svc:8767/cbac/v1/authorize"
    # openapi() rather than app.routes: an included router stays nested there,
    # so app.routes lists the router, not the paths it serves.
    assert "/cbac/v1/authorize" in main.app.openapi()["paths"]

    # A service mounted under an ingress prefix: one slash, not two.
    asyncio.run(_authorize({}, cbac_url="https://gw.example.com/agentdna/"))
    assert posted[-1] == "https://gw.example.com/agentdna/cbac/v1/authorize"


def test_guard_without_a_configured_service_fails_closed(monkeypatch):
    """No CBAC_URL means nowhere to ask, which must never read as permission."""
    from cbac.authorize import _authorize

    monkeypatch.delenv("CBAC_URL", raising=False)
    result = asyncio.run(_authorize({}))

    assert result.decision == "error"  # every caller treats non-"allow" as blocked
    assert "no decision service configured" in result.message


def test_authorize_failure_fails_closed(monkeypatch):
    async def boom(session, **kwargs):
        raise RuntimeError("policy lookup failed")

    install_cbac(monkeypatch, verify_cbac=boom)
    response = asyncio.run(main.authorize_cbac(stub_request(AUTHORIZE_BODY)))

    assert response.headers["X-CBAC-Decision"] == "error"
    body = envelope(response)
    assert body["data"]["decision"] == "error"
    assert "policy lookup failed" in body["message"]
    # The pipeline blew up, so the request did not succeed -- but still HTTP
    # 200, because a 5xx invites the retry that double-folds trust.
    assert body["success"] is False
    assert response.status_code == 200


def test_compute_lhi_endpoint_is_gone():
    """Trust is folded in by /cbac/v1/authorize; there is no second call."""
    assert not hasattr(main, "compute_lhi")
    assert not any("compute-lhi" in path for path in main.app.openapi()["paths"])


# ── Decision audit log ────────────────────────────────────────────────────────


def decision_row(record_id=1, **overrides):
    fields = {
        "id": record_id,
        "agent_id": "did:agent",
        "decision": "allow",
        "reason": "Tier 1 allow",
        "intended_action": "The agent wants to read pull requests.",
        "user_intent": "show me the PRs",
        "callee_name": "github_tool",
        "callee_type": "tool",
        "error_code": ec.TIER1_GAP_ALLOW,
        # Per-row unique in the real table (uuid4-salted), so vary it by id.
        "interaction_hash": f"{record_id:064x}",
        "intent_id": None,
        "created_at": datetime(2026, 8, 17, 9, 14, 22, tzinfo=timezone.utc),
    }
    return SimpleNamespace(**{**fields, **overrides})


def install_repository(monkeypatch, calls=None, records=(), one=None):
    @asynccontextmanager
    async def _fake_get_session():
        yield SimpleNamespace()

    async def fake_get_cbac_decisions(session, agent_id, limit, offset):
        if calls is not None:
            calls.append({"agent_id": agent_id, "limit": limit, "offset": offset})
        return list(records)

    async def fake_get_cbac_decision(session, decision_id):
        return one

    monkeypatch.setattr(main, "get_session", _fake_get_session)
    monkeypatch.setattr(main, "get_cbac_decisions", fake_get_cbac_decisions)
    monkeypatch.setattr(main, "get_cbac_decision", fake_get_cbac_decision)


def list_decisions(**params):
    """Over the real request path: the paging defaults and their bounds are
    declared as Query(...), so a direct call would never resolve them."""
    return TestClient(main.app).get(f"{main.API_PREFIX}/decisions", params=params)


def test_list_decisions_returns_serialized_rows(monkeypatch):
    install_repository(monkeypatch, records=[decision_row(2), decision_row(1)])
    payload = list_decisions(agent_id="did:agent").json()["data"]

    assert payload["agent_id"] == "did:agent"
    assert payload["count"] == 2
    assert [d["id"] for d in payload["decisions"]] == [2, 1]  # newest first
    assert payload["decisions"][0]["created_at"] == "2026-08-17T09:14:22+00:00"
    assert payload["decisions"][0]["reason"] == "Tier 1 allow"


def test_list_decisions_defaults_and_forwards_paging(monkeypatch):
    calls = []
    install_repository(monkeypatch, calls=calls)
    list_decisions(agent_id="did:agent")
    list_decisions(agent_id="did:agent", limit=10, offset=20)

    assert calls == [
        {"agent_id": "did:agent", "limit": 100, "offset": 0},
        {"agent_id": "did:agent", "limit": 10, "offset": 20},
    ]


def test_list_decisions_clamps_limit(monkeypatch):
    """An oversized page is clamped, not rejected — a caller asking for
    everything gets a large page rather than a 422 to learn about."""
    calls = []
    install_repository(monkeypatch, calls=calls)
    list_decisions(agent_id="did:agent", limit=100000)

    assert calls[0]["limit"] == main.MAX_DECISION_LIMIT


@pytest.mark.parametrize(
    "params",
    [
        {},  # agent_id is required
        {"agent_id": ""},  # ...and must not be blank
        {"agent_id": "did:agent", "limit": "abc"},  # limit must be an int
        {"agent_id": "did:agent", "limit": 0},  # ...and at least 1
        {"agent_id": "did:agent", "offset": -1},  # offset must not be negative
    ],
)
def test_list_decisions_rejects_bad_query(params):
    """Driven over the real request path: FastAPI validates before the handler
    runs, so calling the handler directly would skip the thing under test."""
    response = TestClient(main.app).get(f"{main.API_PREFIX}/decisions", params=params)

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["message"] == "request validation failed"
    # Same envelope as every other response, not FastAPI's own {"detail": ...}.
    assert set(body) == {"success", "message", "data"}


def test_read_decision_returns_the_row(monkeypatch):
    install_repository(monkeypatch, one=decision_row(42, decision="deny"))
    response = asyncio.run(main.read_cbac_decision(42))

    payload = envelope(response)["data"]
    assert payload["id"] == 42
    assert payload["decision"] == "deny"


def test_read_decision_404s_when_absent(monkeypatch):
    install_repository(monkeypatch, one=None)
    response = asyncio.run(main.read_cbac_decision(999))

    assert response.status_code == 404
    assert envelope(response)["success"] is False
    assert "999" in envelope(response)["message"]


# ── LHI trust scores ──────────────────────────────────────────────────────────


def lhi_row(
    agent_id="did:agent", callee_name="github_tool", callee_type="tool", **overrides
):
    fields = {
        "agent_id": agent_id,
        "callee_name": callee_name,
        "callee_type": callee_type,
        "intent_score": 0.9,
        "policy_score": 0.8,
        "hallucination_score": 0.95,
        "trust": 0.87,
        "created_at": datetime(2026, 8, 17, 9, 14, 22, tzinfo=timezone.utc),
    }
    return SimpleNamespace(**{**fields, **overrides})


def install_lhi_repository(monkeypatch, calls=None, records=()):
    @asynccontextmanager
    async def _fake_get_session():
        yield SimpleNamespace()

    async def fake_get_latest_trust_for_agents(session, agent_ids):
        if calls is not None:
            calls.append(list(agent_ids))
        return list(records)

    monkeypatch.setattr(main, "get_session", _fake_get_session)
    monkeypatch.setattr(
        main, "get_latest_trust_for_agents", fake_get_latest_trust_for_agents
    )


def test_lhi_scores_groups_edges_by_agent(monkeypatch):
    install_lhi_repository(
        monkeypatch,
        records=[
            lhi_row("did:a", callee_name="github_tool"),
            lhi_row("did:a", callee_name="slack_tool", trust=0.5),
            lhi_row("did:b"),
        ],
    )
    response = asyncio.run(
        main.read_lhi_scores(LHIScoresRequest(agent_ids=["did:a", "did:b"]))
    )

    payload = envelope(response)["data"]
    assert {e["callee_name"] for e in payload["agents"]["did:a"]} == {
        "github_tool",
        "slack_tool",
    }
    assert len(payload["agents"]["did:b"]) == 1
    assert payload["agents"]["did:b"][0]["trust"] == 0.87


def test_lhi_scores_includes_agents_with_no_history(monkeypatch):
    """An agent with no trust record yet is still a key, mapped to []."""
    install_lhi_repository(monkeypatch, records=[lhi_row("did:a")])
    response = asyncio.run(
        main.read_lhi_scores(LHIScoresRequest(agent_ids=["did:a", "did:no-history"]))
    )

    assert envelope(response)["data"]["agents"]["did:no-history"] == []


def test_lhi_scores_forwards_requested_ids(monkeypatch):
    calls = []
    install_lhi_repository(monkeypatch, calls=calls)
    asyncio.run(main.read_lhi_scores(LHIScoresRequest(agent_ids=["did:a", "did:b"])))

    assert calls == [["did:a", "did:b"]]


@pytest.mark.parametrize(
    "body",
    [
        {},  # agent_ids is required
        {"agent_ids": []},  # ...and must not be empty
        {"agent_ids": "did:agent"},  # ...and must be a list, not a string
        {"agent_ids": ["did:a", 42]},  # entries must be strings
        {"agent_ids": ["did:a", ""]},  # ...and not blank
        {"agent_ids": [f"did:{i}" for i in range(MAX_LHI_AGENTS + 1)]},  # too many
    ],
)
def test_lhi_scores_rejects_bad_batch(body):
    """Schema-enforced now, so this goes over the real request path — and the
    engine is deliberately not stubbed: nothing may reach it."""
    response = TestClient(main.app).post(f"{main.API_PREFIX}/lhi-scores", json=body)

    assert response.status_code == 422
    assert response.json()["success"] is False


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
            PrecomputePolicyRequest(
                agent_id="did:agent", policy="allowed-actions: read"
            )
        )
    )

    assert calls == [{"agent_id": "did:agent", "policy": "allowed-actions: read"}]
    assert envelope(response)["data"] == {"agent_id": "did:agent", "chunks_stored": 7}


def test_precompute_without_policy_passes_none(monkeypatch):
    # None is the signal to read the agent's card from the Provenance Layer.
    calls = []
    install_cbac(monkeypatch, index_policy=_recording_precompute(calls))
    asyncio.run(main.precompute_policy(PrecomputePolicyRequest(agent_id="did:agent")))

    assert calls == [{"agent_id": "did:agent", "policy": None}]


def test_precompute_rejects_non_string_policy(monkeypatch):
    """The engine must never see a non-string policy — the schema stops it
    before the handler, so nothing is stubbed here on purpose."""
    response = TestClient(main.app).post(
        f"{main.API_PREFIX}/policies/precompute",
        json={"agent_id": "did:agent", "policy": {"a": 1}},
    )

    assert response.status_code == 422
    assert response.json()["success"] is False
