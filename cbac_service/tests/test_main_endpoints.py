import asyncio
import base64
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("sentence_transformers")

from agentdna.types import AgentCard

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


# ── Decision audit log ────────────────────────────────────────────────────────


def stub_query_request(**params):
    """A Request whose only used surface is `.query_params`."""
    return SimpleNamespace(query_params={k: str(v) for k, v in params.items()})


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


def test_list_decisions_returns_serialized_rows(monkeypatch):
    install_repository(monkeypatch, records=[decision_row(2), decision_row(1)])
    response = asyncio.run(
        main.list_cbac_decisions(stub_query_request(agent_id="did:agent"))
    )

    payload = json.loads(response.body)
    assert payload["agent_id"] == "did:agent"
    assert payload["count"] == 2
    assert [d["id"] for d in payload["decisions"]] == [2, 1]  # newest first
    assert payload["decisions"][0]["created_at"] == "2026-08-17T09:14:22+00:00"
    assert payload["decisions"][0]["reason"] == "Tier 1 allow"


def test_list_decisions_defaults_and_forwards_paging(monkeypatch):
    calls = []
    install_repository(monkeypatch, calls=calls)
    asyncio.run(main.list_cbac_decisions(stub_query_request(agent_id="did:agent")))
    asyncio.run(
        main.list_cbac_decisions(
            stub_query_request(agent_id="did:agent", limit=10, offset=20)
        )
    )

    assert calls == [
        {"agent_id": "did:agent", "limit": 100, "offset": 0},
        {"agent_id": "did:agent", "limit": 10, "offset": 20},
    ]


def test_list_decisions_clamps_limit(monkeypatch):
    """An oversized page is clamped, not rejected — a caller asking for
    everything gets a large page rather than a 400 to learn about."""
    calls = []
    install_repository(monkeypatch, calls=calls)
    asyncio.run(
        main.list_cbac_decisions(stub_query_request(agent_id="did:agent", limit=100000))
    )

    assert calls[0]["limit"] == main.MAX_DECISION_LIMIT


def test_list_decisions_requires_agent_id(monkeypatch):
    install_repository(monkeypatch)
    response = asyncio.run(main.list_cbac_decisions(stub_query_request()))

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "agent_id is required"


def test_list_decisions_rejects_bad_paging(monkeypatch):
    install_repository(monkeypatch)
    bad_type = asyncio.run(
        main.list_cbac_decisions(stub_query_request(agent_id="did:agent", limit="abc"))
    )
    out_of_range = asyncio.run(
        main.list_cbac_decisions(stub_query_request(agent_id="did:agent", offset=-1))
    )

    assert bad_type.status_code == 400
    assert "integers" in json.loads(bad_type.body)["error"]
    assert out_of_range.status_code == 400


def test_read_decision_returns_the_row(monkeypatch):
    install_repository(monkeypatch, one=decision_row(42, decision="deny"))
    response = asyncio.run(main.read_cbac_decision(42))

    payload = json.loads(response.body)
    assert payload["id"] == 42
    assert payload["decision"] == "deny"


def test_read_decision_404s_when_absent(monkeypatch):
    install_repository(monkeypatch, one=None)
    response = asyncio.run(main.read_cbac_decision(999))

    assert response.status_code == 404
    assert "999" in json.loads(response.body)["error"]


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


# ── Local policy source (CBAC_LOCAL_POLICY_DIR) ───────────────────────────────


def test_local_policy_provenance_reads_agent_file(tmp_path):
    (tmp_path / "github-worker.md").write_text("# Policy\n\nMay open issues.\n")

    record = main._LocalPolicyProvenance(tmp_path).get_latest_provenance_record(
        "github-worker"
    )
    card = AgentCard(**record)

    assert card.id == "github-worker"
    assert base64.b64decode(card.policy).decode() == "# Policy\n\nMay open issues.\n"


def test_make_provenance_switches_on_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CBAC_LOCAL_POLICY_DIR", str(tmp_path))
    assert isinstance(main._make_provenance(), main._LocalPolicyProvenance)

    monkeypatch.delenv("CBAC_LOCAL_POLICY_DIR")
    assert not isinstance(main._make_provenance(), main._LocalPolicyProvenance)
