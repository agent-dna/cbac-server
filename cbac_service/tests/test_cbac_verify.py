import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

import cbac_service.cbac as cbac_mod
from cbac_service import error_codes as ec
from cbac_service.cbac import CBAC

AGENT_ID = "did:agent"


CALLEE = {"callee_name": "github_tool", "callee_type": "tool"}


def _make_provenance(tmp_path, policy_text="Agents may read pull requests."):
    policy_b64 = base64.b64encode(policy_text.encode()).decode()
    return SimpleNamespace(
        config_dir=str(tmp_path),
        get_latest_provenance_record=lambda actor_id: {
            "type": "agent",
            "id": actor_id,
            "metadata": {},
            "policy": policy_b64,
        },
    )


def make_verify_cbac(
    tmp_path, monkeypatch, policy_text="Agents may read pull requests."
):
    """CBAC instance with the policy pipeline stubbed out (no NLI/encoder model
    load), so it deterministically falls through Tier 1/2/3 to Tier 3's
    no-backend "advise" — leaving `hallucination_score` as the only real model
    call, which is the thing under test here.
    """
    cbac = CBAC(provenance=_make_provenance(tmp_path, policy_text))
    monkeypatch.setattr(cbac, "_classify_chunks", lambda chunks: (chunks, []))
    monkeypatch.setattr(
        cbac,
        "_get_encoder",
        lambda: SimpleNamespace(
            encode=lambda texts, normalize_embeddings=True: np.zeros((len(texts), 4))
        ),
    )
    monkeypatch.setattr(
        cbac,
        "_nli_scores",
        lambda premise, hypothesis: {"contradiction": 0.0, "entailment": 0.0},
    )

    # DB search layer stubbed to in-memory so the pipeline runs without Postgres:
    # cache always fresh, one dummy chunk, and equal allowed/forbidden scores
    # (gap 0) so Tier 1 escalates to Tier 2, which — NLI stubbed — reaches Tier 3.
    async def _hash_matches(session, agent_id, policy_hash):
        return True

    async def _get_chunks(session, agent_id):
        return ["allowed chunk"]

    async def _search(
        session, agent_id, intent_vec, *args, chunk_type="allowed", **kwargs
    ):
        return [SimpleNamespace(score=0.5, chunk_text="allowed chunk")]

    monkeypatch.setattr(cbac_mod, "policy_hash_matches", _hash_matches)
    monkeypatch.setattr(cbac_mod, "get_policy_chunks", _get_chunks)
    monkeypatch.setattr(cbac_mod, "vector_search", _search)
    monkeypatch.setattr(cbac_mod, "hybrid_search", _search)
    return cbac


def test_hallucination_score_attached_when_reached(
    rows, decisions, tmp_path, monkeypatch
):
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
        )
    )
    assert result.decision == "advise"
    assert result.hallucination_score is not None
    assert 0.0 <= result.hallucination_score <= 1.0
    # Check-1 always runs when user_intent is supplied, so intent_score is set...
    assert result.intent_score is not None
    assert 0.0 <= result.intent_score <= 1.0
    # ...but this pipeline is stubbed to fall through to Tier 3, which has no
    # numeric signal, so policy_score stays None.
    assert result.policy_score is None


def test_hallucination_score_none_without_user_intent(
    rows, decisions, tmp_path, monkeypatch
):
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent=None,
        )
    )
    assert result.decision == "advise"
    assert result.hallucination_score is None
    assert result.intent_score is None


def test_hallucination_scoring_failure_does_not_change_decision(
    rows, decisions, tmp_path, monkeypatch
):
    cbac = make_verify_cbac(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("HHEM unavailable")

    monkeypatch.setattr(cbac, "hallucination_score", boom)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
        )
    )
    assert result.decision == "advise"
    assert result.hallucination_score is None


def test_intent_score_is_one_minus_contradiction(tmp_path, monkeypatch):
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cbac,
        "_nli_scores",
        lambda premise, hypothesis: {"contradiction": 0.3, "entailment": 0.1},
    )
    drift, intent_score = asyncio.run(cbac._check1_drift("user intent", "agent action"))
    assert drift is None  # 0.3 < contradiction_threshold, no deny
    assert intent_score == pytest.approx(0.7)


def test_policy_score_normalized_on_tier1_decision(tmp_path, monkeypatch):
    cbac = make_verify_cbac(tmp_path, monkeypatch)

    # Tier-1 search: allowed cosine 1.0, forbidden 0.0 -> gap 1.0, clamps above allow_gap.
    async def _vsearch(
        session, agent_id, intent_vec, *args, chunk_type="allowed", **kwargs
    ):
        score = 1.0 if chunk_type == "allowed" else 0.0
        return [SimpleNamespace(score=score, chunk_text=f"{chunk_type} chunk")]

    monkeypatch.setattr(cbac_mod, "vector_search", _vsearch)

    decision, _reason, error_code, policy_score = asyncio.run(
        cbac._tiered_decision(None, AGENT_ID, "intent text")
    )
    assert decision == "allow"
    assert error_code == ec.TIER1_GAP_ALLOW
    assert policy_score == pytest.approx(1.0)


def test_hallucination_score_not_computed_on_early_hard_fail(
    rows, decisions, tmp_path, monkeypatch
):
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cbac,
        "hallucination_score",
        lambda *a, **k: pytest.fail(
            "hallucination_score must not run on an early hard-fail"
        ),
    )
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="",
            user_intent="Please show me the pull requests",
        )
    )
    assert result.decision == "deny"
    assert result.hallucination_score is None


# ── Trust folded in at decision time ──────────────────────────────────────────


def test_reached_decision_folds_trust(rows, decisions, tmp_path, monkeypatch):
    """A decision — here the Tier-3 'advise' gray zone — writes one row and
    stamps the resulting trust onto the result. No second call needed."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
            **CALLEE,
        )
    )
    assert result.decision == "advise"
    assert result.trust is not None
    assert 0.0 <= result.trust <= 1.0
    assert len(rows) == 1
    assert (rows[0].callee_name, rows[0].callee_type) == ("github_tool", "tool")
    # Tier 3 produces no numeric policy signal — stored NULL, renormalized away.
    assert rows[0].policy_score is None
    assert rows[0].trust == pytest.approx(result.trust)


def test_drift_deny_also_records(rows, decisions, tmp_path, monkeypatch):
    """A denied call never runs, but it is still evidence: an agent probing
    forbidden actions must not keep pristine trust."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cbac,
        "_nli_scores",
        lambda premise, hypothesis: {"contradiction": 0.9, "entailment": 0.0},
    )
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="close all pull requests",
            user_intent="Do not close anything",
            **CALLEE,
        )
    )
    assert result.decision == "deny"
    # Only Check-1 ran, so intent is the single observed component.
    assert result.trust == pytest.approx(1.0 - 0.9)
    assert len(rows) == 1
    assert rows[0].policy_score is None and rows[0].hallucination_score is None


def test_no_callee_name_folds_no_trust_but_is_audited(
    rows, decisions, tmp_path, monkeypatch
):
    """An empty edge key would pool every caller into one junk edge, so no
    trust row — but the verdict is still audited. This asymmetry is the whole
    reason cbac_decisions exists separately from lhi_records."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None, agent_id=AGENT_ID, intended_action="read pull requests"
        )
    )
    assert result.decision == "advise"
    assert result.trust is None
    assert rows == []
    assert len(decisions) == 1
    assert decisions[0].callee_name is None


def test_early_hard_fail_folds_no_trust_but_is_audited(
    rows, decisions, tmp_path, monkeypatch
):
    """Nothing was measured, so there is no honest trust update — but a deny
    was still issued and has to be answerable for."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(session=None, agent_id=AGENT_ID, intended_action="", **CALLEE)
    )
    assert result.decision == "deny"
    assert result.trust is None
    assert rows == []
    assert len(decisions) == 1
    assert decisions[0].decision == "deny"


def test_trust_failure_does_not_change_decision(rows, decisions, tmp_path, monkeypatch):
    """A DB hiccup during the trust write must never turn a valid decision
    into a blocked call."""
    import cbac_service.cbac as mod

    cbac = make_verify_cbac(tmp_path, monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mod, "insert_lhi_record", boom)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
            **CALLEE,
        )
    )
    assert result.decision == "advise"
    assert result.trust is None


# ── Decision audit log ────────────────────────────────────────────────────────


def test_decision_is_recorded_with_its_context(rows, decisions, tmp_path, monkeypatch):
    """A logged row has to explain itself: what was attempted, on whose behalf,
    against which callee — not just the verdict."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
            **CALLEE,
        )
    )

    assert len(decisions) == 1
    record = decisions[0]
    assert record.agent_id == AGENT_ID
    assert record.decision == result.decision
    assert record.reason == result.reason
    assert record.intended_action == "read pull requests"
    assert record.user_intent == "Please show me the pull requests"
    assert (record.callee_name, record.callee_type) == ("github_tool", "tool")


def test_decision_records_the_flattened_action(rows, decisions, tmp_path, monkeypatch):
    """A structured intended_action is stored as the text the scorers saw, so
    the row justifies the verdict rather than restating the request."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action={"action": "read", "params": {"repo": "payments"}},
            **CALLEE,
        )
    )

    stored = decisions[0].intended_action
    assert "read" in stored and "payments" in stored
    # Flattened, not repr'd — no dict syntax leaks into the audit trail.
    assert "{" not in stored


def test_drift_deny_is_recorded(rows, decisions, tmp_path, monkeypatch):
    cbac = make_verify_cbac(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cbac,
        "_nli_scores",
        lambda premise, hypothesis: {"contradiction": 0.9, "entailment": 0.0},
    )
    asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="close all pull requests",
            user_intent="Do not close anything",
            **CALLEE,
        )
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "deny"
    assert "drift" in decisions[0].reason.lower()


def test_infra_failure_deny_is_recorded(rows, decisions, tmp_path, monkeypatch):
    """The path that folds no trust because it is not the agent's fault still
    produced a deny the caller has to be able to explain later."""
    cbac = make_verify_cbac(tmp_path, monkeypatch)

    def boom(agent_id):
        raise RuntimeError("provenance node down")

    monkeypatch.setattr(cbac, "_get_latest_agent_policy", boom)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            **CALLEE,
        )
    )

    assert result.decision == "deny"
    assert rows == []  # not the agent's fault — no trust penalty
    assert len(decisions) == 1
    assert "provenance node down" in decisions[0].reason


def test_record_failure_does_not_change_decision(
    rows, decisions, tmp_path, monkeypatch
):
    """Failing to write down what was decided must not change what was
    decided — the audit write is never a gate."""
    import cbac_service.cbac as mod

    cbac = make_verify_cbac(tmp_path, monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mod, "insert_cbac_decision", boom)
    result = asyncio.run(
        cbac.verify_cbac(
            session=None,
            agent_id=AGENT_ID,
            intended_action="read pull requests",
            user_intent="Please show me the pull requests",
            **CALLEE,
        )
    )

    assert result.decision == "advise"
    assert result.trust is not None  # trust fold still happened
    assert decisions == []
