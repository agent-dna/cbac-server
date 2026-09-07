import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sentence_transformers")

from cbac_service.cbac import CBAC
from cbac_service.config import LHI_LAMBDA_DOWN, LHI_LAMBDA_UP, LHI_WEIGHTS

SCORES = {
    "intent_score": 0.9,
    "policy_score": 0.8,
    "hallucination_score": 0.95,
}

# compute_lhi only hands the session to the repository functions, which the
# `rows` fixture replaces with in-memory fakes — so an opaque stub suffices.
SESSION = SimpleNamespace()


def weighted_mean(intent_score, policy_score, hallucination_score) -> float:
    """Renormalized mean over the observed components: s = Σ wᵢxᵢ / Σ wᵢ.

    Callers always pass at least one observed value — the nothing-observed case
    is asserted through `compute` instead, which is where the production code
    decides to write no record at all.
    """
    observed = [
        (value, weight)
        for value, weight in zip(
            (intent_score, policy_score, hallucination_score), LHI_WEIGHTS, strict=True
        )
        if value is not None
    ]
    return sum(v * w for v, w in observed) / sum(w for _, w in observed)


def make_cbac(tmp_path):
    return CBAC(provenance=SimpleNamespace(config_dir=str(tmp_path)))  # type: ignore[arg-type]


def compute(cbac, callee_name="github_tool", callee_type="tool", **overrides):
    return asyncio.run(
        cbac.compute_lhi(
            SESSION, "did:agent", callee_name, callee_type, **{**SCORES, **overrides}
        )
    )


def trust_of(cbac, **kwargs) -> float:
    """`compute` for the paths that must yield a trust value — narrows away the
    `None` that only the nothing-observed path returns."""
    trust = compute(cbac, **kwargs)
    assert trust is not None
    return trust


def edge_history(records, callee_name, callee_type):
    return [
        r.trust
        for r in records
        if (r.callee_name, r.callee_type) == (callee_name, callee_type)
    ]


def test_first_interaction_returns_weighted_mean(rows, tmp_path):
    cbac = make_cbac(tmp_path)
    assert compute(cbac) == pytest.approx(weighted_mean(**SCORES))


def test_weights_are_scale_invariant(rows, tmp_path):
    """The renormalizing denominator means only the weight *ratios* matter, so
    dropping the old 4th (output) weight needed no retuning: (0.3, 0.3, 0.2)
    with all three observed is exactly the rescaled (0.375, 0.375, 0.25) mean.
    Checked independently of the implementation's zip, which weighted_mean
    mirrors."""
    expected = 0.375 * 0.9 + 0.375 * 0.8 + 0.25 * 0.95
    assert compute(make_cbac(tmp_path)) == pytest.approx(expected)
    assert sum(LHI_WEIGHTS) == pytest.approx(0.8)  # unnormalized on purpose


def test_improving_scores_raise_trust_slowly(rows, tmp_path):
    cbac = make_cbac(tmp_path)
    prev = trust_of(cbac, **{k: 0.5 for k in SCORES})
    trust = trust_of(cbac)
    s = weighted_mean(**SCORES)
    assert s is not None and prev is not None and trust is not None
    assert trust == pytest.approx(LHI_LAMBDA_UP * prev + (1 - LHI_LAMBDA_UP) * s)
    assert prev < trust < s


def test_degrading_scores_drop_trust_fast(rows, tmp_path):
    cbac = make_cbac(tmp_path)
    prev = compute(cbac)
    trust = compute(cbac, **{k: 0.2 for k in SCORES})
    assert prev is not None and trust is not None
    assert trust == pytest.approx(LHI_LAMBDA_DOWN * prev + (1 - LHI_LAMBDA_DOWN) * 0.2)
    assert trust < prev


def test_zero_component_costs_exactly_its_weight(rows, tmp_path):
    """A zero component lowers s by its renormalized weight, not to 0 — one
    weak signal must not annihilate an otherwise compliant interaction."""
    cbac = make_cbac(tmp_path)
    trust = trust_of(cbac, hallucination_score=0.0)
    normalized_weight = LHI_WEIGHTS[2] / sum(LHI_WEIGHTS)
    mean = weighted_mean(**SCORES)
    assert trust is not None and mean is not None
    assert trust == pytest.approx(mean - normalized_weight * 0.95)
    assert trust > 0.0


# ── Missing components: renormalize, never substitute ─────────────────────────


def test_missing_component_renormalizes_rather_than_skipping(rows, tmp_path):
    """policy_score is None exactly on Tier-3 gray-zone decisions. Skipping
    those records would exclude precisely the interactions trust arbitrates,
    so the mean renormalizes over what was observed instead."""
    cbac = make_cbac(tmp_path)
    trust = compute(cbac, policy_score=None)

    expected = (LHI_WEIGHTS[0] * 0.9 + LHI_WEIGHTS[2] * 0.95) / (
        LHI_WEIGHTS[0] + LHI_WEIGHTS[2]
    )
    assert trust == pytest.approx(expected)
    assert len(rows) == 1
    # Stored as NULL, never a substituted value — the record stays honest.
    assert rows[0].policy_score is None
    assert rows[0].intent_score == 0.9


def test_only_intent_observed_still_records(rows, tmp_path):
    """One observed component is enough to write a record, and with a single
    component the mean is that component."""
    cbac = make_cbac(tmp_path)
    trust = compute(cbac, policy_score=None, hallucination_score=None)
    assert trust == pytest.approx(0.9)
    assert len(rows) == 1


def test_nothing_observed_writes_nothing(rows, tmp_path):
    """A record with no measurement behind it would be fiction."""
    cbac = make_cbac(tmp_path)
    assert compute(cbac, **{k: None for k in SCORES}) is None
    assert rows == []


def test_out_of_range_score_raises_and_writes_nothing(rows, tmp_path):
    cbac = make_cbac(tmp_path)
    with pytest.raises(ValueError, match="intent_score"):
        compute(cbac, intent_score=1.2)
    assert rows == []


# ── Per-edge bookkeeping ──────────────────────────────────────────────────────


def test_trust_is_tracked_per_callee_edge(rows, tmp_path):
    cbac = make_cbac(tmp_path)
    compute(cbac)
    trust_other = compute(
        cbac, callee_name="slack_agent", callee_type="agent", **{k: 0.5 for k in SCORES}
    )
    assert trust_other == pytest.approx(0.5)


def test_same_name_different_type_are_separate_edges(rows, tmp_path):
    """(callee_name, callee_type) is the edge key: the same name with a
    different type starts a fresh trust history."""
    cbac = make_cbac(tmp_path)
    compute(cbac, callee_name="helper", callee_type="tool")
    trust_agent = compute(
        cbac, callee_name="helper", callee_type="agent", **{k: 0.5 for k in SCORES}
    )
    assert trust_agent == pytest.approx(0.5)  # first interaction for helper:agent
    assert len(edge_history(rows, "helper", "tool")) == 1
    assert len(edge_history(rows, "helper", "agent")) == 1


def test_history_is_appended_not_replaced(rows, tmp_path):
    """Every decision adds a row; the edge's trust history is the ordered
    trust column, and the latest row is the current trust the next EMA reads."""
    cbac = make_cbac(tmp_path)
    t1 = compute(cbac, **{k: 0.5 for k in SCORES})
    t2 = compute(cbac)
    t3 = compute(cbac, hallucination_score=0.0)

    history = edge_history(rows, "github_tool", "tool")
    assert history == [pytest.approx(t1), pytest.approx(t2), pytest.approx(t3)]
    # Each row keeps its component scores, so the EMA chain is auditable.
    assert rows[1].intent_score == SCORES["intent_score"]
    assert rows[2].hallucination_score == 0.0


def test_ema_continues_across_instances(rows, tmp_path):
    """Trust state lives in the DB, not the CBAC object — a fresh instance
    (new process, restarted service) continues the same EMA chain."""
    prev = trust_of(make_cbac(tmp_path))
    trust = trust_of(make_cbac(tmp_path))
    s = weighted_mean(**SCORES)
    assert prev is not None and trust is not None and s is not None
    assert trust == pytest.approx(LHI_LAMBDA_UP * prev + (1 - LHI_LAMBDA_UP) * s)


def test_hallucination_score_orientation(tmp_path):
    pytest.importorskip("transformers")
    cbac = make_cbac(tmp_path)
    try:
        contradicted = cbac.hallucination_score(
            "The capital of France is Berlin.", "The capital of France is Paris."
        )
        grounded = cbac.hallucination_score(
            "I am in California", "I am in United States."
        )
    except Exception as exc:
        pytest.skip(f"HHEM model unavailable: {exc}")
    assert 0.0 <= contradicted <= 1.0
    assert 0.0 <= grounded <= 1.0
    assert contradicted < 0.5 < grounded
