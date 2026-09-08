"""Is the dataset itself sound?

Nothing here loads a model or measures the pipeline — it checks that the ground
truth is well-formed, so that a failure elsewhere in this directory is a finding
about `cbac_service` and not about a mislabelled fixture. It is therefore the
one file in `tests/eval/` that runs without `--run-eval`.

It exists because the policy documents live in `scripts/cbac_benchmark/policies/`
and are shared with the DB-backed benchmark runner. Gold labels are matched into
those files by substring, so editing a policy can silently detach every label on
it. That failure is invisible: a marker matching nothing simply stops
contributing, the affected chunks quietly become `neutral`, and the recall
numbers improve.
"""

from __future__ import annotations

import pytest

from cbac_service.chunking import flatten_policy_chunks
from cbac_service.tests.eval.datasets import (
    ALL_POLICIES,
    ALLOW,
    DENY,
    POLICY_BY_ID,
    _norm,
)
from cbac_service.tests.eval.intents import (
    ADVERSARIAL_CASES,
    E2E_CASES,
    INTENT_CASES,
)
from cbac_service.tests.eval.test_tiered_decision import (
    BLOCKING_CATEGORIES,
    GRANTING_CATEGORIES,
)


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.id)
def test_every_gold_marker_matches_a_chunk(policy):
    """A marker that matches nothing is not a neutral loss — it removes a span
    from the scored set and makes the classifier look better than it is."""
    chunks = [_norm(c) for c in flatten_policy_chunks(policy.text)]
    dead = [m for m, _ in policy.gold if not any(_norm(m) in c for c in chunks)]
    assert not dead, (
        f"{policy.id} ({policy.filename}): {len(dead)} gold marker(s) match no chunk — "
        f"the document has changed out from under its labels: {dead}"
    )


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.id)
def test_every_policy_produces_chunks(policy):
    assert flatten_policy_chunks(policy.text), (
        f"{policy.id} ({policy.filename}) chunks to nothing at all"
    )


def test_every_case_names_a_known_policy():
    referenced = {c.policy_id for c in INTENT_CASES}
    referenced |= {c.policy_id for c in ADVERSARIAL_CASES}
    referenced |= {c.policy_id for c in E2E_CASES}
    assert not referenced - set(POLICY_BY_ID), (
        f"cases reference policies that are not in the corpus: "
        f"{sorted(referenced - set(POLICY_BY_ID))}"
    )


def test_every_policy_has_intents():
    """A policy nothing exercises contributes to the classifier numbers but not
    to the decision numbers, which quietly weights the two differently."""
    uncovered = set(POLICY_BY_ID) - {c.policy_id for c in INTENT_CASES}
    assert not uncovered, f"policies with no intent cases: {sorted(uncovered)}"


@pytest.mark.parametrize(
    ("categories", "required_gold"),
    [(BLOCKING_CATEGORIES, DENY), (GRANTING_CATEGORIES, ALLOW)],
    ids=["blocking", "granting"],
)
def test_rate_categories_hold_one_verdict(categories, required_gold):
    """The block and allow rates count *every* case in their categories, not
    only the ones whose gold matches. A single `allow` filed under `near_miss`
    therefore caps the block rate below 1.0 no matter how the pipeline behaves,
    and the number stops meaning what its name says.
    """
    wrong = [
        (c.policy_id, c.category, c.gold, c.action)
        for c in INTENT_CASES
        if c.category in categories and c.gold != required_gold
    ]
    assert not wrong, (
        f"{len(wrong)} case(s) in {categories} are not gold `{required_gold}`; "
        f"move them to another category rather than changing the gold: {wrong[:5]}"
    )
