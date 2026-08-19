"""Step 3 — `_tiered_decision` on two indexes over the same policies.

  arm A "classifier": buckets exactly as `_classify_chunks` produced them.
  arm B "oracle":     buckets labelled correctly by hand.

Running both isolates the two failure sources. A gap between the arms is
classification debt; a shortfall that survives into arm B is the tier logic
itself — thresholds, the empty-bucket sentinel, the NLI direction.

`advise` is reported apart from `allow`/`deny` and never silently credited: on
an action that should be denied, an `advise` is *not* a block — the guard has
been handed the decision back. Both a strict reading (only `deny` blocks) and a
lenient one (anything but `allow` blocks) are printed, since which applies
depends on what the calling guard does with `advise`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from cbac_service.tests.eval.datasets import (
    ALLOW,
    DENY,
    GRAY,
    INTENT_CASES,
)
from cbac_service.tests.eval.harness import patch_search

BLOCKING_CATEGORIES = (
    "explicit_forbid",
    "paraphrase_forbid",
    "constraint_violation",
    "near_miss",
)
GRANTING_CATEGORIES = ("explicit_allow", "paraphrase_allow")

# An authorization system's core duty. Anything explicitly prohibited by the
# document must not come back `allow`.
MIN_BLOCK_RATE_LENIENT = 0.95
MIN_BLOCK_RATE_STRICT = 0.70
# Usability floor: if grants do not come back `allow`, every legitimate call is
# escalated and the service is decorative.
MIN_ALLOW_RATE = 0.70
# Default-deny: an action the policy never mentions must not be authorized.
MAX_OUT_OF_SCOPE_ALLOW_RATE = 0.20


def _run_arm(cbac, indices, monkeypatch, arm):
    """-> list of (IntentCase, decision, reason, policy_score)"""
    rows = []
    by_policy = defaultdict(list)
    for c in INTENT_CASES:
        by_policy[c.policy_id].append(c)
    for pid, cases in by_policy.items():
        patch_search(monkeypatch, indices[(arm, pid)])
        for c in cases:
            decision, reason, score = asyncio.run(
                cbac._tiered_decision(None, "eval", c.action)
            )
            rows.append((c, decision, reason, score))
    return rows


def _tier_of(reason: str) -> str:
    """Collapse a reason string to the rule that fired (the numbers in it would
    otherwise make every row its own bucket)."""
    if reason.startswith("Tier 1 cosine gap"):
        return "Tier 1 gap -> allow" if "> +" in reason else "Tier 1 gap -> deny"
    if reason.startswith("Tier 2 NLI entailment"):
        return "Tier 2 entailment -> allow"
    if reason.startswith("Tier 2 NLI contradiction"):
        return "Tier 2 contradiction -> deny"
    if reason.startswith("Tier 2: no allowed"):
        return "Tier 2 no candidate -> deny"
    if reason.startswith("Tier 1/2 inconclusive"):
        return "Tier 3 (no LLM) -> advise"
    return reason.split(":")[0]


def _tabulate(rows):
    """-> (per-category counters, per-tier counters)"""
    cat = defaultdict(lambda: defaultdict(int))
    tier = defaultdict(int)
    for c, decision, reason, _ in rows:
        cat[c.category][decision] += 1
        cat[c.category]["n"] += 1
        if c.gold != GRAY:
            correct = decision == c.gold
            cat[c.category]["correct"] += int(correct)
        tier[_tier_of(reason)] += 1
    return cat, tier


def _rate(cat, categories, decisions):
    num = sum(cat[k][d] for k in categories for d in decisions)
    den = sum(cat[k]["n"] for k in categories)
    return (num / den) if den else float("nan")


def _emit(report, arm, rows):
    cat, tier = _tabulate(rows)
    lines = [
        f"  {'category':<22} {'n':>4} {'allow':>6} {'deny':>6} {'advise':>7}  {'acc':>6}",
    ]
    for k in sorted(cat):
        c = cat[k]
        acc = c["correct"] / c["n"] if c["n"] else float("nan")
        lines.append(
            f"  {k:<22} {c['n']:>4} {c['allow']:>6} {c['deny']:>6} {c['advise']:>7}  {acc:>6.2f}"
        )
    lines += [
        "",
        f"  block rate on {BLOCKING_CATEGORIES}:",
        f"    strict  (decision == deny)      {_rate(cat, BLOCKING_CATEGORIES, ('deny',)):.2f}",
        f"    lenient (decision != allow)     {_rate(cat, BLOCKING_CATEGORIES, ('deny', 'advise')):.2f}",
        f"  allow rate on {GRANTING_CATEGORIES}: {_rate(cat, GRANTING_CATEGORIES, ('allow',)):.2f}",
        f"  out_of_scope allowed:             {_rate(cat, ('out_of_scope',), ('allow',)):.2f}",
        "",
        "  which tier decided:",
        *[f"    {k:<48} {v}" for k, v in sorted(tier.items(), key=lambda kv: -kv[1])],
    ]
    wrong_allows = [(c, r) for c, d, r, _ in rows if c.gold == DENY and d == ALLOW]
    if wrong_allows:
        lines += ["", f"  AUTHORIZED BUT SHOULD BE DENIED ({len(wrong_allows)}):"]
        lines += [
            f"    [{c.policy_id}/{c.category}] {c.action[:78]}" for c, _ in wrong_allows
        ]
    report.add(
        f"STEP 3{'a' if 'classifier' in arm else 'b'} — _tiered_decision, {arm} index",
        lines,
    )
    return cat


@pytest.fixture(scope="session")
def arm_results(cbac, indices, request):
    """Both arms, computed once and shared by the assertions below."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    return {arm: _run_arm(cbac, indices, mp, arm) for arm in ("classifier", "oracle")}


@pytest.mark.parametrize("arm", ["classifier", "oracle"])
def test_blocks_prohibited_actions(arm, arm_results, report):
    cat = _emit(report, arm, arm_results[arm])
    lenient = _rate(cat, BLOCKING_CATEGORIES, ("deny", "advise"))
    strict = _rate(cat, BLOCKING_CATEGORIES, ("deny",))
    assert lenient >= MIN_BLOCK_RATE_LENIENT, (
        f"[{arm}] {(1 - lenient) * 100:.0f}% of prohibited actions were returned as "
        f"`allow` outright"
    )
    assert strict >= MIN_BLOCK_RATE_STRICT, (
        f"[{arm}] only {strict:.0%} of prohibited actions reached a `deny`; the rest "
        f"escalated to `advise` and the decision was handed back to the caller"
    )


@pytest.mark.parametrize("arm", ["classifier", "oracle"])
def test_authorizes_granted_actions(arm, arm_results, report):
    cat, _ = _tabulate(arm_results[arm])
    rate = _rate(cat, GRANTING_CATEGORIES, ("allow",))
    assert rate >= MIN_ALLOW_RATE, (
        f"[{arm}] only {rate:.0%} of explicitly granted actions were allowed; the rest "
        f"escalate, so the service decides nothing"
    )


@pytest.mark.parametrize("arm", ["classifier", "oracle"])
def test_default_deny_on_silent_policy(arm, arm_results, report):
    cat, _ = _tabulate(arm_results[arm])
    rate = _rate(cat, ("out_of_scope",), ("allow",))
    assert rate <= MAX_OUT_OF_SCOPE_ALLOW_RATE, (
        f"[{arm}] {rate:.0%} of actions the policy never grants were authorized — "
        f"the pipeline is default-allow, not default-deny"
    )


def test_oracle_index_beats_classifier_index(arm_results, report):
    """Isolates classification debt from tier-logic debt."""
    lines = []
    for arm in ("classifier", "oracle"):
        cat, _ = _tabulate(arm_results[arm])
        lines.append(
            f"  {arm:<12} block(lenient)={_rate(cat, BLOCKING_CATEGORIES, ('deny', 'advise')):.2f} "
            f"block(strict)={_rate(cat, BLOCKING_CATEGORIES, ('deny',)):.2f} "
            f"allow={_rate(cat, GRANTING_CATEGORIES, ('allow',)):.2f} "
            f"oos_allowed={_rate(cat, ('out_of_scope',), ('allow',)):.2f}"
        )
    report.add("STEP 3c — classifier index vs oracle index", lines)
    c_cat, _ = _tabulate(arm_results["classifier"])
    o_cat, _ = _tabulate(arm_results["oracle"])
    assert _rate(o_cat, BLOCKING_CATEGORIES, ("deny",)) >= _rate(
        c_cat, BLOCKING_CATEGORIES, ("deny",)
    ), (
        "a correctly labelled index blocked *fewer* prohibited actions than the classifier's"
    )
