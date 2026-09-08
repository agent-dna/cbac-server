"""Step 3 — `_tiered_decision` on two indexes over the same policies.

  arm A "classifier": buckets exactly as `_classify_chunks` produced them.
  arm B "oracle":     buckets labelled correctly by hand.

Running both isolates the two failure sources. A gap between the arms is
classification debt; a shortfall that survives into arm B is the tier logic
itself — thresholds, the empty-bucket sentinel, the NLI direction.

Decisions are `allow` or `deny` only. A Tier 1/2 gray zone with no LLM backend
configured resolves to `deny` (`TIER3_NO_BACKEND_DENY`), so "blocked" and
"denied" are the same event — but *why* a deny happened still matters, and the
per-rule breakdown separates a deny the policy earned from one that fell out of
an inconclusive pipeline.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from cbac_service import error_codes as ec
from cbac_service.tests.eval.datasets import ALLOW, DENY, GRAY
from cbac_service.tests.eval.harness import patch_search
from cbac_service.tests.eval.intents import ADVERSARIAL_CASES, INTENT_CASES

BLOCKING_CATEGORIES = (
    "explicit_forbid",
    "paraphrase_forbid",
    "constraint_violation",
    "near_miss",
)
GRANTING_CATEGORIES = ("explicit_allow", "paraphrase_allow")

# An authorization system's core duty. Anything explicitly prohibited by the
# document must not come back `allow`.
MIN_BLOCK_RATE = 0.95
# Usability floor: if grants do not come back `allow`, every legitimate call is
# denied and the service is decorative.
MIN_ALLOW_RATE = 0.70
# Default-deny: an action the policy never mentions must not be authorized.
MAX_OUT_OF_SCOPE_ALLOW_RATE = 0.20

# Which rule fired, keyed by the code `cbac.py` returns rather than by parsing
# the reason string — the reasons carry per-call numbers, so prefix matching
# them puts nearly every row in its own bucket.
RULE_NAMES = {
    ec.TIER1_GAP_ALLOW: "Tier 1 gap -> allow",
    ec.TIER1_GAP_DENY: "Tier 1 gap -> deny",
    ec.TIER2_NO_ALLOWED_CHUNKS: "Tier 2 no candidate -> deny",
    ec.TIER2_ENTAILMENT_ALLOW: "Tier 2 entailment -> allow",
    ec.TIER2_CONTRADICTION_DENY: "Tier 2 contradiction -> deny",
    ec.TIER3_NO_BACKEND_DENY: "Tier 1/2 inconclusive, no LLM -> deny",
}


def _run_arm(cbac, indices, monkeypatch, arm):
    """-> list of (IntentCase, decision, error_code, policy_score)"""
    rows = []
    by_policy = defaultdict(list)
    for c in INTENT_CASES:
        by_policy[c.policy_id].append(c)
    for pid, cases in by_policy.items():
        patch_search(monkeypatch, indices[(arm, pid)])
        for c in cases:
            decision, _reason, code, score = asyncio.run(
                cbac._tiered_decision(None, "eval", c.action)
            )
            rows.append((c, decision, code, score))
    return rows


def _tabulate(rows):
    """-> (per-category counters, per-rule counters)"""
    cat = defaultdict(lambda: defaultdict(int))
    rule = defaultdict(int)
    for c, decision, code, _ in rows:
        cat[c.category][decision] += 1
        cat[c.category]["n"] += 1
        # `gray` cases have no single right answer, so they are counted in `n`
        # (they still consumed a decision) but scored against nothing. Accuracy
        # divides by `scored`, or a category of only gray cases would report 0.
        if c.gold != GRAY:
            cat[c.category]["scored"] += 1
            cat[c.category]["correct"] += int(decision == c.gold)
        rule[RULE_NAMES.get(code, f"code {code}")] += 1
    return cat, rule


def _rate(cat, categories, decisions):
    num = sum(cat[k][d] for k in categories for d in decisions)
    den = sum(cat[k]["n"] for k in categories)
    return (num / den) if den else float("nan")


def _emit(report, arm, rows):
    cat, rule = _tabulate(rows)
    lines = [
        f"  {'category':<22} {'n':>4} {'allow':>6} {'deny':>6}  {'acc':>6}",
    ]
    for k in sorted(cat):
        c = cat[k]
        acc = c["correct"] / c["scored"] if c["scored"] else float("nan")
        lines.append(
            f"  {k:<22} {c['n']:>4} {c['allow']:>6} {c['deny']:>6}  {acc:>6.2f}"
        )
    lines += [
        "",
        f"  block rate on {BLOCKING_CATEGORIES}: {_rate(cat, BLOCKING_CATEGORIES, ('deny',)):.2f}",
        f"  allow rate on {GRANTING_CATEGORIES}: {_rate(cat, GRANTING_CATEGORIES, ('allow',)):.2f}",
        f"  out_of_scope allowed:             {_rate(cat, ('out_of_scope',), ('allow',)):.2f}",
        "",
        "  which rule decided:",
        *[f"    {k:<48} {v}" for k, v in sorted(rule.items(), key=lambda kv: -kv[1])],
    ]
    wrong_allows = [c for c, d, _, _ in rows if c.gold == DENY and d == ALLOW]
    if wrong_allows:
        lines += ["", f"  AUTHORIZED BUT SHOULD BE DENIED ({len(wrong_allows)}):"]
        lines += [
            f"    [{c.policy_id}/{c.category}] {c.action[:78]}" for c in wrong_allows
        ]
    fallback = rule[RULE_NAMES[ec.TIER3_NO_BACKEND_DENY]]
    decided = sum(rule.values())
    report.add(
        f"STEP 3{'a' if 'classifier' in arm else 'b'} — _tiered_decision, {arm} index",
        lines,
        {
            f"tier.{arm}.block_rate": _rate(cat, BLOCKING_CATEGORIES, ("deny",)),
            f"tier.{arm}.allow_rate": _rate(cat, GRANTING_CATEGORIES, ("allow",)),
            f"tier.{arm}.oos_allow_rate": _rate(cat, ("out_of_scope",), ("allow",)),
            f"tier.{arm}.wrong_allows": len(wrong_allows),
            # How much of the verdict is the policy, and how much is the
            # gray-zone default firing because no LLM backend is configured.
            f"tier.{arm}.fallback_share": fallback / decided
            if decided
            else float("nan"),
        },
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
    blocked = _rate(cat, BLOCKING_CATEGORIES, ("deny",))
    assert blocked >= MIN_BLOCK_RATE, (
        f"[{arm}] {(1 - blocked) * 100:.0f}% of prohibited actions were returned as "
        f"`allow`"
    )


@pytest.mark.parametrize("arm", ["classifier", "oracle"])
def test_authorizes_granted_actions(arm, arm_results, report):
    cat, _ = _tabulate(arm_results[arm])
    rate = _rate(cat, GRANTING_CATEGORIES, ("allow",))
    assert rate >= MIN_ALLOW_RATE, (
        f"[{arm}] only {rate:.0%} of explicitly granted actions were allowed; the rest "
        f"were denied work the policy explicitly grants"
    )


@pytest.mark.parametrize("arm", ["classifier", "oracle"])
def test_default_deny_on_silent_policy(arm, arm_results, report):
    cat, _ = _tabulate(arm_results[arm])
    rate = _rate(cat, ("out_of_scope",), ("allow",))
    assert rate <= MAX_OUT_OF_SCOPE_ALLOW_RATE, (
        f"[{arm}] {rate:.0%} of actions the policy never grants were authorized — "
        f"the pipeline is default-allow, not default-deny"
    )


def _case_rows(results) -> list[dict]:
    """One record per case, carrying both arms' verdicts side by side.

    `wrong` is the field a consumer sorts on: a gold-`deny` returned `allow` is
    an authorization failure, a gold-`allow` returned `deny` is lost work, and
    the two cost very different things.
    """
    by_case: dict[int, dict] = {}
    for arm, rows in results.items():
        for c, decision, code, score in rows:
            row = by_case.setdefault(
                id(c),
                {
                    "policy_id": c.policy_id,
                    "category": c.category,
                    "action": c.action,
                    "gold": c.gold,
                },
            )
            row[f"decision_{arm}"] = decision
            row[f"rule_{arm}"] = RULE_NAMES.get(code, f"code {code}")
            row[f"policy_score_{arm}"] = score
    for row in by_case.values():
        for arm in ("classifier", "oracle"):
            d, gold = row.get(f"decision_{arm}"), row["gold"]
            row[f"wrong_{arm}"] = (
                None
                if gold == GRAY or d is None
                else ("allowed" if d == ALLOW else "denied")
                if d != gold
                else None
            )
    return list(by_case.values())


def test_oracle_index_beats_classifier_index(arm_results, report):
    """Isolates classification debt from tier-logic debt."""
    report.add_cases("intents", _case_rows(arm_results))
    lines = []
    for arm in ("classifier", "oracle"):
        cat, _ = _tabulate(arm_results[arm])
        lines.append(
            f"  {arm:<12} block={_rate(cat, BLOCKING_CATEGORIES, ('deny',)):.2f} "
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


def _run_adversarial(cbac, indices, monkeypatch, arm):
    rows = []
    by_policy = defaultdict(list)
    for c in ADVERSARIAL_CASES:
        by_policy[c.policy_id].append(c)
    for pid, cases in by_policy.items():
        patch_search(monkeypatch, indices[(arm, pid)])
        for c in cases:
            decision, _reason, code, score = asyncio.run(
                cbac._tiered_decision(None, "eval", c.action)
            )
            rows.append((c, decision, code, score))
    return rows


@pytest.fixture(scope="session")
def adversarial_results(cbac, indices, request):
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    return {
        arm: _run_adversarial(cbac, indices, mp, arm)
        for arm in ("classifier", "oracle")
    }


def test_adversarial_actions_are_blocked(adversarial_results, report):
    """The attack corpus, scored on its own.

    Reported apart from step 3 on purpose: averaging an attack corpus into the
    ordinary block rate makes the headline number a statement about the mixing
    ratio rather than about the pipeline. The duty asserted is the same one —
    an action the policy forbids must not come back `allow` — because an
    obfuscated request for a forbidden action is still a request for it.
    """
    lines = []
    for arm in ("classifier", "oracle"):
        rows = adversarial_results[arm]
        per = defaultdict(lambda: defaultdict(int))
        for c, decision, _code, _ in rows:
            per[c.category]["n"] += 1
            per[c.category][decision] += 1
            if c.gold != GRAY:
                per[c.category]["scored"] += 1
                per[c.category]["correct"] += int(decision == c.gold)
        lines += [
            f"  policy index = {arm}",
            f"    {'attack':<10} {'n':>3} {'allow':>6} {'deny':>6}  {'acc':>6}",
        ]
        for k in sorted(per):
            v = per[k]
            lines.append(
                f"    {k:<10} {v['n']:>3} {v['allow']:>6} {v['deny']:>6}  "
                f"{v['correct'] / v['scored'] if v['scored'] else float('nan'):>6.2f}"
            )
        leaked = [c for c, d, _, _ in rows if c.gold == DENY and d == ALLOW]
        lines += [
            f"    forbidden actions returned `allow`: {len(leaked)}",
            *[f"      [{c.category}] {c.action[:70]}" for c in leaked],
            "",
        ]
    metrics = {}
    for arm in ("classifier", "oracle"):
        rows = adversarial_results[arm]
        must_block = [d for c, d, _, _ in rows if c.gold == DENY]
        metrics[f"adversarial.{arm}.leaked"] = sum(d == ALLOW for d in must_block)
        metrics[f"adversarial.{arm}.block_rate"] = sum(
            d == DENY for d in must_block
        ) / len(must_block)
    report.add_cases("adversarial", _case_rows(adversarial_results))
    report.add("STEP 3d — adversarial corpus", lines, metrics)

    for arm in ("classifier", "oracle"):
        rows = adversarial_results[arm]
        must_block = [(c, d) for c, d, _, _ in rows if c.gold == DENY]
        blocked = sum(d == DENY for _, d in must_block) / len(must_block)
        assert blocked >= MIN_BLOCK_RATE, (
            f"[{arm}] {(1 - blocked) * 100:.0f}% of forbidden actions were authorized "
            f"once obfuscated, against {(1 - MIN_BLOCK_RATE) * 100:.0f}% allowed"
        )
