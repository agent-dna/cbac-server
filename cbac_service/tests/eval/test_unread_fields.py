"""Stage 6 — frontmatter the parser reads and the decision never consults.

`parse_skill_md` builds a `SkillsCard` with `expires_at`, `can_delegate_to` and
`constraints`. `flatten_policy_chunks` then chunks `raw_frontmatter`, so those
fields reach the index only as more prose for the encoder to embed, and nothing
in `cbac.py` compares a date, checks an allowlist, or does arithmetic.

The measurement is a controlled substitution: take a real policy, change exactly
one of those fields, re-index, and re-decide the same actions. If every verdict
is identical, the field did not reach the decision — which is a stronger claim
than reading the code, because it also covers the possibility that the value
changed some embedding enough to matter.
"""

from __future__ import annotations

import asyncio

from cbac_service.chunking import flatten_policy_chunks
from cbac_service.tests.eval.corpus import ALLOWED, FORBIDDEN, spec_by_id
from cbac_service.tests.eval.harness import build_index, patch_search

# Long expired, and expired before it was issued — a card no verifier should
# accept under any reading.
PAST = "2024-01-01T00:00:00Z"


def _verdicts(
    cbac, monkeypatch, spec, policy_text
) -> dict[str, tuple[str, int | None]]:
    """Index `policy_text` as this spec's policy, then decide its actions."""
    chunks = flatten_policy_chunks(policy_text)
    allowed, forbidden = cbac._classify_chunks(chunks)
    index = build_index(
        cbac,
        list(allowed) + list(forbidden),
        [ALLOWED] * len(allowed) + [FORBIDDEN] * len(forbidden),
    )
    patch_search(monkeypatch, index)
    out = {}
    for intent, action in spec.actions:
        result = asyncio.run(cbac._decide(None, "eval", action.text, intent.text))
        out[f"{intent.id}/{action.id}"] = (result.decision, result.error_code)
    return out


def test_an_expired_card_decides_the_same_as_a_live_one(cbac, report, monkeypatch):
    """`expires-at` is required frontmatter and is parsed into a datetime.

    Nothing compares it against the clock, so a card that expired two years ago
    authorizes exactly what it authorized when it was issued. An authorization
    system that cannot expire a credential has no revocation story: withdrawing
    a capability means re-issuing the card *and* re-indexing, and until the
    re-index lands the old grants stand.
    """
    from cbac_service.tests.eval.corpus import _card

    spec = spec_by_id()["payments"]
    live = _verdicts(cbac, monkeypatch, spec, _card(spec))
    expired = _verdicts(cbac, monkeypatch, spec, _card(spec, expires_at=PAST))

    differing = {k: (live[k], expired[k]) for k in live if live[k] != expired[k]}
    lines = [
        f"  policy: {spec.id}, expires-at {PAST} vs 2026-12-31",
        f"  verdicts that changed: {len(differing)}/{len(live)}",
        *[f"    {k:<28}{v[0]} -> {v[1]}" for k, v in differing.items()],
        "",
        "  An expired card is enforced as a live one. Revoking a capability",
        "  requires re-issuing the policy and re-indexing it; letting the card",
        "  lapse does nothing.",
    ]
    report.add(
        "STAGE 6a — card expiry", lines, {"unread.expiry.changed": len(differing)}
    )
    assert not differing, (
        "expiry is enforced after all — this test documents the opposite and "
        "needs rewriting against what the pipeline now does"
    )


def test_a_delegation_allowlist_changes_nothing(cbac, report, monkeypatch):
    """`can-delegate-to` names the agents a coordinator may hand work to.

    It is parsed into `SkillsCard.can_delegate_to` and never read. A hand-off is
    judged as prose like any other action, so whether the target agent is on the
    list is decided by how similar the sentence is to the policy's grants.
    """
    from cbac_service.tests.eval.corpus import _card

    spec = spec_by_id()["delegation"]
    listed = _verdicts(
        cbac, monkeypatch, spec, _card(spec, can_delegate_to=list(spec.can_delegate_to))
    )
    empty = _verdicts(cbac, monkeypatch, spec, _card(spec, can_delegate_to=[]))

    differing = {k: (listed[k], empty[k]) for k in listed if listed[k] != empty[k]}
    handoffs = [f"{i.id}/{a.id}" for i, a in spec.actions if "hand_off" in a.callee]
    leaked = [
        k
        for k in handoffs
        if listed[k][0] == "allow"
        and any(a.policy == "deny" for i, a in spec.actions if f"{i.id}/{a.id}" == k)
    ]
    lines = [
        f"  policy: {spec.id}, can-delegate-to {list(spec.can_delegate_to)} vs []",
        f"  verdicts that changed: {len(differing)}/{len(listed)}",
        (
            f"  hand-offs to an agent not on the list, authorized: "
            f"{len(leaked)}/{len(handoffs)}"
        ),
        *[f"    {k}" for k in leaked],
        "",
        "  Emptying the allowlist changes nothing, so the list is not what",
        "  decides a hand-off. Delegation is scored as prose.",
    ]
    report.add(
        "STAGE 6b — delegation allowlist",
        lines,
        {
            "unread.delegation.changed": len(differing),
            "unread.delegation.leaked_handoffs": len(leaked),
        },
    )
    assert not differing, (
        "the delegation allowlist reaches the decision after all — this test "
        "documents the opposite and needs rewriting"
    )


def test_constraints_are_not_arithmetic(run, report):
    """`constraints:` is parsed into `SkillsCard.constraints` and embedded as
    prose. No layer compares a number against a limit, so a granted capability
    with a broken threshold is judged by how much it reads like the grant — and
    it reads exactly like it, because it *is* it.

    The rate is computed in stage 3b; this restates it beside the two fields
    above because all three are the same failure: a field the parser
    understands that the decision does not.
    """
    rows = [c for c in run.cases if c.basis == "constraint"]
    lines = [f"  {len(rows)} actions break a stated numeric or scope limit", ""]
    metrics = {}
    for arm in ("structured", "oracle"):
        blocked = sum(c.arms[arm].final_decision == "deny" for c in rows)
        lines.append(f"    {arm:<14}blocked {blocked}/{len(rows)}")
        metrics[f"unread.constraints.blocked.{arm}"] = blocked / len(rows)
    lines += [
        "",
        *[
            f"    {'BLOCKED' if c.arms['oracle'].final_decision == 'deny' else 'ALLOWED'}"
            f"  {c.spec_id}/{c.intent_id}  {c.call[:58]}"
            for c in rows
        ],
    ]
    report.add("STAGE 6c — numeric and scope limits", lines, metrics)
