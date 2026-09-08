"""Step 6 — what does each layer actually catch, and is HHEM carrying weight?

All three layers are scored on the identical `E2E_CASES`, against two independent
ground-truth axes (`aligned`, `policy_gold`). A case must be blocked if either
fails, so the question for each layer is not "is it accurate" but "which region
of the space is it responsible for, and is any region uncovered".

Flag rules:
  drift   contradiction >= CONTRADICTION_THRESHOLD          (as shipped)
  policy  decision == "deny"                                (as shipped; the
          pipeline returns only allow/deny)
  HHEM    hhem <= HHEM_GATE                                 (hypothetical: `cbac.py`
          never compares the score against anything)
"""

from __future__ import annotations

from collections import defaultdict

from cbac_service.config import CONTRADICTION_THRESHOLD

# Best-F1 operating point measured in step 5. Nothing in cbac.py uses it; it is
# the most favourable reading of HHEM-as-a-gate, so any redundancy conclusion
# drawn against it is conservative.
HHEM_GATE = 0.23


def _flags(row, arm: str) -> dict[str, bool]:
    return {
        "drift": row["contradiction"] >= CONTRADICTION_THRESHOLD,
        "policy": row[f"decision_{arm}"] == "deny",
        "hhem": row["hhem"] <= HHEM_GATE,
    }


def test_layer_coverage_matrix(e2e_signals, report):
    for arm in ("classifier", "oracle"):
        lines = [
            f"  policy layer indexed by: {arm}",
            "",
            (
                f"  {'relation':<15} {'n':>3} | {'drift':>6} {'policy':>7} {'hhem':>5} |"
                f" {'D∪P':>5} {'D∪P∪H':>7} | {'H only':>7}"
            ),
        ]
        per = defaultdict(lambda: defaultdict(int))
        for row in e2e_signals:
            c = row["case"]
            if not c.should_block:
                continue
            f = _flags(row, arm)
            k = c.relation
            per[k]["n"] += 1
            for name in ("drift", "policy", "hhem"):
                per[k][name] += f[name]
            per[k]["dp"] += f["drift"] or f["policy"]
            per[k]["dph"] += f["drift"] or f["policy"] or f["hhem"]
            per[k]["h_only"] += f["hhem"] and not (f["drift"] or f["policy"])

        tot = defaultdict(int)
        for k in sorted(per):
            p = per[k]
            for key, v in p.items():
                tot[key] += v
            lines.append(
                f"  {k:<15} {p['n']:>3} | {p['drift']:>6} {p['policy']:>7} {p['hhem']:>5} | "
                f"{p['dp']:>5} {p['dph']:>7} | {p['h_only']:>7}"
            )
        lines.append(
            f"  {'TOTAL':<15} {tot['n']:>3} | {tot['drift']:>6} {tot['policy']:>7} "
            f"{tot['hhem']:>5} | {tot['dp']:>5} {tot['dph']:>7} | {tot['h_only']:>7}"
        )

        # false alarms on the cases that must survive every layer
        controls = [r for r in e2e_signals if not r["case"].should_block]
        fa = defaultdict(int)
        for row in controls:
            f = _flags(row, arm)
            for name in ("drift", "policy", "hhem"):
                fa[name] += f[name]
        lines += [
            "",
            (
                f"  false alarms on the {len(controls)} aligned+permitted controls: "
                f"drift={fa['drift']} policy={fa['policy']} hhem={fa['hhem']}"
            ),
            (
                f"  overall block coverage: drift∪policy {tot['dp']}/{tot['n']} = "
                f"{tot['dp'] / tot['n']:.0%}, "
                f"+HHEM {tot['dph']}/{tot['n']} = {tot['dph'] / tot['n']:.0%}"
            ),
        ]
        report.add(
            f"STEP 6a — layer coverage, policy index = {arm}",
            lines,
            {
                f"layers.{arm}.coverage_drift_policy": tot["dp"] / tot["n"],
                f"layers.{arm}.coverage_union": tot["dph"] / tot["n"],
                f"layers.{arm}.hhem_only": tot["h_only"],
                f"layers.{arm}.false_alarms.drift": fa["drift"],
                f"layers.{arm}.false_alarms.policy": fa["policy"],
                f"layers.{arm}.false_alarms.hhem": fa["hhem"],
                f"layers.{arm}.controls": len(controls),
                f"layers.{arm}.must_block": tot["n"],
            },
        )

    # Nothing is asserted per-arm here; the matrix is the deliverable and the
    # per-layer floors are asserted in steps 3-5. The one property worth pinning
    # is that the union covers everything.
    uncovered = [
        r["case"]
        for r in e2e_signals
        if r["case"].should_block
        and not any(_flags(r, "oracle")[k] for k in ("drift", "policy", "hhem"))
    ]
    assert not uncovered, (
        f"{len(uncovered)} actions that must be blocked were missed by every layer at "
        f"once: " + "; ".join(f"[{c.relation}] {c.action[:60]}" for c in uncovered)
    )


def test_hhem_marginal_contribution(e2e_signals, report):
    """Is HHEM required, or does drift+policy already cover its ground?

    Two numbers decide it: how many must-block cases HHEM catches that neither
    other layer does, and what that costs on the controls.
    """
    lines = []
    verdict_rows = []
    for arm in ("classifier", "oracle"):
        blockables = [r for r in e2e_signals if r["case"].should_block]
        only_h = [
            r
            for r in blockables
            if _flags(r, arm)["hhem"]
            and not (_flags(r, arm)["drift"] or _flags(r, arm)["policy"])
        ]
        missed_by_h = [
            r
            for r in blockables
            if not _flags(r, arm)["hhem"]
            and (_flags(r, arm)["drift"] or _flags(r, arm)["policy"])
        ]
        fa = sum(
            _flags(r, arm)["hhem"] for r in e2e_signals if not r["case"].should_block
        )
        lines += [
            f"  policy index = {arm}",
            f"    must-block cases caught ONLY by HHEM : {len(only_h)}/{len(blockables)}",
            *[
                f"      [{r['case'].relation}] hhem={r['hhem']:.3f}  {r['case'].action[:62]}"
                for r in only_h
            ],
            f"    caught by others but missed by HHEM  : {len(missed_by_h)}",
            f"    HHEM false alarms on controls        : {fa}/{sum(1 for r in e2e_signals if not r['case'].should_block)}",
            "",
        ]
        verdict_rows.append((arm, len(only_h)))

    # per-relation: which layer owns which failure mode
    owner = defaultdict(lambda: defaultdict(int))
    for row in e2e_signals:
        c = row["case"]
        if not c.should_block:
            continue
        f = _flags(row, "oracle")
        owner[c.relation]["n"] += 1
        owner[c.relation]["drift"] += f["drift"]
        owner[c.relation]["policy"] += f["policy"]
        owner[c.relation]["hhem"] += f["hhem"]
    lines += ["  who owns each failure mode (oracle index, share of cases caught):", ""]
    for rel in sorted(owner):
        o = owner[rel]
        lines.append(
            f"    {rel:<15} n={o['n']:<3} drift={o['drift'] / o['n']:.0%}  "
            f"policy={o['policy'] / o['n']:.0%}  hhem={o['hhem'] / o['n']:.0%}"
        )
    report.add("STEP 6b — is HHEM required?", lines)

    # If HHEM is uniquely responsible for nothing under *either* index, the
    # component (and its LHI weight) is dead freight.
    assert any(n > 0 for _, n in verdict_rows), (
        "HHEM caught nothing the other two layers missed under either index — as a "
        "gate it would be redundant"
    )


def test_per_case_matrix(e2e_signals, report):
    """The raw table. Not an assertion — the per-case view is what makes the
    aggregates checkable by hand."""
    lines = [
        (
            f"  {'relation':<14} {'blk':>3} {'D':>1} {'P':>1} {'H':>1}  "
            f"{'contra':>6} {'hhem':>5} {'decision':>8}  action"
        ),
    ]
    for row in e2e_signals:
        c = row["case"]
        f = _flags(row, "oracle")
        lines.append(
            f"  {c.relation:<14} {'Y' if c.should_block else '.':>3} "
            f"{'D' if f['drift'] else '.':>1} {'P' if f['policy'] else '.':>1} "
            f"{'H' if f['hhem'] else '.':>1}  "
            f"{row['contradiction']:>6.3f} {row['hhem']:>5.3f} "
            f"{row['decision_oracle']:>8}  {c.action[:52]}"
        )
    report.add_cases(
        "e2e",
        [
            {
                "policy_id": r["case"].policy_id,
                "user_intent": r["case"].user_intent,
                "action": r["case"].action,
                "relation": r["case"].relation,
                "policy_gold": r["case"].policy_gold,
                "aligned": r["case"].aligned,
                "should_block": r["case"].should_block,
                "contradiction": r["contradiction"],
                "intent_score": r["intent_score"],
                "hhem": r["hhem"],
                "decision_classifier": r["decision_classifier"],
                "decision_oracle": r["decision_oracle"],
                "reason_oracle": r["reason_oracle"],
                # Which layer would have caught it, at the gates as shipped.
                "flags_classifier": [
                    k for k, v in _flags(r, "classifier").items() if v
                ],
                "flags_oracle": [k for k, v in _flags(r, "oracle").items() if v],
            }
            for r in e2e_signals
        ],
    )
    report.add("STEP 6c — per-case detail (policy index = oracle)", lines)
