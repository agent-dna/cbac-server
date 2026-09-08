"""Step 2 — can `_classify_chunks` separate grants from prohibitions?

Scored against the labels a human reader assigns to each span, on both dataset
shapes. The metric that matters for an authorization system is **forbidden
recall**: a prohibition filed as `allowed` is not merely a lost signal — it
becomes a *grant*, and Tier 1's `max_allowed` will then be maximised by the very
rule that bans the action.

`neutral` spans (identifiers, issuance metadata, descriptive prose) are counted
separately. The classifier has no third bucket, so they must land somewhere;
where they land is measured because everything filed as `allowed` raises the
allowed side of the Tier-1 gap.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from cbac_service.tests.eval.datasets import (
    ALL_POLICIES,
    ALLOWED,
    FORBIDDEN,
    NEUTRAL,
    STRESS_POLICIES,
    STRUCTURED_POLICIES,
    UNSTRUCTURED_POLICIES,
)
from cbac_service.tests.eval.harness import Binary

# Requirements, set from what the task needs rather than from current behaviour.
# Structured cards carry the answer in the chunk text itself ("forbidden-actions:
# ..."), so anything below near-perfect recall there is a defect, not a hard
# language problem. Unstructured prose is genuinely harder; 0.70 is the level at
# which the forbidden bucket is populated enough for the Tier-1 gap to mean
# anything at all.
MIN_FORBIDDEN_RECALL_STRUCTURED = 0.90
MIN_FORBIDDEN_RECALL_UNSTRUCTURED = 0.70


def _score(cbac, chunked, classified, policies):
    """-> (overall Binary for 'is forbidden', per-policy rows, misses, leakage)"""
    overall = Binary()
    rows: list[str] = []
    misses: list[tuple[str, str]] = []
    leak: Counter[str] = Counter()

    for p in policies:
        chunks = chunked[p.id]
        predicted_forbidden = set(classified[p.id][1])
        per = Binary()
        gold_counts: Counter[str] = Counter()

        for c in chunks:
            gold = p.label_for(c)
            gold_counts[gold] += 1
            if gold == "mixed":
                continue  # unlabelable by construction — a chunking failure
            pred = c in predicted_forbidden
            if gold == NEUTRAL:
                leak["forbidden" if pred else "allowed"] += 1
                continue
            actual = gold == FORBIDDEN
            per.add(pred, actual)
            overall.add(pred, actual)
            if actual and not pred:
                misses.append((p.id, c))

        rows.append(
            f"  {p.id:<20} {p.style:<14} chunks={len(chunks):<3} "
            f"gold(a/f/n/mixed)="
            f"{gold_counts[ALLOWED]}/{gold_counts[FORBIDDEN]}/"
            f"{gold_counts[NEUTRAL]}/{gold_counts['mixed']:<3} {per.row()}"
        )
    return overall, rows, misses, leak


def _emit(report, title, overall, rows, misses, leak, group):
    lines = [*rows, "", f"  OVERALL  {overall.row()}"]
    total_leak = sum(leak.values())
    if total_leak:
        lines.append(
            f"  neutral spans (no gold polarity): {total_leak} -> "
            f"{leak['allowed']} filed allowed, {leak['forbidden']} filed forbidden"
        )
    if misses:
        lines += ["", f"  prohibitions filed as ALLOWED ({len(misses)}):"]
        lines += [f"    [{pid}] {c[:96]}" for pid, c in misses]
    report.add(
        title,
        lines,
        {
            f"classify.forbidden_recall.{group}": overall.recall,
            f"classify.forbidden_precision.{group}": overall.precision,
            f"classify.missed_prohibitions.{group}": overall.fn,
            f"classify.neutral_filed_allowed.{group}": leak["allowed"],
            f"classify.neutral_filed_forbidden.{group}": leak["forbidden"],
        },
    )


def test_classify_structured(cbac, chunked, classified, report):
    overall, rows, misses, leak = _score(cbac, chunked, classified, STRUCTURED_POLICIES)
    _emit(
        report,
        "STEP 2a — _classify_chunks on structured skill cards",
        overall,
        rows,
        misses,
        leak,
        "structured",
    )
    assert overall.recall >= MIN_FORBIDDEN_RECALL_STRUCTURED, (
        f"forbidden recall {overall.recall:.2f} < {MIN_FORBIDDEN_RECALL_STRUCTURED}; "
        f"{overall.fn} prohibitions were filed in the allowed bucket, where they act "
        f"as grants"
    )


def test_classify_unstructured(cbac, chunked, classified, report):
    overall, rows, misses, leak = _score(
        cbac, chunked, classified, UNSTRUCTURED_POLICIES
    )
    _emit(
        report,
        "STEP 2b — _classify_chunks on unstructured developer policies",
        overall,
        rows,
        misses,
        leak,
        "unstructured",
    )
    assert overall.recall >= MIN_FORBIDDEN_RECALL_UNSTRUCTURED, (
        f"forbidden recall {overall.recall:.2f} < {MIN_FORBIDDEN_RECALL_UNSTRUCTURED}; "
        f"{overall.fn} prohibitions were filed in the allowed bucket"
    )


def test_classify_stress(cbac, chunked, classified, report):
    """The degenerate and broken documents, reported without a floor.

    No recall requirement is asserted here. A policy whose frontmatter never
    closes, or that names the same action in both lists, has no single correct
    bucketing to hold a classifier to — the numbers are here to be read, and
    the failures they expose belong to the documents as much as to the model.
    """
    overall, rows, misses, leak = _score(cbac, chunked, classified, STRESS_POLICIES)
    _emit(
        report,
        "STEP 2b2 — _classify_chunks on degenerate policies (report only)",
        overall,
        rows,
        misses,
        leak,
        "stress",
    )


def test_forbidden_bucket_is_populated(cbac, chunked, classified, report):
    """Independent of accuracy: does every policy end up with a non-empty
    forbidden bucket?

    Tier 1 substitutes `forbidden_score = 0.0` when the bucket is empty, which
    turns `gap = allowed - forbidden` back into a raw, uncalibrated similarity.
    An empty bucket is therefore not a degraded signal but an inverted one, so
    it is worth counting on its own.
    """
    empty, lines = [], []
    for p in (*STRUCTURED_POLICIES, *UNSTRUCTURED_POLICIES, *STRESS_POLICIES):
        allowed, forbidden = classified[p.id]
        gold_f = sum(1 for c in chunked[p.id] if p.label_for(c) == FORBIDDEN)
        # A degenerate fixture is listed but not charged: `only-forbidden` has
        # no grants by construction, and the point of the others is that they
        # are unreadable.
        if not forbidden and p not in STRESS_POLICIES:
            empty.append(p.id)
        lines.append(
            f"  {p.id:<20} gold_forbidden={gold_f:<3} classified_forbidden={len(forbidden):<3} "
            f"classified_allowed={len(allowed):<3}"
            + ("   <-- EMPTY, Tier-1 gap degenerates" if not forbidden else "")
        )
    report.add(
        "STEP 2c — forbidden-bucket population",
        lines,
        {"classify.empty_forbidden_buckets": len(empty)},
    )
    assert not empty, (
        f"{len(empty)} policies produced an empty forbidden bucket ({', '.join(empty)}), "
        "collapsing Tier 1 to an absolute-similarity test"
    )


def test_chunker_does_not_merge_opposite_polarities(chunked, report):
    """A chunk containing both a grant and a prohibition cannot be labelled by
    any classifier. Counted as a chunking defect, not a classifier one."""
    lines, bad, total_mixed = [], defaultdict(list), 0
    for p in (*STRUCTURED_POLICIES, *UNSTRUCTURED_POLICIES, *STRESS_POLICIES):
        mixed = [c for c in chunked[p.id] if p.label_for(c) == "mixed"]
        total_mixed += len(mixed)
        # `malformed-frontmatter` produces `- read_file forbidden-actions:` — a
        # grant glued to the prohibition header — because its frontmatter never
        # closes. That is the fixture working as intended, not a chunker defect.
        if mixed and p not in STRESS_POLICIES:
            bad[p.id] = mixed
        lines.append(f"  {p.id:<20} {p.style:<14} mixed-polarity chunks: {len(mixed)}")
    for pid, chunks in bad.items():
        lines += [f"    [{pid}] {c[:120]!r}" for c in chunks]
    report.add(
        "STEP 2d — chunk polarity purity",
        lines,
        {
            # Corpus-wide, including the fixtures the assertion excuses.
            "classify.mixed_polarity_chunks": total_mixed,
            "classify.mixed_polarity_chunks.charged": sum(len(v) for v in bad.values()),
        },
    )
    assert not bad, (
        "chunks contain grants and prohibitions in the same unit: "
        + ", ".join(f"{k} ({len(v)})" for k, v in bad.items())
    )


def test_recall_split_by_how_the_prohibition_is_written(
    cbac, chunked, classified, report
):
    """Does the classifier read the language, or the frontmatter key?

    Every gold-forbidden chunk in the corpus, split by whether the chunk text
    literally carries `forbidden-actions:`. The two recalls answer different
    questions: one measures whether the model recognises a prohibition, the
    other measures whether it recognises a label. A wide gap means a policy is
    only as enforceable as its formatting — the same rule written as a sentence
    stops being a prohibition.
    """
    buckets = {"by_label": Binary(), "by_prose": Binary()}
    prose_misses: list[tuple[str, str]] = []

    for p in (*STRUCTURED_POLICIES, *UNSTRUCTURED_POLICIES, *STRESS_POLICIES):
        predicted = set(classified[p.id][1])
        for c in chunked[p.id]:
            gold = p.label_for(c)
            if gold == "mixed":
                continue
            key = "by_label" if "forbidden-actions:" in c.lower() else "by_prose"
            buckets[key].add(c in predicted, gold == FORBIDDEN)
            if gold == FORBIDDEN and key == "by_prose" and c not in predicted:
                prose_misses.append((p.id, c))

    lines = [
        f"  carrying `forbidden-actions:`  {buckets['by_label'].row()}",
        f"  stated in prose               {buckets['by_prose'].row()}",
        "",
        (
            f"  prose prohibitions filed as grants: {len(prose_misses)} across "
            f"{len({pid for pid, _ in prose_misses})} of {len(ALL_POLICIES)} policies"
        ),
    ]
    report.add(
        "STEP 2e — recall by how the prohibition is written",
        lines,
        {
            "classify.forbidden_recall.by_label": buckets["by_label"].recall,
            "classify.forbidden_recall.by_prose": buckets["by_prose"].recall,
            "classify.prose_prohibitions_missed": len(prose_misses),
        },
    )

    # No floor on the gap itself — the requirement is the one already asserted
    # in 2a/2b. This records *why* those fail, so a future fix can be checked
    # against the mechanism rather than only against the aggregate.
    assert buckets["by_prose"].n, "no prose prohibitions in the corpus to score"
