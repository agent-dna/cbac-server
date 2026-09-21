"""Stage 1 — can `_classify_chunks` tell a grant from a prohibition?

Everything downstream is conditional on this. A prohibition filed in the
allowed bucket does not merely lose signal: it becomes the chunk that maximises
Tier 1's `max_allowed` for the very action it bans.

Scored per **capability**, not per chunk. Shapes collapse differently — a prose
paragraph holds five rules in one chunk, a bulleted list holds one each — so a
chunk-level rate rewards a shape for producing fewer, bigger chunks. Every
rendering has the same capabilities to get right, which is what makes the two
halves of a pair comparable at all.
"""

from __future__ import annotations

import math
from collections import defaultdict

from cbac_service.tests.eval.corpus import (
    ALLOWED,
    FORBIDDEN,
    NEUTRAL,
    STRUCTURED,
    UNSTRUCTURED,
    has_prohibition_token,
    specs,
)
from cbac_service.tests.eval.harness import Binary
from cbac_service.tests.eval.runner import (
    CLASSIFIER_ARMS,
    FORBID_FLOOR,
    LOST,
    MIXED_CHUNK,
    OK,
)

# Requirements, not observations. A policy the classifier cannot read is a
# policy the pipeline cannot enforce, and the shape it is written in should not
# decide that.
MIN_FORBIDDEN_RECALL_STRUCTURED = 0.90
MIN_FORBIDDEN_RECALL_UNSTRUCTURED = 0.70


def _recall(caps, gold) -> Binary:
    b = Binary()
    for c in caps:
        b.add(predicted=c.bucket == gold, actual=c.gold == gold)
    return b


def test_capability_classification(run, report):
    lines, metrics = [], {}
    for arm in CLASSIFIER_ARMS:
        caps = [c for c in run.caps if c.arm == arm]
        f, a = _recall(caps, FORBIDDEN), _recall(caps, ALLOWED)
        lost = [c for c in caps if c.status == LOST]
        mixed = [c for c in caps if c.status == MIXED_CHUNK]
        lines += [
            f"  {arm}",
            f"    forbidden  {f.row()}",
            f"    allowed    {a.row()}",
            (
                f"    unreachable before any classifier ran: {len(lost)} severed "
                f"by chunking, {len(mixed)} in a mixed chunk"
            ),
            "",
        ]
        metrics |= {
            f"classify.{arm}.forbidden_recall": f.recall,
            f"classify.{arm}.forbidden_precision": f.precision,
            f"classify.{arm}.allowed_recall": a.recall,
            f"classify.{arm}.lost_to_chunking": len(lost),
            f"classify.{arm}.mixed_chunks": len(mixed),
        }
    report.add("STAGE 1a — capability classification", lines, metrics)

    assert (
        metrics[f"classify.{CLASSIFIER_ARMS[0]}.forbidden_recall"]
        >= MIN_FORBIDDEN_RECALL_STRUCTURED
    ), (
        "a skill card states its prohibitions in a labelled list and they still go missing"
    )
    assert (
        metrics[f"classify.{CLASSIFIER_ARMS[1]}.forbidden_recall"]
        >= MIN_FORBIDDEN_RECALL_UNSTRUCTURED
    ), "the same policy, written in prose, is not read as the same policy"


def _surface(chunk: str) -> str:
    """How a chunk states its polarity, if it states one at all."""
    low = chunk.strip().lower()
    if low.startswith(("forbidden-actions:", "allowed-actions:")):
        return "frontmatter key"
    if has_prohibition_token(chunk):
        return "prose, marked"
    return "prose, unmarked"


def test_label_versus_language(run, report):
    """Is the classifier reading the prohibition, or the convention around it?

    Three surfaces, same sentences underneath: a frontmatter key that names the
    polarity in the chunk's own text, prose that states it in words ("must
    never", "shall not"), and prose that only describes the capability with the
    polarity living in a heading the chunker deleted. If recall tracks the
    surface rather than the content, the pipeline is matching a convention.
    """
    lines, metrics = [f"  {'surface':<20}{'arm':<14}{'recall':>10}"], {}
    for surface in ("frontmatter key", "prose, marked", "prose, unmarked"):
        for arm in CLASSIFIER_ARMS:
            caps = [
                c
                for c in run.caps
                if c.arm == arm
                and c.gold == FORBIDDEN
                and c.chunk
                and _surface(c.chunk) == surface
            ]
            if not caps:
                continue
            hit = sum(c.status == OK for c in caps)
            lines.append(
                f"  {surface:<20}{arm:<14}{hit:>4}/{len(caps):<5}{hit / len(caps):.2f}"
            )
            metrics[
                f"classify.{arm}.recall_{surface.split(',')[0].replace(' ', '_')}"
            ] = hit / len(caps)
    report.add("STAGE 1b — label vs language", lines, metrics)


def test_the_forbidden_floor(run, report):
    """Why the forbidden bucket is empty: the floor is never cleared.

    `_classify_chunks` files a chunk as forbidden only when the entailment of
    "This capability is prohibited and forbidden" beats the allowed hypothesis
    *and* clears a hardcoded 0.40. This is the distribution of that score on the
    chunks that genuinely hold a prohibition — if it sits well below the floor,
    no amount of clearer policy writing changes the bucket.
    """
    by_chunk = {(c.arm, c.text): c for c in run.chunks}
    scores = sorted(
        by_chunk[(c.arm, c.chunk)].forbid_e
        for c in run.caps
        if c.gold == FORBIDDEN and c.chunk and (c.arm, c.chunk) in by_chunk
    )
    n = len(scores)

    def q(f: float) -> float:
        return scores[min(n - 1, int(f * n))]

    cleared = sum(x > FORBID_FLOOR for x in scores)
    lines = [
        f"  forbid_e over the {n} chunks holding a real prohibition",
        (
            f"    median {q(0.5):.3f}   p75 {q(0.75):.3f}   p90 {q(0.90):.3f}   "
            f"max {scores[-1]:.3f}"
        ),
        f"    above the {FORBID_FLOOR} floor: {cleared}/{n} ({cleared / n:.0%})",
        "",
        "  The floor and the two hypothesis strings are literals in cbac.py, not",
        "  config. Nothing a policy author writes moves a chunk over a floor the",
        "  score does not reach.",
    ]
    report.add(
        "STAGE 1b2 — the forbidden floor",
        lines,
        {
            "classify.forbid_e.median": q(0.5),
            "classify.forbid_e.p90": q(0.90),
            "classify.forbid_e.above_floor": cleared / n,
        },
    )


def test_neutral_spans(run, report):
    """Where does normatively inert text land?

    Real policies carry identifiers, issuance metadata and descriptive prose
    that grant nothing. The pipeline has no bucket for them, so every one of
    them inflates the allowed side of Tier 1's gap.
    """
    lines, metrics = [], {}
    for arm in CLASSIFIER_ARMS:
        rows = [c for c in run.chunks if c.arm == arm and c.gold == NEUTRAL]
        leaked = sum(r.bucket == ALLOWED for r in rows)
        lines.append(
            f"  {arm:<14}{leaked:>3}/{len(rows):<4} inert spans filed as a grant"
        )
        metrics[f"classify.{arm}.neutral_filed_allowed"] = leaked
        metrics[f"classify.{arm}.neutral_spans"] = len(rows)
    report.add("STAGE 1c — inert text", lines, metrics)


def test_empty_buckets(run, report):
    """A bucket with nothing in it turns Tier 1's gap into an absolute score.

    `_tiered_decision` substitutes `0.0` for a missing side, so with an empty
    forbidden bucket the "gap" is just `max_allowed` — a quantity no threshold
    was calibrated against.
    """
    lines, metrics = [], {}
    for arm in ("structured", "unstructured", "oracle"):
        empty_f = [s.id for s in specs() if run.buckets[(s.id, arm)][1] == 0]
        empty_a = [s.id for s in specs() if run.buckets[(s.id, arm)][0] == 0]
        lines.append(f"  {arm:<14}empty forbidden: {len(empty_f):<3} {empty_f}")
        if empty_a:
            lines.append(f"  {'':<14}empty allowed  : {len(empty_a):<3} {empty_a}")
        metrics[f"classify.{arm}.empty_forbidden_buckets"] = len(empty_f)
    report.add("STAGE 1d — empty buckets", lines, metrics)


def test_paired_delta(run, report):
    """The controlled comparison: same policy, two shapes.

    A corpus of different documents cannot separate "the shape hurt it" from
    "that is a different policy". These twelve pairs hold the capability
    sentences fixed and vary only the scaffolding, so the per-spec difference is
    attributable to shape.
    """
    per = defaultdict(dict)
    for arm in CLASSIFIER_ARMS:
        for s in specs():
            caps = [c for c in run.caps if c.arm == arm and c.spec_id == s.id]
            per[s.id][arm] = _recall(caps, FORBIDDEN).recall

    lines = [
        f"  {'policy':<17}{'shape':<22}{'card':>6}{'shape':>7}{'delta':>8}",
    ]
    deltas, wins, losses, ties = [], 0, 0, 0
    for s in specs():
        a, b = per[s.id]["structured"], per[s.id]["unstructured"]
        if math.isnan(a) or math.isnan(b):  # no forbidden capabilities to score
            continue
        d = a - b
        deltas.append(d)
        wins += d > 0
        losses += d < 0
        ties += d == 0
        lines.append(f"  {s.id:<17}{s.shape:<22}{a:>6.2f}{b:>7.2f}{d:>+8.2f}")
    mean = sum(deltas) / len(deltas) if deltas else float("nan")
    lines += [
        "",
        (
            f"  mean delta {mean:+.2f} over {len(deltas)} pairs — card better on "
            f"{wins}, worse on {losses}, level on {ties}"
        ),
    ]
    report.add(
        "STAGE 1e — paired structured vs unstructured",
        lines,
        {
            "classify.paired.mean_delta": mean,
            "classify.paired.card_better": wins,
            "classify.paired.card_worse": losses,
        },
    )


def test_by_shape(run, report):
    """Which document shapes the classifier can read.

    Each shape carries the same number of policies, and every policy's other
    rendering is a skill card over the identical sentences, so a shape's row is
    its own paired delta. `lost` counts capabilities no chunk holds whole and
    `mixed` counts those sharing a chunk with the opposite polarity — both are
    chunking defects that cap a shape's recall before classification runs.
    """
    lines = [
        (
            f"  {'shape':<24}{'n':>3}{'recall':>9}{'card':>7}{'delta':>8}"
            f"{'lost':>6}{'mixed':>7}"
        )
    ]
    metrics = {}
    shapes = sorted({c.shape for c in run.caps if c.arm == UNSTRUCTURED})
    for shape in shapes:
        caps = [c for c in run.caps if c.shape == shape and c.gold == FORBIDDEN]
        pids = {c.spec_id for c in caps}
        twin = [
            c
            for c in run.caps
            if c.arm == STRUCTURED and c.spec_id in pids and c.gold == FORBIDDEN
        ]
        r = _recall(caps, FORBIDDEN).recall
        card = _recall(twin, FORBIDDEN).recall
        lines.append(
            f"  {shape:<24}{len(pids):>3}{r:>9.2f}{card:>7.2f}{card - r:>+8.2f}"
            f"{sum(c.status == LOST for c in caps):>6}"
            f"{sum(c.status == MIXED_CHUNK for c in caps):>7}"
        )
        metrics[f"classify.shape.{shape}.forbidden_recall"] = r
    lines += [
        "",
        "  `card` is the same policies read as skill cards, so each row's delta",
        "  is a within-policy comparison and not an artefact of which documents",
        "  happened to land on that shape.",
    ]
    report.add("STAGE 1f — by document shape", lines, metrics)
