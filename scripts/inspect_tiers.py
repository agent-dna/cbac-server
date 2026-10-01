"""Walk one evaluation policy's actions through the tiers — the policy layer.

    uv run python scripts/inspect_tiers.py payments
    uv run python scripts/inspect_tiers.py payments --intent trip
    uv run python scripts/inspect_tiers.py payments --oracle
    uv run python scripts/inspect_tiers.py --list

Prints, for every agent action, against both renderings of the same policy:

    TIER 1  the top allowed and top forbidden chunk the intent vector retrieved,
            their cosine scores, the gap between them and how it compares to
            ALLOW_GAP / DENY_GAP
    TIER 2  the chunk NLI was run against, and the entailment / contradiction
            scores beside their thresholds
    TIER 3  whether it fell through to the fallback

Every tier is shown even when an earlier one decided, and the one that actually
produced the verdict is marked. That is the point of the script: a Tier 1 allow
tells you nothing about whether Tier 2 would have caught the same action, and
the gap between them is usually where a wrong verdict is explained.

The user's request is shown for context but the tiers never read it — this layer
answers only whether the *policy* permits the action, so gold is `policy`.

`--oracle` searches the index a correct chunker and classifier would have built,
taken from the spec. The difference between that and the two real renderings is
classification error rather than tier error.

Loads the encoder and the NLI cross-encoder, so the first run downloads weights.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _pytest.monkeypatch import MonkeyPatch
from agentdna.provenance import Provenance

import cbac_service.cbac as cbac_mod
import cbac_service.error_codes as ec
from cbac_service.cbac import CBAC
from cbac_service.chunking import flatten_policy_chunks
from cbac_service.config import (
    ALLOW_GAP,
    CONTRADICTION_THRESHOLD,
    DENY_GAP,
    ENTAILMENT_THRESHOLD,
    HYBRID_SEARCH_ENABLED,
)
from cbac_service.tests.eval.corpus import Intent, Spec, spec_by_id
from cbac_service.tests.eval.harness import build_index, patch_search

TIER = {
    ec.TIER1_GAP_ALLOW: 1,
    ec.TIER1_GAP_DENY: 1,
    ec.TIER2_NO_ALLOWED_CHUNKS: 2,
    ec.TIER2_ENTAILMENT_ALLOW: 2,
    ec.TIER2_CONTRADICTION_DENY: 2,
    ec.TIER3_NO_BACKEND_DENY: 3,
}
COL = 58


def _elide(text: str | None) -> str:
    if text is None:
        return "(bucket is empty)"
    return text if len(text) <= COL else text[: COL - 1] + "…"


def _search(cbac: CBAC, text: str, chunk_type: str) -> tuple[float, str | None]:
    vec = cbac._get_encoder().encode([text], normalize_embeddings=True)
    hits = asyncio.run(
        cbac_mod.vector_search(None, "eval", vec[0], 1, chunk_type)  # type: ignore[arg-type]
    )
    return (hits[0].score, hits[0].chunk_text) if hits else (0.0, None)


def _show_action(cbac: CBAC, intent: Intent, action: Any) -> None:
    print(f"\n  action: {action.id}  ({action.relation}, gold {action.policy})")
    print(f"    {action.text}")

    decision, _reason, code, _score = asyncio.run(
        cbac._tiered_decision(None, "eval", action.text)  # type: ignore[arg-type]
    )
    fired = TIER.get(code)

    a_score, a_chunk = _search(cbac, action.text, "allowed")
    f_score, f_chunk = _search(cbac, action.text, "forbidden")
    gap = a_score - f_score
    mark = lambda n: "  <-- decided here" if fired == n else ""

    print(f"\n    TIER 1 — cosine gap{mark(1)}")
    print(f"      allowed    {a_score:.3f}  {_elide(a_chunk)}")
    print(f"      forbidden  {f_score:.3f}  {_elide(f_chunk)}")
    print(
        f"      gap        {gap:+.3f}   allow if > +{ALLOW_GAP}, deny if < -{DENY_GAP}"
    )
    if gap > ALLOW_GAP:
        print("      -> allow")
    elif gap < -DENY_GAP:
        print("      -> deny")
    else:
        print("      -> inconclusive, escalate")

    reached = " " if fired == 2 else "  [not reached]"
    source = "hybrid_search" if HYBRID_SEARCH_ENABLED else "vector_search (hybrid off)"
    print(f"\n    TIER 2 — NLI vs the top allowed chunk{mark(2) or reached}")
    if a_chunk is None:
        print("      no allowed chunk to compare against -> deny")
    else:
        s = cbac._nli_scores(a_chunk, action.text)
        print(f"      chunk      {_elide(a_chunk)}   (from {source})")
        print(
            f"      entailment {s.get('entailment', 0.0):.3f}   "
            f"contradiction {s.get('contradiction', 0.0):.3f}   "
            f"neutral {s.get('neutral', 0.0):.3f}"
        )
        print(
            f"      allow if entailment >= {ENTAILMENT_THRESHOLD}, "
            f"deny if contradiction >= {CONTRADICTION_THRESHOLD}"
        )
        if s.get("entailment", 0.0) >= ENTAILMENT_THRESHOLD:
            print("      -> allow")
        elif s.get("contradiction", 0.0) >= CONTRADICTION_THRESHOLD:
            print("      -> deny")
        else:
            print("      -> inconclusive, escalate")

    if fired == 3:
        print("\n    TIER 3 — no llm_backend configured  <-- decided here")
        print("      -> deny (the fallback, not a detection)")

    wrong = action.policy != "gray" and decision != action.policy
    print(
        f"\n    {'!' if wrong else ' '} VERDICT  {decision:<6} {code}   "
        f"gold {action.policy}"
    )


def _show(cbac: CBAC, spec: Spec, only: str | None, oracle: bool) -> None:
    mp = MonkeyPatch()
    arms: list[tuple[str, list[str], list[str]]] = []
    if oracle:
        allowed, forbidden = spec.oracle_buckets
        arms.append(("oracle (a correct index, from the spec)", allowed, forbidden))
    else:
        for rendering in spec.renderings:
            chunks = flatten_policy_chunks(rendering.text)
            a, f = cbac._classify_chunks(chunks)
            arms.append((f"{rendering.kind} ({rendering.shape})", list(a), list(f)))

    try:
        for label, allowed, forbidden in arms:
            rule = "=" * 78
            print(f"\n{rule}\n{spec.id}  —  TIERS, index: {label}\n{rule}")
            print(f"  {len(allowed)} allowed chunks, {len(forbidden)} forbidden chunks")
            patch_search(
                mp,
                build_index(
                    cbac,
                    allowed + forbidden,
                    ["allowed"] * len(allowed) + ["forbidden"] * len(forbidden),
                ),
            )
            for intent in spec.intents:
                if only and intent.id != only:
                    continue
                print(f"\n--- intent: {intent.id} " + "-" * 52)
                print(f"  user asked:  {intent.text}   (the tiers never read this)")
                for action in intent.actions:
                    _show_action(cbac, intent, action)
    finally:
        mp.undo()


def main() -> None:
    specs = spec_by_id()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("spec", nargs="?", default="payments", help="policy id")
    ap.add_argument("--intent", help="show only this intent id")
    ap.add_argument(
        "--oracle", action="store_true", help="search a correct index instead"
    )
    ap.add_argument("--list", action="store_true", help="list the policy ids")
    args = ap.parse_args()

    if args.list:
        for spec in specs.values():
            print(f"  {spec.id:<20}{', '.join(i.id for i in spec.intents)}")
        return
    if args.spec not in specs:
        raise SystemExit(
            f"no policy {args.spec!r} — run with --list to see the {len(specs)} ids"
        )
    spec = specs[args.spec]
    if args.intent and args.intent not in {i.id for i in spec.intents}:
        raise SystemExit(
            f"no intent {args.intent!r} in {spec.id} — "
            f"has {', '.join(i.id for i in spec.intents)}"
        )

    # A test double: the policy is handed in as text and the search layer is
    # pointed at an in-memory index, so neither the chain nor Postgres is used.
    cbac = CBAC(provenance=cast(Provenance, SimpleNamespace()))
    _show(cbac, spec, args.intent, args.oracle)


if __name__ == "__main__":
    main()
