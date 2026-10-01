"""Walk one evaluation policy through the first three steps of `index_policy`.

    uv run python scripts/inspect_policy.py payments
    uv run python scripts/inspect_policy.py --list

Prints, for both renderings of the same policy — the skill card and its
unstructured twin, which carry the identical capability sentences:

    1. the document as stored
    2. `flatten_policy_chunks()`, the chunks the classifier is handed
    3. `_classify_chunks()`, which bucket each one landed in, beside the two
       entailment scores that put it there

Step 3 is where a verdict usually becomes explicable: a prohibition filed in the
allowed bucket is the chunk that then maximises Tier 1's `max_allowed` for the
very action it bans. `forbid_e` is the number to read — a chunk is filed
forbidden only when it beats `allow_e` *and* clears a hardcoded 0.40.

Loads the encoder and the NLI cross-encoder, so the first run downloads weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdna.provenance import Provenance

from cbac_service.cbac import CBAC
from cbac_service.chunking import flatten_policy_chunks
from cbac_service.tests.eval.corpus import Rendering, Spec, spec_by_id

# The two hypotheses `_classify_chunks` scores every chunk against, and the
# floor its forbidden branch has to clear (`cbac.py`). Restated here so the raw
# scores can be shown beside the buckets they produced.
ALLOW_HYPOTHESIS = "This capability is permitted and allowed"
FORBID_HYPOTHESIS = "This capability is prohibited and forbidden"
FORBID_FLOOR = 0.40

# Step 3's chunk column is elided to keep the table aligned. The ellipsis marks
# it, because a chunk the *chunker* severed looks exactly like a truncated one
# and the two mean opposite things — step 2 prints every chunk whole, and the
# line under it names the capabilities that no chunk holds whole.
CHUNK_COL = 72


def _elide(text: str) -> str:
    return text if len(text) <= CHUNK_COL else text[: CHUNK_COL - 1] + "\u2026"


def _show(cbac: CBAC, spec: Spec, rendering: Rendering) -> None:
    rule = "=" * 78
    print(
        f"\n{rule}\n{spec.id}  —  {rendering.kind.upper()}  ({rendering.shape})\n{rule}"
    )

    print("\n--- 1. policy (the document as stored) " + "-" * 39)
    print(rendering.text.rstrip())

    chunks = flatten_policy_chunks(rendering.text)
    print(f"\n--- 2. flatten_policy_chunks() -> {len(chunks)} chunks " + "-" * 34)
    for i, chunk in enumerate(chunks):
        print(f"  [{i:>2}] {chunk}")

    # A capability no single chunk holds whole was severed by
    # `split_by_word_budget`'s hard cut, which fires on a unit over
    # CHUNK_MAX_WORDS with no sentence boundary to split on. It is unreachable
    # from here: no bucket can hold it, however good the classifier is.
    lowered = [c.lower() for c in chunks]
    severed = [
        cap
        for cap in spec.capabilities
        if not any(cap.text.lower() in c for c in lowered)
    ]
    if severed:
        n = len(severed)
        print(
            f"\n  {n} capabilit{'y' if n == 1 else 'ies'} "
            f"survive{'s' if n == 1 else ''} no chunk whole "
            f"(severed by split_by_word_budget):"
        )
        for cap in severed:
            print(f"    {cap.id:<22}{_elide(cap.text)}")
    else:
        print("\n  every capability survives chunking into one chunk")

    allowed, forbidden = cbac._classify_chunks(chunks)
    scores = cbac._nli_scores_batch(
        [(c, h) for c in chunks for h in (ALLOW_HYPOTHESIS, FORBID_HYPOTHESIS)]
    )
    in_forbidden = set(forbidden)
    print(
        f"\n--- 3. _classify_chunks() -> {len(allowed)} allowed / "
        f"{len(forbidden)} forbidden " + "-" * 24
    )
    print(f"  {'':<4}{'bucket':<11}{'gold':<11}{'allow_e':>8}{'forbid_e':>9}  chunk")
    for i, chunk in enumerate(chunks):
        bucket = "forbidden" if chunk in in_forbidden else "allowed"
        gold = rendering.label_for(chunk)
        # `neutral` and `mixed` have no correct bucket: the pipeline has two and
        # a policy carries a third kind of text, and a chunk the chunker severed
        # across both polarities cannot be filed correctly by any classifier.
        wrong = gold not in ("neutral", "mixed") and bucket != gold
        allow_e = scores[2 * i].get("entailment", 0.0)
        forbid_e = scores[2 * i + 1].get("entailment", 0.0)
        print(
            f"  {'!' if wrong else '':<4}{bucket:<11}{gold:<11}"
            f"{allow_e:>8.3f}{forbid_e:>9.3f}  {_elide(chunk)}"
        )
    print(
        f"\n  ! = filed in the wrong bucket. forbidden needs "
        f"forbid_e > allow_e and > {FORBID_FLOOR}"
    )


def main() -> None:
    specs = spec_by_id()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("spec", nargs="?", default="payments", help="policy id")
    ap.add_argument("--list", action="store_true", help="list the policy ids")
    args = ap.parse_args()

    if args.list:
        for spec in specs.values():
            print(f"  {spec.id:<20}{spec.shape:<24}{spec.domain}")
        return
    if args.spec not in specs:
        raise SystemExit(
            f"no policy {args.spec!r} — run with --list to see the {len(specs)} ids"
        )

    spec = specs[args.spec]
    # A test double: nothing here reaches the Provenance Layer, because every
    # policy is handed in as text.
    cbac = CBAC(provenance=cast(Provenance, SimpleNamespace()))
    for rendering in spec.renderings:
        _show(cbac, spec, rendering)


if __name__ == "__main__":
    main()
