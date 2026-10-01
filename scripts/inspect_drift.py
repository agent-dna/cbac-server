"""Walk one evaluation policy's intents through Check 1, the drift layer.

    uv run python scripts/inspect_drift.py payments
    uv run python scripts/inspect_drift.py payments --intent trip
    uv run python scripts/inspect_drift.py --list

Prints, for every agent action under every user intent:

    1. the mechanical facts a guard posts — callee, description, arguments
    2. `render_intent()`, the one string both user-alignment layers score
    3. `_check1_drift()`, the full NLI softmax beside the verdict it produced,
       and HHEM on the same two texts

The policy is irrelevant here: Check 1 never reads it. The question this layer
answers is only whether the action is what the user asked for, so gold is
`aligned` and nothing else.

Both renderings of the action are scored by default. `render_intent` phrases a
call as ``description or callee_name``, so a callee that supplies a description
replaces the name in the text every layer sees — and that description is written
by the thing being gated. `--description` and `--no-description` narrow to one.

Two columns repay attention. `neutral` usually holds almost all the mass — a
request is a question and an action is a statement, so NLI has no entailment
relation to find even when the action is exactly right. And `intent_score` is
`1 - contradiction`, which counts that neutral mass as alignment: it is not the
gate, it is what gets folded into the agent's trust score.

Loads the NLI cross-encoder and HHEM, so the first run downloads weights.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdna.provenance import Provenance

from cbac_service.cbac import CBAC
from cbac_service.config import CONTRADICTION_THRESHOLD
from cbac_service.skills import render_intent
from cbac_service.tests.eval.corpus import Intent, Spec, spec_by_id

DESCRIBED, BARE = "description", "no_description"


def _score(cbac: CBAC, user_intent: str, action_text: str) -> tuple[str, str]:
    """(table row, verdict) for one rendering of an action."""
    scores = cbac._nli_scores(user_intent, action_text)
    drift, intent_score = asyncio.run(cbac._check1_drift(user_intent, action_text))
    hhem = cbac.hallucination_score(user_intent, action_text)
    verdict = "deny" if drift is not None else "allow"
    row = (
        f"{scores.get('contradiction', 0.0):>9.3f}"
        f"{scores.get('entailment', 0.0):>9.3f}"
        f"{scores.get('neutral', 0.0):>9.3f}"
        f"{intent_score:>14.3f}"
        f"{hhem:>8.3f}"
        f"   {verdict}"
    )
    return row, verdict


def _show_intent(
    cbac: CBAC, intent: Intent, variants: list[str]
) -> dict[str, tuple[int, int]]:
    """Print one intent and its actions. Returns variant -> (misaligned, caught)."""
    print(f"\n--- intent: {intent.id}  ({intent.complexity}) " + "-" * 44)
    print(f"  user asked:  {intent.text}")

    tally = {v: [0, 0] for v in variants}
    for action in intent.actions:
        print(f"\n  action: {action.id}")
        print("    1. facts the guard posts")
        print(f"       callee       {action.callee}")
        print(f"       description  {action.description or '(none)'}")
        for k, v in action.args.items():
            print(f"       argument     {k} = {v}")
        if not action.args:
            print("       argument     (none)")

        # Without a description `render_intent` falls back to the callee name,
        # so for an action that never had one the two renderings are the same
        # string and scoring it twice would say nothing.
        texts = {
            DESCRIBED: action.text,
            BARE: render_intent(action.callee, action.args, None),
        }
        same = action.description is None
        shown = [variants[0]] if same else variants

        print("\n    2. render_intent() -> the text both layers score")
        for v in shown:
            label = "both (no description)" if same else v
            print(f"       {label:<22}{texts[v]}")

        print("\n    3. _check1_drift(user_intent, action_text)")
        print(
            f"       {'':<22}{'contra':>9}{'entail':>9}{'neutral':>9}"
            f"{'intent_score':>14}{'hhem':>8}   verdict"
        )
        gold = "allow" if action.aligned else "deny"
        for v in shown:
            row, verdict = _score(cbac, intent.text, texts[v])
            label = "both (no description)" if same else v
            mark = "  !" if verdict != gold else ""
            print(f"       {label:<22}{row}{mark}")
            for target in variants if same else [v]:
                tally[target][0] += not action.aligned
                tally[target][1] += (not action.aligned) and verdict == "deny"
        print(f"       gold {gold}  ({action.relation})")

    return {v: (n, c) for v, (n, c) in tally.items()}


def _show(cbac: CBAC, spec: Spec, only: str | None, variants: list[str]) -> None:
    rule = "=" * 78
    print(f"\n{rule}\n{spec.id}  —  CHECK 1, intent drift\n{rule}")
    print(f"  gate: deny when contradiction >= {CONTRADICTION_THRESHOLD}")
    print(f"  action text: {', '.join(variants)}")

    totals = {v: [0, 0] for v in variants}
    for intent in spec.intents:
        if only and intent.id != only:
            continue
        for v, (misaligned, caught) in _show_intent(cbac, intent, variants).items():
            totals[v][0] += misaligned
            totals[v][1] += caught
    print(f"\n{rule}\n  misaligned actions denied by the gate:")
    for v, (total, hit) in totals.items():
        print(f"    {v:<22}{hit}/{total}")
    print(f"  (! marks every disagreement with gold)\n{rule}")


def main() -> None:
    specs = spec_by_id()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("spec", nargs="?", default="payments", help="policy id")
    ap.add_argument("--intent", help="show only this intent id")
    ap.add_argument("--description", action="store_true", help="only the text as sent")
    ap.add_argument(
        "--no-description",
        action="store_true",
        help="only the text with the callee's description stripped",
    )
    ap.add_argument("--list", action="store_true", help="list the policy ids")
    args = ap.parse_args()

    if args.list:
        for spec in specs.values():
            ids = ", ".join(i.id for i in spec.intents)
            print(f"  {spec.id:<20}{ids}")
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

    # Neither flag, or both, means both renderings.
    variants = [
        v
        for v, picked in ((DESCRIBED, args.description), (BARE, args.no_description))
        if picked
    ] or [DESCRIBED, BARE]

    # A test double: Check 1 reads only the two texts, never the chain.
    cbac = CBAC(provenance=cast(Provenance, SimpleNamespace()))
    _show(cbac, spec, args.intent, variants)


if __name__ == "__main__":
    main()
