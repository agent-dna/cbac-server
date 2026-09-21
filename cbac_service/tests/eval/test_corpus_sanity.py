"""Is the ground truth well-formed? Model-free, so it runs in ordinary pytest.

Everything else in this directory is derived from the specs, so a malformed one
silently corrupts every number downstream. The load-bearing check is the first:
both renderings of a policy must carry the identical capability sentences. That
invariant is what makes a structured-vs-unstructured delta a statement about
document shape — a renderer that paraphrased would turn its own delta into a
wording effect and nothing here would notice.
"""

from __future__ import annotations

from collections import Counter

import pytest

from cbac_service.chunking import flatten_policy_chunks
from cbac_service.skills import parse_skill_md
from cbac_service.tests.eval.corpus import (
    ALLOW,
    BASES,
    COMPLEXITIES,
    DENY,
    EVASIONS,
    GRAY,
    RELATIONS,
    SHAPES,
    STRUCTURED,
    UNSTRUCTURED,
    UNSTRUCTURED_SHAPES,
    specs,
)

SPECS = specs()
ALL_SPECS = pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])


def test_corpus_is_not_empty():
    assert len(SPECS) >= 16, "the paired deltas need enough policies to mean anything"


@ALL_SPECS
def test_both_renderings_carry_every_capability(spec):
    """The invariant the whole paired design rests on."""
    for rendering in spec.renderings:
        low = rendering.text.lower()
        missing = [c.id for c in spec.capabilities if c.text.lower() not in low]
        assert not missing, (
            f"{spec.id}/{rendering.shape} dropped {missing} — the two renderings "
            f"no longer describe the same policy, so their delta is not a shape effect"
        )


@ALL_SPECS
def test_renderings_are_one_of_each_kind(spec):
    kinds = [r.kind for r in spec.renderings]
    assert kinds == [STRUCTURED, UNSTRUCTURED]
    assert spec.shape in SHAPES and spec.shape != "skill-card"


@ALL_SPECS
def test_structured_rendering_parses_as_a_skill_card(spec):
    """`flatten_policy_chunks` only takes the frontmatter path for a card the
    parser accepts; a card that fails silently falls back to prose chunking and
    would be measuring a different thing than it claims to."""
    card = parse_skill_md(spec.renderings[0].text)
    assert [c.text for c in spec.allowed] == card.allowed_actions
    assert [c.text for c in spec.forbidden] == card.forbidden_actions


@ALL_SPECS
def test_capability_ids_and_texts_are_unique(spec):
    ids = [c.id for c in spec.capabilities]
    texts = [c.text.lower() for c in spec.capabilities]
    assert len(set(ids)) == len(ids), f"{spec.id}: duplicate capability id"
    assert len(set(texts)) == len(texts), f"{spec.id}: duplicate capability text"


@ALL_SPECS
def test_no_capability_text_contains_another(spec):
    """Gold labelling reads a chunk's capabilities back by substring. One
    capability nested inside another would make every chunk holding the longer
    one look like it holds both."""
    texts = [c.text.lower() for c in spec.capabilities]
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i != j:
                assert a not in b, f"{spec.id}: {a!r} is a substring of {b!r}"


@ALL_SPECS
def test_oracle_is_the_spec(spec):
    allowed, forbidden = spec.oracle_buckets
    assert allowed == [c.text for c in spec.allowed]
    assert forbidden == [c.text for c in spec.forbidden]


@ALL_SPECS
def test_actions_are_well_formed(spec):
    seen = set()
    for intent, action in spec.actions:
        key = (intent.id, action.id)
        assert key not in seen, f"{spec.id}: duplicate action id {key}"
        seen.add(key)
        assert intent.complexity in COMPLEXITIES
        assert action.relation in RELATIONS
        assert action.policy in (ALLOW, DENY, GRAY)
        # `aligned` is not a free axis: it is the definition of `faithful`.
        # Letting the two disagree would put a case in the drift layer's
        # must-catch set that the relation label says it should pass.
        assert action.aligned == (action.relation == "faithful"), (
            f"{spec.id}/{intent.id}/{action.id}: relation={action.relation} but "
            f"aligned={action.aligned}"
        )
        assert action.text.startswith("The agent wants to ")
        assert action.basis in BASES
        assert (action.basis == "granted") == (action.policy != DENY), (
            f"{spec.id}/{intent.id}/{action.id}: every deny needs a basis, and "
            f"only a deny has one"
        )


@ALL_SPECS
def test_an_action_is_a_function_call(spec):
    """An action is held as the mechanical facts a guard posts, not as prose.

    The guard's `_stringify` sends `arguments` as `dict[str, str]`, and
    `render_intent` phrases each pair as `k = v` into the text the scorers see.
    So every argument has to be a non-empty string and has to survive into the
    rendered action — an argument that does not reach the text is one the
    pipeline never had a chance to judge.

    The space check catches a YAML flow mapping split on a comma
    (`{notes: a, b}` parses as two keys, the second one null), which silently
    truncates the value a case was written to test.
    """
    for intent, action in spec.actions:
        where = f"{spec.id}/{intent.id}/{action.id}"
        for k, v in action.args.items():
            assert " " not in k, (
                f"{where}: argument name {k!r} contains a space — an unquoted "
                f"flow mapping split on a comma"
            )
            assert isinstance(v, str) and v.strip(), (
                f"{where}: argument {k!r} is {v!r}, but the wire format is "
                f"dict[str, str]"
            )
            assert f"{k} = {v}" in action.text, (
                f"{where}: argument {k!r} does not reach the scored text"
            )


@ALL_SPECS
def test_every_complexity_level_is_present(spec):
    levels = {i.complexity for i in spec.intents}
    assert levels == set(COMPLEXITIES), (
        f"{spec.id} is missing {set(COMPLEXITIES) - levels}"
    )


@ALL_SPECS
def test_a_spec_has_cases_on_both_sides(spec):
    golds = {a.policy for _i, a in spec.actions}
    blocks = {a.should_block for _i, a in spec.actions}
    if spec.allowed:
        assert ALLOW in golds, (
            f"{spec.id}: no permitted action — nothing measures a leak"
        )
    assert DENY in golds, f"{spec.id}: no forbidden action to block"
    assert blocks != {False}, f"{spec.id}: nothing that must be blocked"


def test_every_shape_carries_the_same_number_of_policies():
    """A shape used by one policy is confounded with that policy's content —
    the failure mode this corpus exists to remove. Balanced use is what makes a
    per-shape number an estimate rather than an anecdote."""
    counts = Counter(s.shape for s in SPECS)
    assert set(counts) == set(UNSTRUCTURED_SHAPES), (
        f"never exercised: {set(UNSTRUCTURED_SHAPES) - set(counts)}"
    )
    assert len(set(counts.values())) == 1, (
        f"shapes are used unequally, so a per-shape number is not comparable "
        f"across shapes: {dict(counts)}"
    )


def test_evasive_actions_are_denials():
    """An evasion is an attempt to get a forbidden action past the scorers. One
    attached to a permitted action would land in the block rate as a case the
    pipeline is *supposed* to allow."""
    for spec in SPECS:
        for intent, action in spec.actions:
            assert action.evasion in EVASIONS, (
                f"{spec.id}/{intent.id}/{action.id}: unknown evasion {action.evasion!r}"
            )
            if action.evasion:
                assert action.policy == DENY, (
                    f"{spec.id}/{intent.id}/{action.id}: an evasion on a "
                    f"{action.policy} action is not an attack"
                )


def test_every_evasion_is_represented():
    seen = {a.evasion for s in SPECS for _i, a in s.actions if a.evasion}
    missing = set(EVASIONS) - {""} - seen
    assert not missing, f"never exercised: {missing}"


def test_every_relation_is_represented():
    seen = {a.relation for s in SPECS for _i, a in s.actions}
    assert seen == set(RELATIONS), f"never exercised: {set(RELATIONS) - seen}"


@ALL_SPECS
def test_chunking_recovers_most_capabilities(spec):
    """How many capability sentences survive the chunker at all.

    A sentence severed by `split_by_word_budget`'s hard cut is unreachable by
    any classifier, so it caps the recall of the shape that produced it. This
    asserts only that a rendering is not *mostly* destroyed; the exact losses
    are a measured finding, reported by `test_classification.py`.
    """
    for rendering in spec.renderings:
        chunks = [c.lower() for c in flatten_policy_chunks(rendering.text)]
        survived = sum(
            any(c.text.lower() in ch for ch in chunks) for c in spec.capabilities
        )
        assert survived >= len(spec.capabilities) / 2, (
            f"{spec.id}/{rendering.shape}: only {survived}/{len(spec.capabilities)} "
            f"capabilities survive chunking"
        )
