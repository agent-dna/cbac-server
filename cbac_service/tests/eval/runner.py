"""One pass over the corpus, recording what every stage of the pipeline did.

The pipeline's stages are all callable on their own — `_classify_chunks`,
`_check1_drift`, `_tiered_decision`, `hallucination_score` need no session, and
`_decide` needs none either once the three module-level search names are
pointed at an in-memory index. This module calls each of them on the same
inputs and writes down every number, so a wrong verdict can be traced to the
stage that produced it instead of inferred from the verdict alone.

**Every stage runs on every case, including stages `_decide` skips.** `_decide`
returns as soon as Check 1 denies and never reaches the tiers, so on those rows
the policy layer's answer exists nowhere in the pipeline's own output. Without
it there is no way to separate "the drift layer caught this" from "only the
drift layer could have caught this", which is the question the whole per-layer
split exists to answer.

The final verdict comes from `_decide` itself rather than from re-composing the
stage signals here. A harness that re-implements the composition measures its
own copy of the pipeline, and the copy is where it silently stops matching.

Three arms, differing only in the policy index the tiers search:

    structured     the index `_classify_chunks` built from the skill card
    unstructured   the index it built from the same policy's other shape
    oracle         the index a correct chunker and a correct classifier would
                   build — taken from the spec, so it is identical for both
                   renderings and acts as a shared upper bound

Classification is scored per **capability**, not per chunk. Shapes collapse
differently — a legal-prose paragraph holds five rules in one chunk while a
bulleted list holds one each — so a chunk-level rate rewards a shape for
producing fewer, larger chunks. Every rendering has the same capabilities to
get right, which is what makes the paired numbers comparable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import cbac_service.cbac as cbac_mod
from cbac_service.chunking import flatten_policy_chunks
from cbac_service.skills import render_intent
from cbac_service.tests.eval.corpus import (
    ALLOWED,
    FORBIDDEN,
    MIXED,
    STRUCTURED,
    UNSTRUCTURED,
    Action,
    Intent,
    Spec,
    has_prohibition_token,
    specs,
)
from cbac_service.tests.eval.harness import PolicyIndex, build_index, patch_search

ARMS = (STRUCTURED, UNSTRUCTURED, "oracle")
CLASSIFIER_ARMS = (STRUCTURED, UNSTRUCTURED)

# Why a capability is not in the bucket it belongs in.
OK = "ok"
MISFILED = "misfiled"  # the classifier put its chunk in the wrong bucket
MIXED_CHUNK = "mixed_chunk"  # its chunk also carries the opposite polarity
LOST = "lost_to_chunking"  # no chunk contains it at all


@dataclass(frozen=True)
class CapRow:
    """One capability, as one rendering's index actually filed it."""

    spec_id: str
    arm: str
    shape: str
    cap_id: str
    cap_text: str
    gold: str
    bucket: str | None
    status: str
    chunk: str | None
    # Does the chunk state the prohibition as a *label* ("forbidden-actions:",
    # "must not") rather than only in the language of the rule? Splitting recall
    # on this is what separates reading the word from reading the sentence.
    labelled: bool

    @property
    def correct(self) -> bool:
        return self.status == OK


# The two hypotheses `_classify_chunks` scores every chunk against
# (`cbac.py`), and the floor the forbidden branch has to clear. Restated here
# so the raw scores can be reported beside the buckets; `_build` asserts that
# reapplying the rule reproduces `_classify_chunks`, which fires if either the
# wording or the threshold moves in `cbac.py`.
ALLOW_HYPOTHESIS = "This capability is permitted and allowed"
FORBID_HYPOTHESIS = "This capability is prohibited and forbidden"
FORBID_FLOOR = 0.40


@dataclass(frozen=True)
class ChunkRow:
    """One produced chunk, where it landed, and the two numbers that put it
    there. Carries the neutral spans, which have no capability to be scored
    against."""

    spec_id: str
    arm: str
    shape: str
    text: str
    gold: str
    bucket: str
    allow_e: float
    forbid_e: float


@dataclass
class ArmResult:
    decision: str
    error_code: int | None
    reason: str
    policy_score: float | None
    allowed_score: float
    forbidden_score: float
    gap: float
    top_chunk: str | None
    final_decision: str
    final_code: int | None


@dataclass
class CaseRow:
    """One (policy, intent, action), with every stage's signal on it.

    The drift and hallucination signals carry no arm: neither layer reads the
    policy, so they are the same number under all three indices. That is itself
    the finding those layers exist for — they answer a question about the user,
    not about the agent's permissions.
    """

    spec_id: str
    intent_id: str
    complexity: str
    user_intent: str
    action_id: str
    call: str
    action_text: str
    relation: str
    aligned: bool
    policy_gold: str
    evasion: str
    basis: str
    should_block: bool
    contradiction: float
    entailment: float
    neutral: float
    intent_score: float
    drift_denied: bool
    hhem: float
    arms: dict[str, ArmResult] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.spec_id}/{self.intent_id}/{self.action_id}"


@dataclass
class BareRow:
    """The same action with its `description` removed.

    `render_intent` phrases a call as `description or callee_name`, so a callee
    that supplies a description replaces the name in the text every layer scores
    — and that description is written by the thing being gated. Stripping it
    falls back to the de-snaked callee name, which is the only part of the call
    an enforcement point can vouch for.

    Held beside `cases` rather than as a fourth arm on them: the index the tiers
    search and the text they score are independent, and folding one into the
    other is how the corpus this suite replaced became uninterpretable.
    """

    key: str
    spec_id: str
    # False for an action that never carried one, where this row is a duplicate
    # of its case. Kept so every rate covers the whole corpus, not a subset.
    had_description: bool
    action_text: str
    contradiction: float
    intent_score: float
    drift_denied: bool
    hhem: float
    arms: dict[str, ArmResult] = field(default_factory=dict)


@dataclass
class Run:
    caps: list[CapRow]
    chunks: list[ChunkRow]
    cases: list[CaseRow]
    # The same actions with their descriptions stripped, same order as `cases`.
    bare: list[BareRow]
    # (spec_id, arm) -> (n_allowed, n_forbidden) in the index the tiers searched
    buckets: dict[tuple[str, str], tuple[int, int]]
    # The indices themselves, so a test can re-decide a variant of an action
    # against the same policy without rebuilding and re-encoding it.
    indices: dict[tuple[str, str], PolicyIndex]
    # Cases whose Tier 2 answer changes when BM25 fusion is switched on.
    hybrid_flips: list[dict[str, Any]]


# ── Model memoisation ────────────────────────────────────────────────────────


def _memoise(cbac) -> None:
    """Cache the three model calls for the duration of a run.

    The arms differ only in which chunks the tiers search, so the drift NLI, the
    hallucination score and the encoding of an action are identical across all
    three. Without this the second and third arm pay full model cost to
    recompute numbers that cannot have changed.
    """
    if getattr(cbac, "_eval_memoised", False):
        return
    cbac._eval_memoised = True

    # Memoised at the batch entry point, which every NLI caller funnels
    # through (`_nli_scores` is a batch of one). Uncached pairs are still sent
    # as one batch, so batching is preserved.
    nli_cache: dict[tuple[str, str], dict[str, float]] = {}
    raw_batch = cbac._nli_scores_batch

    def nli_batch(pairs):
        missing = [p for p in pairs if p not in nli_cache]
        if missing:
            nli_cache.update(zip(missing, raw_batch(missing)))
        return [nli_cache[p] for p in pairs]

    def nli(premise: str, hypothesis: str):
        return nli_batch([(premise, hypothesis)])[0]

    hhem_cache: dict[tuple[str, str], float] = {}
    raw_hhem = cbac.hallucination_score

    def hhem(source: str, generated: str) -> float:
        k = (source, generated)
        if k not in hhem_cache:
            hhem_cache[k] = raw_hhem(source, generated)
        return hhem_cache[k]

    cbac._nli_scores = nli
    cbac._nli_scores_batch = nli_batch
    cbac.hallucination_score = hhem

    encoder = cbac._get_encoder()
    if not getattr(encoder, "_eval_memoised", False):
        raw_encode = encoder.encode
        enc_cache: dict[tuple[str, ...], Any] = {}

        def encode(sentences, **kw):
            if not isinstance(sentences, list) or len(sentences) != 1:
                return raw_encode(sentences, **kw)
            k = (sentences[0], kw.get("normalize_embeddings", False))
            if k not in enc_cache:
                enc_cache[k] = raw_encode(sentences, **kw)
            return enc_cache[k]

        encoder.encode = encode
        encoder._eval_memoised = True


# ── Index construction ───────────────────────────────────────────────────────


def _cap_rows(spec: Spec, rendering, chunks, forbidden_set) -> list[CapRow]:
    rows = []
    for cap in spec.capabilities:
        gold = ALLOWED if cap in spec.allowed else FORBIDDEN
        holder = next((c for c in chunks if cap.text.lower() in c.lower()), None)
        if holder is None:
            status, bucket, labelled = LOST, None, False
        elif rendering.label_for(holder) == MIXED:
            status, bucket = MIXED_CHUNK, None
            labelled = has_prohibition_token(holder)
        else:
            bucket = FORBIDDEN if holder in forbidden_set else ALLOWED
            status = OK if bucket == gold else MISFILED
            labelled = has_prohibition_token(holder)
        rows.append(
            CapRow(
                spec_id=spec.id,
                arm=rendering.kind,
                shape=rendering.shape,
                cap_id=cap.id,
                cap_text=cap.text,
                gold=gold,
                bucket=bucket,
                status=status,
                chunk=holder,
                labelled=labelled,
            )
        )
    return rows


def _build(
    cbac, spec: Spec
) -> tuple[dict[str, PolicyIndex], list[CapRow], list[ChunkRow]]:
    """The three indices for one spec, plus the classification ground truth."""
    indices: dict[str, PolicyIndex] = {}
    caps: list[CapRow] = []
    chunk_rows: list[ChunkRow] = []

    for rendering in spec.renderings:
        chunks = flatten_policy_chunks(rendering.text)
        allowed, forbidden = cbac._classify_chunks(chunks)
        forbidden_set = set(forbidden)
        # Free from the memo: `_classify_chunks` just scored these same pairs.
        scores = cbac._nli_scores_batch(
            [(c, h) for c in chunks for h in (ALLOW_HYPOTHESIS, FORBID_HYPOTHESIS)]
        )
        e = [
            (
                scores[2 * i].get("entailment", 0.0),
                scores[2 * i + 1].get("entailment", 0.0),
            )
            for i in range(len(chunks))
        ]
        assert [
            c
            for i, c in enumerate(chunks)
            if e[i][1] > e[i][0] and e[i][1] > FORBID_FLOOR
        ] == list(forbidden), (
            "reapplying the classifier rule no longer reproduces _classify_chunks "
            "— the hypotheses or the floor moved in cbac.py"
        )
        indices[rendering.kind] = build_index(
            cbac,
            list(allowed) + list(forbidden),
            [ALLOWED] * len(allowed) + [FORBIDDEN] * len(forbidden),
        )
        caps += _cap_rows(spec, rendering, chunks, forbidden_set)
        chunk_rows += [
            ChunkRow(
                spec_id=spec.id,
                arm=rendering.kind,
                shape=rendering.shape,
                text=c,
                gold=rendering.label_for(c),
                bucket=FORBIDDEN if c in forbidden_set else ALLOWED,
                allow_e=e[i][0],
                forbid_e=e[i][1],
            )
            for i, c in enumerate(chunks)
        ]

    ga, gf = spec.oracle_buckets
    indices["oracle"] = build_index(
        cbac, list(ga) + list(gf), [ALLOWED] * len(ga) + [FORBIDDEN] * len(gf)
    )
    return indices, caps, chunk_rows


# ── The pass ─────────────────────────────────────────────────────────────────


def _arm_result(cbac, text: str, user_intent: str) -> ArmResult:
    decision, reason, code, score = asyncio.run(
        cbac._tiered_decision(None, "eval", text)
    )
    # The two cosine scores the gap is built from. Recomputed rather than parsed
    # back out of `reason`: a reason string is prose for a human, and Tier 2 and
    # Tier 3 do not put them in it at all.
    vec = np.asarray(cbac._get_encoder().encode([text], normalize_embeddings=True))[0]
    a = asyncio.run(cbac_mod.vector_search(None, "eval", vec, 1, ALLOWED))
    f = asyncio.run(cbac_mod.vector_search(None, "eval", vec, 1, FORBIDDEN))
    allowed_score = a[0].score if a else 0.0
    forbidden_score = f[0].score if f else 0.0

    final = asyncio.run(cbac._decide(None, "eval", text, user_intent))
    return ArmResult(
        decision=decision,
        error_code=code,
        reason=reason,
        policy_score=score,
        allowed_score=allowed_score,
        forbidden_score=forbidden_score,
        gap=allowed_score - forbidden_score,
        top_chunk=a[0].chunk_text if a else None,
        final_decision=final.decision,
        final_code=final.error_code,
    )


def _hybrid_flips(cbac, monkeypatch, spec, indices, cases) -> list[dict[str, Any]]:
    """Does BM25 fusion change any Tier 2 answer?

    `HYBRID_SEARCH_ENABLED` defaults to false, so the shipped pipeline never
    calls `hybrid_search` — and neither does the rest of this run. The flag is a
    module-level binding in `cbac.py`, so flipping it is one setattr; anything
    that changes here is a decision that depends on a default nobody set
    deliberately.
    """
    flips = []
    monkeypatch.setattr(cbac_mod, "HYBRID_SEARCH_ENABLED", True)
    try:
        for arm in ARMS:
            patch_search(monkeypatch, indices[arm])
            for case in cases:
                before = case.arms[arm]
                decision, _r, code, _s = asyncio.run(
                    cbac._tiered_decision(None, "eval", case.action_text)
                )
                if decision != before.decision or code != before.error_code:
                    flips.append(
                        {
                            "key": case.key,
                            "arm": arm,
                            "from": [before.decision, before.error_code],
                            "to": [decision, code],
                        }
                    )
    finally:
        monkeypatch.setattr(cbac_mod, "HYBRID_SEARCH_ENABLED", False)
    return flips


def run(cbac, monkeypatch, *, hybrid_check: bool = True) -> Run:
    _memoise(cbac)
    caps: list[CapRow] = []
    chunks: list[ChunkRow] = []
    all_cases: list[CaseRow] = []
    all_bare: list[BareRow] = []
    buckets: dict[tuple[str, str], tuple[int, int]] = {}
    all_indices: dict[tuple[str, str], PolicyIndex] = {}
    flips: list[dict[str, Any]] = []

    for spec in specs():
        indices, spec_caps, spec_chunks = _build(cbac, spec)
        caps += spec_caps
        chunks += spec_chunks
        for arm, idx in indices.items():
            all_indices[(spec.id, arm)] = idx
            buckets[(spec.id, arm)] = (
                sum(t == ALLOWED for t in idx.chunk_types),
                sum(t == FORBIDDEN for t in idx.chunk_types),
            )

        # Stages that never read the policy: one pass per text variant, before
        # the arms. The variants score differently here too — drift and HHEM
        # read the action text, not the policy.
        cases: list[CaseRow] = []
        bare: list[BareRow] = []
        for intent, action in spec.actions:
            cases.append(_user_layers(cbac, spec, intent, action))
            bare.append(_bare_layers(cbac, spec, intent, action))

        for arm in ARMS:
            patch_search(monkeypatch, indices[arm])
            for case, row in zip(cases, bare):
                case.arms[arm] = _arm_result(cbac, case.action_text, case.user_intent)
                row.arms[arm] = _arm_result(cbac, row.action_text, case.user_intent)

        if hybrid_check:
            flips += _hybrid_flips(cbac, monkeypatch, spec, indices, cases)
        all_cases += cases
        all_bare += bare

    return Run(
        caps=caps,
        chunks=chunks,
        cases=all_cases,
        bare=all_bare,
        buckets=buckets,
        indices=all_indices,
        hybrid_flips=flips,
    )


def _user_signals(cbac, user_intent: str, action_text: str):
    """(nli scores, drift denied, intent_score, hhem) for one (intent, action).

    Recomputed per text variant rather than carried across: both layers read the
    action text, so stripping a description changes what they see.
    """
    scores = cbac._nli_scores(user_intent, action_text)
    drift, intent_score = asyncio.run(cbac._check1_drift(user_intent, action_text))
    return (
        scores,
        drift is not None,
        intent_score,
        cbac.hallucination_score(user_intent, action_text),
    )


def _bare_layers(cbac, spec: Spec, intent: Intent, action: Action) -> BareRow:
    text = render_intent(action.callee, action.args, None)
    scores, denied, intent_score, hhem = _user_signals(cbac, intent.text, text)
    return BareRow(
        key=f"{spec.id}/{intent.id}/{action.id}",
        spec_id=spec.id,
        had_description=action.description is not None,
        action_text=text,
        contradiction=scores.get("contradiction", 0.0),
        intent_score=intent_score,
        drift_denied=denied,
        hhem=hhem,
    )


def _user_layers(cbac, spec: Spec, intent: Intent, action: Action) -> CaseRow:
    scores, denied, intent_score, hhem = _user_signals(cbac, intent.text, action.text)
    return CaseRow(
        spec_id=spec.id,
        intent_id=intent.id,
        complexity=intent.complexity,
        user_intent=intent.text,
        action_id=action.id,
        call=action.call,
        action_text=action.text,
        relation=action.relation,
        aligned=action.aligned,
        policy_gold=action.policy,
        evasion=action.evasion,
        basis=action.basis,
        should_block=action.should_block,
        contradiction=scores.get("contradiction", 0.0),
        entailment=scores.get("entailment", 0.0),
        neutral=scores.get("neutral", 0.0),
        intent_score=intent_score,
        drift_denied=denied,
        hhem=hhem,
    )
