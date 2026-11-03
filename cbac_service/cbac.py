import asyncio
import base64
import fnmatch
import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import structlog
from agentdna.provenance import Provenance
from agentdna.types import AgentCard
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from cbac_service import error_codes as ec
from cbac_service import fga
from cbac_service.config import (
    ALLOW_GAP,
    CHUNK_MAX_WORDS,
    CONTRADICTION_THRESHOLD,
    DENY_GAP,
    ENCODER_MODEL,
    ENTAILMENT_THRESHOLD,
    HHEM_MODEL,
    HYBRID_SEARCH_ENABLED,
    INTENT_TTL_SECONDS,
    LHI_LAMBDA_DOWN,
    LHI_LAMBDA_UP,
    LHI_WEIGHTS,
    NLI_BATCH_SIZE,
    NLI_MODEL,
    POLICY_PARSER_MODEL,
    QUERY_PARSER_MODEL,
)
from cbac_service.db.models import CBACIntent, PolicyVersion
from cbac_service.db.repository import (
    get_active_policy_version,
    get_intent,
    get_latest_trust,
    get_policy_chunks,
    insert_cbac_decision,
    insert_intent,
    insert_lhi_record,
    insert_policy_version,
    lock_intent,
    mark_policy_version_active,
    save_policy_chunks,
)
from cbac_service.db.search import SearchResult, hybrid_search, vector_search
from cbac_service.ibac import (
    WHOLE_TOOL,
    NormalizationError,
    canonical,
    fga_user,
    object_id,
    resource_values,
)
from cbac_service.parsers import (
    flatten_policy_chunks,
    parse_policy,
    parse_query,
    split_by_word_budget,
    split_policy_document,
)
from cbac_service.skills import (
    VALID_LLM_DECISIONS,
    CBACResult,
    LLMVerdict,
    _intended_action_text,
    render_intent,
)

# TODO:- Fix return types in CBAC class

logger = structlog.get_logger("cbac_service.cbac")


def _policy_hash(policy: str) -> str:
    """Content hash keying the embedding cache. The precompute side (writes it)
    and the read side (compares it) must derive it identically, or cache
    invalidation breaks — so both go through here."""
    return hashlib.sha256(policy.encode()).hexdigest()


def _interaction_hash(
    agent_id: str,
    decision: str,
    reason: str,
    intended_action: str,
    user_intent: str | None,
    callee_name: str | None,
    callee_type: str | None,
    error_code: int | None,
    salt: str,
    intent_id: str | None = None,
) -> str:
    """Content hash over one full ``cbac_decisions`` row, computed by
    :meth:`CBAC._record_decision` immediately before insert.

    Same construction as :func:`_policy_hash` — sha256 over a canonical
    string — extended here to a JSON array rather than a plain join so a
    value that happens to contain whatever separator we'd otherwise pick
    (a comma, a pipe) can't collide two different rows onto the same hash.
    Field order is fixed by this function, not sorted, since this is a
    positional row fingerprint, not a dict digest. ``intent_id`` —  it is
    appended last (before ``salt``) so a historical hash computed before
    this field existed cannot collide with one that includes it.

    ``salt`` is a fresh random value the caller generates per row (see
    ``_record_decision`` — ``uuid.uuid4()``), not a derived or fixed value.
    Its only job is to guarantee the hash is unique per row: two decisions
    with byte-for-byte identical content (a genuine retry of the same
    interaction) get *different* hashes, deliberately, because the hash is
    now a per-row identifier, not a content fingerprint — do not reuse it
    for dedup/retry detection, since it can no longer do that job. This
    function stays deterministic and unit-testable given a fixed ``salt``;
    the randomness lives at the call site, not in here.
    """
    payload = json.dumps(
        [
            agent_id,
            decision,
            reason,
            intended_action,
            user_intent,
            callee_name,
            callee_type,
            error_code,
            intent_id,
            salt,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _Tier1:
    """What Tier 1 measured for one text — kept so Tier 2 reuses the search."""

    intent_vec: np.ndarray
    allowed_results: list[SearchResult]
    allowed_score: float
    forbidden_score: float
    gap: float
    gap_score: float | None


def _deny(
    reason: str, error_code: int, intent_score: float | None = None
) -> CBACResult:
    return CBACResult(
        decision="deny", reason=reason, error_code=error_code, intent_score=intent_score
    )


def _as_text(value: Any) -> str:
    """A free-text argument as the text the scorers read. Structured values
    are serialized whole, so nothing inside them escapes the checks."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class CBAC:
    def __init__(
        self,
        provenance: Provenance,
        encoder_name: str = ENCODER_MODEL,
        nli_model_name: str = NLI_MODEL,
        # Contract: an async callable taking (intent_text, policy_text) and
        # returning an LLMVerdict — never a bare string. See _tiered_decision's
        # Tier 3 branch for what happens when a backend doesn't honor this.
        llm_backend: Callable[[str, str], Awaitable[LLMVerdict]] | None = None,
        allow_gap: float = ALLOW_GAP,
        deny_gap: float = DENY_GAP,
        hhem_model_name: str = HHEM_MODEL,
        lhi_weights: tuple[float, float, float] = LHI_WEIGHTS,
        lhi_lambda_up: float = LHI_LAMBDA_UP,
        lhi_lambda_down: float = LHI_LAMBDA_DOWN,
    ):
        self.provenance = provenance
        self._encoder_name = encoder_name
        self._nli_model_name = nli_model_name
        self._llm_backend = llm_backend
        self._allow_gap = allow_gap
        self._deny_gap = deny_gap
        self._hhem_model_name = hhem_model_name
        self._lhi_weights = lhi_weights
        self._lhi_lambda_up = lhi_lambda_up
        self._lhi_lambda_down = lhi_lambda_down

        self._encoder = None
        self._nli = None
        self._hhem = None
        self._nli_labels: dict[int, str] = {}

    # ── Model accessors (lazy-loaded) ─────────────────────────────────────────

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._encoder_name)
        return self._encoder

    def _get_nli(self):
        if self._nli is None:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._nli = CrossEncoder(self._nli_model_name)
            try:
                if not self._nli.model:
                    raise ValueError("NLI model is not initialised")
                id2label = self._nli.model.config.id2label

                if id2label is None:
                    raise ValueError("id2label is not initialised")

                self._nli_labels = {i: lbl.lower() for i, lbl in id2label.items()}
            except AttributeError:
                # Fallback for deberta NLI label order (contradiction/entailment/neutral)
                self._nli_labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
        return self._nli

    def _get_hhem(self):
        if self._hhem is None:
            from transformers import AutoModelForSequenceClassification

            # trust_remote_code is required by HHEM-2.1 (custom predict() head).
            self._hhem = AutoModelForSequenceClassification.from_pretrained(
                self._hhem_model_name, trust_remote_code=True
            )
        return self._hhem

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def hallucination_score(self, source_text: str, generated_text: str) -> float:
        """Score how well ``generated_text`` is grounded in ``source_text``.

        Returns a score in [0, 1]: 1 = fully supported, 0 = hallucinated.
        """
        model = self._get_hhem()
        return float(model.predict([(source_text, generated_text)])[0])

    def _nli_scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI cross-encoder on a (premise, hypothesis) pair.

        Returns a dict like {'entailment': 0.82, 'contradiction': 0.05, 'neutral': 0.13}.
        Scores are softmax-normalised probabilities. Thin wrapper over the
        batched path — a single pair is just a batch of one.
        """
        return self._nli_scores_batch([(premise, hypothesis)])[0]

    def _nli_scores_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Batched NLI over many (premise, hypothesis) pairs in one predict() call.

        Same per-pair result shape as calling :meth:`_nli_scores` in a loop,
        but the cross-encoder processes ``pairs`` in mini-batches instead of
        one sample at a time — the loop form never lets the model's internal
        batching do anything useful.
        """
        from scipy.special import softmax as sp_softmax

        if not pairs:
            return []
        nli = self._get_nli()
        raw = nli.predict(
            pairs,
            apply_softmax=False,
            batch_size=NLI_BATCH_SIZE,
            show_progress_bar=False,
        )
        probs = sp_softmax(raw, axis=1)
        return [
            {self._nli_labels.get(i, str(i)): float(row[i]) for i in range(len(row))}
            for row in probs
        ]

    # ── Policy processing ─────────────────────────────────────────────────────

    def _classify_chunks(self, chunks: list[str]) -> tuple[list[str], list[str]]:
        """NLI-classify each chunk as allowed or forbidden."""
        if not chunks:
            return [], []
        pairs: list[tuple[str, str]] = []
        for chunk in chunks:
            pairs.append((chunk, "This capability is permitted and allowed"))
            pairs.append((chunk, "This capability is prohibited and forbidden"))
        scores = self._nli_scores_batch(pairs)

        allowed: list[str] = []
        forbidden: list[str] = []
        for i, chunk in enumerate(chunks):
            allow_e = scores[2 * i].get("entailment", 0.0)
            forbid_e = scores[2 * i + 1].get("entailment", 0.0)
            if forbid_e > allow_e and forbid_e > 0.40:
                forbidden.append(chunk)
            else:
                allowed.append(chunk)
        return allowed, forbidden

    def _get_latest_agent_policy(self, agent_id: str) -> str:
        """Returns the latest decoded policy associated with an agent."""
        actor_card_dict = self.provenance.get_latest_provenance_record(
            actor_id=agent_id
        )
        actor_card = AgentCard(**actor_card_dict)
        try:
            return base64.b64decode(actor_card.policy).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"failed to decode policy for agent {agent_id}: {exc}"
            ) from exc

    # ── Precompute (DB-backed) ────────────────────────────────────────────────

    async def index_policy(
        self,
        session: AsyncSession,
        agent_id: str,
        policy: str | None = None,
    ) -> int:
        """Precompute and cache policy vectors for an agent into Postgres.

        Steps:
        1. Take ``policy`` when supplied, else fetch it from the Provenance Layer.
        2. Flatten it to text chunks.
        3. NLI-classify each chunk into allowed / forbidden buckets.
        4. Encode all chunks with the bi-encoder.
        5. Persist to the policy_chunks table via the repository layer.

        Always recomputes, replacing whatever was indexed before. That makes
        this the *only* way an agent's policy changes after the first decision:
        ``verify_cbac`` reads the indexed copy and consults the Provenance Layer
        only when there is none, so a card republished on chain reaches
        decisions when this runs and not before. Behind
        ``POST /cbac/v1/policies/precompute`` for exactly that reason.

        Passing ``policy`` indexes that text instead of the agent's card —
        which also means it decides real calls until something re-indexes.

        Returns the number of chunks stored.
        """
        if policy is None:
            policy = self._get_latest_agent_policy(agent_id=agent_id)
        if not policy:
            raise RuntimeError(f"No policy found for agent {agent_id}")

        current_hash = _policy_hash(policy)

        # Flatten and classify.
        chunks = flatten_policy_chunks(policy)
        if not chunks:
            raise RuntimeError(f"Policy for agent {agent_id} produced no chunks")

        allowed_chunks, forbidden_chunks = await asyncio.to_thread(
            self._classify_chunks, chunks
        )

        # Encode all chunks.
        all_chunks = allowed_chunks + forbidden_chunks
        chunk_types = (["allowed"] * len(allowed_chunks)) + (
            ["forbidden"] * len(forbidden_chunks)
        )

        encoder = self._get_encoder()
        embeddings = await asyncio.to_thread(
            # asarray is a no-op for the ndarray encode() actually returns; it narrows
            # the declared 5-way union (Tensor | ndarray | dict | ...) to what the DB
            # layer accepts.
            lambda: np.asarray(encoder.encode(all_chunks, normalize_embeddings=True))
        )

        # Persist to DB.
        count = await save_policy_chunks(
            session=session,
            agent_id=agent_id,
            chunks=all_chunks,
            chunk_types=chunk_types,
            embeddings=embeddings,
            policy_hash=current_hash,
        )

        return count

    # ── Check 1: NLI drift ────────────────────────────────────────────────────

    async def _check1_drift(
        self,
        user_intent: str,
        agent_action: str,
    ) -> tuple[tuple[str, str, int] | None, float]:
        """NLI drift check: does the agent's action contradict the user's intent?

        Returns ``((decision, reason, error_code), intent_score)`` if
        contradiction is strong enough to deny, else ``(None, intent_score)``.
        """
        scores = await asyncio.to_thread(self._nli_scores, user_intent, agent_action)
        contradiction = scores.get("contradiction", 0.0)
        intent_score = 1.0 - contradiction
        if contradiction >= CONTRADICTION_THRESHOLD:
            return (
                (
                    "deny",
                    (
                        f"Check 1 drift: user intent {user_intent!r} contradicts agent action "
                        f"{agent_action!r} (NLI contradiction={contradiction:.2f})"
                    ),
                    ec.CHECK1_DRIFT_DENY,
                ),
                intent_score,
            )
        return None, intent_score

    # ── Tiered decision (DB-backed search) ────────────────────────────────────

    async def _tier1_gap(
        self, session: AsyncSession, agent_id: str, intent_text: str
    ) -> _Tier1:
        """Tier 1 inputs: the closest allowed and forbidden chunks and their gap.

        Shared by v1 (:meth:`_tiered_decision`, where the gap can allow or deny)
        and v2 (:meth:`_policy_drift`, where it can only deny).
        """
        # Encode only the intent at runtime (~5 ms on CPU).
        encoder = self._get_encoder()
        intent_vec = await asyncio.to_thread(
            lambda: np.asarray(
                encoder.encode([intent_text], normalize_embeddings=True)
            )[0]
        )

        # Tier 1: cosine gap via pgvector.
        allowed_results = await vector_search(
            session, agent_id, intent_vec, top_k=1, chunk_type="allowed"
        )
        forbidden_results = await vector_search(
            session, agent_id, intent_vec, top_k=1, chunk_type="forbidden"
        )

        allowed_score = allowed_results[0].score if allowed_results else 0.0
        forbidden_score = forbidden_results[0].score if forbidden_results else 0.0
        gap = allowed_score - forbidden_score

        span = self._allow_gap + self._deny_gap
        gap_score = (
            max(0.0, min(1.0, (gap + self._deny_gap) / span)) if span > 0 else None
        )
        return _Tier1(
            intent_vec, allowed_results, allowed_score, forbidden_score, gap, gap_score
        )

    async def _tier2_nli(
        self, session: AsyncSession, agent_id: str, intent_text: str, t1: _Tier1
    ) -> tuple[str | None, dict[str, float]]:
        """Tier 2 inputs: the top allowed chunk and the NLI scores against it.
        ``(None, {})`` when the agent has no allowed chunk."""
        # Use hybrid search if enabled for better candidate selection,
        # otherwise fall back to the vector search result we already have.
        if HYBRID_SEARCH_ENABLED and t1.allowed_results:
            top_results = await hybrid_search(
                session,
                agent_id,
                t1.intent_vec,
                intent_text,
                top_k=1,
                chunk_type="allowed",
            )
            top_chunk = top_results[0].chunk_text if top_results else None
        elif t1.allowed_results:
            top_chunk = t1.allowed_results[0].chunk_text
        else:
            top_chunk = None

        if not top_chunk:
            return None, {}
        scores = await asyncio.to_thread(self._nli_scores, top_chunk, intent_text)
        return top_chunk, scores

    async def _tiered_decision(
        self,
        session: AsyncSession,
        agent_id: str,
        intent_text: str,
    ) -> tuple[str, str, int, float | None]:
        """Tier 1 (cosine gap) → Tier 2 (NLI entailment) → Tier 3 (LLM).

        Uses pgvector search for Tier 1 instead of in-memory numpy operations.
        Returns ``(decision, reason, error_code, policy_score)``.
        """
        t1 = await self._tier1_gap(session, agent_id, intent_text)
        allowed_score, forbidden_score = t1.allowed_score, t1.forbidden_score
        gap, gap_score = t1.gap, t1.gap_score

        if gap > self._allow_gap:
            return (
                "allow",
                (
                    f"Tier 1 cosine gap {gap:+.3f} > +{self._allow_gap} "
                    f"(allowed={allowed_score:.3f}, forbidden={forbidden_score:.3f})"
                ),
                ec.TIER1_GAP_ALLOW,
                gap_score,
            )
        if gap < -self._deny_gap:
            return (
                "deny",
                (
                    f"Tier 1 cosine gap {gap:+.3f} < -{self._deny_gap} "
                    f"(intent closer to forbidden than allowed policy)"
                ),
                ec.TIER1_GAP_DENY,
                gap_score,
            )

        # Tier 2: NLI entailment vs top allowed chunk.
        top_chunk, t2_scores = await self._tier2_nli(session, agent_id, intent_text, t1)
        if not top_chunk:
            return (
                "deny",
                "Tier 2: no allowed policy chunks found",
                ec.TIER2_NO_ALLOWED_CHUNKS,
                None,
            )

        entailment = t2_scores.get("entailment", 0.0)
        contradiction = t2_scores.get("contradiction", 0.0)

        if entailment >= ENTAILMENT_THRESHOLD:
            return (
                "allow",
                f"Tier 2 NLI entailment={entailment:.2f} vs {top_chunk!r}",
                ec.TIER2_ENTAILMENT_ALLOW,
                entailment,
            )
        if contradiction >= CONTRADICTION_THRESHOLD:
            return (
                "deny",
                f"Tier 2 NLI contradiction={contradiction:.2f} vs {top_chunk!r}",
                ec.TIER2_CONTRADICTION_DENY,
                entailment,
            )

        # Tier 3: LLM judgment (optional).
        if self._llm_backend is None:
            return (
                "deny",
                (
                    f"Tier 1/2 inconclusive (gap={gap:+.3f}, "
                    f"entailment={entailment:.2f}, contradiction={contradiction:.2f}); "
                    "no LLM backend configured"
                ),
                ec.TIER3_NO_BACKEND_DENY,
                None,
            )

        # Gather policy text for LLM context.
        all_chunks = await get_policy_chunks(session, agent_id)
        policy_text = "\n".join(all_chunks)

        try:
            verdict = await self._llm_backend(intent_text, policy_text)
        except Exception as e:
            return ("deny", f"Tier 3 LLM error: {e}", ec.TIER3_LLM_ERROR_DENY, None)

        # `llm_backend` is contracted to return an LLMVerdict (see skills.py),
        # not free text — a misbehaving backend (wrong type, or a decision
        # outside the known three) degrades to "deny" rather than raising or
        # silently guessing from prose.
        decision = getattr(verdict, "decision", None)
        reason_text = getattr(verdict, "reason", None)
        if not isinstance(verdict, LLMVerdict) or decision not in VALID_LLM_DECISIONS:
            return (
                "deny",
                f"Tier 3 LLM returned an unrecognised verdict: {verdict!r}",
                ec.TIER3_LLM_MALFORMED_DENY,
                None,
            )

        code = {
            "allow": ec.TIER3_LLM_ALLOW,
            "deny": ec.TIER3_LLM_DENY,
            "advise": ec.TIER3_LLM_ADVISE_DENY,
        }[decision]
        # A backend that itself returns "advise" is folded to deny too — the
        # pipeline no longer has a caller-must-decide state; `decision` (not
        # `final_decision`) still appears in the reason so the audit log shows
        # what the backend actually said.
        final_decision = "deny" if decision == "advise" else decision
        return (final_decision, f"Tier 3 LLM ({decision}): {reason_text}", code, None)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def _fold_trust(
        self,
        session: AsyncSession,
        agent_id: str,
        callee_name: str,
        callee_type: str,
        result: CBACResult,
        mcp_did: str = "",
    ) -> CBACResult:
        """Fold a reached decision's component scores into the edge's trust and
        stamp the new value onto ``result``.

        Never gates the decision: a failed trust update is logged, not raised —
        the same treatment hallucination scoring gets. The decision has already
        been made against the policy, and a DB hiccup here must not turn a valid
        ``allow`` into a blocked call.

        Skipped without a ``callee_name``: the edge key would be empty and every
        caller would pool into one junk edge.
        """
        if not callee_name:
            return result
        try:
            result.trust = await self.compute_lhi(
                session,
                agent_id,
                callee_name,
                callee_type,
                intent_score=result.intent_score,
                policy_score=result.policy_score,
                hallucination_score=result.hallucination_score,
                mcp_did=mcp_did,
            )
        except Exception:
            # The audit write runs next on this same session. A failed insert
            # leaves the transaction pending-rollback, so without this the trust
            # fold takes the decision record down with it -- and `cbac_decisions`
            # is the one table that must have no skip conditions.
            await session.rollback()
            logger.warning(
                "lhi trust update failed",
                callee_name=callee_name,
                decision=result.decision,
                exc_info=True,
            )
        return result

    async def _record_decision(
        self,
        session: AsyncSession,
        agent_id: str,
        intent_text: str,
        user_intent: str | None,
        callee_name: str,
        callee_type: str,
        result: CBACResult,
        intent_id: str | None = None,
    ) -> str | None:
        """Append the verdict to the ``cbac_decisions`` audit log.

        Never gates the decision, on the same grounds as :meth:`_fold_trust`:
        the verdict is already settled, and failing to *write down* what was
        decided must not change what was decided. A failed insert is logged.

        Unlike the trust fold this has no skip conditions — every verdict is
        recorded, including the infrastructure-failure denies and calls with no
        callee, which is exactly what ``lhi_records`` cannot give us.

        ``interaction_hash`` is computed here, over the exact row about to be
        inserted, immediately before the insert — not earlier in the pipeline
        — so it always reflects the final decision/reason/error_code, not an
        intermediate value from a tier that ended up being superseded. A
        fresh ``uuid.uuid4()`` is folded in as a per-row salt so the hash is
        guaranteed unique per row, including for two decisions with
        byte-for-byte identical content (e.g. a caller retry) — this is a
        per-row identifier, not a dedup key; see :func:`_interaction_hash`.

        ``intent_id``

        Returns the ``interaction_hash`` that was written, or ``None`` if the
        insert failed — :meth:`verify_cbac` stamps this onto the result so a
        caller gets the hash back without a second lookup.
        """
        # NULL rather than "" for what the caller never supplied, so
        # "no callee" stays distinguishable from "a callee named ''".
        user_intent_val = user_intent or None
        callee_name_val = callee_name or None
        callee_type_val = callee_type if callee_name else None
        try:
            interaction_hash = _interaction_hash(
                agent_id=agent_id,
                decision=result.decision,
                reason=result.reason,
                intended_action=intent_text,
                user_intent=user_intent_val,
                callee_name=callee_name_val,
                callee_type=callee_type_val,
                error_code=result.error_code,
                salt=str(uuid.uuid4()),
                intent_id=intent_id,
            )
            await insert_cbac_decision(
                session,
                agent_id=agent_id,
                decision=result.decision,
                reason=result.reason,
                intended_action=intent_text,
                user_intent=user_intent_val,
                callee_name=callee_name_val,
                callee_type=callee_type_val,
                error_code=result.error_code,
                interaction_hash=interaction_hash,
                intent_id=intent_id,
            )
            return interaction_hash
        except Exception:
            logger.warning(
                "cbac decision record failed",
                decision=result.decision,
                exc_info=True,
            )
            return None

    async def verify_cbac(
        self,
        session: AsyncSession,
        agent_id: str,
        intended_action: Any,
        user_intent: str | None = None,
        callee_name: str = "",
        callee_type: str = "mcp_tool",
        intent_id: str | None = None,
        mcp_did: str = "",
    ) -> CBACResult:
        """Decide, then write the verdict to the audit log.

        The decision itself is :meth:`_decide`; this wrapper exists so the
        recording happens at exactly one place. ``_decide`` has seven distinct
        return paths, and every one of them must be logged — threading the
        write through each is how a path silently stops being audited.

        ``intent_id`` is the caller's own opaque correlation id (minted
        upstream, at the workflow's first envelope, not by CBAC) — it plays no
        part in the decision itself, so it bypasses ``_decide`` entirely and
        goes straight into the audit write, which is also where it is hashed.

        ``mcp_did`` plays no part in the decision either, but unlike
        ``intent_id`` it lands on the *trust* record rather than the audit row,
        and that write happens inside ``_decide`` — so it does have to travel
        through the pipeline.
        """
        result = await self._decide(
            session,
            agent_id,
            intended_action,
            user_intent,
            callee_name,
            callee_type,
            mcp_did,
        )
        result.interaction_hash = await self._record_decision(
            session,
            agent_id,
            _intended_action_text(intended_action),
            user_intent,
            callee_name,
            callee_type,
            result,
            intent_id,
        )
        return result

    async def _decide(
        self,
        session: AsyncSession,
        agent_id: str,
        intended_action: Any,
        user_intent: str | None = None,
        callee_name: str = "",
        callee_type: str = "mcp_tool",
        mcp_did: str = "",
    ) -> CBACResult:
        """Semantic intent verification against the agent's on-chain policy.

        Fetches the policy, flattens it into allowed/forbidden chunks (cached
        per policy hash), then runs the tiered decision — Tier 1 cosine gap →
        Tier 2 NLI entailment → Tier 3 optional LLM — in
        :meth:`_tiered_decision`; see that method for the per-tier thresholds
        and the ``policy_score`` normalization.

        A Check-1 NLI drift test runs first, but only when ``user_intent`` is
        supplied: it compares ``user_intent`` against the flattened
        ``intended_action`` text and denies immediately on a strong
        contradiction (``CONTRADICTION_THRESHOLD``).

        Once a decision is *reached* — allow or deny alike — its
        component scores are folded into the (agent → callee) trust score via
        :meth:`_fold_trust`, and the result carries the new ``trust``. All three
        components are known at this point, so there is nothing left to wait for
        post-execution. Denials count: an agent probing forbidden actions scores
        badly on policy and loses trust for it.

        The early returns *above* the decision are infrastructure failures
        (policy lookup down, no policy, no chunks), not judgments about the
        agent, so they deliberately fold no trust. They are still *audited* —
        :meth:`verify_cbac` logs every path to ``cbac_decisions`` regardless.

        Parameters
        ----------
        session:
            Async DB session for repository/search calls.
        agent_id:
            Whose policy to fetch (via Provenance Layer).
        intended_action:
            The action the agent wants to perform (any shape, flattened to text).
        user_intent:
            The root user request. Enables Check-1 drift + hallucination score.
        callee_name / callee_type:
            The other end of the edge whose trust this decision updates
            (``"mcp_tool"`` / ``"tool"`` / ``"agent"``). No ``callee_name``
            means no trust update.
        mcp_did:
            The MCP server the call is bound for. Not part of the edge key and
            not scored — recorded on the trust row so the history says which
            server the edge was exercised through.

        Returns
        -------
        CBACResult with ``decision`` in ``{"allow", "deny"}``.
        Fail-closed: any unrecoverable error resolves to ``deny``.
        """
        intent_text = _intended_action_text(intended_action)
        if not intent_text.strip():
            return CBACResult(
                decision="deny",
                reason="Intended action carries no analysable content",
                error_code=ec.GUARD_INTENDED_ACTION_EMPTY,
            )

        # Check 1: NLI drift — only runs when caller supplies the root user intent.
        intent_score: float | None = None
        if user_intent:
            drift, intent_score = await self._check1_drift(user_intent, intent_text)
            if drift is not None:
                decision, reason, error_code = drift
                return await self._fold_trust(
                    session,
                    agent_id,
                    callee_name,
                    callee_type,
                    CBACResult(
                        decision=decision,
                        reason=reason,
                        intent_score=intent_score,
                        error_code=error_code,
                    ),
                    mcp_did,
                )

        all_chunks = await get_policy_chunks(session, agent_id)
        if not all_chunks:
            try:
                policy = await asyncio.to_thread(
                    self._get_latest_agent_policy, agent_id
                )
            except Exception as e:
                return CBACResult(
                    decision="deny",
                    reason=f"Policy lookup failed for agent {agent_id}: {e}",
                    error_code=ec.GUARD_POLICY_LOOKUP_FAILED,
                )
            if not policy:
                return CBACResult(
                    decision="deny",
                    reason=f"No policy available for agent {agent_id}",
                    error_code=ec.GUARD_NO_POLICY_AVAILABLE,
                )
            try:
                await self.index_policy(session, agent_id, policy=policy)
            except Exception as e:
                return CBACResult(
                    decision="deny",
                    reason=f"Policy unavailable for agent {agent_id}: {e}",
                    error_code=ec.GUARD_POLICY_INDEX_FAILED,
                )
            all_chunks = await get_policy_chunks(session, agent_id)

        # Still nothing means the policy indexed to no usable chunks — there is
        # nothing to decide against, which is a deny, not an empty allow.
        if not all_chunks:
            return CBACResult(
                decision="deny",
                reason="Policy carries no analysable content",
                error_code=ec.GUARD_POLICY_NO_CONTENT,
            )

        # Run the tiered decision pipeline (DB-backed search).
        decision, reason, error_code, policy_score = await self._tiered_decision(
            session, agent_id, intent_text
        )

        # Hallucination score — informational, never gates the decision.
        hallucination = None
        if user_intent:
            try:
                hallucination = await asyncio.to_thread(
                    self.hallucination_score, user_intent, intent_text
                )
            except Exception:
                logger.warning(
                    "hallucination scoring failed", decision=decision, exc_info=True
                )
                hallucination = None

        return await self._fold_trust(
            session,
            agent_id,
            callee_name,
            callee_type,
            CBACResult(
                decision=decision,
                reason=reason,
                hallucination_score=hallucination,
                intent_score=intent_score,
                policy_score=policy_score,
                error_code=error_code,
            ),
            mcp_did,
        )

    async def compute_lhi(
        self,
        session: AsyncSession,
        agent_id: str,
        callee_name: str,
        callee_type: str,
        intent_score: float | None,
        policy_score: float | None,
        hallucination_score: float | None,
        mcp_did: str = "",
    ) -> float | None:
        """Update and return the LHI (Local Heuristic Intelligence) trust
        score for one (agent → callee) edge.

        Called from :meth:`verify_cbac` the moment a decision is reached, with
        the three per-decision scores (each in [0, 1], higher is better):

        1. Instantaneous quality ``s`` = weighted **arithmetic** mean over the
           components that were actually *observed*::

               s = Σ_observed wᵢxᵢ / Σ_observed wᵢ

           Renormalizing (rather than skipping the record, or substituting a
           value for a missing component) matters because the components are
           not missing at random: ``policy_score`` is ``None`` exactly on the
           Tier-3 gray-zone decisions, and intent/hallucination are ``None``
           whenever no ``user_intent`` was supplied. Complete-case analysis
           would silently exclude precisely the interactions trust exists to
           arbitrate. Returns ``None`` when *nothing* was observed — a record
           with no measurement behind it would be fiction.

           The mean is deliberately compensatory: the allow/deny gates already
           enforce the hard constraints, so a single weak component costs its
           (renormalized) weight rather than annihilating the score.
        2. Asymmetric EMA against the stored trust for this edge:
           ``T = λ·T_prev + (1−λ)·s`` with λ = ``lhi_lambda_up`` when improving
           (trust builds slowly) and ``lhi_lambda_down`` when degrading (trust
           drops fast). First interaction with a callee: ``T = s``.

        Storage: one ``lhi_records`` row per decision — the table is the full
        trust history for every edge ((agent, callee_name, callee_type)), and
        the current trust is simply the edge's latest row (see
        ``repository.get_latest_trust`` / ``get_trust_history``). Unobserved
        components are stored as NULL, never as a substituted value, so the
        row stays honest about what was measured. ``mcp_did`` rides along on
        the row as descriptive metadata — it names the MCP server the call went
        to, is not part of the edge key, and enters none of the arithmetic
        above.
        """
        scores = {
            "intent": intent_score,
            "policy": policy_score,
            "hallucination": hallucination_score,
        }
        for key, value in scores.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{key}_score must be in [0, 1], got {value}")

        observed = [
            (value, weight)
            for value, weight in zip(scores.values(), self._lhi_weights, strict=True)
            if value is not None
        ]
        if not observed:
            return None

        s = sum(value * weight for value, weight in observed) / sum(
            w for _, w in observed
        )

        prev = await get_latest_trust(session, agent_id, callee_name, callee_type)
        if prev is None:
            trust = s
        else:
            lam = self._lhi_lambda_up if s >= prev else self._lhi_lambda_down
            trust = lam * prev + (1 - lam) * s

        await insert_lhi_record(
            session,
            agent_id=agent_id,
            callee_name=callee_name,
            callee_type=callee_type,
            intent_score=intent_score,
            policy_score=policy_score,
            hallucination_score=hallucination_score,
            trust=trust,
            # NULL rather than "" for a caller that named no MCP server, the
            # same convention the audit row uses for its optional fields.
            mcp_did=mcp_did or None,
        )

        # ponytail: on-chain mirror of the record is parked — the DB is the
        # working copy and nothing reads the card yet. To restore, re-add the
        # `json` / `agentdna.id.get_id` / `repository.agent_has_lhi_records`
        # imports along with this block. Keep it *after* the commit above so a
        # chain failure can never lose the trust update.
        #
        # is_new_agent = not await agent_has_lhi_records(session, agent_id)
        # record = {
        #     "type": "lhi_record",
        #     "agent_id": agent_id,
        #     "callee": {"name": callee_name, "type": callee_type},
        #     "scores": scores,
        #     "trust": trust,
        #     "updated_at": db_record.created_at.isoformat(),
        # }
        # card_id = get_id(f"{agent_id}:cbac")
        # write = (
        #     self.provenance.create_new_provenance_card
        #     if is_new_agent
        #     else self.provenance.append_to_provenance_card
        # )
        # # Chain writes are sync network calls — off the event loop.
        # await asyncio.to_thread(write, card_id=card_id, card_info=json.dumps(record))

        return trust

    # ── v2: IBAC gate + deny-only semantic checks ─────────────────────────────

    async def upload_policy(
        self,
        session: AsyncSession,
        agent_id: str,
        policy: str,
        trusted_data: dict,
        review_required: bool = False,
    ) -> PolicyVersion:
        """Parse a policy document into a new version, and activate it unless
        ``review_required`` holds it ``pending`` for an admin to approve.

        ``review_required`` is decided per upload, by the admin uploading —
        some agents' policies warrant a second reader and some do not. A held
        version changes nothing until approved: the previous active version
        keeps deciding.

        ``policy`` is the whole document — instructions and tool schemas, see
        :func:`~cbac_service.parsers.split_policy_document`. The version row
        stores the two halves apart (``policy`` is the instructions alone,
        which is what gets chunked for policy drift) and hashes the whole
        document. The row is written before activation, so a failed activation
        leaves it ``pending`` and retryable through approval.
        """
        instructions, tools = split_policy_document(policy)
        parsed = await parse_policy(POLICY_PARSER_MODEL, instructions, tools)
        version = await insert_policy_version(
            session,
            agent_id=agent_id,
            policy=instructions,
            policy_hash=_policy_hash(policy),
            tool_schemas=tools,
            trusted_data=trusted_data,
            catalog=parsed.catalog,
            deny_rules=parsed.deny_rules,
            rejected=parsed.rejected,
            unenforceable=parsed.unenforceable,
            parser_model=POLICY_PARSER_MODEL,
            status="pending",
        )
        if not review_required:
            await self.activate_policy_version(session, version)
        return version

    async def activate_policy_version(
        self, session: AsyncSession, version: PolicyVersion
    ) -> None:
        """Make ``version`` the one deciding: re-index its chunks for policy
        drift, swap the exact deny tuples in OpenFGA, then flip the rows.

        The rows flip last, so ``active`` never names a version whose tuples
        are not in the store. Steps that fail leave the previous version
        active; approving again retries the whole sequence.
        """
        previous = await get_active_policy_version(session, version.agent_id)
        await self.index_policy(session, version.agent_id, policy=version.policy)

        def exact_denies(v: PolicyVersion | None) -> list[str]:
            if v is None:
                return []
            return [
                object_id(v.agent_id, r["tool"], r["resource"])
                for r in v.deny_rules
                if r["match"] == "exact"
            ]

        await fga.replace_denies(exact_denies(version), exact_denies(previous))
        await mark_policy_version_active(session, version)

    async def authorize_call(
        self,
        session: AsyncSession,
        agent_id: str,
        intent_id: str,
        user_query: str,
        tool: str,
        arguments: dict[str, Any],
        callee_type: str = "mcp_tool",
        mcp_did: str = "",
    ) -> CBACResult:
        """Decide one tool call against the agent's active v2 policy, then
        fold trust and write the audit row.

        The decision is :meth:`_authorize_call`; this wrapper exists, as
        :meth:`verify_cbac` does for v1, so the recording happens in exactly
        one place whatever path the decision took. The trust fold runs on every
        path: a gate deny measures no component, so it writes no trust row.
        """
        result = await self._authorize_call(
            session, agent_id, intent_id, user_query, tool, arguments
        )
        result = await self._fold_trust(
            session, agent_id, tool, callee_type, result, mcp_did
        )
        result.interaction_hash = await self._record_decision(
            session,
            agent_id,
            render_intent(tool, arguments),
            user_query,
            tool,
            callee_type,
            result,
            intent_id or None,
        )
        return result

    async def _authorize_call(
        self,
        session: AsyncSession,
        agent_id: str,
        intent_id: str,
        user_query: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> CBACResult:
        """The v2 pipeline. Deterministic gate first, semantic checks last.

        1. Intent: bind ``intent_id`` to its query, parsing grants on first sight.
        2. Mapping: the active catalog says which arguments are resources.
        3. Glob deny rules, matched here (OpenFGA cannot wildcard an object).
        4. OpenFGA: every resource needs a live allow tuple and no deny tuple.
        5. Semantic checks on the free-text arguments — deny-only.

        Fail-closed throughout: every error is a deny with its own code.
        """
        if not intent_id or not user_query.strip():
            return _deny(
                "v2 authorize needs an intent_id and a user query", ec.INTENT_MISSING
            )

        version = await get_active_policy_version(session, agent_id)
        if version is None:
            return _deny(
                f"No active v2 policy for agent {agent_id}", ec.MAPPING_NO_ACTIVE_POLICY
            )

        intent, denial = await self._resolve_intent(
            session, version, agent_id, intent_id, user_query
        )
        if denial is not None:
            return denial
        assert intent is not None

        now = datetime.now(timezone.utc)
        if intent.expires_at <= now:
            return _deny(
                f"Intent {intent_id} expired at {intent.expires_at.isoformat()}",
                ec.INTENT_EXPIRED,
            )

        # 2. Mapping.
        entry = version.catalog.get(tool)
        if entry is None:
            return _deny(
                f"Tool {tool!r} is not in the agent's catalog", ec.MAPPING_UNKNOWN_TOOL
            )
        known = set(entry["resources"]) | set(entry["free_text"])
        unknown = sorted(arg for arg in arguments if arg not in known)
        if unknown:
            return _deny(
                f"Arguments {unknown} of {tool!r} are not in the catalog",
                ec.MAPPING_UNKNOWN_ARGUMENT,
            )

        resources: list[str] = []
        if not entry["resources"]:
            resources.append(WHOLE_TOOL)
        for arg in entry["resources"]:
            for value in resource_values(arguments, arg):
                try:
                    resources.append(canonical(value))
                except NormalizationError as exc:
                    return _deny(
                        f"{tool}.{arg}: {exc}", ec.MAPPING_NORMALIZATION_FAILED
                    )
        if not resources:
            return _deny(f"Call to {tool!r} names no resource", ec.MAPPING_NO_RESOURCE)

        # 3. Glob deny rules.
        for resource in resources:
            for rule in version.deny_rules:
                if (
                    rule["match"] == "glob"
                    and rule["tool"] == tool
                    and fnmatch.fnmatchcase(resource, rule["resource"])
                ):
                    return _deny(
                        f"{tool}#{resource} matches deny rule "
                        f"{rule['resource']!r} ({rule['source']!r})",
                        ec.AUTHZ_GLOB_DENY,
                    )

        # 4. OpenFGA.
        user = fga_user(intent_id)
        try:
            refused: tuple[str, bool] | None = None
            for resource in resources:
                obj = object_id(agent_id, tool, resource)
                if not await fga.can_invoke(user, obj, now):
                    refused = (f"{tool}#{resource}", await fga.is_blocked(user, obj))
                    break
        except Exception as exc:
            return _deny(f"OpenFGA check failed: {exc}", ec.AUTHZ_FGA_ERROR)
        if refused is not None and refused[1]:
            return _deny(f"{refused[0]} is blocked by policy", ec.AUTHZ_DENY_TUPLE)
        if refused is not None:
            return _deny(
                f"{refused[0]} was not granted for this intent",
                ec.AUTHZ_NO_ALLOW_TUPLE,
            )

        # 5. Semantic checks.
        texts = [
            (arg, _as_text(arguments[arg]))
            for arg in entry["free_text"]
            if arguments.get(arg) is not None
        ]
        texts = [(arg, text) for arg, text in texts if text.strip()]
        if not texts:
            return CBACResult(
                decision="allow",
                reason=f"Granted: {', '.join(f'{tool}#{r}' for r in resources)}",
                error_code=ec.ALLOW_NO_FREE_TEXT,
            )
        return await self._semantic_checks(session, agent_id, user_query, texts)

    async def _resolve_intent(
        self,
        session: AsyncSession,
        version: PolicyVersion,
        agent_id: str,
        intent_id: str,
        user_query: str,
    ) -> tuple[CBACIntent | None, CBACResult | None]:
        """The intent row for ``intent_id``, creating it on first sight.

        Runs under a per-intent advisory lock and ends the transaction on every
        path, which releases it. The first call parses the query, writes the
        allow tuples, and stores the binding; every later call must carry the
        same query and agent, which is what freezes the grants.
        ponytail: the lock holds a pool connection for one LLM call, so the
        pool size bounds concurrent new intents; a "parsing" status row frees
        the connection if that ever binds.
        """
        await lock_intent(session, intent_id)
        intent = await get_intent(session, intent_id)
        if intent is not None:
            await session.commit()
            if intent.agent_id != agent_id or intent.user_query != user_query:
                return None, _deny(
                    f"Intent {intent_id} is bound to a different query or agent",
                    ec.INTENT_MISMATCH,
                )
            return intent, None

        try:
            grants, dropped = await parse_query(
                QUERY_PARSER_MODEL, user_query, version.trusted_data, version.catalog
            )
        except Exception as exc:
            await session.rollback()
            return None, _deny(f"Query parser failed: {exc}", ec.INTENT_PARSE_FAILED)

        now = datetime.now(timezone.utc)
        objects = [object_id(agent_id, g["tool"], g["resource"]) for g in grants]
        try:
            if objects:
                await fga.write_grants(
                    fga_user(intent_id), objects, now, INTENT_TTL_SECONDS
                )
        except Exception as exc:
            await session.rollback()
            return None, _deny(f"OpenFGA grant write failed: {exc}", ec.AUTHZ_FGA_ERROR)

        intent = await insert_intent(
            session,
            intent_id=intent_id,
            agent_id=agent_id,
            user_query=user_query,
            policy_version_id=version.id,
            grants=grants,
            dropped=dropped,
            parser_model=QUERY_PARSER_MODEL,
            created_at=now,
            expires_at=now + timedelta(seconds=INTENT_TTL_SECONDS),
        )
        await session.commit()
        logger.info(
            "intent parsed", intent_id=intent_id, grants=grants, dropped=dropped
        )
        return intent, None

    async def _policy_drift(
        self, session: AsyncSession, agent_id: str, text: str
    ) -> tuple[CBACResult | None, float | None]:
        """Tier 1 then Tier 2 against the policy chunks, each able only to deny.

        Returns ``(denial or None, policy_score)``. Anything short of a clear
        deny passes: the call already holds an exact grant, and requiring
        positive entailment would deny nearly every free-text argument.
        """
        t1 = await self._tier1_gap(session, agent_id, text)
        if t1.gap < -self._deny_gap:
            return _deny(
                f"Policy drift (Tier 1): gap {t1.gap:+.3f} < -{self._deny_gap} "
                f"for {text!r}",
                ec.SEMANTIC_POLICY_DRIFT_TIER1,
            ), t1.gap_score
        top_chunk, scores = await self._tier2_nli(session, agent_id, text, t1)
        contradiction = scores.get("contradiction", 0.0)
        if top_chunk and contradiction >= CONTRADICTION_THRESHOLD:
            return _deny(
                f"Policy drift (Tier 2): {text!r} contradicts {top_chunk!r} "
                f"(NLI contradiction={contradiction:.2f})",
                ec.SEMANTIC_POLICY_DRIFT_TIER2,
            ), t1.gap_score
        return None, t1.gap_score

    async def _semantic_checks(
        self,
        session: AsyncSession,
        agent_id: str,
        user_query: str,
        texts: list[tuple[str, str]],
    ) -> CBACResult:
        """Intent drift and policy drift over every window of every free-text
        argument; HHEM attached, never gating."""
        windows = [
            (arg, w)
            for arg, text in texts
            for w in split_by_word_budget(text, CHUNK_MAX_WORDS)
        ]
        try:
            scores = await asyncio.to_thread(
                self._nli_scores_batch, [(user_query, w) for _, w in windows]
            )
            contradictions = [s.get("contradiction", 0.0) for s in scores]
            worst = max(range(len(windows)), key=contradictions.__getitem__)
            intent_score = 1.0 - contradictions[worst]
            if contradictions[worst] >= CONTRADICTION_THRESHOLD:
                arg, window = windows[worst]
                return _deny(
                    f"Intent drift: {arg}={window!r} contradicts the user query "
                    f"(NLI contradiction={contradictions[worst]:.2f})",
                    ec.SEMANTIC_INTENT_DRIFT,
                    intent_score=intent_score,
                )

            policy_score: float | None = None
            for _, window in windows:
                denial, score = await self._policy_drift(session, agent_id, window)
                if score is not None:
                    policy_score = (
                        score if policy_score is None else min(policy_score, score)
                    )
                if denial is not None:
                    denial.intent_score = intent_score
                    denial.policy_score = policy_score
                    return denial
        except Exception as exc:
            return _deny(f"Semantic check failed: {exc}", ec.SEMANTIC_ERROR)

        hallucination = None
        try:
            hallucination = await asyncio.to_thread(
                self.hallucination_score,
                user_query,
                " ".join(text for _, text in texts),
            )
        except Exception:
            logger.warning("hallucination scoring failed", exc_info=True)

        return CBACResult(
            decision="allow",
            reason=f"Granted; free text {[arg for arg, _ in texts]} passed both drift checks",
            error_code=ec.ALLOW_FREE_TEXT_PASSED,
            intent_score=intent_score,
            policy_score=policy_score,
            hallucination_score=hallucination,
        )
