"""Repository layer — async DB access for the CBAC policy store.

All SQLAlchemy session handling lives here so `cbac.py` stays clean
of ORM concerns. Every function takes an AsyncSession and returns
plain data (models, booleans, or None).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from cbac_service.config import ENCODER_MODEL, NLI_MODEL

from .models import CBACDecision, LHIRecord, PolicyChunk, PolicyMeta


async def policy_hash_matches(
    session: AsyncSession,
    agent_id: str,
    policy_hash: str,
) -> bool:
    """Check if the stored policy hash matches the current on-chain hash.

    Returns True if cache is valid (no recompute needed), False otherwise.
    Also returns False if no meta row exists for this agent.
    """
    stmt = select(PolicyMeta.policy_hash).where(PolicyMeta.agent_id == agent_id)
    result = await session.execute(stmt)
    stored_hash = result.scalar_one_or_none()
    if stored_hash is None:
        return False
    return stored_hash == policy_hash


async def save_policy_chunks(
    session: AsyncSession,
    agent_id: str,
    chunks: list[str],
    chunk_types: list[str],
    embeddings: np.ndarray,
    policy_hash: str,
    sections: list[str | None] | None = None,
) -> int:
    """Bulk-write policy chunks for an agent, replacing any existing rows.

    Steps:
      1. Delete all existing chunks for this agent.
      2. Insert new chunks with embeddings.
      3. Upsert the policy_meta row.

    Returns the number of chunks inserted.
    """
    # Annotated local rather than reassigning the parameter: `[None] * n` infers
    # list[None], which invariance rejects against list[str | None].
    chunk_sections: list[str | None] = (
        list(sections) if sections is not None else [None] * len(chunks)
    )

    # 1. Delete old chunks for this agent.
    await session.execute(delete(PolicyChunk).where(PolicyChunk.agent_id == agent_id))

    # 2. Insert new chunks.
    rows = []
    for i, (text, ctype, section) in enumerate(
        zip(chunks, chunk_types, chunk_sections, strict=True)
    ):
        row = PolicyChunk(
            agent_id=agent_id,
            chunk_text=text,
            chunk_type=ctype,
            embedding=embeddings[i].tolist(),
            policy_hash=policy_hash,
            section=section,
            chunk_index=i,
        )
        rows.append(row)

    session.add_all(rows)

    # 3. Upsert policy_meta.
    await upsert_policy_meta(
        session=session,
        agent_id=agent_id,
        policy_hash=policy_hash,
        encoder_model=ENCODER_MODEL,
        nli_model=NLI_MODEL,
        chunk_count=len(rows),
    )

    await session.commit()
    return len(rows)


async def get_policy_chunks(
    session: AsyncSession,
    agent_id: str,
    chunk_type: str | None = None,
) -> Sequence[str]:
    """Load the text of all policy chunks for an agent, ordered by chunk_index.

    Optionally filter by chunk_type ('allowed' or 'forbidden'). Only the text
    is selected — no caller reads the embeddings, and fetching them costs 384
    floats per row.
    """
    stmt = (
        select(PolicyChunk.chunk_text)
        .where(PolicyChunk.agent_id == agent_id)
        .order_by(PolicyChunk.chunk_index)
    )
    if chunk_type is not None:
        stmt = stmt.where(PolicyChunk.chunk_type == chunk_type)

    result = await session.execute(stmt)
    return result.scalars().all()


async def get_policy_meta(
    session: AsyncSession,
    agent_id: str,
) -> PolicyMeta | None:
    """Get the policy meta row for an agent, or None if not cached."""
    stmt = select(PolicyMeta).where(PolicyMeta.agent_id == agent_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_policy_meta(
    session: AsyncSession,
    agent_id: str,
    policy_hash: str,
    encoder_model: str = ENCODER_MODEL,
    nli_model: str = NLI_MODEL,
    chunk_count: int = 0,
) -> PolicyMeta:
    """Insert or update the policy_meta row for an agent."""
    stmt = insert(PolicyMeta).values(
        agent_id=agent_id,
        policy_hash=policy_hash,
        encoder_model=encoder_model,
        nli_model=nli_model,
        chunk_count=chunk_count,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["agent_id"],
        set_={
            "policy_hash": stmt.excluded.policy_hash,
            "encoder_model": stmt.excluded.encoder_model,
            "nli_model": stmt.excluded.nli_model,
            "chunk_count": stmt.excluded.chunk_count,
            "cached_at": func.now(),
        },
    ).returning(PolicyMeta)

    # populate_existing: RETURNING maps onto any instance already in the identity
    # map, and without this it keeps that instance's stale attribute values.
    result = await session.execute(stmt, execution_options={"populate_existing": True})
    return result.scalar_one()


async def delete_policy_chunks(
    session: AsyncSession,
    agent_id: str,
) -> int:
    """Delete all chunks and meta for an agent. Returns rows deleted."""
    result = await session.execute(
        delete(PolicyChunk).where(PolicyChunk.agent_id == agent_id)
    )
    await session.execute(delete(PolicyMeta).where(PolicyMeta.agent_id == agent_id))
    await session.commit()
    return result.rowcount  # type: ignore[return-value]


# ── LHI trust records ──────────────────────────────────────────────────────────


async def get_latest_trust(
    session: AsyncSession,
    agent_id: str,
    callee_name: str,
    callee_type: str,
) -> float | None:
    """Current trust for one agent→callee edge: the latest record's value.

    None when the edge has no history yet (first interaction).
    """
    stmt = (
        select(LHIRecord.trust)
        .where(
            LHIRecord.agent_id == agent_id,
            LHIRecord.callee_name == callee_name,
            LHIRecord.callee_type == callee_type,
        )
        .order_by(LHIRecord.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def agent_has_lhi_records(session: AsyncSession, agent_id: str) -> bool:
    """Whether any LHI record exists for this agent (any edge) — decides
    create-vs-append for the agent's provenance card."""
    stmt = select(LHIRecord.id).where(LHIRecord.agent_id == agent_id).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def insert_lhi_record(
    session: AsyncSession,
    agent_id: str,
    callee_name: str,
    callee_type: str,
    intent_score: float | None,
    policy_score: float | None,
    hallucination_score: float | None,
    trust: float,
) -> LHIRecord:
    """Append one interaction's scores + resulting trust. Commits.

    An unmeasured component is stored as NULL — `trust` already renormalizes
    over the observed ones, so nothing here substitutes a value.
    """
    record = LHIRecord(
        agent_id=agent_id,
        callee_name=callee_name,
        callee_type=callee_type,
        intent_score=intent_score,
        policy_score=policy_score,
        hallucination_score=hallucination_score,
        trust=trust,
    )
    session.add(record)
    await session.commit()
    return record


async def get_latest_trust_for_agents(
    session: AsyncSession,
    agent_ids: Sequence[str],
) -> Sequence[LHIRecord]:
    """Latest trust record per (agent_id, callee_name, callee_type) edge, for
    a batch of agents in one round trip.

    Trust is tracked per edge, not per agent — an agent with several callees
    comes back with several rows. An agent with no history at all is simply
    absent from the result; the caller fills in the gap.
    """
    if not agent_ids:
        return []
    stmt = (
        select(LHIRecord)
        .distinct(LHIRecord.agent_id, LHIRecord.callee_name, LHIRecord.callee_type)
        .where(LHIRecord.agent_id.in_(agent_ids))
        .order_by(
            LHIRecord.agent_id,
            LHIRecord.callee_name,
            LHIRecord.callee_type,
            LHIRecord.id.desc(),
        )
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_trust_history(
    session: AsyncSession,
    agent_id: str,
    callee_name: str,
    callee_type: str,
    limit: int = 100,
) -> Sequence[LHIRecord]:
    """Trust history for one edge, newest first."""
    stmt = (
        select(LHIRecord)
        .where(
            LHIRecord.agent_id == agent_id,
            LHIRecord.callee_name == callee_name,
            LHIRecord.callee_type == callee_type,
        )
        .order_by(LHIRecord.id.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


# ── CBAC decisions ────────────────────────────────────────────────────────────


async def insert_cbac_decision(
    session: AsyncSession,
    agent_id: str,
    decision: str,
    reason: str,
    intended_action: str,
    user_intent: str | None,
    callee_name: str | None,
    callee_type: str | None,
    error_code: int | None,
    interaction_hash: str,
) -> CBACDecision:
    """Append one authorization verdict to the audit log. Commits.

    ``intended_action`` is the flattened action text the scorers actually saw,
    not the caller's raw payload. Values are stored exactly as given — the
    caller decides what "absent" means and passes None for it.

    ``error_code`` identifies which layer/condition of the pipeline produced
    ``reason`` (see ``cbac_service.error_codes``). ``interaction_hash`` is a
    salted sha256 of this same row's content, computed by the caller
    (``CBAC._interaction_hash``) immediately before this call — passed in
    rather than computed here so the repository layer stays pure DB access
    with no hashing/business logic of its own. The salt makes it unique per
    row, not a content-derived dedup key.
    """
    record = CBACDecision(
        agent_id=agent_id,
        decision=decision,
        reason=reason,
        intended_action=intended_action,
        user_intent=user_intent,
        callee_name=callee_name,
        callee_type=callee_type,
        error_code=error_code,
        interaction_hash=interaction_hash,
    )
    session.add(record)
    await session.commit()
    return record


async def get_cbac_decisions(
    session: AsyncSession,
    agent_id: str,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[CBACDecision]:
    """Decision history for one agent, newest first."""
    stmt = (
        select(CBACDecision)
        .where(CBACDecision.agent_id == agent_id)
        .order_by(CBACDecision.id.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_cbac_decision(
    session: AsyncSession,
    decision_id: int,
) -> CBACDecision | None:
    """One decision by id, or None if there is no such row."""
    stmt = select(CBACDecision).where(CBACDecision.id == decision_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_cbac_decision_by_hash(
    session: AsyncSession,
    interaction_hash: str,
) -> CBACDecision | None:
    """One decision by its interaction_hash, or None if there is no such row.

    ``interaction_hash`` is salted per row (see ``CBAC._interaction_hash``),
    so in practice each value identifies exactly one row. ``.limit(1)`` here
    is defensive, not load-bearing — there is no database-level uniqueness
    constraint on the column, so this still degrades gracefully (returns the
    newest match) rather than raising if that were ever violated.
    """
    stmt = (
        select(CBACDecision)
        .where(CBACDecision.interaction_hash == interaction_hash)
        .order_by(CBACDecision.id.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
