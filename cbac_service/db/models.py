"""ORM models for the CBAC vector/policy store."""

from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class PolicyChunk(Base):
    """One text chunk from an agent's policy, with its embedding vector."""

    __tablename__ = "policy_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'allowed' | 'forbidden'
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=False)
    policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_policy_chunks_agent_hash", "agent_id", "policy_hash"),)

    def __repr__(self) -> str:
        return (
            f"<PolicyChunk(id={self.id}, agent_id={self.agent_id!r}, "
            f"chunk_type={self.chunk_type!r}, chunk_index={self.chunk_index})>"
        )


class PolicyMeta(Base):
    """Lightweight per-agent metadata for cache invalidation.

    One row per agent. Compared against on-chain policy hash at runtime
    to decide whether re-embedding is needed.
    """

    __tablename__ = "policy_meta"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    encoder_model: Mapped[str] = mapped_column(
        String, nullable=False, default="BAAI/bge-small-en-v1.5"
    )
    nli_model: Mapped[str] = mapped_column(
        String, nullable=False, default="cross-encoder/nli-deberta-v3-small"
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (UniqueConstraint("agent_id", name="uq_policy_meta_agent_id"),)

    def __repr__(self) -> str:
        return (
            f"<PolicyMeta(agent_id={self.agent_id!r}, "
            f"policy_hash={self.policy_hash!r}, chunks={self.chunk_count})>"
        )


class LHIRecord(Base):
    """One LHI trust update — a single completed agent→callee interaction.

    Append-only: the table *is* the trust history the JSON file's per-edge
    ``scores`` list used to approximate. The current trust for an edge
    (agent_id, callee_name, callee_type) is the ``trust`` of its latest row;
    the EMA chain is reconstructable because every row stores the value it
    produced.
    """

    __tablename__ = "lhi_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    callee_name: Mapped[str] = mapped_column(String, nullable=False)
    callee_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'tool' | 'agent' | 'mcp'
    # Nullable on purpose: a component the pipeline could not measure is stored
    # as NULL, never as a substituted value, so the record stays honest about
    # what was observed. `trust` renormalizes over whichever ones are present.
    intent_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Latest-row-per-edge is the hot query; id disambiguates same-timestamp rows.
    __table_args__ = (
        Index("ix_lhi_records_edge", "agent_id", "callee_name", "callee_type", "id"),
    )

    def __repr__(self) -> str:
        return (
            f"<LHIRecord(id={self.id}, agent_id={self.agent_id!r}, "
            f"edge={self.callee_name!r}:{self.callee_type!r}, trust={self.trust:.3f})>"
        )


class CBACDecision(Base):
    """One authorization verdict — the audit log of what was decided and why.

    Append-only, and *complete*: every decision `verify_cbac` reaches lands here,
    including the infrastructure-failure denies and calls that carry no callee.
    That is what separates it from `lhi_records`, which looks similar but is a
    trust history and is deliberately skipped without a callee, without a
    measured component, and on those same infra-failure paths — so it can never
    answer "what did we decide for this agent, and why".
    """

    __tablename__ = "cbac_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str] = mapped_column(
        String, nullable=False
    )  # 'allow' | 'deny' (historical rows may carry 'advise', retired)
    # Text, not String: a Tier-2 reason embeds the matched policy chunk verbatim.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    # The *flattened* action text — what the scorers actually saw, so the row
    # explains the verdict rather than just restating the request.
    intended_action: Mapped[str] = mapped_column(Text, nullable=False)
    # NULL when the caller supplied none, rather than an empty string, so
    # "not provided" stays distinguishable from "provided as empty".
    user_intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    callee_name: Mapped[str | None] = mapped_column(String, nullable=True)
    callee_type: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # 'tool' | 'agent' | 'mcp'
    # Which layer/condition produced `reason` — see cbac_service.error_codes.
    # Nullable so a row written before this column existed doesn't need a
    # fabricated backfill value; every row inserted going forward has one.
    error_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # sha256 over the row's content fields plus a fresh per-row uuid4 salt
    # (agent_id, decision, reason, intended_action, user_intent, callee_name,
    # callee_type, error_code, salt), computed by CBAC._interaction_hash
    # immediately before insert. The salt makes this a per-row identifier —
    # deliberately unique even across two decisions with byte-for-byte
    # identical content (e.g. a caller retry) — not a content fingerprint,
    # so don't use it to detect that an interaction was decided more than
    # once; it can no longer answer that.
    interaction_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Newest-first-per-agent is the only read pattern; id disambiguates
    # same-timestamp rows. The leftmost prefix also serves agent-only lookups,
    # so no separate index on agent_id is needed.
    __table_args__ = (
        Index("ix_cbac_decisions_agent_id_id", "agent_id", "id"),
        Index("ix_cbac_decisions_interaction_hash", "interaction_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"<CBACDecision(id={self.id}, agent_id={self.agent_id!r}, "
            f"decision={self.decision!r}, error_code={self.error_code})>"
        )
