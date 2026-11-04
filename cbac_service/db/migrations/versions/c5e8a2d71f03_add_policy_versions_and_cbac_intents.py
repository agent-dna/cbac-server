"""add policy_versions and cbac_intents - the v2 gate's state

- policy_versions: one row per v2 policy upload, holding the parser's inputs
  (tool schemas, trusted data) and its validated output (catalog, deny rules)
  next to what validation rejected. status is pending | active | superseded.
- cbac_intents: one row per intent_id, binding it to the user query it was
  first seen with and to the grants parsed from that query.

The tuples themselves live in OpenFGA; these rows are the readable record of
what was written there and why.

Revision ID: c5e8a2d71f03
Revises: a1f3c7d95b26
Create Date: 2026-11-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c5e8a2d71f03"
down_revision: str | None = "a1f3c7d95b26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    jsonb = postgresql.JSONB(astext_type=sa.Text())
    op.create_table(
        "policy_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column("policy_hash", sa.String(length=64), nullable=False),
        sa.Column("tool_schemas", jsonb, nullable=False),
        sa.Column("trusted_data", jsonb, nullable=False),
        sa.Column("catalog", jsonb, nullable=False),
        sa.Column("deny_rules", jsonb, nullable=False),
        sa.Column("rejected", jsonb, nullable=False),
        sa.Column("unenforceable", jsonb, nullable=False),
        sa.Column("parser_model", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_policy_versions_agent_status_id",
        "policy_versions",
        ["agent_id", "status", "id"],
    )
    op.create_table(
        "cbac_intents",
        sa.Column("intent_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("policy_version_id", sa.Integer(), nullable=False),
        sa.Column("grants", jsonb, nullable=False),
        sa.Column("dropped", jsonb, nullable=False),
        sa.Column("parser_model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["policy_version_id"], ["policy_versions.id"]),
        sa.PrimaryKeyConstraint("intent_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("cbac_intents")
    op.drop_index("ix_policy_versions_agent_status_id", table_name="policy_versions")
    op.drop_table("policy_versions")
