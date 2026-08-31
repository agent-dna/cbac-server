"""add cbac_decisions - the authorization audit log

Every verdict `verify_cbac` reaches lands here, including the
infrastructure-failure denies and calls that carry no callee. `lhi_records`
looks similar but is a trust history and skips exactly those cases, so it
cannot answer "what did we decide for this agent, and why".

Revision ID: b7d2a5e91c34
Revises: 9c4e1f80a72b
Create Date: 2026-08-17 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d2a5e91c34"
down_revision: str | None = "9c4e1f80a72b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "cbac_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("intended_action", sa.Text(), nullable=False),
        sa.Column("user_intent", sa.Text(), nullable=True),
        sa.Column("callee_name", sa.String(), nullable=True),
        sa.Column("callee_type", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # Newest-first-per-agent is the only read pattern; the leftmost prefix also
    # serves agent-only lookups, so no separate agent_id index is needed.
    op.create_index(
        "ix_cbac_decisions_agent_id_id",
        "cbac_decisions",
        ["agent_id", "id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_cbac_decisions_agent_id_id", table_name="cbac_decisions")
    op.drop_table("cbac_decisions")
