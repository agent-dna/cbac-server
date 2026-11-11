"""add cbac_intent_failures and policy_versions.tuple_scheme

- cbac_intent_failures: the latest failed query parse per intent_id. A call
  carrying an id whose row is younger than QUERY_PARSER_RETRY_SECONDS is denied
  without running the parser again. In Postgres so every worker sees it.
- policy_versions.tuple_scheme: ibac.TUPLE_SCHEME as of the version's last
  activation. NULL on every existing row, which the gate reads as "deny tuples
  not written under the current scheme": each agent's active version has to be
  re-activated (POST /cbac/v2/policies/{agent_id}/approve) before its calls
  are decided again.

Revision ID: d2a9f4e6b815
Revises: c5e8a2d71f03
Create Date: 2026-11-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2a9f4e6b815"
down_revision: str | None = "c5e8a2d71f03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "cbac_intent_failures",
        sa.Column("intent_id", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("intent_id"),
    )
    op.add_column(
        "policy_versions", sa.Column("tuple_scheme", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("policy_versions", "tuple_scheme")
    op.drop_table("cbac_intent_failures")
