"""drop lhi output_score; make the component scores nullable

LHI is now computed inside verify_cbac, at decision time. `output_score` was
the one component that required waiting for execution, so it is gone. The three
survivors become nullable: the weighted mean renormalizes over whichever ones
were observed, and an unmeasured component is stored as NULL rather than a
substituted value.

Revision ID: 9c4e1f80a72b
Revises: 7b21c9e4d3a8
Create Date: 2026-08-14 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c4e1f80a72b"
down_revision: str | None = "7b21c9e4d3a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

COMPONENT_COLUMNS = ("intent_score", "policy_score", "hallucination_score")


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("lhi_records", "output_score")
    for column in COMPONENT_COLUMNS:
        op.alter_column("lhi_records", column, existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    for column in COMPONENT_COLUMNS:
        # Rows written after the upgrade may hold NULLs the old NOT NULL
        # constraint would reject; 0 is the only value that cannot inflate a
        # historical trust score.
        op.execute(
            sa.text(f"UPDATE lhi_records SET {column} = 0 WHERE {column} IS NULL")
        )
        op.alter_column("lhi_records", column, existing_type=sa.Float(), nullable=False)

    op.add_column(
        "lhi_records",
        sa.Column("output_score", sa.Float(), nullable=False, server_default="1"),
    )
    op.alter_column("lhi_records", "output_score", server_default=None)
