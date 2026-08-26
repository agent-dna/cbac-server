"""add error_code and interaction_hash to cbac_decisions

Adds two columns to the audit log added in b7d2a5e91c34:

- error_code: which layer/condition of the pipeline produced `reason` (see
  cbac_service.error_codes) — machine-stable where `reason` (free text) is
  not.
- interaction_hash: sha256 fingerprint of the row's content fields,
  computed immediately before insert (CBAC._interaction_hash). Indexed for
  dedup/reference lookups; not unique, since the same interaction can
  legitimately be decided more than once.

Both are nullable: existing rows (written before this migration) keep
NULL rather than a fabricated backfill value — every row inserted going
forward has both populated by the application.

Revision ID: 4b5242414c76
Revises: b7d2a5e91c34
Create Date: 2026-08-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b5242414c76"
down_revision: str | None = "b7d2a5e91c34"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "cbac_decisions", sa.Column("error_code", sa.Integer(), nullable=True)
    )
    op.add_column(
        "cbac_decisions",
        sa.Column("interaction_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_cbac_decisions_interaction_hash",
        "cbac_decisions",
        ["interaction_hash"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_cbac_decisions_interaction_hash", table_name="cbac_decisions")
    op.drop_column("cbac_decisions", "interaction_hash")
    op.drop_column("cbac_decisions", "error_code")
