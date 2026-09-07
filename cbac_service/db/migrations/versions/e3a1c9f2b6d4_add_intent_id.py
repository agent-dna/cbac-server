"""add intent_id to cbac_decisions

Adds one column to the audit log:

- intent_id: opaque correlation id the caller generates upstream, at the
  workflow's first envelope, and passes through unchanged all the way to
  this row (and into interaction_hash) — see CBAC._record_decision /
  CBAC._interaction_hash. CBAC never parses or validates it.

Nullable: existing rows (written before this migration) keep NULL rather
than a fabricated backfill value, and a caller that never had one to send
leaves it NULL too — every row inserted going forward carries whatever the
caller supplied, including nothing.

Revision ID: e3a1c9f2b6d4
Revises: 4b5242414c76
Create Date: 2026-08-31 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3a1c9f2b6d4"
down_revision: str | None = "4b5242414c76"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("cbac_decisions", sa.Column("intent_id", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("cbac_decisions", "intent_id")
