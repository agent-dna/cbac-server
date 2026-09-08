"""add mcp_did to lhi_records

Adds one column to the trust history:

- mcp_did: the MCP server the guarded call was bound for, supplied by the
  caller on POST /cbac/v1/authorize and returned by POST /cbac/v1/lhi-scores.

Recorded, not part of the edge key — the edge stays (agent_id, callee_name,
callee_type), so this is descriptive metadata and adding it needs no index and
no change to how trust is read or folded.

Nullable: existing rows keep NULL rather than a fabricated backfill value, and
a caller with no MCP server behind it (an in-process tool guard, a bare curl
probe) leaves it NULL too.

Revision ID: a1f3c7d95b26
Revises: e3a1c9f2b6d4
Create Date: 2026-09-08 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1f3c7d95b26"
down_revision: str | None = "e3a1c9f2b6d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("lhi_records", sa.Column("mcp_did", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("lhi_records", "mcp_did")
