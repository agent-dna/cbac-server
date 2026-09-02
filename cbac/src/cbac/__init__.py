"""CBAC governance layer: the framework-agnostic guard + optional MCP glue.

The guard API is re-exported here for ``from cbac import ...``. The MCP
adapter lives in :mod:`cbac.mcp` and is imported from there — it is one
framework's glue, not something every caller needs.
"""

from .authorize import (
    GovernanceContext,
    authorize,
    cbac_context,
    get_context,
)

__all__ = [
    "GovernanceContext",
    "authorize",
    "cbac_context",
    "get_context",
]
