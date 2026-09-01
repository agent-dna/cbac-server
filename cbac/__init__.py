"""CBAC governance layer: the framework-agnostic guard + optional MCP glue.

The guard API is re-exported here for ``from cbac import ...``.
The MCP integration lives in :mod:`cbac.mcp` and is imported separately
(it needs the ``mcp`` dependency), so importing this package does not
pull it in.
"""

from .guard import (
    GovernanceContext,
    cbac_context,
    cbac_guard,
    get_context,
)

__all__ = [
    "GovernanceContext",
    "cbac_context",
    "cbac_guard",
    "get_context",
]
