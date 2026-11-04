"""An agentdna ``CbacFn`` that asks the CBAC v2 gate (``POST /cbac/v2/authorize``).

agentdna's MCP middleware calls its ``cbac_fn`` with
``(agent_id, mcp_did, tool, args, user_intent, description, intent_id)``:

- ``agent_id`` is the actor of the workflow's latest envelope — the calling
  agent's DID. The v2 policy must be uploaded under that DID.
- ``user_intent`` is the root envelope's payload: what the user asked.
- ``intent_id`` is the workflow id, shared by every agent working on one
  request.

v2 binds an intent id to one agent on first sight and refuses it to any other
(4002). In a multi-agent workflow every member carries the same workflow id,
so the adapter sends ``<workflow id>:<agent DID>`` instead: one intent per
(request, agent), each parsed into grants against that agent's own policy.

The gate's verdict is its status code: only 4301 and 4302 allow. Anything
else — a deny, or one of the guard's 9100s when no verdict was reached — goes
back to the middleware as a refusal, so the call fails closed.
"""

from __future__ import annotations

import os
from typing import Any

from cbac import authorize_call

DEFAULT_CBAC_URL = "http://127.0.0.1:8767"
# A cold service loads its models, and a new intent waits for the query parser.
DEFAULT_TIMEOUT = 600.0

ALLOW_CODES = frozenset({4301, 4302})


def scoped_intent_id(intent_id: str | None, agent_id: str) -> str:
    """``<intent_id>:<agent_id>``, or ``""`` when there is no intent id (the
    gate then denies with 4001)."""
    return f"{intent_id}:{agent_id}" if intent_id else ""


def make_cbac_fn(cbac_url: str | None = None, cbac_timeout: float | None = None):
    """A ``CbacFn`` bound to one service. ``cbac_url`` defaults to ``CBAC_URL``
    or ``http://127.0.0.1:8767``; ``cbac_timeout`` to ``CBAC_TIMEOUT`` or 600s."""
    url = cbac_url or os.environ.get("CBAC_URL") or DEFAULT_CBAC_URL
    timeout = cbac_timeout or float(os.environ.get("CBAC_TIMEOUT") or DEFAULT_TIMEOUT)

    async def cbac_fn(
        agent_id: str,
        mcp_did: str,
        tool: str,
        args: dict[str, Any],
        user_intent: str | None,
        description: str | None,
        intent_id: str | None,
    ) -> tuple[str, int, str]:
        # `description` is unused: v2 maps the call by tool name and arguments,
        # and the tool's description reaches the gate through the uploaded schemas.
        decision, status, interaction_hash = await authorize_call(
            agent_id,
            scoped_intent_id(intent_id, agent_id),
            user_intent or "",
            tool,
            args,
            mcp_did=mcp_did,
            cbac_url=url,
            cbac_timeout=timeout,
        )
        if status in ALLOW_CODES:
            return "allow", status, interaction_hash
        return ("deny" if decision == "allow" else decision), status, interaction_hash

    return cbac_fn
