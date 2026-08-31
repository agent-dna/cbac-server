"""
MCP integration for the CBAC guard (optional).

Requires the ``mcp`` package (declared in the ``dev`` dependency group).

Two enforcement topologies live here, and they are not equivalent:

**Gateway-enforced (preferred).** The agent's MCP client points at a CBAC
gateway instead of at the MCP server. The client's only job is to *label*
each outgoing call with the governance context (:func:`cbac_propagate`);
the gateway extracts that label, calls the decision service, and forwards
or blocks. The developer holds no security logic and cannot switch the
check off -- dropping :func:`cbac_propagate` does not bypass CBAC, it just
makes every call arrive contextless, which a fail-closed gateway denies.
See ``examples/github_agent_workflow/gateway.py`` for the gateway half.

**Client-enforced (fallback).** :func:`cbac_intercept` calls the decision
service from inside the client process. Use it only where no gateway can
sit in the path -- a laptop agent talking straight to a remote MCP server.
It is real enforcement against a *confused* agent, but not against a
developer who declines to install it.

Both share the same governance context and the same one-HTTP-call gate.

The context travels as HTTP headers because that is the only per-call
channel MCP client adapters expose today (``langchain_mcp_adapters``
exposes ``headers``, not ``params._meta``). ``_meta`` is the
protocol-native place for this and is where it should move once clients
let a caller set it; :func:`context_from_headers` is the only piece that
would change.
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote, unquote

from mcp.types import CallToolResult, TextContent

# Re-exported so MCP users import the whole recipe from one place.
from .guard import (
    GovernanceContext,
    authorize_tool_call,
    cbac_context,
    cbac_guard,
    get_context,
)

__all__ = [
    "CBAC_AGENT_HEADER",
    "CBAC_INTENT_HEADER",
    "cbac_context",
    "cbac_guard",
    "cbac_headers",
    "cbac_intercept",
    "cbac_propagate",
    "context_from_headers",
    "denial_body",
]

# The governance context on the wire. Neither is a tool argument, so the LLM
# neither sees nor controls them.
#
# Trust boundary: both values are *client-supplied*, so a gateway treats them
# as evidence, not identity. ``user_intent`` is unforgeable-in-principle
# nowhere -- only the client knows what the user asked -- so CBAC scores it as
# evidence either way. ``agent_id`` decides *whose policy applies* and must be
# pinned by the gateway from its authenticated principal (OAuth subject, mTLS
# SAN, API key) wherever one exists; the header is the fallback for a trusted
# network only.
CBAC_AGENT_HEADER = "X-CBAC-Agent-Id"
CBAC_INTENT_HEADER = "X-CBAC-User-Intent"

# Headers are latin-1 and size-capped by every HTTP stack in the path, while a
# user intent is arbitrary UTF-8 of arbitrary length. Percent-encode, then cap
# well under the usual 8 KB total-header limit. A truncated intent is still a
# usable drift signal; a rejected request is not.
_INTENT_MAX_CHARS = 4096
_PARTIAL_ESCAPE = re.compile("%[0-9A-Fa-f]?$")


def _encode_intent(text: str) -> str:
    """Percent-encode and cap, never cutting a multi-byte escape in half."""
    return _PARTIAL_ESCAPE.sub("", quote(text, safe="")[:_INTENT_MAX_CHARS])


def cbac_headers() -> dict[str, str]:
    """The ambient governance context, encoded as headers. ``{}`` when off.

    :func:`cbac_propagate` intercepts *tool* calls, which is where a LangChain
    client offers a per-call hook. Resources and prompts have no such hook, and
    a raw MCP client has none at all -- so those attach the same headers to the
    connection instead. One encoder either way: a gateway must recover the same
    context however the client got it there.
    """
    ctx = get_context()
    if ctx is None:
        return {}
    return {
        CBAC_AGENT_HEADER: quote(ctx.agent_id, safe=""),
        CBAC_INTENT_HEADER: _encode_intent(ctx.user_intent),
    }


async def cbac_propagate(request, handler):
    """Client-side: label each outgoing MCP call with the governance context.

    Carries **no** security decision and calls no service -- it only attaches
    the ambient :func:`cbac_context` so a CBAC gateway downstream can
    authorize the call with the same inputs an in-process ``@cbac_guard``
    would have had. Pass it as ``tool_interceptors=[cbac_propagate]``.

    Without an open context it is a passthrough; a fail-closed gateway is
    what turns a missing label into a denial.
    """
    headers = cbac_headers()
    if not headers:
        return await handler(request)
    return await handler(
        request.override(headers={**(request.headers or {}), **headers})
    )


def context_from_headers(headers: dict[str, str]) -> GovernanceContext | None:
    """Gateway-side: recover the governance context, or None if unlabelled.

    Accepts any case of header name. A context with an empty ``user_intent``
    is still a context -- deciding what to do about a missing intent is the
    gateway's fail-closed policy, not this parser's.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    agent_id = lowered.get(CBAC_AGENT_HEADER.lower())
    intent = lowered.get(CBAC_INTENT_HEADER.lower())
    if agent_id is None and intent is None:
        return None
    return GovernanceContext(
        agent_id=unquote(agent_id or ""),
        user_intent=unquote(intent or ""),
    )


async def cbac_intercept(request, handler):
    """Client-side *enforcement*: authorize each call before it leaves.

    The fallback topology -- use it only where no gateway can sit in the
    path. All CBAC logic lives in :func:`~cbac.guard.authorize_tool_call`;
    this only maps the MCP request/result types and renders the denial, so
    another framework needs just its own equally thin adapter. Requires an
    open :func:`cbac_context`; without one it is a passthrough.
    """
    # MCPToolCallRequest (verified ≤0.3.2 and upstream main) carries no tool
    # description — only name/args/server_name/headers/runtime — so this reads
    # None today and the de-snaked tool name becomes the verb phrase. A gateway
    # does better: it has the upstream tool's real description locally.
    description = getattr(request, "description", None)
    decision, detail = await authorize_tool_call(
        request.name, request.args, description, callee_type="mcp"
    )
    if decision != "allow":
        return _denied_result(decision, detail)

    return await handler(request)


def denial_body(decision: str, detail: str) -> str:
    """The wire form of a block: ``{"status": "denied"|"error", "error": ...}``.

    Readable JSON, not a raised error, so the agent loop can see the block and
    adapt. Shared by every enforcement point so a client sees the same shape
    whether the block came from ``@cbac_guard``, this module's interceptor, or
    a gateway several hops away.
    """
    status = "denied" if decision == "deny" else "error"
    return json.dumps({"status": status, "error": detail})


def _denied_result(decision: str, detail: str) -> CallToolResult:
    # is_error defaults to False; passing it explicitly is unresolvable for pyright
    # because mcp 2.0 generates the camelCase alias via alias_generator.
    return CallToolResult(
        content=[TextContent(type="text", text=denial_body(decision, detail))]
    )
