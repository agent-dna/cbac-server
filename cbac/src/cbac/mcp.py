"""
MCP integration for the CBAC guard (optional).

Pure stdlib: the context is HTTP headers, so nothing here imports the MCP
SDK -- only the enforcement point that calls it does.

Enforcement is gateway-side. The agent's MCP client points at a CBAC
gateway instead of at the MCP server, and the client's only job is to
*label* each outgoing call with the governance context
(:func:`cbac_propagate`); the gateway extracts that label
(:func:`context_from_headers`), calls the decision service, and forwards
or blocks (:func:`denial_body`). The developer holds no security logic and
cannot switch the check off -- dropping :func:`cbac_propagate` does not
bypass CBAC, it only makes calls arrive unlabelled, and what a gateway
does with an unlabelled call is the gateway's policy. See
``examples/github_agent_workflow/gateway.py`` for the gateway half.

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

# Re-exported so MCP users import the whole recipe from one place.
from .guard import (
    GovernanceContext,
    # authorize_tool_call,  # commented out in guard.py for now
    cbac_context,
    # cbac_guard,  # commented out in guard.py for now
    get_context,
)

__all__ = [
    "CBAC_AGENT_HEADER",
    "CBAC_INTENT_HEADER",
    "CBAC_INTENT_ID_HEADER",
    "cbac_context",
    # "cbac_guard",
    "cbac_headers",
    # "cbac_intercept",
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
# ``intent_id`` is neither: it is the caller's own opaque correlation id, minted
# upstream at the workflow's first envelope. CBAC never parses or validates it,
# only threads it to the audit row and its hash -- so a client that forges one
# corrupts its own trace and nothing else.
CBAC_AGENT_HEADER = "X-CBAC-Agent-Id"
CBAC_INTENT_HEADER = "X-CBAC-User-Intent"
CBAC_INTENT_ID_HEADER = "X-CBAC-Intent-Id"

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
    headers = {
        CBAC_AGENT_HEADER: quote(ctx.agent_id, safe=""),
        CBAC_INTENT_HEADER: _encode_intent(ctx.user_intent),
    }
    # Only when there is one: a workflow that mints no correlation id should
    # not put an empty header on every call it makes.
    if ctx.intent_id:
        headers[CBAC_INTENT_ID_HEADER] = quote(ctx.intent_id, safe="")
    return headers


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
        intent_id=unquote(lowered.get(CBAC_INTENT_ID_HEADER.lower()) or ""),
    )


# Commented out for now: this repo enforces at the gateway, and nothing here
# uses the client-side interceptor.
# async def cbac_intercept(request, handler):
#     """Client-side *enforcement*: authorize each call before it leaves.

#     The fallback topology -- use it only where no gateway can sit in the
#     path. All CBAC logic lives in :func:`~cbac.guard.authorize_tool_call`;
#     this only maps the MCP request/result types and renders the denial, so
#     another framework needs just its own equally thin adapter. Requires an
#     open :func:`cbac_context`; without one it is a passthrough.
#     """
#     # MCPToolCallRequest (verified ≤0.3.2 and upstream main) carries no tool
#     # description — only name/args/server_name/headers/runtime — so this reads
#     # None today and the de-snaked tool name becomes the verb phrase. A gateway
#     # does better: it has the upstream tool's real description locally.
#     description = getattr(request, "description", None)
#     result = await authorize_tool_call(
#         request.name, request.args, description, callee_type="mcp"
#     )
#     if result.decision != "allow":
#         return _denied_result(result.decision, result.message)

#     return await handler(request)


def denial_body(status_code: int | None, interaction_hash: str | None = None) -> str:
    """The wire form of a block, from what :func:`~cbac.guard.authorize` returns.

    Readable JSON, not a raised error, so the agent loop can see the block and
    adapt. Shared by every enforcement point so a client sees the same shape
    however many hops away the gateway is.

    ``"denied"`` when the service reached a verdict and it was not an allow;
    ``"error"`` when it never reached one (``status_code`` is ``None`` -- no
    service configured, a malformed response, a network error). Both stop the
    call; only the reason differs. The code and hash are what a caller follows
    up with: ``GET /cbac/v1/decisions/by-hash/{hash}`` has the full reason, and
    the audit row it came from.
    """
    if status_code is None:
        return json.dumps(
            {"status": "error", "error": "CBAC reached no decision", "hash": None}
        )
    return json.dumps(
        {
            "status": "denied",
            "error": f"CBAC denied this call (status {status_code})",
            "status_code": status_code,
            "hash": interaction_hash,
        }
    )


# def _denied_result(decision: str, detail: str) -> CallToolResult:
#     # is_error defaults to False; passing it explicitly is unresolvable for pyright
#     # because mcp 2.0 generates the camelCase alias via alias_generator.
#     return CallToolResult(
#         content=[TextContent(type="text", text=denial_body(decision, detail))]
#     )
