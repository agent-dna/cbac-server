"""
Framework-agnostic CBAC guard layer.

This module provides the single Policy Enforcement Point (PEP) for
agent actions as a wrapper decorator, plus the ambient governance
context it reads from. It is deliberately independent of any agent
framework (LangGraph, CrewAI, MCP, ...): the guard wraps plain async
Python callables, which is the lowest common denominator every
framework ultimately dispatches to.

The guard does one thing: authorize. It reports the call's facts to the
CBAC decision service (one HTTP call) and either lets the wrapped
function run or short-circuits a denial. It does **not** build
provenance envelopes and does **not** perform the action itself:

    call facts -> authorize (CBAC) -> run the wrapped fn

What it deliberately does *not* do is decide how the action is worded.
The service's tiers score prose, so the phrasing is part of the decision
function: an enforcement point that worded a call its own way would get
its own verdicts. This layer sends the mechanical facts -- callee name,
arguments, description -- and the service renders them, which is also
why an enforcement point in another language needs no port of the
rendering.

The trust score (LHI) is folded in service-side as part of reaching the
decision, so there is exactly one HTTP call per guarded action and no
component score ever round-trips through the client.

Attestation (build/handle envelopes) is handled separately at the
workflow's delegation boundaries, and the wrapped function does its own
work -- e.g. a GitHub tool makes its own HTTP request and returns a
result dict.

There are two ways in, and which one fits depends on whether the caller
already holds the governance context:

- :func:`authorize` takes everything as arguments -- for an enforcement
  point that parsed the context off the wire, such as an MCP gateway.
- :func:`authorize_tool_call` reads it from a ``contextvars.ContextVar``
  set once at the request entry point via :func:`cbac_context` -- for
  in-process calls, which have no way to thread it down to the call
  site. It is never a function argument there, so an LLM can neither
  supply nor forge it.

Static config comes from the decorator's own parameters. Where the
decision service is (``cbac_url``, ``cbac_endpoint``, ``cbac_timeout``)
is an argument to :func:`authorize`, and falls back to ``CBAC_URL`` /
``CBAC_PATH`` / ``CBAC_TIMEOUT`` in the environment for the ambient
entry points, which have nowhere to pass it from.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os

# functools and inspect are only needed by the commented-out decorator below.
from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
    Any,
)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class GovernanceContext:
    """Per-request governance state the guard authorizes against.

    ``agent_id`` (whose policy is checked) and ``user_intent`` are the two
    inputs the CBAC call needs beyond the intended action. ``intent_id`` — CBAC never parses or validates it, only threads it
    through to the audit row and its hash.
    """

    agent_id: str
    user_intent: str = ""
    intent_id: str = ""


_governance_ctx: contextvars.ContextVar[GovernanceContext | None] = (
    contextvars.ContextVar("agentdna_governance_ctx", default=None)
)


def get_context() -> GovernanceContext | None:
    """Return the ambient GovernanceContext, or None when governance is off."""
    return _governance_ctx.get()


@contextlib.contextmanager
def cbac_context(
    agent_id: str,
    user_intent: str = "",
    intent_id: str = "",
) -> Iterator[GovernanceContext]:
    """Open a governance scope. Set once at the request entry point.

    Every ``@cbac_guard``-wrapped callable invoked inside this block
    reads the context ambiently; the caller passes nothing per call.
    ``intent_id`` is opened once here — at the workflow's first envelope —
    and every guarded call inside the scope carries it unchanged.
    """
    holder = GovernanceContext(
        agent_id=agent_id, user_intent=user_intent, intent_id=intent_id
    )
    token = _governance_ctx.set(holder)
    try:
        yield holder
    finally:
        _governance_ctx.reset(token)


# ── Layer configuration ───────────────────────────────────────────────────────


# ── Guard internals ───────────────────────────────────────────────────────────


def _stringify(args: dict[str, Any]) -> dict[str, str]:
    """Argument values as text, which is the only use the service has for them.

    They exist to be phrased into the intent prose, so rendering them here
    keeps the wire trivially JSON-serializable — a ``datetime`` or a dataclass
    argument would otherwise fail to encode and fail the call closed. This is
    mechanical, not phrasing: how the pieces are worded stays server-side.
    """
    return {k: str(v) for k, v in args.items()}


# def _bind_kwargs(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict[str, Any]:
#     """Flatten positional + keyword call args into a name->value dict."""
#     try:
#         bound = sig.bind(*args, **kwargs)
#         bound.apply_defaults()
#         return dict(bound.arguments)
#     except TypeError:
#         # Let the wrapped function raise its own error on the real call.
#         return dict(kwargs)


def _payload(
    agent_id: str,
    callee_name: str,
    args: dict[str, Any],
    user_intent: str | None,
    description: str | None,
    callee_type: str,
    intent_id: str | None = None,
) -> dict[str, Any]:
    """The ``/cbac/v1/authorize`` request body. One definition, every entry point.

    Facts only — the service turns them into the text it scores. ``intent_id``
    is opaque to the service too — it is never worded into anything, only
    stored and hashed.
    """
    return {
        "agent_id": agent_id,
        "callee_name": callee_name,
        "callee_type": callee_type,
        "callee_description": description,
        "arguments": _stringify(args),
        "user_intent": user_intent or None,
        "intent_id": intent_id or None,
    }


@dataclass
class AuthResult:
    """One ``/cbac/v1/authorize`` call's outcome, as the guard layer sees it.

    ``decision`` and ``message`` are the pair every caller has always gotten;
    ``status_code`` (the service's pipeline error code, ``cbac_service.
    error_codes``) and ``hash`` (the row's ``interaction_hash``) ride back
    alongside them so a caller never needs a second lookup to get either.
    Both are ``None`` on the fail-closed paths inside :func:`_authorize`
    itself (no service configured, a malformed response, a network error) —
    there was no service-side verdict to carry a code or a hash.
    """

    decision: str
    message: str
    status_code: int | None = None
    hash: str | None = None


async def _authorize(
    payload: dict[str, Any],
    cbac_url: str = "",
    cbac_endpoint: str = "",
    cbac_timeout: float = 0.0,
) -> AuthResult:
    """One decision call to the CBAC service, fail-closed: any error resolves
    to ``"error"``.

    The reference implementation (the cbac_service package) runs
    ``verify_cbac`` behind this endpoint and returns a decision only -- it
    never executes the action.

    The payload is the call's mechanical facts, not a rendered sentence: the
    service phrases ``callee_name``/``arguments``/``callee_description`` into
    the text its scorers see, so an enforcement point cannot word a call into
    a different verdict. ``intended_action`` overrides that, for a caller with
    phrasing of its own. ``callee_name``/``callee_type`` also name the edge
    whose trust score the service updates as a side effect of deciding; the
    component scores behind that update stay server-side, so there is nothing
    for the caller to carry and nothing to report back afterwards.

    ``requests`` blocks, and the enforcement points that call this serve other
    traffic concurrently -- a gateway multiplexes MCP sessions, and
    ``cbac_timeout`` is generous enough (a cold service loads three transformer
    models) that a blocked event loop would stall every call in flight, not
    just this one. Hence the thread.

    Where the service is and how long to wait for it are arguments, and each
    one left unset falls back to the environment here -- read per call, so a
    process that loads its ``.env`` after importing this module still gets it.
    There is no baked-in host: with no ``CBAC_URL`` there is nowhere to ask,
    which is an ``"error"`` -- every caller treats that as not-allowed -- rather
    than an agent's actions going to whatever host was compiled in.
    """
    import requests  # lazy; transitive dependency of the library already

    cbac_url = cbac_url or os.environ.get("CBAC_URL", "")
    cbac_endpoint = cbac_endpoint or os.environ.get("CBAC_PATH", "/cbac/v1/authorize")
    cbac_timeout = cbac_timeout or float(os.environ.get("CBAC_TIMEOUT", "100"))
    if not cbac_url:
        return AuthResult(
            "error",
            "CBAC: no decision service configured — set CBAC_URL or pass "
            "cbac_url= to authorize()",
        )

    try:
        response = await asyncio.to_thread(
            requests.post,
            f"{cbac_url.rstrip('/')}/{cbac_endpoint.lstrip('/')}",
            json=payload,
            timeout=cbac_timeout,
        )
        # Reading the response stays inside the guard: decoding a body can
        # raise too, and the callers' contract is that this never does.
        body = response.json()
        data = body.get("data") or {}
        decision = data.get("decision")
        message = body.get("message") or ""
        # No decision in the envelope means this was not a verdict at all -- a
        # 422, a proxy's error page, some other service on that URL. Not
        # "advise": that is a real gray-zone verdict and would misreport the
        # reason a call was blocked.
        if not decision:
            return AuthResult(
                "error", message or f"CBAC: unrecognized response {response.text!r}"
            )
        return AuthResult(decision, message, data.get("status_code"), data.get("hash"))
    except Exception as exc:
        return AuthResult("error", str(exc))


# ── Framework-agnostic call gate ──────────────────────────────────────────────
#
# The authorize flow every framework needs, split from any framework's
# request/result types. A framework interceptor supplies only its own glue:
# extract (name, args), execute, detect success, render a denial. See
# `cbac.mcp.cbac_intercept` for the MCP/LangChain adapter.


async def authorize(
    agent_id: str,
    callee_name: str,
    args: dict[str, Any],
    user_intent: str | None = None,
    description: str | None = None,
    callee_type: str = "tool",
    intent_id: str | None = None,
    cbac_url: str = "https://cbac-service.agentdna.io",
    cbac_timeout: float = 300,
) -> tuple[int | None, str | None]:
    """Authorize one call. Every input is an argument (decision only).

    For enforcement points that already hold the governance context as
    values -- a gateway that parsed it off the wire, a job runner that read
    it from a queue message. They have nothing to look up ambiently, and
    routing what they already hold through a contextvar only to read it back
    would be a detour.

    Returns ``(status_code, interaction_hash)``. The code *is* the verdict --
    it is ``cbac_service.error_codes``, where each code names one outcome of
    one pipeline layer -- and which codes an enforcement point treats as
    permission is the enforcement point's call. Never raises, so a caller that
    forgets to check the code fails **open**; that check is its whole job.

    Both are ``None`` in exactly the three fail-closed cases inside
    :func:`_authorize` -- no service configured, a malformed response, a
    network error -- because no service-side verdict was reached to carry
    either. ``None`` is kept rather than a placeholder (``0``, ``""``) so a
    caller can never mistake "no verdict" for a real code or hash, and the
    hash is ``None`` too whenever the service could not write its audit row.

    ``description`` is the callee's own description (schema/docstring first
    line); the service uses it as the verb phrase when it renders the intent,
    and falls back to the de-snaked ``callee_name`` without it. Passing it is
    worth the lookup — an enforcement point that has the callee's real
    description scores far better than one working from the name alone.
    ``callee_type`` labels the other end of the edge (``"tool"``, ``"agent"``,
    ``"mcp"``) whose trust score the service updates while deciding.
    ``intent_id`` is the caller's own opaque correlation id, threaded through
    unchanged to the audit row and its hash.

    ``cbac_url`` defaults to the hosted reference service; pass a different
    one for a self-hosted deployment. ``cbac_timeout`` defaults to 300s — a
    cold service loads three transformer models, and this is generous enough
    to ride that out rather than time out a caller's first request. The path
    on the service (``/cbac/v1/authorize``) is not a parameter here — it is
    not something a caller has a reason to override — but still falls back
    to ``CBAC_PATH`` in the environment inside :func:`_authorize`, for a
    deployment that mounts the service under a different prefix.
    """
    result = await _authorize(
        _payload(
            agent_id,
            callee_name,
            args,
            user_intent,
            description,
            callee_type,
            intent_id,
        ),
        cbac_url,
        cbac_timeout=cbac_timeout,
    )
    return (result.status_code, result.hash)


# Commented out with cbac_intercept, its only caller.
# async def authorize_tool_call(
#     callee_name: str,
#     args: dict[str, Any],
#     description: str | None = None,
#     callee_type: str = "tool",
# ) -> AuthResult:
#     """Authorize one tool call against the *ambient* policy (decision only).

#     :func:`authorize` for in-process callers, which have no way to thread
#     the governance context down to the call site and read it from
#     :func:`cbac_context` instead.

#     ``decision`` is ``"allow"`` when governance is off (no
#     :func:`cbac_context` open), so a caller can always proceed on
#     ``"allow"`` and short-circuit otherwise.
#     """
#     ctx = get_context()
#     if ctx is None:
#         return AuthResult("allow", "")
#     return await authorize(
#         ctx.agent_id,
#         callee_name,
#         args,
#         ctx.user_intent,
#         description,
#         callee_type,
#         intent_id=ctx.intent_id,
#     )


# Commented out for now: the gateway uses authorize() only, and nothing in
# this repo wraps a callable with the decorator.
# # ── The decorator ─────────────────────────────────────────────────────────────


# def cbac_guard(
#     *,
#     action: str | None = None,
#     action_intent: Callable[[dict[str, Any]], str] | None = None,
#     on_deny: str = "return",
#     callee_type: str = "tool",
# ):
#     """Guard an async callable with a CBAC authorization gate.

#     Parameters
#     ----------
#     action:
#         Logical action name used to label the intent; defaults to the
#         function name.
#     action_intent:
#         ``kwargs -> str`` builder that phrases the action itself, scored
#         verbatim. Left unset -- the normal case -- the service renders the
#         prose from the action name, the call's arguments and the docstring,
#         wording it the same way it words every other enforcement point's
#         calls. Set this only for an action those three do not describe.
#     on_deny:
#         ``"return"`` -> a denied call returns ``{"status": "denied", ...}``
#         (readable by an LLM loop); ``"raise"`` -> raises PermissionError.
#     callee_type:
#         What is on the other end of this edge (``"tool"``, ``"agent"``,
#         ``"mcp"``), recorded with the trust score.

#     Behavior
#     --------
#     - No ambient context (:func:`cbac_context` not open): governance is
#       not enabled for this call path, so the guard is a no-op and the
#       wrapped function runs unguarded. Opening a context is a deployment
#       decision an LLM cannot make, so this can never bypass a guard that
#       *is* active -- it only expresses "this path opted out of
#       governance", the same way a route without rate-limiting opts out.
#     - Otherwise: intent (from the call's arguments) -> authorize (one HTTP
#       call to the CBAC decision service) -> a non-allow decision
#       short-circuits -> run the wrapped function -> return the result.

#     The trust score is updated service-side while the decision is made, so
#     the guard has nothing to report afterwards -- and a denied call, which
#     never runs, still leaves a record of having been denied.
#     """
#     if on_deny not in ("return", "raise"):
#         raise ValueError(f"unsupported on_deny: {on_deny!r}")

#     def decorate(fn: Callable[..., Awaitable[Any]]):
#         if not asyncio.iscoroutinefunction(fn):
#             raise TypeError(f"cbac_guard requires an async function, got {fn!r}")

#         action_name = action or fn.__name__
#         sig = inspect.signature(fn)
#         doc = inspect.getdoc(fn)
#         description = doc.splitlines()[0].strip() if doc else None

#         @functools.wraps(fn)
#         async def wrapper(*args, **kwargs):
#             ctx = get_context()

#             # No governance scope open -> governance is opt-in and this
#             # path opted out; run the function exactly as if unguarded.
#             if ctx is None:
#                 return await fn(*args, **kwargs)

#             call_kwargs = _bind_kwargs(sig, args, kwargs)
#             payload = _payload(
#                 ctx.agent_id,
#                 action_name,
#                 call_kwargs,
#                 ctx.user_intent,
#                 description,
#                 callee_type,
#                 ctx.intent_id,
#             )
#             if action_intent:
#                 # Caller phrases this action itself; the service scores that
#                 # text verbatim instead of rendering from the fields above.
#                 payload["intended_action"] = action_intent(call_kwargs)

#             result = await _authorize(payload)

#             if result.decision != "allow":
#                 if result.decision == "deny" and on_deny == "raise":
#                     raise PermissionError(result.message)
#                 status = "denied" if result.decision == "deny" else "error"
#                 return {
#                     "status": status,
#                     "error": result.message,
#                     "status_code": result.status_code,
#                     "hash": result.hash,
#                 }

#             return await fn(*args, **kwargs)

#         # Frameworks (LangChain / MCP) build the LLM-facing tool schema by
#         # introspecting the callable's signature. functools.wraps copies
#         # __name__/__doc__/__annotations__ but leaves the wrapper's own
#         # ``(*args, **kwargs)`` in place, and not every schema generator
#         # follows ``__wrapped__``. Pin the wrapped function's real
#         # signature so the generated schema stays correct.
#         wrapper.__signature__ = sig  # pyright: ignore[reportAttributeAccessIssue]

#         return wrapper

#     return decorate
