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


# ── Guard-side status codes ───────────────────────────────────────────────────
#
# The service's codes (``cbac_service.error_codes``) name an outcome of one
# pipeline layer, and it owns 3000-3999 and 9000-9099 for them. These four name
# the cases where this guard never got a verdict to report, so 9100-9199 is
# reserved here to say so without a code the service could also mint. Same rules
# as the service's: one code per return statement below, and never renumbered --
# a code that reached a log or a dashboard has to keep meaning what it meant.
#
# All four are failures of the *asking*, not of the action, and every one blocks:
# a guard that cannot reach a verdict has not been told yes.
NO_SERVICE_CONFIGURED = 9101  # error — no cbac_url and no CBAC_URL in the env
UNRECOGNIZED_RESPONSE = 9102  # error — 200 body carried no decision at all
TRANSPORT_FAILED = 9103  # error — the POST raised: refused, timed out, bad TLS
VERDICT_WITHOUT_CODE = 9104  # error — a decision came back, but with no code to
# act on, so this guard cannot tell allow from deny


@dataclass
class AuthResult:
    """One ``/cbac/v1/authorize`` call's outcome, as the guard layer sees it.

    ``decision`` and ``message`` are the human-readable pair. ``status_code``
    is the machine-readable one — the service's pipeline code
    (``cbac_service.error_codes``) when a verdict was reached, one of the
    9100s above when it was not — and ``hash`` is the audit row's
    ``interaction_hash``, ``""`` when no row was written to have one.

    Neither is ever ``None``: an enforcement point decides on the code, and a
    code it has to null-check first is one more branch between an agent and a
    blocked action.
    """

    decision: str
    message: str
    status_code: int
    hash: str = ""


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
            NO_SERVICE_CONFIGURED,
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
                "error",
                message or f"CBAC: unrecognized response {response.text!r}",
                UNRECOGNIZED_RESPONSE,
            )
        # A verdict with no code is one this guard cannot act on -- an
        # enforcement point decides by code, and there is nothing to compare.
        # Reported as its own failure rather than folded into the deny codes,
        # because the fix is a service version, not a policy change.
        status_code = data.get("status_code")
        if not isinstance(status_code, int):
            return AuthResult(
                "error",
                message or "CBAC: verdict carried no status code",
                VERDICT_WITHOUT_CODE,
                data.get("hash") or "",
            )
        return AuthResult(decision, message, status_code, data.get("hash") or "")
    except Exception as exc:
        return AuthResult("error", str(exc), TRANSPORT_FAILED)


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
) -> tuple[int, str]:
    """Authorize one call. Every input is an argument (decision only).

    For enforcement points that already hold the governance context as
    values -- a gateway that parsed it off the wire, a job runner that read
    it from a queue message. They have nothing to look up ambiently, and
    routing what they already hold through a contextvar only to read it back
    would be a detour.

    Returns ``(status_code, interaction_hash)``, always both, never ``None``.
    The code *is* the verdict -- ``cbac_service.error_codes`` when the service
    reached one, where each code names the outcome of one pipeline layer -- and
    which codes an enforcement point treats as permission is the enforcement
    point's call. Never raises, so a caller that forgets to check the code
    fails **open**; that check is its whole job.

    When no verdict was reached the code is one of this module's 9100s --
    :data:`NO_SERVICE_CONFIGURED`, :data:`UNRECOGNIZED_RESPONSE`,
    :data:`TRANSPORT_FAILED`, :data:`VERDICT_WITHOUT_CODE` -- rather than
    ``None``, so an enforcement point tests one thing (is this code permission?)
    instead of two, and the reason still reaches a log or a dashboard. A caller
    that needs to tell "blocked by policy" from "could not ask" tests the range,
    not a null.

    ``interaction_hash`` is ``""`` whenever no audit row was written to have
    one -- every no-verdict case, and a service-side failure outside the
    pipeline. It is an id to look a decision up by, so absence is the empty
    string rather than a null to guard against.

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
