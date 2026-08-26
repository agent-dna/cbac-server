"""
Framework-agnostic CBAC guard layer.

This module provides the single Policy Enforcement Point (PEP) for
agent actions as a wrapper decorator, plus the ambient governance
context it reads from. It is deliberately independent of any agent
framework (LangGraph, CrewAI, MCP, ...): the guard wraps plain async
Python callables, which is the lowest common denominator every
framework ultimately dispatches to.

The guard does one thing: authorize. It turns a call into an intent,
asks the CBAC decision service (one HTTP call), and either lets the
wrapped function run or short-circuits a denial. It does **not** build
provenance envelopes and does **not** perform the action itself:

    intent (from call args) -> authorize (CBAC) -> run the wrapped fn

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
- :func:`authorize_tool_call` and ``@cbac_guard`` read it from a
  ``contextvars.ContextVar`` set once at the request entry point via
  :func:`cbac_context` -- for in-process calls, which have no way to
  thread it down to the call site. It is never a function argument
  there, so an LLM can neither supply nor forge it.

Static config comes from the decorator's own parameters, and the layer
config (service URL, timeout) from :func:`configure` at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import (
    Any,
)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class GovernanceContext:
    """Per-request governance state the guard authorizes against.

    ``agent_id`` (whose policy is checked) and ``user_intent`` are the
    only two inputs the CBAC call needs beyond the intended action.
    """

    agent_id: str
    user_intent: str = ""


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
) -> Iterator[GovernanceContext]:
    """Open a governance scope. Set once at the request entry point.

    Every ``@cbac_guard``-wrapped callable invoked inside this block
    reads the context ambiently; the caller passes nothing per call.
    """
    holder = GovernanceContext(agent_id=agent_id, user_intent=user_intent)
    token = _governance_ctx.set(holder)
    try:
        yield holder
    finally:
        _governance_ctx.reset(token)


# ── Layer configuration ───────────────────────────────────────────────────────


@dataclass
class GuardConfig:
    cbac_url: str = "https://cbac-admin.agentdna.io"
    cbac_timeout: float = 100.0


_config = GuardConfig()


def configure(
    cbac_url: str | None = None,
    cbac_timeout: float | None = None,
) -> None:
    """Set layer-wide guard configuration. Call once at startup."""
    if cbac_url is not None:
        _config.cbac_url = cbac_url
    if cbac_timeout is not None:
        _config.cbac_timeout = cbac_timeout


def get_config() -> GuardConfig:
    return _config


# ── Guard internals ───────────────────────────────────────────────────────────


def _action_summary(action_name: str, description: str | None = None) -> str:
    """Verb-only prose for the intended action: "The agent wants to <verb>."

    The verb phrase is the tool's own description (docstring first line)
    when available, else the de-snaked tool name. This is the prefix of
    :func:`render_intent`'s full rendering; kept as its own function so a
    verb-only view stays available to callers that want one.
    """
    verb = (description or action_name.replace("_", " ")).strip().rstrip(".")
    return f"The agent wants to {verb[:1].lower()}{verb[1:]}."


def render_intent(
    action_name: str, kwargs: dict[str, Any], description: str | None = None
) -> str:
    """Render the intended action as prose, parameters included.

    Public because it is the one thing every enforcement point must agree on:
    a guard, an MCP gateway and an ext_authz adapter have to phrase the same
    call the same way, or the policy scores them differently.

    NLI and the policy tiers score natural language far better than raw
    ``tool k=v`` call syntax — snake_case tool names hide their verb from
    the NLI model (measured: a direct "close" vs "do not close"
    contradiction scored 0.01 on call syntax, 0.95 as prose).
    """
    text = _action_summary(action_name, description).rstrip(".")
    if kwargs:
        text += ", with " + ", ".join(f"{k} = {v!s}" for k, v in kwargs.items())
    return text + "."


def _bind_kwargs(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Flatten positional + keyword call args into a name->value dict."""
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        # Let the wrapped function raise its own error on the real call.
        return dict(kwargs)


def _authorize_sync(
    agent_id: str,
    intended_action: Any,
    user_intent: str | None,
    callee_name: str,
    callee_type: str,
    cfg: GuardConfig,
) -> tuple[str, str]:
    """POST to the CBAC decision service.

    The reference implementation (the cbac_service package) runs
    ``verify_cbac`` behind this endpoint and returns a
    decision only -- it never executes the action. The payload is
    exactly that method's inputs.

    ``callee_name``/``callee_type`` name the edge whose trust score the
    service updates as a side effect of deciding. The component scores
    behind that update stay server-side: there is nothing for the caller
    to carry, and nothing to report back afterwards.
    """
    import requests  # lazy; transitive dependency of the library already

    payload = {
        "agent_id": agent_id,
        "intended_action": intended_action,
        "user_intent": user_intent,
        "callee_name": callee_name,
        "callee_type": callee_type,
    }

    response = requests.post(
        f"{cfg.cbac_url.rstrip('/')}/authorize-cbac",
        json=payload,
        timeout=cfg.cbac_timeout,
    )

    return response.headers.get("X-CBAC-Decision", "advise"), response.text


async def _authorize(
    agent_id: str,
    intent_text: str,
    user_intent: str | None,
    callee_name: str,
    callee_type: str,
    cfg: GuardConfig,
) -> tuple[str, str]:
    """One decision call, fail-closed: any error resolves to ``"error"``.

    ``cfg`` is threaded in rather than looked up here: it is set once when
    the process starts, and a caller that forgot to :func:`configure` should
    fail where it can be seen rather than quietly reach the default URL.
    """
    try:
        return await asyncio.to_thread(
            _authorize_sync,
            agent_id,
            intent_text,
            user_intent or None,
            callee_name,
            callee_type,
            cfg,
        )
    except Exception as exc:
        return "error", str(exc)


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
    cfg: GuardConfig | None = None,
) -> tuple[str, str]:
    """Authorize one call. Every input is an argument (decision only).

    For enforcement points that already hold the governance context as
    values -- a gateway that parsed it off the wire, a job runner that read
    it from a queue message. They have nothing to look up ambiently, and
    routing what they already hold through a contextvar only to read it back
    would be a detour.

    Returns ``(decision, detail)``. Fail-closed: any error resolves to
    ``"error"`` and never raises.

    ``description`` is the callee's own description (schema/docstring first
    line) used as the verb phrase of the rendered intent; without it the
    de-snaked ``callee_name`` is the fallback. ``callee_type`` labels the
    other end of the edge (``"tool"``, ``"agent"``, ``"mcp"``) whose trust
    score the service updates while deciding.
    """
    return await _authorize(
        agent_id,
        render_intent(callee_name, args, description),
        user_intent,
        callee_name,
        callee_type,
        cfg or get_config(),
    )


async def authorize_tool_call(
    callee_name: str,
    args: dict[str, Any],
    description: str | None = None,
    callee_type: str = "tool",
) -> tuple[str, str]:
    """Authorize one tool call against the *ambient* policy (decision only).

    :func:`authorize` for in-process callers, which have no way to thread
    the governance context down to the call site and read it from
    :func:`cbac_context` instead.

    ``decision`` is ``"allow"`` when governance is off (no
    :func:`cbac_context` open), so a caller can always proceed on
    ``"allow"`` and short-circuit otherwise.
    """
    ctx = get_context()
    if ctx is None:
        return "allow", ""
    return await authorize(
        ctx.agent_id, callee_name, args, ctx.user_intent, description, callee_type
    )


# ── The decorator ─────────────────────────────────────────────────────────────


def cbac_guard(
    *,
    action: str | None = None,
    action_intent: Callable[[dict[str, Any]], str] | None = None,
    on_deny: str = "return",
    callee_type: str = "tool",
):
    """Guard an async callable with a CBAC authorization gate.

    Parameters
    ----------
    action:
        Logical action name used to label the intent; defaults to the
        function name.
    action_intent:
        ``kwargs -> str`` builder for the intent handed to CBAC. Defaults
        to the action name followed by the call's arguments.
    on_deny:
        ``"return"`` -> a denied call returns ``{"status": "denied", ...}``
        (readable by an LLM loop); ``"raise"`` -> raises PermissionError.
    callee_type:
        What is on the other end of this edge (``"tool"``, ``"agent"``,
        ``"mcp"``), recorded with the trust score.

    Behavior
    --------
    - No ambient context (:func:`cbac_context` not open): governance is
      not enabled for this call path, so the guard is a no-op and the
      wrapped function runs unguarded. Opening a context is a deployment
      decision an LLM cannot make, so this can never bypass a guard that
      *is* active -- it only expresses "this path opted out of
      governance", the same way a route without rate-limiting opts out.
    - Otherwise: intent (from the call's arguments) -> authorize (one HTTP
      call to the CBAC decision service) -> a non-allow decision
      short-circuits -> run the wrapped function -> return the result.

    The trust score is updated service-side while the decision is made, so
    the guard has nothing to report afterwards -- and a denied call, which
    never runs, still leaves a record of having been denied.
    """
    if on_deny not in ("return", "raise"):
        raise ValueError(f"unsupported on_deny: {on_deny!r}")

    def decorate(fn: Callable[..., Awaitable[Any]]):
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"cbac_guard requires an async function, got {fn!r}")

        action_name = action or fn.__name__
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn)
        description = doc.splitlines()[0].strip() if doc else None

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_context()

            # No governance scope open -> governance is opt-in and this
            # path opted out; run the function exactly as if unguarded.
            if ctx is None:
                return await fn(*args, **kwargs)

            call_kwargs = _bind_kwargs(sig, args, kwargs)
            intent_text = (
                action_intent(call_kwargs)
                if action_intent
                else render_intent(action_name, call_kwargs, description)
            )

            decision, detail = await _authorize(
                ctx.agent_id,
                intent_text,
                ctx.user_intent,
                action_name,
                callee_type,
                get_config(),
            )

            if decision != "allow":
                if decision == "deny" and on_deny == "raise":
                    raise PermissionError(detail)
                status = "denied" if decision == "deny" else "error"
                return {"status": status, "error": detail}

            return await fn(*args, **kwargs)

        # Frameworks (LangChain / MCP) build the LLM-facing tool schema by
        # introspecting the callable's signature. functools.wraps copies
        # __name__/__doc__/__annotations__ but leaves the wrapper's own
        # ``(*args, **kwargs)`` in place, and not every schema generator
        # follows ``__wrapped__``. Pin the wrapped function's real
        # signature so the generated schema stays correct.
        wrapper.__signature__ = sig  # pyright: ignore[reportAttributeAccessIssue]

        return wrapper

    return decorate
