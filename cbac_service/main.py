import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote
from uuid import uuid4

import structlog
import uvicorn
from agentdna.provenance import Provenance
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text

# The prose renderer is shared with the guard on purpose: a guard, an MCP
# gateway and this adapter must phrase the same call identically, or the policy
# scores them differently. `cbac` pulls in none of the ML stack.
from cbac.guard import render_intent
from cbac_service.cbac import CBAC
from cbac_service.db.engine import close_db, get_session
from cbac_service.logging_config import setup_logging

# ── App lifecycle ──────────────────────────────────────────────────────────────

setup_logging()
logger = structlog.get_logger("cbac_service.api")

_cbac: CBAC | None = None

# Fail closed when a call arrives without the root user intent: the drift check
# is half the decision, and a caller that omits it gets neither half.
REQUIRE_CONTEXT = os.environ.get("CBAC_REQUIRE_CONTEXT", "true").lower() == "true"


# TODO:- Remove it.
class _LocalPolicyProvenance:
    """Policy source backed by ``<dir>/{agent_id}.md`` instead of the chain.

    Enabled by ``CBAC_LOCAL_POLICY_DIR``. Lets the service decide against a
    policy file with no Provenance Layer key — which is what ``examples/``
    needs to be runnable. Card writes are no-ops; LHI records still persist
    to Postgres, they just are not mirrored on-chain.

    ponytail: duck-types the four ``Provenance`` members ``CBAC`` actually
    calls, rather than subclassing it.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def get_latest_provenance_record(self, actor_id: str) -> dict[str, Any]:
        """Return an AgentCard-shaped dict whose policy is the file's text."""
        text = (self._dir / f"{actor_id}.md").read_text(encoding="utf-8")
        return {
            "type": "agent",
            "id": actor_id,
            "metadata": {},
            "policy": base64.b64encode(text.encode()).decode(),
        }

    def create_new_provenance_card(self, card_id: str, card_info: Any) -> None:
        return None

    def append_to_provenance_card(self, card_id: str, card_info: Any) -> None:
        return None


def _make_provenance() -> Provenance:
    local_dir = os.environ.get("CBAC_LOCAL_POLICY_DIR")
    if local_dir:
        logger.info("using local policy directory", directory=local_dir)
        return cast(Provenance, _LocalPolicyProvenance(Path(local_dir)))
    return Provenance(
        name="cbac-service",
        api_key=os.environ.get("AGENTDNA_API_KEY", ""),
    )


def _get_cbac() -> CBAC:
    global _cbac
    if _cbac is None:
        _cbac = CBAC(provenance=_make_provenance())
    return _cbac


def _bind_request_context(body: dict) -> None:
    """Bind the request identity so every log line from the decision pipeline
    below carries it without threading it through the call chain."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=uuid4().hex[:12],
        agent_id=body.get("agent_id", ""),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shutdown: close the connection pool. The engine builds itself lazily."""
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)

# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Basic healthcheck — verifies DB connectivity."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)


@app.post("/authorize-cbac")
async def authorize_cbac(request: Request) -> PlainTextResponse:
    """Decide whether an agent may perform an intended action.

    The body carries the reason and ``X-CBAC-Decision`` the verdict — that is
    the whole wire contract. The component scores stay server-side: reaching a
    decision also folds them into the (agent → callee) trust score, so there is
    no second call for the caller to make and no score to round-trip.
    """
    body = await request.json()
    _bind_request_context(body)

    try:
        async with get_session() as session:
            result = await _get_cbac().verify_cbac(
                session=session,
                agent_id=body.get("agent_id", ""),
                intended_action=body.get("intended_action"),
                user_intent=body.get("user_intent"),
                callee_name=body.get("callee_name", ""),
                callee_type=body.get("callee_type", "tool"),
            )
        decision, reason = result.decision, result.reason
        logger.info(
            "cbac decision",
            decision=decision,
            reason=reason,
            intent_score=result.intent_score,
            policy_score=result.policy_score,
            hallucination_score=result.hallucination_score,
            trust=result.trust,
        )
    except Exception as exc:
        decision, reason = "error", str(exc)
        logger.exception("cbac authorization failed")

    return PlainTextResponse(reason, headers={"X-CBAC-Decision": decision})


# ── Gateways that are not ours ────────────────────────────────────────────────


def _tools_call(raw: bytes) -> tuple[str, dict[str, Any]] | str | None:
    """Pull ``(tool_name, arguments)`` out of a JSON-RPC body.

    Returns ``None`` when the request carries no tool call and should simply be
    allowed through (an ``initialize``, a ``tools/list``, an empty session
    GET/DELETE), or a ``str`` reason when it must be refused instead.
    """
    if not raw.strip():
        return None  # session traffic with no body — cannot be a tool call
    try:
        payload = json.loads(raw)
    except ValueError:
        # A well-formed tools/call is always valid JSON. If this is not, we
        # cannot say what it asks for, so we do not wave it through.
        return "unparseable JSON-RPC body"
    if isinstance(payload, list):
        # Batching left the MCP spec in 2025-06-18. Authorizing one element and
        # forwarding the rest unchecked is the failure mode to avoid.
        return "batched JSON-RPC is not supported"
    if not isinstance(payload, dict) or payload.get("method") != "tools/call":
        return None
    params = payload.get("params") or {}
    return str(params.get("name", "")), dict(params.get("arguments") or {})


@app.post("/ext-authz")
async def ext_authz(request: Request) -> PlainTextResponse:
    """External authorization, for an MCP gateway this project does not own.

    Envoy, Istio, Gloo, agentgateway and Kong all speak the same shape: the
    gateway forwards the request it is about to proxy, and enforces the status
    code — 2xx forwards, anything else refuses. That is the entire contract, so
    an enterprise plugs CBAC in by pointing its existing gateway here. No CBAC
    code runs in the gateway.

    Identity comes from ``X-CBAC-Agent-Id``, which the **gateway** must set
    from the authenticated principal (a JWT claim, the mTLS SAN, the API key it
    resolved) and must strip from the incoming client request. A client-settable
    agent id is a client choosing its own policy.

    Unlike our own gateway this sees no tool description — only the wire — so
    the intended action reads "close issue" rather than "close an existing
    GitHub issue". Where the gateway can hand over the description, pass it as
    ``X-CBAC-Tool-Description`` and the prose matches again.
    """
    call = _tools_call(await request.body())
    if call is None:
        return PlainTextResponse("", headers={"X-CBAC-Decision": "allow"})
    if isinstance(call, str):
        return PlainTextResponse(
            f"CBAC: {call}", status_code=403, headers={"X-CBAC-Decision": "deny"}
        )

    name, arguments = call
    agent_id = request.headers.get("x-cbac-agent-id", "")
    user_intent = unquote(request.headers.get("x-cbac-user-intent", ""))
    description = request.headers.get("x-cbac-tool-description") or None

    if not agent_id:
        return PlainTextResponse(
            "CBAC: no authenticated agent id (X-CBAC-Agent-Id) — the gateway "
            "must set it from the authenticated principal",
            status_code=403,
            headers={"X-CBAC-Decision": "deny"},
        )
    if REQUIRE_CONTEXT and not user_intent:
        return PlainTextResponse(
            "CBAC: no governance context (X-CBAC-User-Intent missing) — the "
            "MCP client must install cbac_propagate",
            status_code=403,
            headers={"X-CBAC-Decision": "deny"},
        )

    _bind_request_context({"agent_id": agent_id})
    try:
        async with get_session() as session:
            result = await _get_cbac().verify_cbac(
                session=session,
                agent_id=agent_id,
                intended_action=render_intent(name, arguments, description),
                user_intent=user_intent or None,
                callee_name=name,
                callee_type="mcp",
            )
        decision, reason = result.decision, result.reason
        logger.info("cbac decision", decision=decision, reason=reason, tool=name)
    except Exception as exc:
        decision, reason = "error", str(exc)
        logger.exception("ext-authz authorization failed")

    return PlainTextResponse(
        reason,
        status_code=200 if decision == "allow" else 403,
        headers={"X-CBAC-Decision": decision},
    )


@app.post("/precompute-policy")
async def precompute_policy(request: Request) -> JSONResponse:
    """Precompute and cache policy embeddings for an agent.

    Call this after deploying or updating an agent's policy card. Supply
    ``policy`` (the decoded policy text) to embed that instead of reading the
    agent's card from the Provenance Layer.
    """
    body = await request.json()
    agent_id = body.get("agent_id", "")
    policy = body.get("policy")

    if not agent_id:
        return JSONResponse({"error": "agent_id is required"}, status_code=400)
    if policy is not None and not isinstance(policy, str):
        return JSONResponse({"error": "policy must be a string"}, status_code=400)

    try:
        async with get_session() as session:
            count = await _get_cbac().index_policy(
                session=session,
                agent_id=agent_id,
                policy=policy,
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse({"agent_id": agent_id, "chunks_stored": count})


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("CBAC_SERVICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CBAC_SERVICE_PORT", "8767")),
        # None = keep our JSON config; uvicorn's default would re-add its own handlers.
        log_config=None,
    )
