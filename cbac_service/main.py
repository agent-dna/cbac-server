from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
import uvicorn
from agentdna.provenance import Provenance
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text

from cbac_service import error_codes as ec
from cbac_service.cbac import CBAC
from cbac_service.config import (
    AGENTDNA_API_KEY,
    CBAC_SERVICE_HOST,
    CBAC_SERVICE_PORT,
    PROVENANCE_URL,
)
from cbac_service.db.engine import close_db, get_session
from cbac_service.db.models import CBACDecision, LHIRecord
from cbac_service.db.repository import (
    get_cbac_decision,
    get_cbac_decision_by_hash,
    get_cbac_decisions,
    get_latest_trust_for_agents,
)
from cbac_service.entity import (
    AuthorizeRequest,
    LHIScoresRequest,
    PrecomputePolicyRequest,
)
from cbac_service.logging_config import setup_logging
from cbac_service.skills import render_intent

# ── App lifecycle ──────────────────────────────────────────────────────────────

setup_logging()
logger = structlog.get_logger("cbac_service.api")

_cbac: CBAC | None = None


def _get_cbac() -> CBAC:
    """The decision engine, built on first use.

    Policy always comes from the Provenance Layer. A harness that has to decide
    against policy from somewhere else assigns its own engine to ``_cbac``
    before the first request
    """
    global _cbac
    if _cbac is None:
        _cbac = CBAC(
            provenance=Provenance(
                name="cbac-service",
                api_key=AGENTDNA_API_KEY,
                provenance_url=PROVENANCE_URL,
            )
        )
    return _cbac


def _bind_request_context(agent_id: str) -> None:
    """Bind the request identity so every log line from the decision pipeline
    below carries it without threading it through the call chain."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=uuid4().hex[:12],
        agent_id=agent_id,
    )


# ── Wire format ────────────────────────────────────────────────────────────────


def api_response(
    data: Any = None,
    message: str = "",
    success: bool = True,
    status_code: int = 200,
) -> JSONResponse:
    """Every response this service sends: ``{success, message, data}``.

    ``success`` says the request was *processed*, never what the answer was. On
    ``/authorize`` in particular a deny is a perfectly successful call — the
    verdict is ``data.decision``, and reading ``success`` as "allowed" would
    invert the gate. ``message`` is prose for a human; ``data`` is what a
    program reads.
    """
    return JSONResponse(
        {"success": success, "message": message, "data": data},
        status_code=status_code,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shutdown: close the connection pool. The engine builds itself lazily."""
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)

API_PREFIX = "/cbac/v1"
router = APIRouter(prefix=API_PREFIX)

# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """Malformed bodies and query params answer in the same envelope as
    everything else, rather than in FastAPI's own error shape."""
    return api_response(
        # jsonable_encoder because an error's ``ctx`` can hold the original
        # exception object, which json.dumps cannot serialize.
        data={"errors": jsonable_encoder(exc.errors())},
        message="request validation failed",
        success=False,
        status_code=422,
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Basic healthcheck — verifies DB connectivity."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return api_response(data={"status": "ok"}, message="ok")
    except Exception as exc:
        return api_response(
            data={"status": "error"},
            message=str(exc),
            success=False,
            status_code=503,
        )


def _intended_action(body: AuthorizeRequest) -> Any:
    """The action text to score, rendered here unless the caller pre-rendered it.

    An enforcement point sends mechanical facts — ``callee_name``,
    ``arguments``, ``callee_description`` — and this phrases them, so every PEP
    asks about the same call in the same words no matter what protocol it
    speaks or language it is written in. The tiers score prose, so the wording
    is part of the decision and belongs next to the models it was tuned for.

    ``intended_action`` wins when supplied, and is the whole body a caller
    needs: it covers both a caller with phrasing of its own
    (``cbac_guard(action_intent=...)``) and a bare probe like the ``curl``
    examples, which have a sentence and nothing to render from.
    """
    if body.intended_action is not None or not body.callee_name:
        return body.intended_action
    return render_intent(body.callee_name, body.arguments, body.callee_description)


@router.post("/authorize")
async def authorize_cbac(body: AuthorizeRequest) -> JSONResponse:
    """Decide whether an agent may perform an intended action.

    ``data.decision`` is the verdict — ``"allow" | "deny" | "advise" |
    "error"`` — and ``message`` is the reason behind it. The component scores
    stay server-side: reaching a decision also folds them into the (agent →
    callee) trust score, so there is no second call for the caller to make and
    no score to round-trip.

    Always HTTP 200, including the fail-closed ``"error"`` verdict, because a
    decision *was* reached and the caller acts on it. A 5xx would invite the
    retry that a decision must not have: the trust fold is not idempotent, so
    a retried call counts the same evidence twice.
    """
    _bind_request_context(body.agent_id)

    try:
        async with get_session() as session:
            result = await _get_cbac().verify_cbac(
                session=session,
                agent_id=body.agent_id,
                intended_action=_intended_action(body),
                user_intent=body.user_intent,
                callee_name=body.callee_name,
                callee_type=body.callee_type,
                intent_id=body.intent_id,
            )
        decision, reason, status_code = (
            result.decision,
            result.reason,
            result.error_code,
        )
        interaction_hash = result.interaction_hash
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
        # Raised before or around _decide (e.g. the DB session never opens),
        # so there is no CBACResult and no cbac_decisions row — a pipeline
        # error_code would misattribute this to a layer that never ran.
        decision, reason, status_code = "error", str(exc), ec.API_UNHANDLED_EXCEPTION
        interaction_hash = None
        logger.exception("cbac authorization failed")

    response = api_response(
        data={
            "decision": decision,
            "status_code": status_code,
            "hash": interaction_hash,
        },
        message=reason,
        # The gate ran and answered; the answer itself lives in data.decision.
        success=decision != "error",
    )
    # Also a header, so a proxy or access log can route on the verdict without
    # parsing a body. The body stays authoritative.
    response.headers["X-CBAC-Decision"] = decision
    return response


MAX_DECISION_LIMIT = 500


def _decision_json(record: CBACDecision) -> dict:
    """Serialize one audit-log row for the wire."""
    return {
        "id": record.id,
        "agent_id": record.agent_id,
        "decision": record.decision,
        "reason": record.reason,
        "error_code": record.error_code,
        "interaction_hash": record.interaction_hash,
        "intent_id": record.intent_id,
        "intended_action": record.intended_action,
        "user_intent": record.user_intent,
        "callee_name": record.callee_name,
        "callee_type": record.callee_type,
        "created_at": record.created_at.isoformat(),
    }


@router.get("/decisions")
async def list_cbac_decisions(
    agent_id: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """Past authorization verdicts for one agent, newest first.

    Every decision is here — including the ones that never reached the trust
    fold (no callee, or an infrastructure-failure deny), which is what makes
    this the audit log rather than ``lhi_records``.
    """
    # Clamped rather than rejected: a caller asking for everything gets a large
    # page, not a 422 it has to learn about.
    limit = min(limit, MAX_DECISION_LIMIT)

    try:
        async with get_session() as session:
            records = await get_cbac_decisions(
                session, agent_id=agent_id, limit=limit, offset=offset
            )
    except Exception as exc:
        return api_response(message=str(exc), success=False, status_code=500)

    decisions = [_decision_json(record) for record in records]
    return api_response(
        data={
            "agent_id": agent_id,
            "decisions": decisions,
            "count": len(decisions),
        },
        message=f"{len(decisions)} decision(s) for {agent_id}",
    )


@router.get("/decisions/{decision_id}")
async def read_cbac_decision(decision_id: int) -> JSONResponse:
    """One authorization verdict by id."""
    try:
        async with get_session() as session:
            record = await get_cbac_decision(session, decision_id=decision_id)
    except Exception as exc:
        return api_response(message=str(exc), success=False, status_code=500)

    if record is None:
        return api_response(
            message=f"no decision {decision_id}", success=False, status_code=404
        )
    return api_response(data=_decision_json(record))


@router.get("/decisions/by-hash/{interaction_hash}")
async def read_cbac_decision_by_hash(interaction_hash: str) -> JSONResponse:
    """One authorization verdict by its ``interaction_hash``.

    For a middleware dashboard that already holds the hash from a prior
    ``/cbac/v1/authorize`` call (or from streaming ``cbac_decisions`` some other
    way) and wants the full row back without knowing the numeric ``id``.

    ``interaction_hash`` is sha256 hex (64 lowercase hex chars) — rejected
    with 400 if it isn't shaped like one, so a malformed lookup fails fast
    instead of silently returning "not found". Not a uniqueness constraint:
    the same interaction can be decided more than once, so this returns the
    newest matching row — see ``get_cbac_decision_by_hash``.
    """
    if len(interaction_hash) != 64 or any(
        c not in "0123456789abcdef" for c in interaction_hash.lower()
    ):
        return api_response(
            message="interaction_hash must be 64 hex characters",
            success=False,
            status_code=400,
        )

    try:
        async with get_session() as session:
            record = await get_cbac_decision_by_hash(
                session, interaction_hash=interaction_hash.lower()
            )
    except Exception as exc:
        return api_response(message=str(exc), success=False, status_code=500)

    if record is None:
        return api_response(
            message=f"no decision with interaction_hash {interaction_hash}",
            success=False,
            status_code=404,
        )
    return api_response(data=_decision_json(record))


def _lhi_record_json(record: LHIRecord) -> dict:
    """Serialize one trust-history row for the wire — the edge fields
    (agent_id) are omitted since the caller already keys the response by them."""
    return {
        "callee_name": record.callee_name,
        "callee_type": record.callee_type,
        "intent_score": record.intent_score,
        "policy_score": record.policy_score,
        "hallucination_score": record.hallucination_score,
        "trust": record.trust,
        "created_at": record.created_at.isoformat(),
    }


@router.post("/lhi-scores")
async def read_lhi_scores(body: LHIScoresRequest) -> JSONResponse:
    """Current trust for a batch of agents, one entry per caller→callee edge.

    Trust is tracked per (agent_id, callee_name, callee_type) edge (see the
    LHI section in CLAUDE.md), not per agent, so an agent with several
    callees comes back with several edges. An agent with no history yet is
    still a key in the response, mapped to an empty list, rather than
    omitted — the caller doesn't have to distinguish "no history" from
    "didn't ask".

    A POST for what is a read because the batch of ids is the request body;
    a fleet dashboard's id list does not belong in a query string.
    """
    try:
        async with get_session() as session:
            records = await get_latest_trust_for_agents(
                session, agent_ids=body.agent_ids
            )
    except Exception as exc:
        return api_response(message=str(exc), success=False, status_code=500)

    agents: dict[str, list[dict]] = {agent_id: [] for agent_id in body.agent_ids}
    for record in records:
        agents[record.agent_id].append(_lhi_record_json(record))

    edges = sum(len(v) for v in agents.values())
    return api_response(
        data={"agents": agents},
        message=f"{edges} trust edge(s) across {len(agents)} agent(s)",
    )


@router.post("/policies/precompute")
async def precompute_policy(body: PrecomputePolicyRequest) -> JSONResponse:
    """Precompute and cache policy embeddings for an agent.

    Call this after deploying or updating an agent's policy card. Supply
    ``policy`` (the decoded policy text) to embed that instead of reading the
    agent's card from the Provenance Layer.
    """
    _bind_request_context(body.agent_id)

    try:
        async with get_session() as session:
            count = await _get_cbac().index_policy(
                session=session,
                agent_id=body.agent_id,
                policy=body.policy,
            )
    except Exception as exc:
        return api_response(message=str(exc), success=False, status_code=500)

    return api_response(
        data={"agent_id": body.agent_id, "chunks_stored": count},
        message=f"indexed {count} chunk(s) for {body.agent_id}",
    )


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=CBAC_SERVICE_HOST,
        port=CBAC_SERVICE_PORT,
        # None = keep our JSON config; uvicorn's default would re-add its own handlers.
        log_config=None,
    )
