import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import structlog
import uvicorn
from agentdna.provenance import Provenance
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text

from cbac_service.cbac import CBAC
from cbac_service.db.engine import close_db, get_session
from cbac_service.logging_config import setup_logging

# ── App lifecycle ──────────────────────────────────────────────────────────────

setup_logging()
logger = structlog.get_logger("cbac_service.api")

_cbac: CBAC | None = None


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
