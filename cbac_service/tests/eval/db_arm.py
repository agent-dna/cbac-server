"""The optional real-Postgres arm (`--eval-db`).

Everything else in this suite replaces the search layer with numpy and Okapi
BM25. The stand-ins compute the same quantities — cosine over the same
normalised embeddings, RRF over the same two lists with the same constants —
but pgvector's HNSW index is approximate and pg_textsearch tokenises and
saturates differently, so "the same quantity" is a claim, not a guarantee.

This arm indexes every rendering through the production `index_policy` and runs
the production `verify_cbac` against it. Only divergences are reported: a
matching verdict says nothing new, and a differing one says the in-memory
numbers do not transfer.

No Provenance stub is needed — `index_policy` takes the policy text directly.
"""

from __future__ import annotations

import asyncio

from cbac_service.db.engine import close_db, get_session
from cbac_service.db.repository import delete_policy_chunks
from cbac_service.tests.eval.corpus import specs


async def _run(cbac) -> dict[str, tuple[str, int | None]]:
    out: dict[str, tuple[str, int | None]] = {}
    async with get_session() as session:
        for spec in specs():
            for rendering in spec.renderings:
                agent_id = f"eval:{spec.id}:{rendering.kind}"
                await delete_policy_chunks(session, agent_id)
                await cbac.index_policy(session, agent_id, policy=rendering.text)
                for intent, action in spec.actions:
                    result = await cbac.verify_cbac(
                        session,
                        agent_id,
                        action.text,
                        user_intent=intent.text,
                    )
                    key = f"{spec.id}/{intent.id}/{action.id}|{rendering.kind}"
                    out[key] = (result.decision, result.error_code)
                await delete_policy_chunks(session, agent_id)
    await close_db()
    return out


def compare(cbac, run) -> tuple[str, list[str]]:
    """(title, lines) for the report — the in-memory arm against Postgres."""
    db = _with_fresh_models(cbac)
    lines, diverged = [], 0
    for case in run.cases:
        for arm in ("structured", "unstructured"):
            key = f"{case.key}|{arm}"
            if key not in db:
                continue
            got = db[key]
            want = (case.arms[arm].final_decision, case.arms[arm].final_code)
            if got != want:
                diverged += 1
                if len(lines) < 40:
                    lines.append(f"    {arm:<14}{case.key:<44}{want} -> {got}")
    return (
        "REAL POSTGRES — where the in-memory search layer does not transfer",
        [
            (
                f"  {diverged} of {len(db)} verdicts differ against pgvector "
                f"+ pg_textsearch"
            ),
            "",
            *lines,
        ],
    )


def _with_fresh_models(cbac):
    """`runner._memoise` replaced the model calls with caches keyed on text.
    Those results are still correct here — same models, same inputs — so the DB
    arm reuses them and pays only for the search layer it is testing."""
    return asyncio.run(_run(cbac))
