"""OpenFGA access for the v2 gate: the allow/deny tuples and the check.

Thin on purpose. Object and user ids arrive already built by
:mod:`cbac_service.ibac`; this module only moves tuples in and out. The
authorization model is ``fga_model.json``::

    tool_invocation.can_invoke = [user with non_expired_grant] but not blocked
    tool_invocation.blocked    = [user:*]

so an allow tuple carries its own expiry and an exact deny tuple on the same
object beats it. What a user and an object stand for is
:mod:`cbac_service.ibac`'s business.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from openfga_sdk import ClientConfiguration, OpenFgaClient
from openfga_sdk.client.models import (
    ClientCheckRequest,
    ClientTuple,
    ClientWriteRequest,
)
from openfga_sdk.client.models.write_conflict_opts import (
    ClientWriteRequestOnDuplicateWrites,
    ClientWriteRequestOnMissingDeletes,
    ConflictOptions,
)
from openfga_sdk.credentials import CredentialConfiguration, Credentials
from openfga_sdk.models import CreateStoreRequest, RelationshipCondition

from cbac_service.config import OPENFGA_API_TOKEN, OPENFGA_API_URL, OPENFGA_STORE_NAME

_MODEL_PATH = Path(__file__).with_name("fga_model.json")

# OpenFGA's default per-request tuple limit (OPENFGA_MAX_TUPLES_PER_WRITE).
_MAX_TUPLES_PER_WRITE = 100

# Re-writing a tuple that exists, or deleting one that does not, is not an
# error here: the Postgres row is the record of what *should* be in the store,
# and replaying it must converge rather than fail half way.
_IDEMPOTENT: Any = {
    "conflict": ConflictOptions(
        on_duplicate_writes=ClientWriteRequestOnDuplicateWrites.IGNORE,
        on_missing_deletes=ClientWriteRequestOnMissingDeletes.IGNORE,
    )
}

_client: OpenFgaClient | None = None
_lock = asyncio.Lock()


def _rfc3339(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


async def ensure_store() -> OpenFgaClient:
    """The client, bound to this environment's store and the current model.

    Finds the store named ``OPENFGA_STORE_NAME`` or creates it, so a restart
    keeps every tuple. The model is written on every first connect.
    ponytail: one model version per process start; OpenFGA keeps them all and
    checks against the id written here. Compare against the latest model first
    if the version list ever needs to stay short.
    """
    global _client
    if _client is not None:
        return _client
    if not OPENFGA_STORE_NAME:
        raise RuntimeError("OPENFGA_STORE_NAME is not set")
    async with _lock:
        if _client is None:
            credentials = (
                Credentials(
                    method="api_token",
                    configuration=CredentialConfiguration(api_token=OPENFGA_API_TOKEN),
                )
                if OPENFGA_API_TOKEN
                else None
            )
            client = OpenFgaClient(
                ClientConfiguration(api_url=OPENFGA_API_URL, credentials=credentials)
            )
            try:
                found = await client.list_stores({"name": OPENFGA_STORE_NAME})
                stores = [s for s in found.stores or [] if s.name == OPENFGA_STORE_NAME]
                if stores:
                    store_id = stores[0].id
                else:
                    store_id = (
                        await client.create_store(
                            CreateStoreRequest(name=OPENFGA_STORE_NAME)
                        )
                    ).id
                client.set_store_id(store_id)
                model = json.loads(_MODEL_PATH.read_text())
                written = await client.write_authorization_model(model)
                client.set_authorization_model_id(written.authorization_model_id)
            except Exception:
                await client.close()
                raise
            _client = client
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def _write(
    writes: Sequence[ClientTuple] = (), deletes: Sequence[ClientTuple] = ()
) -> None:
    """Writes then deletes, each in batches of the server's tuple limit."""
    client = await ensure_store()
    for batch, is_write in [(list(writes), True), (list(deletes), False)]:
        for i in range(0, len(batch), _MAX_TUPLES_PER_WRITE):
            chunk = batch[i : i + _MAX_TUPLES_PER_WRITE]
            body = ClientWriteRequest(
                writes=chunk if is_write else None,
                deletes=None if is_write else chunk,
            )
            # The SDK annotates `options` as scalars only, but reads the
            # ConflictOptions object out of it (options_to_conflict_info).
            await client.write(body, _IDEMPOTENT)


async def write_grants(
    grants: Sequence[tuple[str, str]], grant_time: datetime, ttl_seconds: int
) -> None:
    """Allow tuples for one intent, live for ``ttl_seconds`` from ``grant_time``.

    Each grant is a ``(user, object)`` pair, as built by
    :func:`~cbac_service.ibac.fga_user` and
    :func:`~cbac_service.ibac.object_id`. Pairs are deduplicated: OpenFGA
    refuses a request that carries a tuple twice, even with duplicate writes
    set to ignore.
    """
    condition = RelationshipCondition(
        name="non_expired_grant",
        context={
            "grant_time": _rfc3339(grant_time),
            "grant_duration": f"{ttl_seconds}s",
        },
    )
    await _write(
        writes=[
            ClientTuple(
                user=user, relation="can_invoke", object=obj, condition=condition
            )
            for user, obj in dict.fromkeys(grants)
        ]
    )


def _deny(obj: str) -> ClientTuple:
    return ClientTuple(user="user:*", relation="blocked", object=obj)


async def replace_denies(add: Sequence[str], remove: Sequence[str]) -> None:
    """Swap one policy version's exact deny tuples for the next one's.

    New tuples are written before old ones are deleted, so a rule present in
    both versions is never briefly missing. Both lists are deduplicated: two
    rules can name the same tool and value, or two spellings of it, and
    OpenFGA refuses a request that carries a tuple twice.
    """
    keep = set(add)
    await _write(
        writes=[_deny(obj) for obj in dict.fromkeys(add)],
        deletes=[_deny(obj) for obj in dict.fromkeys(remove) if obj not in keep],
    )


async def can_invoke(user: str, obj: str, now: datetime) -> bool:
    """Whether ``user`` holds a live allow tuple on ``obj`` and no deny blocks it."""
    client = await ensure_store()
    response = await client.check(
        ClientCheckRequest(
            user=user,
            relation="can_invoke",
            object=obj,
            context={"current_time": _rfc3339(now)},
        )
    )
    return bool(response.allowed)


async def is_blocked(user: str, obj: str) -> bool:
    """Whether a deny tuple covers ``obj``. Only asked after a failed
    :func:`can_invoke`, to say which of the two reasons denied the call."""
    client = await ensure_store()
    response = await client.check(
        ClientCheckRequest(user=user, relation="blocked", object=obj)
    )
    return bool(response.allowed)
