import asyncio
from datetime import UTC, datetime

from cbac_service import fga


class _Client:
    def __init__(self):
        self.requests = []

    async def write(self, body, options):
        self.requests.append(body)


def test_writes_never_carry_a_tuple_twice(monkeypatch):
    """OpenFGA refuses a request that repeats a tuple, even with duplicates
    set to ignore — so a policy with two rules on one value must still
    activate, and two grants on one object must still be written."""
    client = _Client()

    async def ensure_store():
        return client

    monkeypatch.setattr(fga, "ensure_store", ensure_store)
    asyncio.run(fga.replace_denies(["a", "b", "a"], ["c", "c", "a"]))
    asyncio.run(fga.write_grants("user:u", ["x", "x"], datetime.now(UTC), 60))

    writes, deletes, grants = client.requests
    assert [t.object for t in writes.writes] == ["a", "b"]
    assert [t.object for t in deletes.deletes] == ["c"]
    assert [t.object for t in grants.writes] == ["x"]
