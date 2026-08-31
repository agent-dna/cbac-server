from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


@pytest.fixture
def rows(monkeypatch):
    """In-memory stand-in for the lhi_records table (one entry per decision),
    patched over the repository functions cbac.py imports by name.

    Shared because `verify_cbac` now writes a trust record itself, so the
    verify tests need the same fakes the LHI tests do.
    """
    pytest.importorskip("sentence_transformers")
    import cbac_service.cbac as cbac_mod

    records: list[SimpleNamespace] = []

    async def fake_get_latest_trust(session, agent_id, callee_name, callee_type):
        for record in reversed(records):
            if (record.agent_id, record.callee_name, record.callee_type) == (
                agent_id,
                callee_name,
                callee_type,
            ):
                return record.trust
        return None

    async def fake_insert_lhi_record(session, **kwargs):
        record = SimpleNamespace(
            id=len(records) + 1, created_at=datetime.now(timezone.utc), **kwargs
        )
        records.append(record)
        return record

    monkeypatch.setattr(cbac_mod, "get_latest_trust", fake_get_latest_trust)
    monkeypatch.setattr(cbac_mod, "insert_lhi_record", fake_insert_lhi_record)
    return records


@pytest.fixture
def decisions(monkeypatch):
    """In-memory stand-in for the cbac_decisions table.

    `verify_cbac` records on every path, so any test that calls it needs this
    or it reaches the real repository.
    """
    pytest.importorskip("sentence_transformers")
    import cbac_service.cbac as cbac_mod

    records: list[SimpleNamespace] = []

    async def fake_insert_cbac_decision(session, **kwargs):
        record = SimpleNamespace(
            id=len(records) + 1, created_at=datetime.now(timezone.utc), **kwargs
        )
        records.append(record)
        return record

    monkeypatch.setattr(cbac_mod, "insert_cbac_decision", fake_insert_cbac_decision)
    return records
