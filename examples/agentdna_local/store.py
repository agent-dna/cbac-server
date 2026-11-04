"""The local stand-in for AgentDNA's hosted state: one SQLite file.

Everything the hosted services would hold lives here — each actor's DID and
public key, the registry of users, agents and tools, provenance cards and the
states appended to them, and auth evidence. It is one file so that separate
processes (the agent, the MCP server, the CBAC service) see the same actors and
can verify each other's signatures.

Each operation opens its own connection: the relay signs from worker threads,
and a per-call connection is the simplest thing that is safe there. WAL mode
and a busy timeout let several processes write without "database is locked".
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

ENV_DIR = "AGENTDNA_LOCAL_DIR"
DB_NAME = "agentdna-local.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS actors (
    did         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    public_key  BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS registry (
    did         TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- user | agent | tool
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    card_id         TEXT PRIMARY KEY,
    owner_did       TEXT NOT NULL,
    parent_card_id  TEXT,
    created_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS card_states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id     TEXT NOT NULL REFERENCES cards(card_id),
    data        TEXT NOT NULL,
    tx_id       TEXT NOT NULL,
    signer_did  TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_card_states_card ON card_states(card_id, id);
CREATE TABLE IF NOT EXISTS auth_evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT,
    record      TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class CardExistsError(Exception):
    """A card id that is already taken."""


class CardNotFoundError(Exception):
    """A card id that does not exist."""


def default_dir() -> str:
    """``$AGENTDNA_LOCAL_DIR``, or ``~/.agentdna-local``."""
    return os.environ.get(ENV_DIR) or str(Path.home() / ".agentdna-local")


class LocalStore:
    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self.dir = Path(directory or default_dir()).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / DB_NAME
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ── actors: DID → public key ─────────────────────────────────────────────
    def put_actor(self, did: str, name: str, public_key: bytes) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO actors (did, name, public_key, created_at) "
                "VALUES (?, ?, ?, ?)",
                (did, name, public_key, time.time()),
            )

    def public_key(self, did: str) -> bytes | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT public_key FROM actors WHERE did = ?", (did,)
            ).fetchone()
        return None if row is None else bytes(row["public_key"])

    # ── registry: what the hosted register-* endpoints record ─────────────────
    def register(self, did: str, kind: str, name: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO registry (did, kind, name, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(did) DO UPDATE SET kind = excluded.kind, name = excluded.name",
                (did, kind, name, time.time()),
            )

    def registration(self, did: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT did, kind, name FROM registry WHERE did = ?", (did,)
            ).fetchone()
        return None if row is None else dict(row)

    # ── provenance cards ─────────────────────────────────────────────────────
    def create_card(
        self,
        card_id: str,
        owner_did: str,
        data: str,
        parent_card_id: str | None = None,
    ) -> str:
        """Create a card with its first state; returns the state's tx id."""
        tx_id = uuid.uuid4().hex
        now = time.time()
        with closing(self._connect()) as conn, conn:
            try:
                conn.execute(
                    "INSERT INTO cards (card_id, owner_did, parent_card_id, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (card_id, owner_did, parent_card_id, now),
                )
            except sqlite3.IntegrityError as exc:
                raise CardExistsError(card_id) from exc
            conn.execute(
                "INSERT INTO card_states (card_id, data, tx_id, signer_did, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (card_id, data, tx_id, owner_did, now),
            )
        return tx_id

    def card_exists(self, card_id: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM cards WHERE card_id = ?", (card_id,)
            ).fetchone()
        return row is not None

    def append_state(self, card_id: str, signer_did: str, data: str) -> str:
        tx_id = uuid.uuid4().hex
        with closing(self._connect()) as conn, conn:
            if (
                conn.execute(
                    "SELECT 1 FROM cards WHERE card_id = ?", (card_id,)
                ).fetchone()
                is None
            ):
                raise CardNotFoundError(card_id)
            conn.execute(
                "INSERT INTO card_states (card_id, data, tx_id, signer_did, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (card_id, data, tx_id, signer_did, time.time()),
            )
        return tx_id

    def card_states(self, card_id: str) -> list[dict[str, Any]]:
        """Every state of a card, oldest first, shaped like the chain's
        ``get_nft_states`` rows: the written text is under ``data``."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT card_id, data, tx_id, signer_did, created_at FROM card_states "
                "WHERE card_id = ? ORDER BY id",
                (card_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── auth evidence ────────────────────────────────────────────────────────
    def add_evidence(self, record: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO auth_evidence (request_id, record, created_at) VALUES (?, ?, ?)",
                (record.get("request_id"), json.dumps(record), time.time()),
            )

    def evidence(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT record FROM auth_evidence ORDER BY id"
            ).fetchall()
        return [json.loads(row["record"]) for row in rows]
