"""``LocalProvenance``: ``agentdna.provenance.Provenance`` without the chain.

Same constructor, same methods, same errors where callers depend on them
(``AgentDNA.record`` matches ``child nftId <id> already exists locally``). What
changes is where things live:

- **Identity.** Each actor name gets an Ed25519 key, kept as a PEM file in the
  local directory. Its DID is a CIDv1 (dag-pb, sha2-256, base32), so it starts
  with ``bafy`` like a Rubix DID, computed from the raw public key.
- **Cards and signatures.** Cards, card states, and every DID's public key are
  rows in the shared SQLite store (:mod:`agentdna_local.store`), so a signature
  made in one process verifies in another.

Nothing here opens a network connection.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from agentdna.helpers import _hash_content, canonicalize_envelope
from agentdna.id import get_agent_card_id
from agentdna.log import configure_logging, get_logger
from agentdna.types import Envelope
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .store import CardExistsError, CardNotFoundError, LocalStore, default_dir

# CIDv1 header: version 1, dag-pb codec (0x70), sha2-256 multihash (0x12, 32 bytes).
_CID_V1_DAG_PB_SHA256 = bytes([0x01, 0x70, 0x12, 0x20])


def did_from_public_key(public_key: bytes) -> str:
    """The DID of a raw Ed25519 public key: ``b`` + base32 of a CIDv1."""
    digest = hashlib.sha256(public_key).digest()
    encoded = base64.b32encode(_CID_V1_DAG_PB_SHA256 + digest).decode("ascii")
    return "b" + encoded.lower().rstrip("=")


def raw_public_key(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def _key_path(directory: Path, name: str) -> Path:
    # Names can be emails or have spaces; the hash keeps two names that
    # sanitize the same from sharing a key.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:48] or "actor"
    tag = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return directory / "keys" / f"{safe}-{tag}.pem"


def load_or_create_key(
    directory: str | os.PathLike[str], name: str
) -> Ed25519PrivateKey:
    """The actor's private key, created on first use. One key per actor name."""
    path = _key_path(Path(directory), name)
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{path} does not hold an Ed25519 key")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # O_EXCL: if another process created it first, use theirs.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return load_or_create_key(directory, name)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)
    return key


class ProvenanceError(Exception):
    """A card operation the chain would also refuse. Callers match its text,
    as they match the chain connector's, so the messages follow the original."""


class LocalProvenance:
    def __init__(
        self,
        name: str,
        provenance_url: str = "",
        api_key: str = "",
        config_path: str = "",
        timeout_in_seconds: int = 10000,
        logger: Any = None,
        actor_type: str = "",
    ):
        if name == "":
            raise NameError("'name' attribute cannot be empty")

        self.name = name
        self.config_dir = config_path or default_dir()
        self.store = LocalStore(self.config_dir)
        # Never contacted: kept so code that reads it (logs, the evidence sink)
        # names the local store rather than a chain connector.
        self.provenance_url = self.store.path.as_uri()

        self._key = load_or_create_key(self.config_dir, name)
        public_key = raw_public_key(self._key)
        self.provenance_id = did_from_public_key(public_key)
        self.store.put_actor(self.provenance_id, name, public_key)

        if logger is None:
            configure_logging()
            logger = get_logger("agentdna.provenance").bind(
                actor_name=name, actor_type=actor_type or "unknown"
            )
        self.logger = logger.bind(actor_id=self.provenance_id)

    # ── cards ────────────────────────────────────────────────────────────────
    def create_new_provenance_card(
        self,
        card_id: str,
        card_info: str,
        card_value: float = 0.001,
    ) -> dict[str, Any]:
        self.logger.info(
            "provenance_card.create.start", card_id=card_id, card_value=card_value
        )
        try:
            tx_id = self.store.create_card(card_id, self.provenance_id, card_info)
        except CardExistsError:
            self.logger.error(
                "provenance_card.create.failed", card_id=card_id, error="exists"
            )
            raise ProvenanceError(
                f"failed to create provenance card of id: {card_id}, err: card already exists"
            ) from None
        self.logger.info(
            "provenance_card.create.success", card_id=card_id, card_value=card_value
        )
        return {"status": True, "nft_id": card_id, "tx_id": tx_id}

    def create_new_child_provenance_card(
        self, parent_card_id: str, card_info: str, child_nft_id: str
    ) -> tuple[str, str]:
        if not self.store.card_exists(parent_card_id):
            error = f"parent card {parent_card_id} not found"
            self.logger.error(
                "provenance_card.create_child.failed",
                parent_card_id=parent_card_id,
                error=error,
            )
            raise ProvenanceError(f"Child NFT creation failed: {error}")
        try:
            tx_id = self.store.create_card(
                child_nft_id,
                self.provenance_id,
                card_info,
                parent_card_id=parent_card_id,
            )
        except CardExistsError:
            # AgentDNA.record matches this exact text and appends instead.
            error = f"child nftId {child_nft_id} already exists locally"
            self.logger.error(
                "provenance_card.create_child.failed",
                parent_card_id=parent_card_id,
                error=error,
            )
            raise ProvenanceError(f"Child NFT creation failed: {error}") from None
        self.logger.info(
            "provenance_card.create_child.success",
            parent_card_id=parent_card_id,
            card_id=child_nft_id,
            tx_id=tx_id,
        )
        return child_nft_id, tx_id

    def append_to_provenance_card(self, card_id: str, card_info: str) -> dict[str, Any]:
        self.logger.info("provenance_card.append.start", card_id=card_id)
        try:
            tx_id = self.store.append_state(card_id, self.provenance_id, card_info)
        except CardNotFoundError:
            self.logger.error(
                "provenance_card.append.failed", card_id=card_id, error="not_found"
            )
            raise ProvenanceError(
                f"failed to append to provenance card {card_id}, err: card not found"
            ) from None
        self.logger.info("provenance_card.append.success", card_id=card_id)
        return {"status": True, "message": "appended", "tx_id": tx_id}

    def provenance_card_history(self, actor_id: str) -> list[dict[str, Any]]:
        """Every state of the actor's card, oldest first; ``[]`` for none."""
        if actor_id == "":
            self.logger.error("provenance_card.history.failed", error="empty_actor_id")
            raise ValueError("agent_id cannot be empty")

        if not actor_id.startswith("bafy"):
            self.logger.error(
                "provenance_card.history.failed",
                actor_id=actor_id,
                error="invalid_actor_id_format",
            )
            raise ValueError(
                f"invalid agent_id provided: {actor_id}, it should start with 'bafy'"
            )

        result = self.store.card_states(get_agent_card_id(agent_id=actor_id))
        self.logger.info(
            "provenance_card.history.success", actor_id=actor_id, count=len(result)
        )
        return result

    def get_latest_provenance_record(self, actor_id: str) -> dict[str, Any]:
        history = self.provenance_card_history(actor_id=actor_id)
        if len(history) == 0:
            self.logger.error(
                "provenance_card.latest.failed",
                actor_id=actor_id,
                error="no_records_found",
            )
            raise ValueError(f"no provenance records found for actor {actor_id}")
        data = history[-1].get("data")
        if data is None:
            raise ValueError("latest provenance record does not contain data")
        return json.loads(data)

    # ── envelopes ────────────────────────────────────────────────────────────
    def sign_envelope(self, envelope: Envelope) -> str:
        """Ed25519 signature over the canonical envelope, hex encoded."""
        self.logger.info(
            "envelope.sign.start", envelope_from=envelope.from_, epoch=envelope.epoch
        )
        message = canonicalize_envelope(envelope)
        signature_hex = self._key.sign(message.encode("utf-8")).hex()
        self.logger.info(
            "envelope.sign.success",
            envelope_from=envelope.from_,
            epoch=envelope.epoch,
            signature=signature_hex,
        )
        return signature_hex

    def verify_envelope(self, envelope: Envelope) -> bool:
        """Checks the actor, the signature's format, the intrinsic hash, and
        the signature against the actor's public key from the shared store."""
        if envelope.from_ == "":
            self.logger.error("envelope.verify.missing_actor", epoch=envelope.epoch)
            return False

        if envelope.signature == "":
            self.logger.error(
                "envelope.verify.missing_signature",
                envelope_from=envelope.from_,
                epoch=envelope.epoch,
            )
            return False

        expected_hash = _hash_content(envelope)
        if envelope.hash != expected_hash:
            self.logger.error(
                "envelope.verify.hash_mismatch",
                envelope_from=envelope.from_,
                expected=expected_hash,
                actual=envelope.hash,
            )
            return False

        try:
            signature_bytes = bytes.fromhex(envelope.signature)
        except ValueError:
            self.logger.error(
                "envelope.verify.invalid_signature_format",
                envelope_from=envelope.from_,
                epoch=envelope.epoch,
                raw_signature=envelope.signature,
            )
            return False

        public_key = self.store.public_key(envelope.from_)
        if public_key is None:
            self.logger.error(
                "envelope.verify.unknown_actor",
                envelope_from=envelope.from_,
                epoch=envelope.epoch,
            )
            return False

        message = canonicalize_envelope(envelope)
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                signature_bytes, message.encode("utf-8")
            )
        except InvalidSignature:
            self.logger.error(
                "envelope.verify.signature_invalid",
                envelope_from=envelope.from_,
                epoch=envelope.epoch,
            )
            return False
        return True
