"""LocalProvenance: identities, signatures, cards. No network, no chain."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from agentdna.helpers import _hash_content
from agentdna.id import get_agent_card_id
from agentdna.types import Envelope
from agentdna_local import LocalProvenance, did_from_public_key
from agentdna_local.provenance import ProvenanceError, raw_public_key


def _envelope(sender: LocalProvenance, payload: str = "hello") -> Envelope:
    envelope = Envelope(
        from_=sender.provenance_id, payload=payload, epoch=1, run_id="run"
    )
    envelope.hash = _hash_content(envelope)
    envelope.signature = sender.sign_envelope(envelope)
    return envelope


def test_did_is_a_bafy_cid_of_the_public_key(local_dir) -> None:
    actor = LocalProvenance("alice", config_path=str(local_dir))
    assert actor.provenance_id.startswith("bafy")
    assert actor.provenance_id == did_from_public_key(raw_public_key(actor._key))


def test_one_key_per_name_survives_a_restart(local_dir) -> None:
    first = LocalProvenance("alice", config_path=str(local_dir))
    again = LocalProvenance("alice", config_path=str(local_dir))
    other = LocalProvenance("bob", config_path=str(local_dir))
    assert first.provenance_id == again.provenance_id
    assert first.provenance_id != other.provenance_id


def test_signature_verifies_from_another_instance(local_dir) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    # A second instance stands in for another process: it shares only the store.
    verifier = LocalProvenance("verifier", config_path=str(local_dir))
    assert verifier.verify_envelope(_envelope(alice))


@pytest.mark.parametrize(
    "tamper",
    [
        lambda e: replace(e, payload="changed"),  # hash no longer matches
        lambda e: replace(e, signature="00" * 64),  # wrong signature
        lambda e: replace(e, signature="not-hex"),
        lambda e: replace(e, signature=""),
        lambda e: replace(e, from_=""),
    ],
)
def test_tampered_envelopes_fail(local_dir, tamper) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    assert not alice.verify_envelope(tamper(_envelope(alice)))


def test_a_signature_by_someone_else_fails(local_dir) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    mallory = LocalProvenance("mallory", config_path=str(local_dir))
    envelope = _envelope(alice)
    envelope.signature = mallory.sign_envelope(envelope)
    assert not alice.verify_envelope(envelope)


def test_unknown_actor_fails(local_dir, tmp_path) -> None:
    stranger = LocalProvenance("stranger", config_path=str(tmp_path / "elsewhere"))
    alice = LocalProvenance("alice", config_path=str(local_dir))
    assert not alice.verify_envelope(_envelope(stranger))


def test_cards_create_append_and_history(local_dir) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    card_id = get_agent_card_id(alice.provenance_id)
    alice.create_new_provenance_card(card_id, '{"v": 1}')
    alice.append_to_provenance_card(card_id, '{"v": 2}')

    history = alice.provenance_card_history(alice.provenance_id)
    assert [state["data"] for state in history] == ['{"v": 1}', '{"v": 2}']
    assert alice.get_latest_provenance_record(alice.provenance_id) == {"v": 2}

    with pytest.raises(ProvenanceError, match="already exists"):
        alice.create_new_provenance_card(card_id, "{}")
    with pytest.raises(ProvenanceError, match="not found"):
        alice.append_to_provenance_card("no-such-card", "{}")


def test_history_checks_the_actor_id(local_dir) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    with pytest.raises(ValueError, match="cannot be empty"):
        alice.provenance_card_history("")
    with pytest.raises(ValueError, match="should start with 'bafy'"):
        alice.provenance_card_history("did:example:123")
    assert alice.provenance_card_history(alice.provenance_id) == []
    with pytest.raises(ValueError, match="no provenance records"):
        alice.get_latest_provenance_record(alice.provenance_id)


def test_child_card_errors_match_what_record_parses(local_dir) -> None:
    alice = LocalProvenance("alice", config_path=str(local_dir))
    alice.create_new_provenance_card("parent", "{}")

    child_id, tx_id = alice.create_new_child_provenance_card("parent", "{}", "child-1")
    assert child_id == "child-1" and tx_id

    with pytest.raises(ProvenanceError) as exc:
        alice.create_new_child_provenance_card("parent", "{}", "child-1")
    # AgentDNA.record switches to an append on exactly this text.
    match = re.search(r"child nftId (\S+) already exists locally", str(exc.value))
    assert match and match.group(1) == "child-1"

    with pytest.raises(ProvenanceError, match="parent card"):
        alice.create_new_child_provenance_card("missing", "{}", "child-2")
