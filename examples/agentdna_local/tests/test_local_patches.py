"""agentdna with the local patches installed: the real AgentDNA class, end to
end, with every hosted call replaced. The autouse ``no_network`` fixture fails
any test that tries to leave the machine."""

from __future__ import annotations

import json
from pathlib import Path

import agentdna_local
import pytest
from agentdna.auth import sink
from agentdna.core import AgentDNA
from agentdna.error import RESULT_OK
from agentdna.id import get_agent_card_id
from agentdna.types import VERIFY_HEAVY
from agentdna_local import LocalProvenance, LocalStore, whitelist_check

from agentdna import admin, core


@pytest.fixture
def policy_file(tmp_path) -> Path:
    path = tmp_path / "SKILLS.md"
    path.write_text("# Skills\n\n- Read the configured repository.\n", encoding="utf-8")
    return path


def test_install_replaces_every_hosted_entry_point(local_dir) -> None:
    assert core.Provenance is LocalProvenance
    assert core.get_default_config_dir() == str(local_dir)
    assert core.originate is agentdna_local.originate
    assert admin.request_agent_whitelist_check is whitelist_check
    assert agentdna_local.install() == str(local_dir)  # idempotent


def test_user_agent_chain_builds_verifies_and_records(local_dir, policy_file) -> None:
    user = AgentDNA(name="user@example.local", type="user")
    agent = AgentDNA(name="local-agent", type="agent", agent_policy_file=policy_file)
    tool = AgentDNA(name="local-tool", type="tool")

    request = user.build(payload="Summarise the repository")
    step = agent.build(payload="calling a tool", previous_workflows=request)
    # Another actor (here the tool) verifies the whole chain from the store.
    assert tool.verify(step, mode=VERIFY_HEAVY) == RESULT_OK

    card_id, _ = user.record(step)
    assert card_id == step.id
    # A second record of the same workflow appends rather than failing.
    assert user.record(step)[0] == step.id
    assert len(LocalStore(local_dir).card_states(step.id)) == 2


def test_agent_registration_stores_the_policy_as_its_card(
    local_dir, policy_file
) -> None:
    agent = AgentDNA(name="local-agent", type="agent", agent_policy_file=policy_file)
    record = agent.provenance.get_latest_provenance_record(agent.get_actor_id())
    assert record["type"] == "actor_card"
    assert record["id"] == agent.get_actor_id()

    # Re-registering with the same file adds nothing; a changed file appends.
    AgentDNA(name="local-agent", type="agent", agent_policy_file=policy_file)
    card_id = get_agent_card_id(agent.get_actor_id())
    assert len(LocalStore(local_dir).card_states(card_id)) == 1
    policy_file.write_text("# Skills\n\n- Changed.\n", encoding="utf-8")
    AgentDNA(name="local-agent", type="agent", agent_policy_file=policy_file)
    assert len(LocalStore(local_dir).card_states(card_id)) == 2


def test_agent_registration_needs_a_policy_file(local_dir, tmp_path) -> None:
    with pytest.raises(ValueError, match="policy path"):
        AgentDNA(name="no-policy", type="agent")
    with pytest.raises(FileNotFoundError):
        AgentDNA(
            name="no-policy", type="agent", agent_policy_file=tmp_path / "missing.md"
        )


def test_whitelist_is_registered_agents(local_dir, policy_file, monkeypatch) -> None:
    agent = AgentDNA(name="local-agent", type="agent", agent_policy_file=policy_file)
    other = AgentDNA(name="other-agent", type="agent", agent_policy_file=policy_file)
    tool = AgentDNA(name="local-tool", type="tool")

    assert whitelist_check("", agent.get_actor_id())
    assert not whitelist_check("", tool.get_actor_id())
    assert not whitelist_check("", "bafyunknown")

    monkeypatch.setenv("AGENTDNA_LOCAL_WHITELIST", "local-agent")
    assert whitelist_check("", agent.get_actor_id())
    assert not whitelist_check("", other.get_actor_id())

    monkeypatch.setenv("AGENTDNA_LOCAL_WHITELIST", other.get_actor_id())
    assert whitelist_check("", other.get_actor_id())
    assert not whitelist_check("", agent.get_actor_id())


def test_whitelist_check_in_the_mcp_middleware_is_patched(local_dir) -> None:
    checks = pytest.importorskip("agentdna.mcp.server.checks")
    assert checks.request_agent_whitelist_check is whitelist_check


def test_login_is_local(local_dir, monkeypatch) -> None:
    monkeypatch.setenv("AGENTDNA_LOCAL_USER_EMAIL", "someone@example.local")
    user = AgentDNA.login()
    assert user.name == "someone@example.local"
    assert user.run_id
    assert user.build(payload="hi").envelope.run_id == user.run_id


def test_auth_evidence_goes_to_the_store(local_dir, monkeypatch, tmp_path) -> None:
    class Evidence:
        def as_dict(self) -> dict:
            return {"request_id": "r-1", "kind": "test"}

    user = AgentDNA(name="user@example.local", type="user")
    evidence_file = tmp_path / "evidence.jsonl"
    monkeypatch.setenv(sink.ENV_ON, "1")
    monkeypatch.setenv(sink.ENV_FILE, str(evidence_file))

    sink.send(Evidence(), user)

    assert LocalStore(local_dir).evidence() == [{"request_id": "r-1", "kind": "test"}]
    assert json.loads(evidence_file.read_text()) == {
        "request_id": "r-1",
        "kind": "test",
    }
