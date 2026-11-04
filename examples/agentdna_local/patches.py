"""Point every hosted call agentdna makes at local stand-ins.

agentdna talks to four hosted services. Each is reached through one function
or class, and :func:`install` swaps each one at runtime — ``agentdna/`` itself
is not edited:

=========================================  ==========================================
hosted                                     local
=========================================  ==========================================
Rubix chain (``Provenance``)               :class:`~agentdna_local.provenance.LocalProvenance`
provenance URL ``/core/v1/register-*``     rows in the store's ``registry`` table
admin server whitelist                     registered agents, narrowed by
                                           ``AGENTDNA_LOCAL_WHITELIST``
auth server login (``originate``)          a fresh run id and a local email
auth-evidence post (``sink``)              rows in the store's ``auth_evidence`` table
=========================================  ==========================================

and the default config directory becomes the local directory, so identities,
keys and the store all live in one place (``AGENTDNA_LOCAL_DIR``).

Call :func:`install` before the first ``AgentDNA(...)`` is created: a module
that already did ``from agentdna.core import Provenance`` keeps what it got,
which is why the patch targets the names each module looks up at call time.
"""

from __future__ import annotations

import base64
import getpass
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from agentdna.card import build_actor_card_payload
from agentdna.id import get_agent_card_id

from .provenance import LocalProvenance
from .store import ENV_DIR, LocalStore, default_dir

ENV_WHITELIST = "AGENTDNA_LOCAL_WHITELIST"
ENV_USER_EMAIL = "AGENTDNA_LOCAL_USER_EMAIL"

_installed = False


# ── registration (AgentDNA.__register_*) ─────────────────────────────────────
def _store(dna: Any) -> LocalStore:
    return LocalStore(dna.config_dir)


def _register_user(self: Any) -> None:
    _store(self).register(self.get_actor_id(), "user", self.name)
    self.logger.info("agentdna.register_user.success", backend="local")


def _register_tool(self: Any) -> None:
    _store(self).register(self.get_actor_id(), "tool", self.name)
    self.logger.info("agentdna.register_tool.success", backend="local")


def _register_agent(self: Any) -> None:
    """Record the agent, and store its policy file as its agent card — what the
    hosted register-agent endpoint does with the uploaded file. A changed
    policy file is appended as a new card state; an unchanged one is not."""
    policy_path = self._AgentDNA__agent_policy_path
    if policy_path is None:
        raise ValueError("agent policy path is not defined")
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    actor_id = self.get_actor_id()
    store = _store(self)
    store.register(actor_id, "agent", self.name)

    policy_b64 = base64.b64encode(policy_path.read_bytes()).decode("utf-8")
    card = json.dumps(build_actor_card_payload(actor_id=actor_id, policy=policy_b64))
    card_id = get_agent_card_id(actor_id)
    states = store.card_states(card_id)
    if not states:
        store.create_card(card_id, actor_id, card)
    elif json.loads(states[-1]["data"]).get("policy") != policy_b64:
        store.append_state(card_id, actor_id, card)
    self.logger.info(
        "agentdna.register_agent.success", backend="local", policy_path=str(policy_path)
    )


# ── admin server whitelist ───────────────────────────────────────────────────
def whitelist_check(agentdna_admin_url: str, agent_id: str) -> bool:
    """A registered agent is whitelisted. ``AGENTDNA_LOCAL_WHITELIST`` (comma
    separated agent names or DIDs) narrows that to the agents it lists."""
    entry = LocalStore(default_dir()).registration(agent_id)
    if entry is None or entry["kind"] != "agent":
        return False
    allowed = {
        item.strip()
        for item in os.environ.get(ENV_WHITELIST, "").split(",")
        if item.strip()
    }
    return not allowed or agent_id in allowed or entry["name"] in allowed


# ── auth server login ────────────────────────────────────────────────────────
def originate(*, auth_server_url: str = "", timeout: float = 300) -> tuple[str, str]:
    """A new run id, and ``AGENTDNA_LOCAL_USER_EMAIL`` (default
    ``<login name>@localhost``) as the signed-in human. No browser, no server."""
    email = os.environ.get(ENV_USER_EMAIL) or f"{getpass.getuser()}@localhost"
    return uuid.uuid4().hex, email


# ── auth evidence ────────────────────────────────────────────────────────────
def _make_send(sink: Any):
    def send(evidence: Any, dna: Any) -> None:
        """The original's file append, and the post replaced by a store row."""
        record = evidence.as_dict()
        path = os.environ.get(sink.ENV_FILE, "")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if sink._posting():
            LocalStore(getattr(dna, "config_dir", None) or default_dir()).add_evidence(
                record
            )

    return send


def install(directory: str | os.PathLike[str] | None = None) -> str:
    """Patch agentdna to run locally; returns the local directory. Idempotent.

    ``directory`` overrides ``AGENTDNA_LOCAL_DIR`` (default ``~/.agentdna-local``),
    and is written back to the environment so every later lookup agrees.
    """
    global _installed
    local_dir = str(Path(directory or default_dir()).expanduser())
    os.environ[ENV_DIR] = local_dir
    LocalStore(local_dir)

    if _installed:
        return local_dir

    import agentdna.admin
    import agentdna.config
    import agentdna.core
    import agentdna.origination
    import agentdna.provenance
    from agentdna.auth import sink

    def get_default_config_dir() -> str:
        return default_dir()

    for module in (agentdna.config, agentdna.core, agentdna.provenance):
        module.get_default_config_dir = get_default_config_dir

    agentdna.core.Provenance = LocalProvenance
    agentdna.provenance.Provenance = LocalProvenance

    agentdna.core.AgentDNA._AgentDNA__register_user = _register_user
    agentdna.core.AgentDNA._AgentDNA__register_agent = _register_agent
    agentdna.core.AgentDNA._AgentDNA__register_tool = _register_tool

    agentdna.core.originate = originate
    agentdna.origination.originate = originate

    agentdna.admin.request_agent_whitelist_check = whitelist_check
    # checks.py imports fastmcp; it is only there when the MCP extras are.
    try:
        checks = importlib.import_module("agentdna.mcp.server.checks")
    except ImportError:
        pass
    else:
        checks.request_agent_whitelist_check = whitelist_check

    sink.send = _make_send(sink)

    _installed = True
    return local_dir
