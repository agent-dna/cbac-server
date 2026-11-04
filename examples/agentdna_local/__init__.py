"""Run agentdna and the CBAC v2 gate on one machine, with no hosted services.

    import agentdna_local
    agentdna_local.install()        # before the first AgentDNA(...)

See README.md beside this package for what is replaced and how.
"""

from .cbac_v2 import ALLOW_CODES, make_cbac_fn, scoped_intent_id
from .patches import install, originate, whitelist_check
from .policy import (
    actor_did,
    policy_document,
    server_tools,
    upload_agent_policy,
    upload_policy,
)
from .provenance import LocalProvenance, did_from_public_key
from .store import LocalStore, default_dir

__all__ = [
    "ALLOW_CODES",
    "LocalProvenance",
    "LocalStore",
    "actor_did",
    "default_dir",
    "did_from_public_key",
    "install",
    "make_cbac_fn",
    "originate",
    "policy_document",
    "scoped_intent_id",
    "server_tools",
    "upload_agent_policy",
    "upload_policy",
    "whitelist_check",
]
