"""The CBAC decision service, run against policy files on disk.

In production `cbac_service` resolves an agent's policy from the Provenance
Layer, which needs an API key and a registered actor id. This demo has neither,
so it starts the same app with a policy source that reads `policies/{agent_id}.md`
instead — and the substitution lives here, in the demo, rather than as a branch
inside the service.

Nothing else about the service changes: the same pipeline, the same endpoints,
the same Postgres. Only where the policy text comes from.

Run from the **repo root**, with the root venv (the service's ML dependencies
live there, not in this directory's venv):

    DATABASE_URL="postgresql+asyncpg://cbac_user:cbac_pass@localhost:5432/cbac" \\
    HYBRID_SEARCH_ENABLED=true \\
      uv run python examples/github_agent_workflow/service.py

Env: DATABASE_URL, HYBRID_SEARCH_ENABLED, CBAC_SERVICE_HOST/PORT,
CBAC_POLICY_DIR (defaults to ./policies beside this file).
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any, cast

import uvicorn
from agentdna.provenance import Provenance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cbac_service import main
from cbac_service.cbac import CBAC

POLICY_DIR = Path(
    os.environ.get("CBAC_POLICY_DIR", Path(__file__).resolve().parent / "policies")
)
HOST = os.environ.get("CBAC_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CBAC_SERVICE_PORT", "8767"))


class LocalPolicyProvenance:
    """Policy source backed by ``<dir>/{agent_id}.md`` instead of the chain.

    Duck-types the three ``Provenance`` members ``CBAC`` actually calls rather
    than subclassing it. Card writes are no-ops — LHI records still persist to
    Postgres, they just are not mirrored on-chain, and nothing in this demo
    reads the card back.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def get_latest_provenance_record(self, actor_id: str) -> dict[str, Any]:
        """Return an AgentCard-shaped dict whose policy is the file's text."""
        text = (self._dir / f"{actor_id}.md").read_text(encoding="utf-8")
        return {
            "type": "agent",
            "id": actor_id,
            "metadata": {},
            "policy": base64.b64encode(text.encode()).decode(),
        }

    def create_new_provenance_card(self, card_id: str, card_info: Any) -> None:
        return None

    def append_to_provenance_card(self, card_id: str, card_info: Any) -> None:
        return None


if __name__ == "__main__":
    if not POLICY_DIR.is_dir():
        raise SystemExit(f"[service] no policy directory at {POLICY_DIR}")

    # The app builds its engine on first use and keeps it in this module global,
    # so assigning one is all it takes for the service to never reach the chain.
    # Cheap: CBAC loads its three models lazily, on the first decision. Done
    # here rather than at import so importing this file configures nothing.
    main._cbac = CBAC(provenance=cast(Provenance, LocalPolicyProvenance(POLICY_DIR)))

    policies = sorted(p.stem for p in POLICY_DIR.glob("*.md"))
    print(f"[service] {HOST}:{PORT} — policies from {POLICY_DIR}")
    print(f"[service] agent ids served: {', '.join(policies) or '(none found)'}")
    uvicorn.run(main.app, host=HOST, port=PORT, log_config=None)
