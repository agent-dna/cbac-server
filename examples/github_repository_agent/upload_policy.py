"""Upload the GitHub agent's CBAC v2 policy to the local CBAC service.

The document is `cbac_policies/github_agent.md` as its Policy Instructions and
the tools `mcp_server.py` serves as its Tool Schema, stored under the agent's
local DID (the `agent_id` the MCP middleware sends). The configured repository
goes along as trusted data, so a request that only says "the repository" can
still be granted it.

Run with the CBAC service up (examples/agentdna_local/service.py):

    CBAC_POLICY_KEY=... python upload_policy.py [--review-required]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from config import settings
from mcp_server import mcp

from agentdna_local import upload_agent_policy

POLICY = Path(__file__).resolve().parent / "cbac_policies" / "github_agent.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--review-required",
        action="store_true",
        help="hold the version pending until it is approved",
    )
    args = parser.parse_args()
    reply = asyncio.run(
        upload_agent_policy(
            settings.github_agent_name,
            POLICY,
            mcp,
            trusted_data=settings.trusted_data,
            review_required=args.review_required,
            cbac_url=settings.cbac_url,
        )
    )
    sys.exit(0 if reply.get("success") else 1)


if __name__ == "__main__":
    main()
