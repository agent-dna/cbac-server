"""Upload the CBAC v2 policies of the two agents that call the RSS MCP server.

The security and technology agents each sign their own tool calls, so the MCP
middleware sends each one's DID as the CBAC `agent_id`, and each needs its own
policy: `cbac_policies/rss_<agent>.md` as the Policy Instructions and the tools
`mcp_server.py` serves as the Tool Schema. The configured feeds go along as
trusted data. The orchestrator calls no tools and needs none.

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

POLICIES = Path(__file__).resolve().parent / "cbac_policies"
AGENTS = {
    "rss_security_agent": settings.security_agent_name,
    "rss_technology_agent": settings.technology_agent_name,
}


async def upload_all(review_required: bool) -> bool:
    ok = True
    for policy, agent_name in AGENTS.items():
        reply = await upload_agent_policy(
            agent_name,
            POLICIES / f"{policy}.md",
            mcp,
            trusted_data=settings.trusted_data,
            review_required=review_required,
            cbac_url=settings.cbac_url,
        )
        ok = ok and bool(reply.get("success"))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--review-required",
        action="store_true",
        help="hold each version pending until it is approved",
    )
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(upload_all(args.review_required)) else 1)


if __name__ == "__main__":
    main()
