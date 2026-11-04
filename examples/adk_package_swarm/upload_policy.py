"""Upload the CBAC v2 policies of the three swarm members that call tools.

Each specialist signs its own tool calls, so the MCP middleware sends that
member's DID as the CBAC `agent_id`, and each needs its own policy:
`cbac_policies/<member>.md` as the Policy Instructions and every tool
`mcp_server.py` serves as the Tool Schema — the policy itself says which of
them the member may use. The coordinator calls no tools and needs none.

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
MEMBERS = {
    "package_agent": settings.package_agent_name,
    "security_agent": settings.security_agent_name,
    "ecosystem_agent": settings.ecosystem_agent_name,
}


async def upload_all(review_required: bool) -> bool:
    ok = True
    for member, agent_name in MEMBERS.items():
        reply = await upload_agent_policy(
            agent_name,
            POLICIES / f"{member}.md",
            mcp,
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
