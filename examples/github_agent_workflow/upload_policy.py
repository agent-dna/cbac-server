"""
Upload the demo's v2 policy document: ``policies/{agent_id}.md`` as its
``Policy Instructions`` section, and every tool and resource template exactly
as the gateway lists them as its ``Tool Schema`` section.

The v2 gate matches a call to its catalog entry by name, and the gateway is
what names the calls (``github_close_issue``, server prefix included), so the
schemas come from the gateway rather than from each upstream — and are listed
fresh on every upload rather than kept in the policy file, because DeepWiki's
schemas are a third party's to change.

Run (with the gateway and the CBAC service already up):

    CBAC_POLICY_KEY=... python upload_policy.py

Env: CBAC_POLICY_KEY (required), CBAC_URL, CBAC_AGENT_ID, GATEWAY_URL.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import requests
from agent import CBAC_AGENT_ID, CBAC_URL, GATEWAY_URL
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

POLICY = Path(__file__).resolve().parent / "policies" / f"{CBAC_AGENT_ID}.md"


async def gateway_schemas() -> list[dict]:
    async with Client(StreamableHttpTransport(GATEWAY_URL)) as client:
        tools = await client.list_tools()
        templates = await client.list_resource_templates()
    schemas = [
        {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
        for t in tools
    ]
    # The gateway authorizes resources/read as a call to the template's name
    # with one argument, the URI, so each template is uploaded as a tool-like
    # schema with that one argument.
    schemas += [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": {
                "type": "object",
                "properties": {"uri": {"type": "string"}},
            },
        }
        for t in templates
    ]
    return schemas


def policy_document(instructions: str, tools: list[dict]) -> str:
    return (
        f"---\nPolicy Instructions\n{instructions.strip()}\n"
        f"---\nTool Schema\n{json.dumps(tools, indent=2)}\n---\n"
    )


def main() -> None:
    instructions = POLICY.read_text(encoding="utf-8")
    body = {
        "agent_id": CBAC_AGENT_ID,
        "policy": policy_document(instructions, asyncio.run(gateway_schemas())),
    }
    response = requests.post(
        f"{CBAC_URL.rstrip('/')}/cbac/v2/policies",
        json=body,
        headers={"X-CBAC-Key": os.environ["CBAC_POLICY_KEY"]},
        # The policy parser is an LLM call over every schema.
        timeout=900,
    )
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
