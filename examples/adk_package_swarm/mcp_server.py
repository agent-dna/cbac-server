"""Package-intelligence tools, as an AgentDNA-protected MCP server.

The tools themselves are in package_tools.py; every one uses public data that
needs no API key. AgentDNA's middleware requires every tool call to carry a
signed IntentWorkflow from a whitelisted agent, has the CBAC v2 gate authorize
the call against the user's request and the calling member's policy, and signs
each answer.

    python mcp_server.py
"""

from __future__ import annotations

from config import settings
from fastmcp import FastMCP
from package_tools import TOOLS

from agentdna import AgentDNA
from agentdna.mcp.server.fastmcp import AgentDNAMCPMiddleware
from agentdna_local import make_cbac_fn

mcp_server_dna = AgentDNA(
    name=settings.mcp_server_name,
    type="tool",
)

# Every tool call is authorized by the CBAC v2 gate (POST /cbac/v2/authorize)
# at CBAC_URL, against the policy uploaded for the calling member's DID.
cbac_fn = make_cbac_fn(settings.cbac_url)

mcp = FastMCP("package-intelligence")
mcp.add_middleware(AgentDNAMCPMiddleware(mcp_server_dna, cbac_fn=cbac_fn))

for tool in TOOLS:
    mcp.tool(tool)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port, path="/mcp"
    )
