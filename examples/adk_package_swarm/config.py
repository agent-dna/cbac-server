from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

import agentdna_local
from dotenv import load_dotenv

# Puts GOOGLE_API_KEY in the environment, where Google ADK reads it.
load_dotenv()
# Every hosted AgentDNA call goes to the local store instead (AGENTDNA_LOCAL_DIR).
# Runs before any module creates an AgentDNA identity: they all import this one.
agentdna_local.install()


class _Drop(logging.Filter):
    """Drops log records whose message contains any of `fragments`."""

    def __init__(self, *fragments: str) -> None:
        super().__init__()
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(fragment in record.getMessage() for fragment in self.fragments)


# Keep the output readable. ADK announces every experimental feature it uses
# internally, suggests a context cache (not worth it for short runs), and probes
# for Google Cloud mTLS on each MCP connection, which a local server doesn't need.
# This runs before ADK is imported, because ADK warns about some features while
# importing. Only these messages are dropped; every other ADK warning still shows.
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\] feature", category=UserWarning)
logging.getLogger("google_adk.google.adk.runners").addFilter(_Drop("context_cache_config"))
logging.getLogger("google_adk.google.adk.tools.mcp_tool.mcp_session_manager").addFilter(
    _Drop("Failed to configure mTLS")
)


@dataclass(frozen=True)
class Settings:
    # Gemini, through Google ADK
    gemini_model: str = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"

    # The package-intelligence MCP server
    package_mcp_url: str = os.getenv("PACKAGE_MCP_URL") or "http://127.0.0.1:8680/mcp"
    mcp_host: str = os.getenv("MCP_HOST") or "127.0.0.1"
    mcp_port: int = int(os.getenv("MCP_PORT") or "8680")
    http_timeout_seconds: float = float(os.getenv("HTTP_TIMEOUT_SECONDS") or "30")

    # AgentDNA identities, kept in the local store. Each name is one Ed25519 key.
    user_name: str = os.getenv("USER_NAME") or "local-user@example.local"
    mcp_server_name: str = os.getenv("MCP_SERVER_NAME") or "package-intelligence-mcp"
    # The local CBAC decision service (examples/agentdna_local/service.py).
    cbac_url: str = os.getenv("CBAC_URL") or "http://127.0.0.1:8767"
    coordinator_agent_name: str = os.getenv("COORDINATOR_AGENT_NAME") or "package-coordinator"
    package_agent_name: str = os.getenv("PACKAGE_AGENT_NAME") or "package-agent"
    security_agent_name: str = os.getenv("SECURITY_AGENT_NAME") or "package-security-agent"
    ecosystem_agent_name: str = os.getenv("ECOSYSTEM_AGENT_NAME") or "package-ecosystem-agent"


settings = Settings()
