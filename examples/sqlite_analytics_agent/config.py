from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import agentdna_local
from dotenv import load_dotenv

load_dotenv()
# Every hosted AgentDNA call goes to the local store instead (AGENTDNA_LOCAL_DIR).
# Runs before any module creates an AgentDNA identity: they all import this one.
agentdna_local.install()


def _json_env(name: str) -> dict:
    """Read a JSON object from the environment. Empty or unreadable means none."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Settings:
    agent_id: str = "sqlite-analytics-agent"
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama").lower()
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    database_path: Path = Path(os.getenv("SQLITE_DATABASE_PATH", "data/analytics.db"))

    # AgentDNA identities, kept in the local store. Each name is one Ed25519 key.
    user_name: str = os.getenv("USER_NAME") or "local-user@example.local"
    sqlite_agent_name: str = os.getenv("SQLITE_AGENT_NAME") or "sqlite-analytics-agent"
    sqlite_mcp_url: str = os.getenv("SQLITE_MCP_URL", "http://127.0.0.1:8544/mcp")
    mcp_timeout_seconds: int = 3000
    mcp_server_name: str = os.getenv("MCP_SERVER_NAME", "sqlite-analytics-mcp")
    # The local CBAC decision service (examples/agentdna_local/service.py).
    cbac_url: str = os.getenv("CBAC_URL") or "http://127.0.0.1:8767"

    # Headers the agent sends to the MCP server. Nothing here needs them to
    # reach SQLite - they exist so a run can carry a credential, which is what
    # AgentDNA's authentication evidence observes.
    #
    #   SQLITE_MCP_HEADERS='{"Authorization": "Bearer <token>"}'
    sqlite_mcp_headers: dict = field(default_factory=lambda: _json_env("SQLITE_MCP_HEADERS"))

    mcp_host: str = os.getenv("MCP_HOST", "127.0.0.1")
    mcp_port: int = int(os.getenv("MCP_PORT", "8544"))

settings = Settings()