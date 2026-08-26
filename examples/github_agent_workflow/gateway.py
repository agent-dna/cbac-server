"""
The CBAC MCP gateway: a Policy Enforcement Point the agent developer does not
control.

It is an MCP server to the agent and an MCP client to the real MCP server, so
every ``tools/call`` passes through it. It reads the governance context off the
call, asks the CBAC decision service, and forwards or blocks. Nothing about
this depends on the agent, the framework, or the upstream server cooperating:

                                        ┌─> github-mcp ──> api.github.com
    agent ──MCP──> gateway (PEP) ──MCP──┤     (local stand-in)
                      │                 └─> mcp.deepwiki.com
                      │                       (real third party)
                      └── POST /authorize-cbac  (CBAC decision service)

Why here and not in the agent process: an interceptor inside the agent is
opt-in, and a security control the developer can delete is not a control. The
agent's *only* job is to label its calls (``cbac_propagate``), which carries no
decision. Skip the label and the calls still reach the gateway — contextless,
and denied.

The gateway is also where the upstream credentials live, and it is the only
service with a public address — every upstream sits on a private network
behind it. Between them those two facts are what make bypass pointless rather
than merely inconvenient: the agent cannot route to an upstream, and could not
authenticate to it if it could.

Run:

    python gateway.py

Env: MCP_UPSTREAM_URL, MCP_THIRDPARTY_URL, CBAC_GATEWAY_HOST/PORT, CBAC_URL,
CBAC_AGENT_ID, CBAC_REQUIRE_CONTEXT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cbac import configure, get_config
from cbac.guard import authorize
from cbac.mcp import context_from_headers, denial_body

load_dotenv()

UPSTREAM_URL = os.environ.get(
    "MCP_UPSTREAM_URL",
    f"http://{os.environ.get('MCP_SERVER_HOST', '127.0.0.1')}"
    f":{os.environ.get('MCP_SERVER_PORT', '8765')}/mcp/",
)
# A real third-party MCP server: public, no auth, and nobody in this repo owns
# it. The local stand-in above only *represents* an undecoratable server; this
# one actually is one. Same gateway, same middleware, same policy.
THIRDPARTY_URL = os.environ.get("MCP_THIRDPARTY_URL", "https://mcp.deepwiki.com/mcp")
GATEWAY_HOST = os.environ.get("CBAC_GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("CBAC_GATEWAY_PORT", "8766"))
CBAC_URL = os.environ.get("CBAC_URL", "http://127.0.0.1:8767")
CBAC_TIMEOUT = float(os.environ.get("CBAC_TIMEOUT", "600"))
# Whose policy applies. Pinned here, not taken from the caller: see the
# trust-boundary note in CBACMiddleware.
AGENT_ID = os.environ.get("CBAC_AGENT_ID", "github-worker")
REQUIRE_CONTEXT = os.environ.get("CBAC_REQUIRE_CONTEXT", "true").lower() == "true"

configure(cbac_url=CBAC_URL, cbac_timeout=CBAC_TIMEOUT)


class CBACMiddleware(Middleware):
    """Authorize every ``tools/call`` crossing this gateway. Fail-closed.

    ``agent_id`` is pinned by the gateway rather than read from the caller.
    The header is client-supplied, and *whose policy applies* is exactly the
    thing a compromised client would lie about — so it is a deployment fact
    here, not an input. ``user_intent`` is the opposite: only the client can
    know what the user asked, so it is unavoidably client-supplied and CBAC
    treats it as evidence (a drift signal), never as identity.

    ponytail: one pinned id for the whole gateway. A multi-tenant gateway maps
    the authenticated principal (OAuth subject, mTLS SAN, API key) to an agent
    id instead — same middleware, one lookup where ``self._agent_id`` is read.
    """

    def __init__(self, agent_id: str, require_context: bool = True) -> None:
        self._agent_id = agent_id
        self._require_context = require_context

    async def _decide(self, callee_name, args, description, error) -> None:
        """The gate itself. Raises ``error`` unless the policy allows the call.

        Every MCP primitive that acts on the world or returns the server's data
        comes through here — only the exception type differs, because that is
        all that differs. A gate wired to ``tools/call`` alone is not a gate:
        the same capability is usually reachable as a resource or a prompt, and
        the client picks which one to use.
        """
        ctx = context_from_headers(get_http_headers())
        user_intent = ctx.user_intent if ctx else ""

        # No label => the agent skipped cbac_propagate, or something is calling
        # the gateway directly. Deny rather than silently degrade to a
        # policy-only check: the drift check is half the decision.
        if self._require_context and not user_intent:
            raise error(
                denial_body(
                    "deny",
                    "CBAC: call arrived with no governance context "
                    "(X-CBAC-User-Intent missing) — the MCP client must "
                    "install cbac_propagate",
                )
            )

        decision, detail = await authorize(
            self._agent_id,
            callee_name,
            args,
            user_intent,
            description,
            callee_type="mcp",
            cfg=get_config(),
        )

        if decision != "allow":
            raise error(denial_body(decision, detail))

    async def on_call_tool(self, context, call_next):
        call = context.message
        _, description = await _describe(context, "get_tool", call.name)
        await self._decide(
            call.name, dict(call.arguments or {}), description, ToolError
        )
        return await call_next(context)

    async def on_read_resource(self, context, call_next):
        """``resources/read`` returns the server's data, so it is an action.

        The trust edge is keyed on the resource's registered name, not its URI.
        A URI carries its arguments inside it — ``github://acme/api/…`` — so
        keying on it would mint a new edge per repository and the trust history
        would never accumulate anywhere.
        """
        uri = str(context.message.uri)
        name, description = await _describe_resource(context, uri)
        await self._decide(
            name or uri,
            {"uri": uri},
            # Last resort, never the bare URI: with no description the URI
            # itself lands in the verb slot and the policy scores near-noise.
            description or "read the resource",
            ResourceError,
        )
        return await call_next(context)

    # ``prompts/get`` is deliberately not gated. Prompts are user-controlled in
    # MCP's model — the person picks one, the model does not reach for it — so
    # checking it for drift from what the user asked is circular. It also
    # returns text rather than acting, and the actions that follow are tool
    # calls, which are gated. Residual risk, if you want it later: a server
    # free to do real work while building a template does that work ungated.


async def _describe(context, getter: str, key: str) -> tuple[str | None, str | None]:
    """The upstream object's registered name and first line of description.

    ``getter`` is ``get_tool``, ``get_resource`` or ``get_prompt``. A
    gateway-side advantage: both are available locally here, while a
    client-side interceptor only has what was on the wire. NLI scores "close an
    existing GitHub issue" far better than a de-snaked "close issue".
    """
    try:
        obj = await getattr(context.fastmcp_context.fastmcp, getter)(key)
        return obj.name, (obj.description or "").splitlines()[0] or None
    except Exception:
        return None, None


async def _describe_resource(context, uri: str) -> tuple[str | None, str | None]:
    """Name and description for a *concrete* resource URI.

    ``get_resource`` resolves URIs that were registered literally. A templated
    resource — ``github://{owner}/{name}/collaborators`` — is registered as a
    pattern, so a concrete URI misses it and has to be matched against the
    templates instead.

    Worth the extra lookup: the description is what carries the verb. Without
    it the URI lands in the verb slot, and a forbidden read that should score a
    clear -0.113 cosine gap scores -0.054 and lands in the gray zone.
    """
    name, description = await _describe(context, "get_resource", uri)
    if description is not None:
        return name, description
    try:
        server = context.fastmcp_context.fastmcp
        templates = await server.list_resource_templates()
    except Exception:
        return name, None
    for template in templates:
        if template.matches(uri) is not None:
            first = (template.description or "").splitlines()[0]
            return template.name, first or None
    return name, None


# Two upstreams behind one gateway — the shape a real egress PEP has. FastMCP
# prefixes each server's tools with its name (``github_close_issue``,
# ``deepwiki_ask_question``), so the agent can tell them apart and the gateway
# knows which credential to attach.
#
# A config, not a Client: FastMCP then gives each proxied session its own
# client rather than sharing one across concurrent callers.
gateway = FastMCP.as_proxy(
    {
        "mcpServers": {
            # No credential here either — the local stand-in needs none. A
            # real upstream's token (GitHub OAuth, …) goes in this dict, which
            # is the point: it lives in the gateway and never reaches the agent.
            "github": {"url": UPSTREAM_URL},
            # No credential of ours to attach — DeepWiki is open. When the
            # third party does need one (GitHub, Notion, Slack), the token goes
            # here, in the gateway, and never reaches the agent process.
            "deepwiki": {"url": THIRDPARTY_URL},
        }
    },
    name="cbac-gateway",
)
gateway.add_middleware(CBACMiddleware(AGENT_ID, require_context=REQUIRE_CONTEXT))


if __name__ == "__main__":
    print(f"[gateway] {GATEWAY_HOST}:{GATEWAY_PORT} -> {UPSTREAM_URL} (github)")
    print(f"[gateway] {GATEWAY_HOST}:{GATEWAY_PORT} -> {THIRDPARTY_URL} (deepwiki)")
    print(
        f"[gateway] cbac={CBAC_URL} agent_id={AGENT_ID} "
        f"require_context={REQUIRE_CONTEXT}"
    )
    gateway.run(transport="http", host=GATEWAY_HOST, port=GATEWAY_PORT)
