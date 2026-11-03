"""The deterministic half of the v2 gate: catalog, normalization, tuple ids.

Pure functions, no I/O. Everything here runs identically when a tuple is
*written* (from a parser's output) and when a tool call is *checked*, so a
resource only ever matches itself after the same normalization.

A **catalog** says, per tool, which top-level arguments are *resources* —
identifiers the call acts on (a recipient, a path, a URL, an id) that OpenFGA
matches exactly — and which are *free text*, which nothing can match and which
get the semantic checks instead. The policy parser writes it from the tool
schemas; :func:`validate_catalog` is what keeps that LLM output honest.
"""

from __future__ import annotations

import hashlib
import posixpath
import unicodedata
from email.utils import getaddresses
from typing import Any
from urllib.parse import urlsplit, urlunsplit

KINDS = ("email", "path", "url", "id")

# JSON-schema string formats that identify an argument as a resource of one
# kind. A property carrying one of these cannot be classified as free text:
# free text is only scored, and an address does not contradict anything.
FORMAT_KINDS = {"email": "email", "idn-email": "email", "uri": "url", "iri": "url"}
# Identifier formats with no dedicated kind: still must be a resource.
RESOURCE_FORMATS = {*FORMAT_KINDS, "hostname", "idn-hostname", "uuid"}


class NormalizationError(ValueError):
    """A value that cannot be put in canonical form. Always a deny."""


def normalize(kind: str, value: Any) -> str:
    """Canonical form of one resource value, or :class:`NormalizationError`.

    Rejects rather than guesses: a value that could mean two things (a comma
    list of recipients, a look-alike character) fails, and a failure is a
    deny. A false deny is safe; a second spelling that slips past an exact
    match is not.
    """
    if not isinstance(value, str):
        raise NormalizationError(f"{kind} resource must be a string, got {value!r}")
    if kind == "email":
        return _email(value)
    if kind == "path":
        return _path(value)
    if kind == "url":
        return _url(value)
    if kind == "id":
        text = unicodedata.normalize("NFKC", value).strip()
        if not text or not text.isascii() or any(c.isspace() for c in text):
            raise NormalizationError(f"invalid id {value!r}")
        return text
    raise NormalizationError(f"unknown resource kind {kind!r}")


def _email(value: str) -> str:
    # getaddresses splits "Bob <bob@x>, eve@y" into two — exactly one is the
    # only acceptable answer. ASCII-only rejects homoglyph domains outright.
    addresses = [addr for _, addr in getaddresses([value]) if addr]
    if len(addresses) != 1:
        raise NormalizationError(f"expected one email address, got {value!r}")
    addr = addresses[0].strip().lower()
    local, at, domain = addr.rpartition("@")
    if not at or not local or "." not in domain or not addr.isascii():
        raise NormalizationError(f"invalid email address {value!r}")
    if any(c.isspace() for c in addr):
        raise NormalizationError(f"invalid email address {value!r}")
    return addr


def _path(value: str) -> str:
    text = value.strip()
    if not text.startswith("/") or "\x00" in text:
        raise NormalizationError(f"path must be absolute, got {value!r}")
    # normpath resolves ".." lexically, so docs/../etc/passwd checks as
    # /etc/passwd. It cannot see symlinks: the tool may still resolve one.
    return posixpath.normpath(text)


def _url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:
        raise NormalizationError(f"invalid url {value!r}: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not host:
        raise NormalizationError(f"url must be http(s) with a host, got {value!r}")
    if parts.username or parts.password:
        raise NormalizationError(f"url must not carry credentials, got {value!r}")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise NormalizationError(f"invalid url host {value!r}") from exc
    netloc = host if port is None else f"{host}:{port}"
    # The query stays: it is where data leaves through a URL (?d=SECRET), so a
    # grant for the page must not cover the page plus a payload.
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def object_id(agent_id: str, action: str, resource: str) -> str:
    """The OpenFGA object for ``action#resource``, scoped to one agent.

    Hashed rather than spelled out: raw emails, paths and URLs run into
    OpenFGA's id character and length limits, and an escaping bug there would
    be a bypass. Including ``agent_id`` keeps one agent's deny tuple from ever
    matching another agent's call. The readable form lives in Postgres.
    """
    digest = hashlib.sha256(f"{agent_id}\x00{action}#{resource}".encode()).hexdigest()
    return f"tool_invocation:{digest}"


def fga_user(intent_id: str) -> str:
    """The OpenFGA user an intent's grants are written to. Hashed for the
    same reason as :func:`object_id` — the intent id comes off the wire."""
    return f"user:{hashlib.sha256(intent_id.encode()).hexdigest()}"


def tool_action(tool: str) -> str:
    """The action of a tool with no resource argument: the tool itself."""
    return f"tool:{tool}"


def resource_values(arguments: dict[str, Any], arg: str) -> list[Any]:
    """The values of one resource argument; a list is checked item by item."""
    if arg not in arguments or arguments[arg] is None:
        return []
    value = arguments[arg]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def action_kinds(catalog: dict[str, dict]) -> dict[str, str]:
    """``action -> kind`` over a validated catalog; ``"*"`` for tool actions."""
    kinds: dict[str, str] = {}
    for tool, entry in catalog.items():
        if not entry["resources"]:
            kinds[tool_action(tool)] = "*"
        for res in entry["resources"]:
            kinds[res["action"]] = res["kind"]
    return kinds


def _summary(tool: dict) -> str:
    """The first line of a tool's description, bounded — what the query
    parser reads to match "tell Bob" to an invite tool."""
    description = tool.get("description")
    if not isinstance(description, str):
        return ""
    return description.strip().partition("\n")[0][:200]


def _top_level_props(tool: dict) -> dict[str, dict]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    return props if isinstance(props, dict) else {}


def _formats(prop: dict) -> set[str]:
    """String formats on a property or, for an array, on its items."""
    found = {prop.get("format")}
    items = prop.get("items")
    if isinstance(items, dict):
        found.add(items.get("format"))
    return {f for f in found if isinstance(f, str)}


def validate_catalog(
    entries: list[dict], tools: list[dict]
) -> tuple[dict[str, dict], list[dict]]:
    """Check a parser-written catalog against the tool schemas, in code.

    Returns ``(catalog, rejected)``: ``catalog`` maps each accepted tool name
    to ``{"resources": [{arg, action, kind}], "free_text": [arg],
    "description": str}``, and
    ``rejected`` lists ``{tool, reason}``. A rejected tool is simply absent
    from the catalog, so every call to it is denied — the safe direction for
    any mistake the LLM made.

    The rules that matter most:

    - **Coverage.** Every top-level property of the tool's schema is either a
      resource or free text. An argument the mapping forgot (``bcc``) would
      otherwise be an open channel.
    - **Formats.** A property typed ``email``/``uri``/… must be a resource of
      the matching kind; marking it free text would turn its exact check off.
    - **One kind per action**, so a grant for an action normalizes one way.
    """
    by_name: dict[str, dict] = {
        t["name"]: t for t in tools if isinstance(t.get("name"), str)
    }
    catalog: dict[str, dict] = {}
    rejected: list[dict] = []
    kinds: dict[str, str] = {}

    for entry in entries:
        name = str(entry.get("tool"))
        reason = _entry_problem(entry, by_name, catalog, kinds)
        if reason:
            rejected.append({"tool": name, "reason": reason})
            continue
        resources = [
            {"arg": r["arg"], "action": r["action"], "kind": r["kind"]}
            for r in entry["resources"]
        ]
        for r in resources:
            kinds[r["action"]] = r["kind"]
        catalog[name] = {
            "resources": resources,
            "free_text": list(entry["free_text"]),
            "description": _summary(by_name[name]),
        }

    for name in by_name:
        if name not in catalog and not any(r["tool"] == name for r in rejected):
            rejected.append({"tool": name, "reason": "not classified by the parser"})
    return catalog, rejected


def _entry_problem(
    entry: dict,
    by_name: dict[str, dict],
    catalog: dict[str, dict],
    kinds: dict[str, str],
) -> str | None:
    name = entry.get("tool")
    if name not in by_name:
        return "no such tool in the uploaded schemas"
    if name in catalog:
        return "classified twice"
    props = _top_level_props(by_name[name])
    resources = entry.get("resources") or []
    free_text = entry.get("free_text") or []

    resource_args = [r.get("arg") for r in resources]
    if len(set(resource_args)) != len(resource_args):
        return "an argument is listed as a resource twice"
    overlap = set(resource_args) & set(free_text)
    if overlap:
        return f"classified as both resource and free text: {sorted(overlap)}"
    unknown = (set(resource_args) | set(free_text)) - set(props)
    if unknown:
        return f"arguments not in the schema: {sorted(map(str, unknown))}"
    uncovered = set(props) - set(resource_args) - set(free_text)
    if uncovered:
        return f"arguments left unclassified: {sorted(uncovered)}"

    local: dict[str, str] = {}
    for r in resources:
        action, kind = r.get("action"), r.get("kind")
        if kind not in KINDS:
            return f"unknown kind {kind!r} for {r.get('arg')!r}"
        if not isinstance(action, str) or not action or "#" in action:
            return f"invalid action {action!r} for {r.get('arg')!r}"
        if any(c.isspace() for c in action) or action.startswith("tool:"):
            return f"invalid action {action!r} for {r.get('arg')!r}"
        if kinds.get(action, local.get(action, kind)) != kind:
            return f"action {action!r} used with two kinds"
        local[action] = kind
        want = {FORMAT_KINDS[f] for f in _formats(props[r["arg"]]) if f in FORMAT_KINDS}
        if want and kind not in want:
            return f"{r['arg']!r} has a {sorted(want)} format but kind {kind!r}"
    for arg in free_text:
        if _formats(props[arg]) & RESOURCE_FORMATS:
            return f"{arg!r} is typed as an identifier but marked free text"
    return None
