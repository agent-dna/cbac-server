"""The deterministic half of the v2 gate: catalog, canonical resources, tuple ids.

Pure functions, no I/O. Everything here runs identically when a tuple is
*written* (from a parser's output) and when a tool call is *checked*, so a
resource only ever matches itself in the same canonical form.

A capability is IBAC's *(tool, resource)*: one tool, and one value of one of
its resource arguments. A **catalog** says, per tool, which top-level
arguments are *resources* — the values the call acts on, checked exactly — and
which are *free text*, which nothing can match and which get the semantic
checks instead. The policy parser writes it from the tool schemas;
:func:`validate_catalog` checks its shape against those schemas.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

# The resource of a tool that has no resource argument: the tool as a whole.
WHOLE_TOOL = "*"


class NormalizationError(ValueError):
    """A value that has no canonical form as a resource. Always a deny."""


def canonical(value: Any) -> str:
    """The form one resource value is compared in.

    Domain-neutral by design: a string is Unicode-NFC-normalized and trimmed,
    a number or boolean is its JSON spelling (so ``3`` and ``"3"`` agree), and
    nothing else is interpreted. Two spellings of the same thing — letter
    case, a display name, ``..`` in a path — are two different resources.
    Under exact allow-listing that can only cost a false deny, never a false
    allow, because a value matches a grant only by being spelled the way the
    grant was. Objects, lists and ``null`` are not resources.
    """
    if isinstance(value, bool | int | float):
        return json.dumps(value)
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value).strip()
        if text:
            return text
    raise NormalizationError(f"not a resource value: {value!r}")


def object_id(agent_id: str, tool: str, resource: str) -> str:
    """The OpenFGA object for ``tool#resource``, scoped to one agent.

    Hashed rather than spelled out: raw values run into OpenFGA's id character
    and length limits, and an escaping bug there would be a bypass. Including
    ``agent_id`` keeps one agent's deny tuple from ever matching another
    agent's call. The readable form lives in Postgres.
    """
    digest = hashlib.sha256(f"{agent_id}\x00{tool}#{resource}".encode()).hexdigest()
    return f"tool_invocation:{digest}"


def fga_user(intent_id: str) -> str:
    """The OpenFGA user an intent's grants are written to. Hashed for the
    same reason as :func:`object_id` — the intent id comes off the wire."""
    return f"user:{hashlib.sha256(intent_id.encode()).hexdigest()}"


def resource_values(arguments: dict[str, Any], arg: str) -> list[Any]:
    """The values of one resource argument; a list is checked item by item."""
    if arg not in arguments or arguments[arg] is None:
        return []
    value = arguments[arg]
    return list(value) if isinstance(value, list | tuple) else [value]


def _summary(tool: dict) -> str:
    """The first line of a tool's description, bounded — what the query
    parser reads to pick a tool by what it does."""
    description = tool.get("description")
    if not isinstance(description, str):
        return ""
    return description.strip().partition("\n")[0][:200]


def _top_level_props(tool: dict) -> dict[str, dict]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    return props if isinstance(props, dict) else {}


def validate_catalog(
    entries: list[dict], tools: list[dict]
) -> tuple[dict[str, dict], list[dict]]:
    """Check a parser-written catalog against the tool schemas, in code.

    Returns ``(catalog, rejected)``: ``catalog`` maps each accepted tool name
    to ``{"resources": [arg], "free_text": [arg], "description": str}``, and
    ``rejected`` lists ``{tool, reason}``. A rejected tool is simply absent
    from the catalog, so every call to it is denied — the safe direction for
    any mistake the LLM made.

    The rule that matters most is **coverage**: every top-level property of the
    tool's schema is either a resource or free text, exactly once. An argument
    the mapping forgot would otherwise be an open channel. Which of the two an
    argument is stays the parser's judgment — nothing here second-guesses it.
    """
    by_name: dict[str, dict] = {
        t["name"]: t for t in tools if isinstance(t.get("name"), str)
    }
    catalog: dict[str, dict] = {}
    rejected: list[dict] = []

    for entry in entries:
        name = str(entry.get("tool"))
        reason = _entry_problem(entry, by_name, catalog)
        if reason:
            rejected.append({"tool": name, "reason": reason})
            continue
        catalog[name] = {
            "resources": list(entry["resources"]),
            "free_text": list(entry["free_text"]),
            "description": _summary(by_name[name]),
        }

    for name in by_name:
        if name not in catalog and not any(r["tool"] == name for r in rejected):
            rejected.append({"tool": name, "reason": "not classified by the parser"})
    return catalog, rejected


def _entry_problem(
    entry: dict, by_name: dict[str, dict], catalog: dict[str, dict]
) -> str | None:
    name = entry.get("tool")
    if name not in by_name:
        return "no such tool in the uploaded schemas"
    if name in catalog:
        return "classified twice"
    resources = entry.get("resources")
    free_text = entry.get("free_text")
    if not isinstance(resources, list) or not isinstance(free_text, list):
        return "resources and free_text must be lists of argument names"
    if not all(isinstance(a, str) for a in [*resources, *free_text]):
        return "resources and free_text must be lists of argument names"
    if len(set(resources)) != len(resources) or len(set(free_text)) != len(free_text):
        return "an argument is listed twice"

    props = _top_level_props(by_name[name])
    overlap = set(resources) & set(free_text)
    if overlap:
        return f"classified as both resource and free text: {sorted(overlap)}"
    unknown = (set(resources) | set(free_text)) - set(props)
    if unknown:
        return f"arguments not in the schema: {sorted(unknown)}"
    uncovered = set(props) - set(resources) - set(free_text)
    if uncovered:
        return f"arguments left unclassified: {sorted(uncovered)}"
    return None
