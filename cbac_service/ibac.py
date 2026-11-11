"""The deterministic half of the v2 gate: catalog, canonical resources, tuple ids.

Pure functions, no I/O. Everything here runs identically when a tuple is
*written* (from a parser's output) and when a tool call is *checked*, so a
resource only ever matches itself in the same canonical form.

A capability is *(tool, argument, resource)*: one tool, one of its resource
arguments, and one value for that argument. A **catalog** says, per tool,
which top-level arguments are *resources* — the values the call acts on,
checked exactly — and which are *free text*, which nothing can match and which
get the semantic checks instead. The policy parser writes it from the tool
schemas; :func:`validate_catalog` checks its shape against those schemas.

Allowing is strict and denying is loose, and one OpenFGA object carries both::

    allow  (fga_user(intent, resource), can_invoke, object_id(agent, tool, argument, resource))
    deny   (user:*,                     blocked,    object_id(agent, tool, argument, resource))

The object names a *class* of spellings (:func:`deny_key` of the resource);
the user names one exact spelling. An allow tuple therefore matches only the
value spelled as it was granted, in the argument it was granted for, while a
deny tuple blocks every spelling in the class. A deny rule names no argument,
so it is written once per resource argument of its tool. Both tuples sit on
the same object, which is what the model's ``can_invoke = grant but not
blocked`` needs to let a deny beat a grant.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import unicodedata
from typing import Any

# The resource of a tool that has no resource argument: the tool as a whole.
# Such a tool's one capability is (tool, "*", "*").
WHOLE_TOOL = "*"

# What :func:`object_id` and :func:`fga_user` hash. Bump it whenever either
# changes: tuples written under another number can never match again. A policy
# version records the number its deny tuples were written under, and the gate
# denies every call (4010) until the version is re-activated under this one.
TUPLE_SCHEME = 1


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
    grant was. Deny rules compare more loosely, see :func:`deny_key`. Objects,
    lists and ``null`` are not resources.
    """
    if isinstance(value, bool | int | float):
        return json.dumps(value)
    if isinstance(value, str):
        text = unicodedata.normalize("NFC", value).strip()
        if text:
            return text
    raise NormalizationError(f"not a resource value: {value!r}")


def deny_key(resource: str) -> str:
    """The form a canonical resource is compared in by deny rules: NFKC, then
    casefolded.

    Looser than :func:`canonical` on purpose, and used only to deny. Letter
    case and compatibility spellings (fullwidth letters, ligatures) of a
    forbidden value are forbidden too, so ``Production`` does not get past a
    rule on ``production``. Loose matching here can only add denies: allowing
    still needs the exact spelling that was granted.
    """
    folded = unicodedata.normalize("NFKC", resource).casefold()
    # Casefolding can leave a string that is no longer normalized.
    return unicodedata.normalize("NFKC", folded)


def glob_denies(pattern: str, resource: str) -> bool:
    """Whether a glob deny rule covers ``resource``, both in :func:`deny_key`
    form. ``*`` and ``?`` are the only wildcards: ``[`` is escaped to a
    literal, because NFKC can produce one from a fullwidth bracket and a
    character class would match less than the literal does."""
    return fnmatch.fnmatchcase(
        deny_key(resource), deny_key(pattern).replace("[", "[[]")
    )


def _digest(*parts: str) -> str:
    """sha256 over the parts as a JSON array, so no value can pose as a
    boundary between two of them."""
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def object_id(agent_id: str, tool: str, argument: str, resource: str) -> str:
    """The OpenFGA object for one resource argument of one tool holding
    ``resource`` in any spelling :func:`deny_key` treats as the same.

    ``argument`` is part of the object, so a value granted for one argument
    is not granted for another: a grant for ``from`` does not let the same
    value through as ``to``. Including ``agent_id`` keeps one agent's deny
    tuple from ever matching another agent's call.

    Hashed rather than spelled out: raw values run into OpenFGA's id character
    and length limits, and an escaping bug there would be a bypass. The
    readable form lives in Postgres.
    """
    return f"tool_invocation:{_digest(agent_id, tool, argument, deny_key(resource))}"


def fga_user(intent_id: str, resource: str) -> str:
    """The OpenFGA user one grant is written to: this intent, holding exactly
    this canonical ``resource``.

    The exact spelling lives here because :func:`object_id` holds only its
    class. A call spelled any other way asks as a different user and finds no
    allow tuple. Hashed for the same reasons as :func:`object_id` — the
    intent id comes off the wire.
    """
    return f"user:{_digest(intent_id, resource)}"


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
