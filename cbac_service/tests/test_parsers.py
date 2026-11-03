import asyncio
import json

import pytest

from cbac_service import parsers
from cbac_service.ibac import tool_action

TOOLS = [
    {
        "name": "send_invite",
        "description": "Send a calendar invite.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "format": "email"},
                "body": {"type": "string"},
            },
        },
    },
    {
        "name": "read_file",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {"name": "delete_mailbox", "inputSchema": {"type": "object", "properties": {}}},
]

POLICY = """\
The assistant manages the calendar.
Never invite anyone at competitor.com.
Never read anything under /etc.
Never delete the mailbox.
Payments are limited to 2000 USD."""

CATALOG_OUT = [
    {
        "tool": "send_invite",
        "resources": [{"arg": "to", "action": "calendar:invite", "kind": "email"}],
        "free_text": ["body"],
    },
    {
        "tool": "read_file",
        "resources": [{"arg": "path", "action": "file:read", "kind": "path"}],
        "free_text": [],
    },
    {"tool": "delete_mailbox", "resources": [], "free_text": []},
]


def fake_llm(monkeypatch, answer):
    calls = []

    async def chat_json(model, system, user, schema):
        calls.append({"model": model, "user": user, "schema": schema})
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(parsers, "chat_json", chat_json)
    return calls


def test_parse_policy_validates_every_rule(monkeypatch):
    fake_llm(
        monkeypatch,
        {
            "catalog": CATALOG_OUT,
            "deny_rules": [
                {
                    "action": "calendar:invite",
                    "match": "glob",
                    "resource": "*@Competitor.com",
                    "source": "Never invite anyone at competitor.com.",
                },
                {
                    "action": "file:read",
                    "match": "glob",
                    "resource": "/etc/*",
                    "source": "Never  read anything\nunder /etc.",  # whitespace differs
                },
                {
                    "action": tool_action("delete_mailbox"),
                    "match": "exact",
                    "resource": "*",
                    "source": "Never delete the mailbox.",
                },
                # Not a quote of the policy: an invented rule is rejected.
                {
                    "action": "file:read",
                    "match": "exact",
                    "resource": "/home/me/secrets",
                    "source": "Never read secrets.",
                },
                # An action the catalog does not have.
                {
                    "action": "email:send",
                    "match": "exact",
                    "resource": "x@y.io",
                    "source": "The assistant manages the calendar.",
                },
                # A glob without a wildcard.
                {
                    "action": "file:read",
                    "match": "glob",
                    "resource": "/etc",
                    "source": "Never read anything under /etc.",
                },
            ],
            "unenforceable": [
                {"source": "Payments are limited to 2000 USD.", "why": "numeric limit"}
            ],
        },
    )
    parsed = asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))

    assert set(parsed.catalog) == {"send_invite", "read_file", "delete_mailbox"}
    assert [(r["action"], r["match"], r["resource"]) for r in parsed.deny_rules] == [
        ("calendar:invite", "glob", "*@competitor.com"),
        ("file:read", "glob", "/etc/*"),
        ("tool:delete_mailbox", "exact", "*"),
    ]
    reasons = [r["reason"] for r in parsed.rejected]
    assert len(reasons) == 3
    assert any("verbatim" in r for r in reasons)
    assert any("not in the catalog" in r for r in reasons)
    assert any("glob" in r for r in reasons)
    assert parsed.unenforceable == [
        {"source": "Payments are limited to 2000 USD.", "why": "numeric limit"}
    ]


def test_parse_policy_schema_is_closed(monkeypatch):
    calls = fake_llm(
        monkeypatch, {"catalog": [], "deny_rules": [], "unenforceable": []}
    )
    asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))
    schema = calls[0]["schema"]

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from objects(value)

    for obj in objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    tool_enum = schema["properties"]["catalog"]["items"]["properties"]["tool"]["enum"]
    assert tool_enum == ["delete_mailbox", "read_file", "send_invite"]


def test_parse_policy_rejects_malformed_output(monkeypatch):
    fake_llm(monkeypatch, {"catalog": "nope"})
    with pytest.raises(TypeError):
        asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))


CATALOG = {
    "send_invite": {
        "resources": [{"arg": "to", "action": "calendar:invite", "kind": "email"}],
        "free_text": ["body"],
    },
    "read_calendar": {
        "resources": [{"arg": "day", "action": "calendar:read", "kind": "id"}],
        "free_text": [],
    },
    "list_events": {"resources": [], "free_text": []},
}
TRUSTED = {"contacts": {"Bob": "bob@company.com", "Eve": "eve@evil.com"}}
QUERY = "Check today's calendar and tell Bob the meeting is at 3pm"


def test_parse_query_keeps_only_grounded_grants(monkeypatch):
    calls = fake_llm(
        monkeypatch,
        {
            "grants": [
                {"action": "calendar:read", "resource": "today"},
                {"action": "calendar:invite", "resource": "Bob@Company.com"},
                {"action": "calendar:invite", "resource": "bob@company.com"},  # dup
                # Named nowhere: the model invented it.
                {"action": "calendar:invite", "resource": "attacker@evil.com"},
                {"action": "calendar:invite", "resource": "bob@company.com, x@y.io"},
                {"action": "email:send", "resource": "bob@company.com"},
                {"action": tool_action("list_events"), "resource": "anything"},
            ]
        },
    )
    grants, dropped = asyncio.run(
        parsers.parse_query("ollama:m", QUERY, TRUSTED, CATALOG)
    )
    assert grants == [
        {"action": "calendar:read", "resource": "today"},
        {"action": "calendar:invite", "resource": "bob@company.com"},
        {"action": "tool:list_events", "resource": "*"},
    ]
    assert len(dropped) == 3
    assert {d["reason"] for d in dropped} >= {
        "resource is not grounded",
        "action not in the catalog",
    }
    # The vocabulary and the trusted data reach the model; the enum pins actions.
    assert "calendar:invite: resource: an email address" in calls[0]["user"]
    assert json.dumps(TRUSTED, indent=1, sort_keys=True) in calls[0]["user"]
    action_enum = calls[0]["schema"]["properties"]["grants"]["items"]["properties"][
        "action"
    ]["enum"]
    assert action_enum == ["calendar:invite", "calendar:read", "tool:list_events"]


def test_parse_query_with_empty_catalog_skips_the_model(monkeypatch):
    calls = fake_llm(monkeypatch, {"grants": []})
    assert asyncio.run(parsers.parse_query("ollama:m", QUERY, {}, {})) == ([], [])
    assert calls == []


def test_chat_json_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown parser backend"):
        asyncio.run(parsers.chat_json("openai:gpt", "s", "u", {}))


# ── the policy document ──────────────────────────────────────────────────────

DOCUMENT_TOOLS = [{"name": "read_file", "inputSchema": {"properties": {"path": {}}}}]


def document(*sections):
    return "\n".join(["---", *(x for s in sections for x in (s, "---"))]) + "\n"


def test_split_policy_document():
    text = document(
        "Policy Instructions\nNever read /etc.\n\nNever delete mail.",
        "## Tool Schema\n```json\n" + json.dumps(DOCUMENT_TOOLS) + "\n```",
    )
    instructions, tools = parsers.split_policy_document(text)
    assert instructions == "Never read /etc.\n\nNever delete mail."
    assert tools == DOCUMENT_TOOLS


def test_split_policy_document_takes_a_tools_list_result():
    text = document(
        "Tool Schema\n" + json.dumps({"tools": DOCUMENT_TOOLS}),
        "Policy Instructions\nNever read /etc.",
    )
    assert parsers.split_policy_document(text) == ("Never read /etc.", DOCUMENT_TOOLS)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        (document("Policy Instructions\nNever."), "Tool Schema section is missing"),
        (document("Tool Schema\n" + json.dumps(DOCUMENT_TOOLS)), "Instructions"),
        (document("Policy Instructions\nx", "Tool Schema\nnot json"), "not JSON"),
        (document("Policy Instructions\nx", "Tool Schema\n[]"), "no tools"),
        (document("Policy Instructions\nx", "Tool Schema\n[{}]"), "no string name"),
        (document("Policy Instructions\nx", "Notes\ny"), "unknown section"),
        (
            document("Policy Instructions\nx", "Policy Instructions\ny"),
            "appears twice",
        ),
    ],
)
def test_split_policy_document_rejects(text, why):
    with pytest.raises(parsers.PolicyDocumentError, match=why):
        parsers.split_policy_document(text)
