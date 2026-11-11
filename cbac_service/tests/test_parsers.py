import asyncio
import io
import json

import pytest

from cbac_service import parsers

# Neutral tools: a resource argument that names a target, one that takes an
# integer, free text, and a tool with no arguments at all.
TOOLS = [
    {
        "name": "act_on",
        "description": "Act on a target.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "note": {"type": "string"},
            },
        },
    },
    {
        "name": "open_item",
        "inputSchema": {"type": "object", "properties": {"item": {"type": "integer"}}},
    },
    {"name": "reset", "inputSchema": {"type": "object", "properties": {}}},
]

POLICY = """\
The agent manages items.
Never act on anything in the restricted area.
Never open item 7.
Never reset.
Act on at most three targets a day."""

CATALOG_OUT = [
    {"tool": "act_on", "resources": ["target"], "free_text": ["note"]},
    {"tool": "open_item", "resources": ["item"], "free_text": []},
    {"tool": "reset", "resources": [], "free_text": []},
]


def fake_llm(monkeypatch, answer):
    calls = []

    async def chat_json(model, system, user, schema, **limits):
        calls.append({"model": model, "user": user, "schema": schema, "limits": limits})
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(parsers, "chat_json", chat_json)
    return calls


def rule(tool, match, resource, source):
    return {"tool": tool, "match": match, "resource": resource, "source": source}


def test_parse_policy_validates_every_rule(monkeypatch):
    fake_llm(
        monkeypatch,
        {
            "catalog": CATALOG_OUT,
            "deny_rules": [
                rule(
                    "act_on",
                    "glob",
                    "restricted/*",
                    "Never act on anything in\nthe restricted  area.",  # spacing differs
                ),
                rule("open_item", "exact", "7", "Never open item 7."),
                # "*" on a tool without resources is its only resource: exact.
                rule("reset", "glob", "*", "Never reset."),
                # "*" on a tool with resources is a pattern over every value.
                rule(
                    "act_on",
                    "exact",
                    "*",
                    "Never act on anything in the restricted area.",
                ),
                # Not a quote of the policy: an invented rule is rejected.
                rule("act_on", "exact", "x", "Never act on x."),
                # A tool the catalog does not have.
                rule("missing", "exact", "x", "The agent manages items."),
                # A glob without a wildcard.
                rule(
                    "act_on",
                    "glob",
                    "restricted",
                    "Never act on anything in the restricted area.",
                ),
                # A concrete value for a tool that has no resource arguments.
                rule("reset", "exact", "now", "Never reset."),
            ],
            "unenforceable": [
                {"source": "Act on at most three targets a day.", "why": "a count"}
            ],
        },
    )
    parsed = asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))

    assert set(parsed.catalog) == {"act_on", "open_item", "reset"}
    assert [(r["tool"], r["match"], r["resource"]) for r in parsed.deny_rules] == [
        ("act_on", "glob", "restricted/*"),
        ("open_item", "exact", "7"),
        ("reset", "exact", "*"),
        ("act_on", "glob", "*"),
    ]
    reasons = [r["reason"] for r in parsed.rejected]
    assert len(reasons) == 4
    assert any("verbatim" in r for r in reasons)
    assert any("not in the catalog" in r for r in reasons)
    assert any("glob" in r for r in reasons)
    assert any("no resource arguments" in r for r in reasons)
    assert parsed.unenforceable == [
        {"source": "Act on at most three targets a day.", "why": "a count"}
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
    props = schema["properties"]
    names = ["act_on", "open_item", "reset"]
    assert props["catalog"]["items"]["properties"]["tool"]["enum"] == names
    assert props["deny_rules"]["items"]["properties"]["tool"]["enum"] == names


def test_parse_policy_rejects_malformed_output(monkeypatch):
    fake_llm(monkeypatch, {"catalog": "nope"})
    with pytest.raises(TypeError):
        asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))


CATALOG = {
    "act_on": {"resources": ["target"], "free_text": ["note"], "description": "Act."},
    "open_item": {"resources": ["item"], "free_text": [], "description": ""},
    "reset": {"resources": [], "free_text": [], "description": "Start over."},
}
TRUSTED = {"aliases": {"the main one": "target-A"}}
QUERY = "Open item 42 and act on the main one"


def grant(tool, argument, resource):
    return {"tool": tool, "argument": argument, "resource": resource}


def test_parse_query_keeps_only_grounded_grants(monkeypatch):
    calls = fake_llm(
        monkeypatch,
        {
            "grants": [
                grant("open_item", "item", "42"),
                grant("act_on", "target", "target-A"),  # via the trusted data
                grant("act_on", "target", " target-A "),  # duplicate once canonical
                # Named nowhere: the model invented it.
                grant("act_on", "target", "target-Z"),
                grant("act_on", "target", "   "),
                # A value packed under an argument the tool does not have.
                grant("act_on", "target, note", "target-A"),
                grant("missing", "x", "42"),
                grant("reset", "anything", "anything"),
            ]
        },
    )
    grants, dropped = asyncio.run(
        parsers.parse_query("ollama:m", QUERY, TRUSTED, CATALOG)
    )
    assert grants == [
        grant("open_item", "item", "42"),
        grant("act_on", "target", "target-A"),
        grant("reset", "*", "*"),
    ]
    assert len(dropped) == 4
    assert {d["reason"] for d in dropped} >= {
        "resource is not grounded",
        "argument is not a resource of the tool",
        "tool not in the catalog",
    }
    # The tool list and the trusted data reach the model; the enum pins tools.
    user = calls[0]["user"]
    assert '- act_on (resource arguments: target) — "Act."' in user
    assert "- reset (resource arguments: none)" in user
    assert json.dumps(TRUSTED, indent=1, sort_keys=True) in user
    tool_enum = calls[0]["schema"]["properties"]["grants"]["items"]["properties"][
        "tool"
    ]["enum"]
    assert tool_enum == ["act_on", "open_item", "reset"]


def test_parse_query_with_empty_catalog_skips_the_model(monkeypatch):
    calls = fake_llm(monkeypatch, {"grants": []})
    assert asyncio.run(parsers.parse_query("ollama:m", QUERY, {}, {})) == ([], [])
    assert calls == []


@pytest.mark.parametrize(
    ("query", "trusted", "resource", "kept"),
    [
        # Inside the query, but not as a whole span.
        ("read /tmp/x", {}, "/", False),
        ("move $100 to savings", {}, "1", False),
        ("tell Bobby", {}, "bob", False),
        # A whole span: nothing but punctuation, space or the ends around it.
        ("email bob@company.com.", {}, "bob@company.com", True),
        ("raise it on acme/api", {}, "acme/api", True),
        ("Email BOB@company.com", {}, "bob@company.com", True),
        # Trusted data grounds a whole leaf, never a piece of one or a key.
        (
            "email him",
            {"contacts": {"Bob": "bob@company.com"}},
            "bob@company.com",
            True,
        ),
        ("email him", {"contacts": {"Bob": "bob@company.com"}}, "bob", False),
        ("email him", {"contacts": {"Bob": "bob@company.com"}}, "company.com", False),
        ("open it", {"items": [3]}, "3", True),
    ],
)
def test_grounding_needs_a_whole_span_or_a_whole_leaf(
    monkeypatch, query, trusted, resource, kept
):
    fake_llm(monkeypatch, {"grants": [grant("act_on", "target", resource)]})
    grants, dropped = asyncio.run(
        parsers.parse_query("ollama:m", query, trusted, CATALOG)
    )
    assert [g["resource"] for g in grants] == ([resource] if kept else [])
    assert [d["reason"] for d in dropped] == (
        [] if kept else ["resource is not grounded"]
    )


def test_each_parser_runs_under_its_own_limits(monkeypatch):
    # The query parser's come from config; the policy parser keeps long ones.
    monkeypatch.setattr(parsers, "QUERY_PARSER_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(parsers, "QUERY_PARSER_MAX_TOKENS", 300)
    calls = fake_llm(monkeypatch, {"grants": []})
    asyncio.run(parsers.parse_query("ollama:m", QUERY, TRUSTED, CATALOG))
    assert calls[0]["limits"] == {"timeout": 7.0, "max_tokens": 300}

    calls = fake_llm(
        monkeypatch, {"catalog": [], "deny_rules": [], "unenforceable": []}
    )
    asyncio.run(parsers.parse_policy("ollama:m", POLICY, TOOLS))
    assert calls[0]["limits"] == {"timeout": 600, "max_tokens": 16000}


def test_chat_json_gives_up_at_its_deadline(monkeypatch):
    async def never(name, system, user, schema, max_tokens):
        await asyncio.sleep(60)

    monkeypatch.setattr(parsers, "_anthropic", never)
    with pytest.raises(TimeoutError, match="within 0.01s"):
        asyncio.run(
            parsers.chat_json("anthropic:m", "s", "u", {}, timeout=0.01, max_tokens=8)
        )


def test_chat_json_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown parser backend"):
        asyncio.run(
            parsers.chat_json("openai:gpt", "s", "u", {}, timeout=1, max_tokens=8)
        )


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


def fake_urlopen(monkeypatch, reply):
    """Answer the HTTP backends with ``reply``; returns what they sent."""
    sent = {}

    def urlopen(request, timeout):
        sent.update(
            json.loads(request.data),
            auth=request.get_header("Authorization"),
            socket_timeout=timeout,
        )
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(parsers.urllib.request, "urlopen", urlopen)
    return sent


def test_nvidia_backend_strips_inline_reasoning(monkeypatch):
    content = '<think>\nhmm {"x": 0}\n</think>\n{"ok": "yes"}'
    sent = fake_urlopen(
        monkeypatch,
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
    )
    monkeypatch.setattr(parsers, "NVIDIA_API_KEY", "k")
    schema = {"type": "object"}
    out = asyncio.run(
        parsers.chat_json(
            "nvidia:nvidia/some-model", "s", "u", schema, timeout=9, max_tokens=70
        )
    )
    assert out == {"ok": "yes"}
    assert sent["model"] == "nvidia/some-model"
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert sent["auth"] == "Bearer k"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert (sent["max_tokens"], sent["socket_timeout"]) == (70, 9)


def test_ollama_backend_caps_the_answer_and_refuses_a_cut_one(monkeypatch):
    reply = {"done_reason": "stop", "message": {"content": '{"ok": "yes"}'}}
    sent = fake_urlopen(monkeypatch, reply)
    call = ("ollama:some-model", "s", "u", {"type": "object"})
    out = asyncio.run(parsers.chat_json(*call, timeout=9, max_tokens=70))
    assert out == {"ok": "yes"}
    assert (sent["options"]["num_predict"], sent["socket_timeout"]) == (70, 9)

    # Cut off at the cap: half an answer is an error, not a shorter answer.
    reply["done_reason"] = "length"
    with pytest.raises(RuntimeError, match="length"):
        asyncio.run(parsers.chat_json(*call, timeout=9, max_tokens=70))
