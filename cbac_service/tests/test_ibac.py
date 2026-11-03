import pytest

from cbac_service.ibac import (
    NormalizationError,
    action_kinds,
    fga_user,
    normalize,
    object_id,
    resource_values,
    tool_action,
    validate_catalog,
)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("email", "Bob@Company.com", "bob@company.com"),
        ("email", "  bob@company.com ", "bob@company.com"),
        ("email", '"Bob" <bob@company.com>', "bob@company.com"),
        ("path", "/docs/../etc/passwd", "/etc/passwd"),
        ("path", "/docs//report.pdf", "/docs/report.pdf"),
        ("url", "HTTPS://Example.COM/page?d=1", "https://example.com/page?d=1"),
        ("url", "https://example.com", "https://example.com/"),
        ("url", "https://bücher.example/x", "https://xn--bcher-kva.example/x"),
        ("id", "  acme/api ", "acme/api"),
        ("id", "ｔｏｄａｙ", "today"),  # fullwidth folds under NFKC
    ],
)
def test_normalize(kind, value, expected):
    assert normalize(kind, value) == expected


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("email", "bob@company.com, eve@evil.com"),  # two recipients in one string
        ("email", "bob@cоmpany.com"),  # Cyrillic "о"
        ("email", "bob"),
        ("email", ""),
        ("path", "docs/report.pdf"),  # relative
        ("url", "ftp://example.com/x"),
        ("url", "https://user:pw@example.com/"),
        ("url", "not a url"),
        ("id", "two words"),
        ("id", "ünïcode"),
        ("id", ""),
        ("email", ["bob@company.com"]),  # not a string
        ("nope", "x"),
    ],
)
def test_normalize_rejects(kind, value):
    with pytest.raises(NormalizationError):
        normalize(kind, value)


def test_url_query_is_part_of_the_resource():
    # A grant for the page must not cover the page plus an exfiltration payload.
    page = normalize("url", "https://example.com/page")
    leak = normalize("url", "https://example.com/page?d=SECRET")
    assert page != leak


def test_object_id_is_agent_scoped_and_opaque():
    a = object_id("agent-a", "email:send", "bob@company.com")
    assert a == object_id("agent-a", "email:send", "bob@company.com")
    assert a != object_id("agent-b", "email:send", "bob@company.com")
    assert a != object_id("agent-a", "email:send", "eve@company.com")
    assert a.startswith("tool_invocation:")
    assert "bob" not in a
    assert fga_user("intent-1") != fga_user("intent-2")
    assert fga_user("intent-1").startswith("user:")


def test_resource_values_checks_lists_item_by_item():
    args = {"to": ["a@x.io", "b@x.io"], "cc": "c@x.io", "bcc": None}
    assert resource_values(args, "to") == ["a@x.io", "b@x.io"]
    assert resource_values(args, "cc") == ["c@x.io"]
    assert resource_values(args, "bcc") == []
    assert resource_values(args, "missing") == []


TOOLS = [
    {
        "name": "send_email",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string", "format": "email"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    },
    {
        "name": "read_file",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {"name": "list_events", "inputSchema": {"type": "object", "properties": {}}},
]


def entry(tool, resources=(), free_text=()):
    return {"tool": tool, "resources": list(resources), "free_text": list(free_text)}


def res(arg, action, kind):
    return {"arg": arg, "action": action, "kind": kind}


GOOD = [
    entry("send_email", [res("to", "email:send", "email")], ["subject", "body"]),
    entry("read_file", [res("path", "file:read", "path")]),
    entry("list_events"),
]


def test_validate_catalog_accepts_a_covering_catalog():
    catalog, rejected = validate_catalog(GOOD, TOOLS)
    assert rejected == []
    assert set(catalog) == {"send_email", "read_file", "list_events"}
    assert action_kinds(catalog) == {
        "email:send": "email",
        "file:read": "path",
        tool_action("list_events"): "*",
    }


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        # `body` left out: an unclassified argument would be an open channel.
        (
            entry("send_email", [res("to", "email:send", "email")], ["subject"]),
            "unclassified",
        ),
        # `to` is typed format=email, so marking it free text turns its check off.
        (entry("send_email", [], ["to", "subject", "body"]), "identifier"),
        (
            entry("send_email", [res("to", "email:send", "id")], ["subject", "body"]),
            "format",
        ),
        (
            entry(
                "send_email",
                [res("to", "email:send", "email")],
                ["to", "subject", "body"],
            ),
            "both",
        ),
        (
            entry(
                "send_email",
                [res("bcc", "email:send", "email")],
                ["to", "subject", "body"],
            ),
            "not in the schema",
        ),
        (entry("read_file", [res("path", "file read", "path")]), "invalid action"),
        (entry("read_file", [res("path", "file:read", "blob")]), "unknown kind"),
        (entry("delete_mailbox"), "no such tool"),
    ],
)
def test_validate_catalog_rejects(bad, why):
    catalog, rejected = validate_catalog([bad], TOOLS)
    assert bad["tool"] not in catalog
    assert any(why in r["reason"] for r in rejected if r["tool"] == bad["tool"])


def test_validate_catalog_rejects_one_action_with_two_kinds():
    entries = [
        entry("send_email", [res("to", "shared:act", "email")], ["subject", "body"]),
        entry("read_file", [res("path", "shared:act", "path")]),
    ]
    catalog, rejected = validate_catalog(entries, TOOLS)
    assert "send_email" in catalog
    assert "read_file" not in catalog
    assert any("two kinds" in r["reason"] for r in rejected)


def test_validate_catalog_reports_tools_the_parser_skipped():
    catalog, rejected = validate_catalog(GOOD[:1], TOOLS)
    assert set(catalog) == {"send_email"}
    skipped = {r["tool"] for r in rejected}
    assert skipped == {"read_file", "list_events"}
