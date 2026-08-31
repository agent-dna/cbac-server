---
agent-did: did:rubix:test-malformed-agent-001
agent-name: malformed-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - read_file
forbidden-actions:
  - delete_file
  - delete_production_database

NOTE: this fixture deliberately never closes the frontmatter block with a
second "---" delimiter. parse_skill_md's text.split("---", 2) then returns
only 2 parts, triggers "skill.md must have a closing --- line after
frontmatter", and flatten_policy_chunks's except clause silently swallows
that and re-chunks this entire file as plain prose instead — the structured
allowed-actions/forbidden-actions lists above are never read as structured
data at all, only as leftover text mixed into the plain-text chunker.

The agent may read any file in the working directory.

The agent must never delete a file. The agent must never delete or drop the
production database under any circumstances.
