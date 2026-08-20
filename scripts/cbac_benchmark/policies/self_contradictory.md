---
agent-did: did:rubix:test-contradict-agent-001
agent-name: contradictory-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - archive_old_tickets
  - delete_ticket
forbidden-actions:
  - delete_ticket
---

# Contradictory Agent Policy

Note: `delete_ticket` appears in both `allowed-actions` and
`forbidden-actions` above — a policy authoring mistake that is easy to make
when a policy is edited by more than one person over time.

## Body text reinforces the contradiction

The agent may archive tickets older than 90 days, which includes deleting
the underlying ticket record to free up storage.

Separately: the agent must never delete a ticket record under any
circumstances — tickets are legally retained for audit purposes.
