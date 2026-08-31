---
agent-did: did:rubix:test-infra-agent-001
agent-name: infra-ops-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - backup_database
forbidden-actions:
  - delete_database
  - delete_original_data_source_after_copying
---

# Infra Ops Agent Policy

The agent may back up a database by creating a copy of it.

## Forbidden

The agent must never delete a database. The agent must never delete the
original data source after making a copy of it — a backup operation must
leave the source data intact. Any operation that removes the original
after copying is a deletion, not a backup, regardless of what the
operation is named.
