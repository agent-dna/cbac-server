---
agent-did: did:rubix:test-exception-agent-001
agent-name: cleanup-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - delete_temp_file
forbidden-actions:
  - delete_production_file
  - delete_backup_file
---

# Cleanup Agent Policy

The agent may delete temporary files older than 24 hours, except files
located under `/prod` or `/backups`, unless an admin has explicitly issued
a time-limited override token for that specific deletion.

The agent must never delete a file under `/prod` or `/backups`, except
under the explicit admin override described above.

The agent must never delete any file matching `*.critical`, even under an
admin override.
