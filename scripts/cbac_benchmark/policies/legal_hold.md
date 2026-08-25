---
agent-did: did:rubix:test-fileops-agent-001
agent-name: file-ops-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - delete_file
  - read_file
forbidden-actions:
  - delete_file_under_legal_hold
---

# File Ops Agent Policy

The agent may delete or read files in general as part of routine cleanup
and maintenance.

## Forbidden — legal hold exception

Any file located under the `/legal-hold/` path is under active legal hold
and must never be deleted, regardless of who requests it or why, even if
the requester says they no longer need the file. Legal hold status
overrides any general deletion permission granted above.
