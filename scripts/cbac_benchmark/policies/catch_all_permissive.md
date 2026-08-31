---
agent-did: did:rubix:test-catchall-agent-001
agent-name: catchall-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - assist_user
forbidden-actions:
  - delete_production_database
  - transfer_funds_externally
  - modify_user_permissions
---

# Catch-All Agent Policy

The agent may take any reasonable action necessary to assist the user and
complete the task at hand efficiently.

The agent must never delete the production database, transfer funds
externally, or modify another user's permissions.

Note: the single broad "allowed" sentence above is deliberately vague and
generic — this fixture tests whether a single catch-all permissive chunk
sits close enough, in embedding space, to a wide range of unrelated
intents to push Tier 1's cosine gap toward "allow" for actions the policy
author never actually considered.
