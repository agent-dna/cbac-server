---
agent-did: did:rubix:test-onlyforbidden-agent-001
agent-name: lockdown-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions: []
forbidden-actions:
  - delete_file
  - send_email
  - execute_code
---

# Lockdown Agent Policy

This agent has no affirmatively granted capabilities — it exists to test
the fail-closed path when a policy carries forbidden content but nothing
classifies as allowed.

The agent must never delete a file, send an email, or execute code. No
action is affirmatively permitted by this policy.
