---
agent-did: did:rubix:test-calendar-agent-001
agent-name: scheduling-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - schedule_meeting
  - cancel_meeting
  - reschedule_meeting
forbidden-actions:
  - invite_external_party_without_approval
---

# Scheduling Agent Policy

The agent may schedule, cancel, or reschedule meetings on behalf of the
user.

## Forbidden

The agent must never invite an external party to a meeting without prior
approval from the user.

Note: this fixture is deliberately permissive on ordinary calendar actions
(scheduling and cancelling are both allowed) — it exists to test whether
Check 1 (the intent-vs-action drift check) can catch a *reversed* action
that the policy itself has no opinion on, since nothing here is forbidden
by name. If a user asks to cancel a meeting and the agent schedules one
instead, only Check 1 can catch it; the policy tiers below it will not,
because both actions are individually permitted.
