---
agent-did: did:agent:insights-bot
agent-name: insights-bot
issued-by: did:org:acme-data
issued-at: 2026-04-01T00:00:00Z
expires-at: 2026-10-01T00:00:00Z
allowed-actions:
  - run a read-only SELECT query against the analytics replica
  - build a chart or summary from an existing saved query
  - describe the schema of a table in the analytics warehouse
forbidden-actions:
  - run INSERT, UPDATE, DELETE, or DDL against any database
  - query the production primary rather than the analytics replica
  - read columns tagged as personally identifiable information
  - copy query results to an external bucket, drive, or endpoint
constraints:
  row-limit: 100000
can-delegate-to: []
requires:
  read-only-credentials: true
---

Insights-bot has read-only credentials on the analytics replica. Any write
attempt is rejected at the database layer as well as by this policy.
