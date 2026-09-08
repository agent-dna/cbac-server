---
agent-did: did:agent:deploy-bot
agent-name: deploy-bot
issued-by: did:org:acme-platform
issued-at: 2026-02-01T00:00:00Z
expires-at: 2026-08-01T00:00:00Z
allowed-actions:
  - read pod logs from the staging namespace
  - restart a deployment in the staging namespace
  - scale a staging deployment between 1 and 5 replicas
  - open a pull request against the infrastructure repository
forbidden-actions:
  - apply any change to the production namespace
  - delete a persistent volume claim
  - read or print the contents of a Kubernetes secret
  - disable, silence, or delete an alerting rule
  - grant cluster-admin or edit any RBAC binding
constraints:
  namespace-allowlist: staging
can-delegate-to:
  - did:agent:test-runner
requires:
  change-ticket: production changes require an approved change ticket
---

Deploy-bot reconciles the staging environment against the manifests in the
infrastructure repository.

Escalation to production is performed by a human release manager, never by the
agent.
