---
agent-did: did:rubix:test-docsharing-agent-001
agent-name: document-sharing-agent
issued-by: did:rubix:test-admin-001
issued-at: 2026-01-01T00:00:00Z
expires-at: 2027-01-01T00:00:00Z
allowed-actions:
  - share_document_with_named_colleague
forbidden-actions:
  - share_document_publicly
  - share_document_with_anyone_with_the_link
constraints:
  share_document_with_named_colleague:
    visibility: internal
---

# Document Sharing Agent Policy

The agent may share a document with a specific, named colleague inside the
company.

## Forbidden

The agent must never make a document public, and must never set a
document's sharing visibility to "anyone with the link." Any request to
share a document must name a specific internal recipient, not a broad or
public audience.
