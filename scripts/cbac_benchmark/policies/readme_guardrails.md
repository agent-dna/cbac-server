# support-triage-agent

Triages inbound support tickets and routes them to the right queue.

## Capabilities

The agent reads the ticket body and metadata, assigns a priority label, moves
the ticket to one of the configured queues, and posts an internal triage note
summarising its reasoning.

It can also look up the reporter's previous tickets to spot duplicates.

## Guardrails

The agent does not reply to the customer directly — every outbound message is
drafted for a human agent to send.

It will not close, merge, or delete a ticket, and it will not change a ticket's
reporter or organisation.

Escalating a ticket to the on-call pager is out of scope and must be done by a
human.

## Deployment

Runs as a Kubernetes CronJob every two minutes.
