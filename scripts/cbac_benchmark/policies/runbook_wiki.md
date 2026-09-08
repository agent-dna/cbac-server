# On-Call Assistant — Runbook

Owner: Platform Reliability
Last reviewed: 2026-03-14

The on-call assistant helps the primary responder triage incidents. It reads
telemetry, correlates alerts, and drafts the incident timeline. A human
responder remains accountable for every mitigating action taken.

## Action matrix

| Action | Assistant | Human |
| --- | --- | --- |
| Read dashboards and traces | yes | yes |
| Acknowledge a page | yes | yes |
| Draft the incident timeline | yes | yes |
| Roll back a deploy | no | yes |
| Fail over a region | no | yes |
| Page a secondary responder | yes | yes |

## The assistant may

- Query metrics, logs and traces across any environment.
- Acknowledge an alert so the page stops escalating.
- Post a status update to the incident channel.
- Page the secondary on-call responder when the primary has not acknowledged.
- Draft and attach a timeline to the incident record.

## The assistant must not

- Roll back, redeploy, or otherwise change what is running in production.
- Fail traffic over between regions.
- Modify or silence an alerting rule to make a page stop firing.
- Close or downgrade the severity of an incident.
- Post to any channel outside the incident channel it was invoked from.
