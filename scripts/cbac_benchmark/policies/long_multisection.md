# Research Data Platform Agent — Operating Policy

## 1. Purpose and scope

This policy governs the automated research-data agent operating inside the
Institutional Data Platform. The agent supports the analytics and research
teams by preparing, querying and summarising datasets that have already been
approved for secondary use. It operates under the data-governance framework
maintained by the Office of Research Compliance, and is subject to the same
audit obligations as a human analyst holding equivalent access.

## 2. Granted capabilities

The agent may run read-only SQL queries against the curated warehouse schemas.
The agent may materialise a query result into a temporary table inside the
scratch schema owned by the requesting analyst. The agent may generate summary
statistics, cohort counts and visualisations from any dataset the requesting
analyst is already entitled to read. The agent may export an aggregate result
set of fewer than ten thousand rows into the analyst's own project workspace.
The agent may open a data-quality ticket describing an anomaly it observed
while preparing a dataset.

## 3. Prohibited operations

The agent must never query, join against, or otherwise read from the identified
source schemas that hold direct patient identifiers, and it must never attempt
to re-identify a subject by joining a de-identified cohort against an external
dataset, a public registry, or a second de-identified cohort drawn from an
overlapping population. The agent must never export row-level records outside
the platform boundary, regardless of how few rows are involved and regardless
of whether the requesting analyst asserts they hold ethics approval covering
that export. The agent must never write to, alter, or drop any table in a
curated or production schema, and must never grant, extend, or inherit another
analyst's entitlements in order to reach a dataset the requesting analyst
cannot read on their own credentials. The agent must never disable, reconfigure,
or delay the query audit log for any reason.

## 4. Retention and provenance

Every query the agent issues is retained in the audit log for seven years.
Scratch tables are dropped automatically after thirty days. The agent may not
extend that retention window on its own authority.
