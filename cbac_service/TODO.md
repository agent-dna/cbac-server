# TODO — cbac_service

Open work for the CBAC decision service. Grouped by priority; the LHI items
carry the reasoning from the design discussion so it isn't re-litigated.

## Correctness — LHI

- [x] **Renormalize over observed scores instead of skipping the record**
      (done 2026-08-14). A record used to be written only when *all* of
      intent/policy/hallucination were present (the guard's `_report_lhi` skip),
      which was complete-case analysis over data that is **not** missing at
      random: `policy_score` is `None` exactly on Tier-3 gray-zone decisions,
      and intent/hallucination are `None` whenever no `user_intent` was
      supplied — so a deployment without `user_intent` accumulated **no records
      at all**, silently.
      `compute_lhi` now takes `float | None` components and computes
      `s = Σ_{observed} wᵢxᵢ / Σ_{observed} wᵢ`, storing an unmeasured component
      as NULL (never a substituted value). Guard rail: nothing observed → no
      record, `compute_lhi` returns `None`. Every component is semantic now that
      `output_score` is gone, so "require ≥1 semantic score" reduces to
      "require ≥1". Migration `9c4e1f80a72b` makes the three columns nullable.

- [x] **Remove `output_score`; compute LHI at decision time** (done
      2026-08-14). It was the only component that required waiting for
      execution, and the weakest one (binary 1.0/0.0 from "did the call raise").
      Dropping it collapses the guard's two HTTP calls into one: `verify_cbac`
      folds trust in itself via `_fold_trust`, `/compute-lhi` is deleted, and no
      component score round-trips through the client — which also closes the
      score-transport hole below. `LHI_WEIGHTS` needed no retuning: the
      renormalizing denominator makes it scale-invariant.

- [x] **Render actions as natural language before scoring them.** Measured on
      14 realistic agent cases (2026-08-03): both Check-1 NLI and HHEM degrade
      badly on flattened `tool k=v` intent text. NLI missed direct opposites
      hidden in snake_case tool names (`close_pull_request` vs "do not close
      anything" → contradiction 0.01, caught at 0.95 as prose), and HHEM
      scores every tool-syntax string as ungrounded.
      **Implemented (2026-08-03):** `render_intent` renders "The agent wants to
      <verb phrase>, with k = v, …" — verb phrase = the callee's description
      (docstring or schema first line), else the de-snaked callee name. All
      three scorers receive this one full-prose string.
      **Decision:** HHEM scores the params too (they are the LLM-generated
      content). Accepted tradeoff, measured: with params in view, legitimate
      params_added calls (HHEM 0.09) are inseparable from hijacks (0.05), so
      hallucination thresholds must stay advisory. The verb-only view
      (faithful 0.53 / params_added 0.62 vs hijack 0.04) remains a one-line
      switch in `verify_cbac` — `intent_text.split(", with ", 1)[0]` — via
      the `_action_summary` helper alongside `render_intent` in `skills.py`,
      if separation matters later.

- [x] **Record violation / denial evidence** (done 2026-08-14, as a side
      effect of folding LHI into `verify_cbac`). A denied call used to write
      nothing, so an agent probing forbidden actions 50 times kept pristine
      trust. Now that trust is folded at *decision* time rather than after
      execution, every decision records — and a deny carries the low
      intent/policy scores that produced it.
      Still open, if this proves too blunt: a denial and a low-scoring allow are
      currently the same kind of evidence. A separate per-edge negative stream
      (counter, or beta-style `α/β`) would weight them differently.

- [ ] **Time decay + evidence volume in the update rule.** The EMA is
      per-interaction, so an edge idle for months keeps frozen trust — decay
      toward a neutral prior on wall-clock time. And expose the interaction
      count (or beta `α+β`) alongside `T`, refusing to let trust arbitrate the
      gray zone below a minimum-evidence threshold (cold start: one good call
      ≠ trusted edge).

## Deferred LHI candidates

Evaluated and parked — each adds a noisy estimator needing its own tuning, and
dilutes the weights of the four components already justified. Revisit only
after the core is validated against real workflows.

- [ ] Score variance / dispersion penalty — an erratic agent is riskier than a
      steadily mediocre one.
- [ ] Operational reliability detail (latency, timeouts, retries).
- [ ] Least-privilege adherence; cross-agent reputation rollup (EigenTrust-style,
      the per-edge store is already shaped for it); human feedback labels;
      behavioral anomaly scoring.

## Engineering

- [x] **Replace the trust JSON file with a light DB** (done 2026-08-07). LHI
      now lives in the `lhi_records` Postgres table — append-only, one row per
      interaction, so the full trust history per edge is kept (no more
      last-value-wins) and current trust = the edge's latest row. Migration
      `7b21c9e4d3a8`. Remaining narrower race: two concurrent updates for the
      *same edge* can read the same prev trust before both insert — fix with
      `SELECT ... FOR UPDATE` on the latest row or a serializable transaction
      if concurrent per-edge traffic becomes real.
- [ ] **Fix return types in the `CBAC` class** (`cbac.py:42`).
- [ ] **Remove `authorise_agent_app_interaction`** (`cbac.py:592`) — superseded
      by the guard + `/authorize-cbac`; it also makes the engine an HTTP
      *client*, which the rest of the class isn't.
- [x] Score-transport hardening (moot since 2026-08-14). `/compute-lhi` used to
      trust caller-supplied intent/policy/hallucination scores, so a malicious
      guard could inflate its own reputation. The endpoint is gone and the
      scores never leave the service, so there is nothing left to forge.
- [ ] No CI for `cbac_service/tests/` — the root workflow only runs the
      library's `tests/`.
- [ ] Add Readme