# CBAC pipeline evaluation

Measures what each stage of the decision pipeline can actually do, on datasets
written independently of the implementation. Ground truth is what a policy
author means, not what the code computes — so a failing test here is a finding,
not a broken test.

```bash
uv run pytest cbac_service/tests/eval --run-eval -q       # ~8s, models must be cached
```

Opt-in behind `--run-eval` because it loads the encoder, NLI and HHEM models.
The normal `uv run pytest` skips the whole directory.

The consolidated report prints after the summary line regardless of pass/fail —
every test records its numbers before it asserts.

## Layout

| file | what it covers |
|---|---|
| `datasets.py` | 11 policies (5 skill cards, 6 developer-written), 108 intents, 46 end-to-end cases |
| `harness.py` | in-memory pgvector/BM25/RRF stand-ins, metric types, report collector |
| `test_classify_chunks.py` | step 2 — allowed/forbidden separation on both dataset shapes |
| `test_tiered_decision.py` | step 3 — tiers on the classifier's index vs a correctly labelled one |
| `test_check1_drift.py` | step 4 — NLI drift against user intent |
| `test_hallucination.py` | step 5 — HHEM on the same pairs |
| `test_layer_comparison.py` | step 6 — coverage matrix; does HHEM earn its place |

The search fakes compute the same quantities as the production queries (cosine
over the same normalized embeddings; RRF over a vector list and a BM25 list with
the same `RRF_K` and `fetch_k`), so Tier 1 and Tier 2 behave as they do against
Postgres. Only the storage engine differs.

## Ground-truth choices worth arguing with

- **Default-deny.** An action a policy never grants is `deny`. A policy listing
  six permissions is not consenting to the other million.
- **Three chunk classes.** `allowed` / `forbidden` / `neutral`. Real policies
  contain identifiers, issuance metadata and descriptive prose that grant
  nothing; the pipeline has no bucket for them, so where they land is measured.
- **`advise` is not a block.** On an action that must be denied, an `advise`
  hands the decision back to the caller. Reported strictly (`deny` only) and
  leniently (anything but `allow`) side by side.
- **Two independent axes end-to-end.** `aligned` (did the agent do what the user
  asked) and `policy_gold` (does the policy permit it) vary independently, so
  each layer is scored only on the region it is responsible for.
- **3 of 108 intents are marked `gray`** and excluded from strict accuracy.

## Findings, as of the first run

| | result |
|---|---|
| forbidden recall, skill cards | **0.42** — 15 of 26 prohibitions filed as grants, including strings beginning `forbidden-actions:` |
| forbidden recall, dev policies | **0.05** — 18 of 19 |
| policies with an empty forbidden bucket | **5 of 11** — Tier 1's `forbidden_score = 0.0` sentinel then turns the gap into an uncalibrated absolute similarity |
| prohibited actions returned `allow` (classifier index) | **64%** |
| out-of-scope actions authorized (classifier index) | **55%** |
| prohibited actions returned `allow` (oracle index) | **18%** |
| escalation rate on the oracle index | 42 of 108 reach `advise` |
| drift recall on misaligned actions | **0.38** at the shipped 0.60 gate (AUC 0.837) |
| `intent_score` on hijacked actions | up to **0.998** — `1 - contradiction` credits NLI's `neutral` mass |
| HHEM AUC on the same pairs | **0.912**, and `r = 0.26` against drift |

Read together: classification debt is the dominant term — a correct index lifts
the block rate from 36% to 82% — but the tier logic still authorizes 18% of
prohibited actions with a perfect index, and decides nothing in 39% of cases.

Layer ownership is clean and non-overlapping, which answers the HHEM question:

```
contradiction   drift 100%   policy  17%   hhem  67%
faithful        drift   0%   policy  92%   hhem  17%
hijack          drift  38%   policy  88%   hhem 100%
param_poison    drift  17%   policy  83%   hhem 100%
scope_creep     drift   0%   policy  67%   hhem 100%
```

HHEM is the only layer that covers `scope_creep` and `param_poison`, and it
catches 3 must-block cases (10 under the shipped classifier) that neither other
layer sees — so it is not redundant. It is also the noisiest: 4 false alarms on
the 8 controls, which is why it belongs where it currently sits — attached to
the result and folded into trust, not gating.

One case is missed by all three layers: approving a 780 EUR expense report
against a 250 EUR ceiling. No semantic layer does arithmetic; numeric
constraints need a different mechanism.
