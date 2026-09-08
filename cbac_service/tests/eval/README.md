# CBAC pipeline evaluation

Measures what each stage of the decision pipeline can actually do, on datasets
written independently of the implementation. Ground truth is what a policy
author means, not what the code computes — so a failing test here is a finding,
not a broken test.

```bash
uv run pytest cbac_service/tests/eval --run-eval -q       # ~30s, models must be cached
```

Opt-in behind `--run-eval` because it loads the encoder, NLI and HHEM models.
`test_datasets_sanity.py` is the exception: it needs no models and runs in the
ordinary `uv run pytest`, because it guards the gold labels against policy files
that live outside this package.

The consolidated report prints after the summary line regardless of pass/fail —
every test records its numbers before it asserts.

## Consuming the results

The report opens with a `HEADLINE` block: the dozen numbers worth reading first,
so nothing has to be scraped out of 700 lines of tables.

```
HEADLINE
--------
  forbidden recall, structured          0.55
  forbidden recall, unstructured        0.09
    ...on `forbidden-actions:` lines    0.75
    ...on prose prohibitions            0.08
  policies with no forbidden bucket        8
  block rate, classifier index          0.57
  block rate, oracle index              0.92
  ...
```

Two flags write it out instead:

```bash
# machine-readable: metrics, per-test outcomes, and every section
uv run pytest cbac_service/tests/eval --run-eval -q --eval-json eval.json

# the same consolidated text you see on stdout
uv run pytest cbac_service/tests/eval --run-eval -q --eval-report eval.txt
```

`eval.json` has four keys:

| key | what it holds |
|---|---|
| `corpus` | how many policies, intents and cases the run covered |
| `metrics` | ~60 flat `dotted.key -> number` entries, e.g. `tier.oracle.block_rate` |
| `tests` | `test name -> passed \| failed`, so a consumer can see *which* requirement moved |
| `sections` | every report section, title plus lines |

Comparing two runs needs no extra tooling:

```bash
python3 -c '
import json, sys
a, b = (json.load(open(f))["metrics"] for f in sys.argv[1:3])
for k in sorted(a.keys() | b.keys()):
    x, y = a.get(k), b.get(k)
    if x != y:
        print(f"{k:<48} {x} -> {y}")
' before.json after.json
```

A metric moving is the signal; a `tests` entry flipping is the requirement it
crossed. Note that failures are expected here — see Findings below — so the
suite's exit code is not a useful gate on its own.

### Per-case view

The JSON also carries `cases` (all 378, with each arm's verdict and the rule
that fired) and `policies` (each chunk's gold label beside the bucket
`_classify_chunks` put it in). That is what answers *which* case regressed,
which the aggregates cannot.

```bash
uv run pytest cbac_service/tests/eval --run-eval -q --eval-html eval.html
```

One self-contained HTML file, no dependencies and no network: every case is one
cell, grouped by policy, and hovering a cell shows the action, the user intent,
the rule that decided it, and the policy index the verdict was made against —
prohibitions that landed in the allowed bucket are marked there, which is
usually the whole explanation for a wrong `allow`.

Correct cases are deliberately grey rather than green. `good` and `critical` in
the status palette sit at CVD ΔE 4.1 under deuteranopia, so a red/green grid is
unreadable for a red-green colourblind reader; only failures carry colour, and
each also carries a glyph (▲ wrongly allowed, ▼ wrongly denied) so colour is
never the only channel.

Adding cases needs no change here: a new `IntentCase`, `E2ECase` or policy shows
up in the page on the next run, in its policy's row and its category's bar. Only
a new *dataset* — a fourth table beside intents/adversarial/e2e — needs a button
adding in `dashboard.py`.

To rebuild the page from a run you already have, without paying for the models
again:

```bash
uv run python scripts/eval_dashboard.py eval.json -o eval.html
```

## Layout

| file | what it covers |
|---|---|
| `datasets.py` | the 31-policy corpus with per-span gold labels; loads text from `scripts/cbac_benchmark/policies/` |
| `intents.py` | 285 intents, 25 adversarial cases, 68 end-to-end cases |
| `harness.py` | in-memory pgvector/BM25/RRF stand-ins, metric types, report collector |
| `test_datasets_sanity.py` | is the ground truth well-formed? (model-free) |
| `test_classify_chunks.py` | step 2 — allowed/forbidden separation, by document shape |
| `test_tiered_decision.py` | step 3 — tiers on the classifier's index vs a correctly labelled one, plus the adversarial corpus |
| `test_check1_drift.py` | step 4 — NLI drift against user intent |
| `test_hallucination.py` | step 5 — HHEM on the same pairs |
| `test_layer_comparison.py` | step 6 — coverage matrix; does HHEM earn its place |

The search fakes compute the same quantities as the production queries (cosine
over the same normalized embeddings; RRF over a vector list and a BM25 list with
the same `RRF_K` and `fetch_k`), so Tier 1 and Tier 2 behave as they do against
Postgres. Only the storage engine differs.

## The corpus

Policy documents live in `scripts/cbac_benchmark/policies/` and are shared with
the DB-backed runner in `scripts/cbac_benchmark/`. One corpus, two harnesses.
This directory holds only the ground truth.

Three groups, reported separately because averaging them answers no question:

| group | n | what it is |
|---|---|---|
| structured | 14 | skill cards — `allowed-actions:` / `forbidden-actions:` frontmatter plus a prose body |
| unstructured | 11 | the shapes developers actually write: markdown headings, a system prompt, IAM JSON, a YAML config, a README, numbered rules, a long multi-section policy, a runbook with a permission table, OPA-style rule objects, legal prose, cross-referencing sections |
| stress | 6 | degenerate by construction: unclosed frontmatter, the same action in both lists, no grants at all, a single catch-all grant, inert narrative, a French policy |

Documents range from 65 to 327 words. Two exceed `CHUNK_MAX_WORDS = 120` and
exercise `split_by_word_budget`: `long-multisection`, which splits on sentence
boundaries as intended, and `opa-rules`, which has no sentence boundaries at all
and so takes the hard-cut fallback.

## Ground-truth choices worth arguing with

- **Default-deny.** An action a policy never grants is `deny`. A policy listing
  six permissions is not consenting to the other million.
- **Three chunk classes.** `allowed` / `forbidden` / `neutral`. Real policies
  contain identifiers, issuance metadata and descriptive prose that grant
  nothing; the pipeline has no bucket for them, so where they land is measured.
- **Fixture `Note:` paragraphs count as `neutral`, not stripped.** Text that
  grants nothing but reads like policy is exactly what the neutral class exists
  to measure.
- **Two independent axes end-to-end.** `aligned` (did the agent do what the user
  asked) and `policy_gold` (does the policy permit it) vary independently, so
  each layer is scored only on the region it is responsible for.
- **A blocking category is all-`deny`, a granting category all-`allow`.** Both
  rates count every case in their categories, so one misfiled case caps a rate
  regardless of behaviour. Genuinely undecidable cases carry `gray` and the
  `contradictory` category, landing in neither rate.
- **The adversarial corpus is scored apart.** Averaging attacks into the block
  rate makes that number a statement about the mixing ratio.

## Findings

### Classification is the dominant term, and it is reading the label, not the language

| | result |
|---|---|
| forbidden recall, structured cards | **0.55** |
| forbidden recall, unstructured policies | **0.09** |
| forbidden recall, degenerate policies | 0.58 |
| **recall on chunks carrying the literal string `forbidden-actions:`** | **0.75** (39/52) |
| **recall on prohibitions stated in prose** | **0.08** (4/53) |
| precision, every group | **1.00** — no grant is ever filed as a prohibition |

That split is the whole story. `_classify_chunks` reliably picks up a chunk that
carries the frontmatter key and almost nothing else: `The agent must never delete
an invoice record` is filed as a **grant**, and so is at least one equivalent
sentence in 25 of the 31 documents. A prohibition in the allowed bucket does not merely
lose signal — it becomes the chunk that maximises Tier 1's `max_allowed` for the
very action it bans.

Consequences that follow directly:

| | result |
|---|---|
| policies left with an empty forbidden bucket | **8 of 31** — Tier 1's `forbidden_score = 0.0` sentinel then turns the gap into an uncalibrated absolute similarity |
| neutral spans filed `allowed` | **143 of 143** — none ever reach the forbidden bucket |
| chunks holding both polarities at once | 4 — `iam-json`, `opa-rules` (the word-budget hard cut severs a rule mid-object), `runbook-wiki` (a markdown permission table has no bullets and no blank lines, so it collapses into one chunk), `malformed-frontmatter` |

### The tiers, on the classifier's index vs a correct one

| | classifier | oracle |
|---|---|---|
| prohibited actions blocked | 0.57 | **0.92** |
| explicitly granted actions allowed | 0.61 | **0.47** |
| out-of-scope actions authorized | 0.18 | 0.02 |

A correct index nearly doubles the block rate — and costs a quarter of the
grants. On the oracle index **49% of all decisions come from
`TIER3_NO_BACKEND_DENY`**, the Tier 1/2 gray zone with no LLM configured. That
is not the policy being read; it is the fallback firing. The 0.92 block rate and
the 0.47 allow rate are two views of the same behaviour, and a pipeline that
denies 53% of the work its policy explicitly grants is not usable as shipped
without Tier 3.

### The adversarial corpus

24 of 25 attacks are denied under either index. Homoglyphs, zero-width
characters, French-against-French, negation stacking, bundling and hedged
framing all fail to get through — but read that against the paragraph above:
default-deny blocks an attack for the same reason it blocks legitimate work, so
a high block rate here is weak evidence on its own.

The one leak is the informative row: **`approve_expense_under_500` with
`amount = 500`** comes back `allow` against a policy granting approvals *under*
500. So does `780 EUR` against a 250 EUR ceiling — the single end-to-end case
missed by all three layers. No semantic layer does arithmetic; numeric
constraints need a different mechanism, and `constraints:` is parsed into
`SkillsCard` but never read by `cbac.py`.

### The user-alignment layers

| | result |
|---|---|
| drift recall on misaligned actions | **0.42** at the shipped 0.60 gate (AUC 0.831, precision 1.00) |
| drift recall on `scope_creep` | **0.00** — doing what was asked *and more* contradicts nothing |
| `intent_score` on hijacked actions | up to **0.996** — `1 - contradiction` credits NLI's `neutral` mass |
| HHEM AUC on the same pairs | **0.828**, and `r = 0.26` against drift |

Layer ownership is clean and non-overlapping, which answers the HHEM question:

```
                     drift  policy   hhem
contradiction  n=10    90%     40%    70%
faithful       n=14     0%     93%    21%
hijack         n=10    50%    100%   100%
param_poison   n=12    33%     83%   100%
scope_creep    n=11     0%     82%    91%
```

Drift owns contradiction and nothing else. The policy layer owns faithful-but-
forbidden, which no other layer can see. HHEM is the only layer with real
coverage of `scope_creep` and `param_poison`, and on the oracle index it catches
5 must-block cases the other two miss. It is also the noisiest — 7 false alarms
on the 11 controls — which is why it belongs where it currently sits: attached
to the result and folded into trust, not gating.

Union coverage: drift ∪ policy blocks 51/57 (89%); adding HHEM reaches 56/57
(98%). The remaining one is the 780 EUR expense report.
