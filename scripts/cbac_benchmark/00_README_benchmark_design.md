# benchmarking design for CBAC

_no happy paths_

Each entry in `intents.yaml` has: `id`, `category`, `policy_file`,
`user_intent` (can also be omitted), `intended_action`, and `hypothesis` 

## the check points in the CBAC pipeline

1. **drift** (`_check1_drift`): runs *only if `user_intent` is
   thruthful*. NLI(`user_intent` as premise, `intended_action` as hypothesis).
   `contradiction >= 0.60` immediately denies, skipping every later stage
   entirely
2. **policy fetch + chunk cache**: `skill.md` → `flatten_policy_chunks` →
   each chunk NLI-classified as allowed/forbidden at index time
   (`_classify_chunks`): forbidden **only if** `forbid_entailment >
   allow_entailment` **and** `forbid_entailment > 0.40`; every other chunk defaults to **allowed**.
3. **(tier 1) cosine gap**: embed the intent (BGE-small-en-v1.5), compare to
   the single nearest allowed chunk and single nearest forbidden chunk
   (`top_k=1` each side).if `gap > 0.12` → allow.if `gap < -0.08` → deny.
   everything else → Tier 2.
4. **(tier 2) NLI entailment**: cross-encoder NLI (`nli-deberta-v3-small`)
   with the *top allowed chunk* as premise, intent as hypothesis.if
   `entailment >= 0.55` → allow.if `contradiction >= 0.60` → deny. everything else → Tier 3.
5. **(tier 3) optional LLM**: only if a backend is configured; it must return
   an `LLMVerdict` (see `skills.py`) whose `decision` is `allow`, `deny` or
   `advise`. An `advise` is folded to `deny`, as is a malformed verdict — the
   pipeline has no caller-must-decide state. No backend configured → `deny`
   (`TIER3_NO_BACKEND_DENY`), which is where roughly half of all decisions
   currently land.


---

## Category index

| Category | Targets | Why it matters |
|---|---|---|
| `NEG` | Tier 2 NLI, negation/scope | because NLI models are known to be weak at negation and exception scoping |
| `SYN` | Tier 1 + 2, lexical vs semantic | Synonyms/paraphrase not lexically present in the policy |
| `UNI` | Encoder robustness | Unicode/homoglyph/zero-width obfuscation of the action text |
| `LANG` | Encoder cross-lingual alignment | `bge-small-en-v1.5` is English-centric |
| `DIL` | Tier 1 cosine | Padding/dilution shifts the embedding centroid |
| `BUN` | Whole pipeline | A benign primary action bundled with a forbidden secondary one |
| `HED` | Tier 2 NLI | Hedged/hypothetical/fictional framing of a forbidden action |
| `BND` | Tier 1/2 thresholds | Deliberately near the 0.12 / 0.08 / 0.55 / 0.60 boundaries |
| `NUM` | Whole pipeline | Semantic models don't do arithmetic — quantity blindness |
| `TYPE` | Input handling | Non-string / malformed `intended_action` shapes |
| `EMPTY` | Input handling | Degenerate / empty inputs |
| `LEN` | Tier 1 cosine | Extreme-length intents diluting or overloading the signal |
| `DRIFT` | Check 1 | Contradiction detection, and the no-`user_intent` bypass |
| `LLM` | Tier 3 verdict handling | What a backend's verdict does to the decision |
| `POLICY` | Chunking / classification | Malformed, vague, contradictory, or lopsided policy files |
| `DETERM` | Consistency | Same semantic policy, different formatting → different decision? |

## Policy fixtures (`policies/`)

This directory is the **shared policy corpus**. The cases in `intents.yaml`
reference a subset of it by filename; the eval suite in
`cbac_service/tests/eval/` reads every file in it and carries the per-span gold
labels for each. Adding a file here is free for this runner (it only loads what
`intents.yaml` names) but an edit to an existing one can detach the eval's gold
labels — `cbac_service/tests/eval/test_datasets_sanity.py` fails when that
happens, and it runs in the ordinary `uv run pytest`.

- `baseline_strict.md`: straigh forward allow/forbid list. 
- `vague_narrative.md`: heavy neutral prose, no clear permission language.
  Targets the "unclassified chunk defaults to allowed" behavior.
- `malformed_frontmatter.md` — broken YAML delimiter. it should
  fallback to plain-text chunking (`flatten_policy_chunks`'s `except`).
- `self_contradictory.md` — the same action both permitted and forbidden in
  different places. 
- `only_forbidden_no_allowed.md` — no allowed content at all. Should hit the
  fail-closed "no allowed policy chunks found" path (more like a sanity check)
- `catch_all_permissive.md` — one broad, vague "may do what's needed"
  sentence as the entire allowed side.
- `non_english_policy.md` — same policy content, written in French.
- `exception_carveouts.md` — nested exceptions ("X, except when Y, unless
  Z").
- `numeric_constraint.md` — a policy expressed as a dollar threshold, which
  nothing in the pipeline can actually evaluate as a number.
- `calendar_actions.md`, `contract_review.md`, `discount_policy.md`,
  `doc_sharing.md`, `infra_actions.md`, `legal_hold.md` — ordinary cards in
  other domains, one intent case each here and a full grid in the eval suite.

The remaining files (`payments_card.md`, `devops_card.md`, `support_card.md`,
`clinical_card.md`, `analytics_card.md`, `md_headings.md`,
`prose_systemprompt.txt`, `iam_json.json`, `yaml_config.yaml`,
`readme_guardrails.md`, `numbered_rules.md`, `long_multisection.md`,
`runbook_wiki.md`, `opa_rules.json`, `terms_prose.txt`, `cross_reference.md`)
are the eval suite's own corpus and are not referenced by `intents.yaml`. They
cover document shapes the fixtures above do not: markdown headings, a system
prompt, IAM JSON, a YAML config, a README, numbered rules, a policy long enough
to exceed `CHUNK_MAX_WORDS`, a permission table, OPA-style rule objects, legal
prose, and grants whose exceptions live in a different section.

## running this

This corpus assumes a `cbac-server` instance with these policies loaded per
agent (one agent per fixture is simplest) and calls `verify_cbac` or
`POST /cbac/v1/authorize` 
