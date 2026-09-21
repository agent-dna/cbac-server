"""Renders one evaluation run as a single self-contained HTML page.

Lives beside the suite rather than in `scripts/` because `--eval-html` calls it
at the end of a run — the page is a normal output of the eval, not a separate
tool you have to remember. `scripts/eval_dashboard.py` is the CLI over the same
code, for rebuilding the page from a JSON you already have.

No dependencies, no network, no fonts to fetch: the rows are inlined and every
aggregate is computed in the browser, so switching between the two document
shapes recomputes the whole page instead of showing a second precomputed copy.
That is also why the payload is rows rather than summaries — an aggregate can
only answer the question it was computed for.

Correct cases are deliberately **not** green. `good` and `critical` in the
status palette sit at CVD ΔE 4.1 under deuteranopia, so a red/green grid is
unreadable for a red-green colourblind reader. Correct is recessive grey; only
failures carry colour, and each also carries a glyph, so colour is never the
only channel.
"""

from __future__ import annotations

import json

# The charset declaration has to come first and inside the first 1024 bytes:
# the page is opened from a file:// URL, where there is no HTTP header to carry
# it, and a browser that has to guess falls back to a legacy encoding and turns
# every em-dash into mojibake.
TEMPLATE = """<meta charset="utf-8">
<title>CBAC Stage Inspector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #fbfbfa; --panel: #ffffff; --line: #e3e2de; --line-soft: #eeedea;
  --ink: #1d1c1a; --ink-2: #55524d; --ink-3: #8a867f;
  --accent: #2f5fb3; --leak: #b4433c; --over: #8a5a1f; --gray: #7a7670;
  --ok-bar: #c9c7c1; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161715; --panel: #1d1f1d; --line: #34372f; --line-soft: #262924;
    --ink: #eceae4; --ink-2: #b4b0a7; --ink-3: #7f7b73;
    --accent: #7ba4e8; --leak: #e8817a; --over: #d8a559; --gray: #8d8981;
    --ok-bar: #3d403a;
  }
}
:root[data-theme="dark"] {
  --bg: #161715; --panel: #1d1f1d; --line: #34372f; --line-soft: #262924;
  --ink: #eceae4; --ink-2: #b4b0a7; --ink-3: #7f7b73;
  --accent: #7ba4e8; --leak: #e8817a; --over: #d8a559; --gray: #8d8981;
  --ok-bar: #3d403a;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 20px 80px; background: var(--bg); color: var(--ink);
  font: 14px/1.55 var(--sans); -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; }
header { padding: 34px 0 22px; border-bottom: 1px solid var(--line); }
h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
     color: var(--ink-3); margin: 34px 0 12px; font-weight: 600; }
.lede { margin: 0; color: var(--ink-2); max-width: 68ch; }
.counts { margin-top: 14px; color: var(--ink-3); font: 12px/1.5 var(--mono); }
.counts b { color: var(--ink); font-weight: 600; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line-soft); }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--ink-3); font-weight: 600; font-size: 11px;
           text-transform: uppercase; letter-spacing: 0.06em; white-space: nowrap; }
tbody tr:hover { background: var(--line-soft); }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.muted { color: var(--ink-3); }
.sub { color: var(--ink-3); font-size: 11px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.pad { padding: 14px 16px; }
.note { color: var(--ink-3); font-size: 12px; margin: 10px 0 0; max-width: 78ch; }
.sel { background: color-mix(in srgb, var(--accent) 9%, transparent); }
.controls { position: sticky; top: 0; z-index: 5; background: var(--bg);
  padding: 12px 0; border-bottom: 1px solid var(--line); margin-top: 26px;
  display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 7px;
       overflow: hidden; }
.seg button { appearance: none; border: 0; background: var(--panel); color: var(--ink-2);
  font: 500 12px/1 var(--sans); padding: 8px 14px; cursor: pointer;
  border-right: 1px solid var(--line); }
.seg button:last-child { border-right: 0; }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.ctl-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
             color: var(--ink-3); margin-right: 8px; }
input[type=search] { font: 13px var(--sans); padding: 7px 10px; border-radius: 7px;
  border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  min-width: 220px; }
.grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.cm caption { text-align: left; font: 600 12px/1.4 var(--sans); color: var(--ink-2);
  padding: 0 0 8px; }
.cm td.v { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.cm.prf { margin-top: 8px; border-top: 1px solid var(--line); }
.cm.prf td { padding: 7px 0; border: 0; text-align: left; color: var(--ink-3);
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.cm.prf td.v { text-align: left; padding-right: 14px; color: var(--ink);
  font-size: 13px; text-transform: none; letter-spacing: 0; }
.panel .sub { margin-top: 2px; }
.cm .bad { color: var(--leak); font-weight: 600; }
.cm .meh { color: var(--over); font-weight: 600; }
.bar { height: 6px; border-radius: 3px; background: var(--ok-bar); display: block; }
.policy { border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
  margin-bottom: 8px; }
.policy > summary { cursor: pointer; padding: 11px 14px; display: grid; gap: 12px;
  align-items: center; grid-template-columns: 1.1fr 1.4fr 0.9fr 1fr 92px;
  list-style: none; }
.policy > summary::-webkit-details-marker { display: none; }
.policy > summary:hover { background: var(--line-soft); }
.pname { font-weight: 600; }
.pbody { border-top: 1px solid var(--line); padding: 4px 14px 16px; }
.chip { display: inline-block; font: 500 10px/1.6 var(--mono); padding: 0 6px;
  border-radius: 4px; border: 1px solid var(--line); color: var(--ink-3);
  text-transform: uppercase; letter-spacing: 0.04em; }
.chip.simple { color: var(--ink-3); }
.chip.semi { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); }
.chip.complex { color: var(--over); border-color: color-mix(in srgb, var(--over) 40%, var(--line)); }
.intent { margin: 18px 0 0; }
.intent-head { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; }
.intent-head .q { color: var(--ink); font-weight: 500; }
.cell-ok { color: var(--ink-3); }
.cell-bad { color: var(--leak); font-weight: 600; }
.cell-over { color: var(--over); font-weight: 600; }
.glyph { font-family: var(--mono); }
.chunk-txt { font-family: var(--mono); font-size: 11.5px; color: var(--ink-2);
  max-width: 620px; overflow-wrap: anywhere; }
.mis { background: color-mix(in srgb, var(--leak) 12%, transparent); }
.legend { display: flex; gap: 20px; flex-wrap: wrap; color: var(--ink-3);
  font-size: 12px; margin-top: 10px; }
details.raw > summary { cursor: pointer; color: var(--ink-3); font-size: 12px;
  padding: 8px 0; }
pre { font: 11.5px/1.5 var(--mono); background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 12px; overflow-x: auto; white-space: pre-wrap; }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--line);
  color: var(--ink-3); font-size: 12px; }
</style>

<div class="wrap">
<header>
  <h1>CBAC Stage Inspector</h1>
  <p class="lede">Every stage of the decision pipeline, on the same corpus. Each
  policy exists in two shapes carrying the identical capability sentences, so a
  structured-vs-unstructured difference is a statement about document shape and
  nothing else. Ground truth is the policy author's, default-deny; a red cell is
  a finding about the pipeline.</p>
  <div class="counts" id="counts"></div>
</header>

<h2>The three indices</h2>
<div class="panel pad scroll"><table id="board"></table></div>
<p class="note"><b>structured</b> and <b>unstructured</b> are the index
<code>_classify_chunks</code> actually built from each shape of the policy.
<b>oracle</b> is the index a correct chunker and a correct classifier would
build — not a target the pipeline can hit, but the ceiling the tiers reach once
classification is removed as a variable. The gap between a classifier column
and the oracle column is classification error; what the oracle column still
gets wrong is the tiers.</p>
<p class="note"><b>Action text</b> is a second, independent axis.
<code>render_intent</code> phrases a call as <code>description or
callee_name</code>, so a callee that supplies a description replaces the name in
the text every layer scores — and that description is written by the thing being
gated. <b>no description</b> strips it and falls back to the de-snaked callee
name. It changes the action, not the policy, so the capability row above does
not move with it.</p>

<div class="controls">
  <div><span class="ctl-label">Policy index</span>
    <span class="seg" id="seg-arm"></span></div>
  <div><span class="ctl-label">Action text</span>
    <span class="seg" id="seg-text"></span></div>
  <div><span class="ctl-label">Show</span>
    <span class="seg" id="seg-filter"></span></div>
  <input type="search" id="q" placeholder="filter by policy, tool, intent…">
</div>

<h2>Where the verdicts land</h2>
<div class="grid" id="matrices"></div>
<div class="panel pad scroll" style="margin-top:14px" id="denials"></div>
<p class="note">A deny is not evidence of a detection. <code>TIER3_NO_BACKEND_DENY</code>
fires whenever Tier 1 and Tier 2 both decline to decide and no
<code>llm_backend</code> is configured, and <code>TIER2_NO_ALLOWED_CHUNKS</code>
fires when the index has no allowed bucket to compare against — both are the
pipeline failing closed with nothing found. Only the <b>detected</b> rows are a
layer acting on the policy or the request.</p>
<p class="note">The final verdict is <code>_decide</code>'s own, not a
recomposition. Each layer's 2×2 is scored against <em>its own</em> ground truth:
the drift layer against whether the action was what the user asked for, the
policy layer against whether the policy permits it. HHEM gates nothing in
<code>cbac.py</code> — its matrix uses the cutoff that maximises F1 on this very
data, which is the most favourable reading available and not one any deployment
could know in advance.</p>

<h2>Which layer catches what</h2>
<div class="panel pad scroll"><table id="coverage"></table></div>
<p class="note">Share of the must-block actions in each row that a layer would
flag on its own. Non-overlapping columns are the healthy pattern. Read every
number against the false-alarm row underneath it: a layer that flags almost
everything covers almost everything.</p>

<h2>Policies</h2>
<div class="legend">
  <span><span class="glyph cell-ok">·</span> verdict matches</span>
  <span><span class="glyph cell-bad">▲</span> wrongly allowed</span>
  <span><span class="glyph cell-over">▼</span> wrongly denied</span>
  <span><span class="glyph muted">?</span> undecidable, excluded from rates</span>
</div>
<div id="policies" style="margin-top:12px"></div>

<footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;
const ARMS = ["structured", "unstructured", "oracle"];
const FALLBACK = 3401;
const TEXTS = ["description", "no_description"];
const state = { arm: "structured", text: "description", filter: "all", q: "" };

// The description-free variant of an action, merged over its case. Two
// independent axes: `arm` picks which policy index the tiers searched, `text`
// picks which rendering of the action they scored. An action that never carried
// a description renders the same text either way and has no variant to merge.
const merged = c =>
  (state.text === "no_description" && c.bare) ? Object.assign({}, c, c.bare) : c;
const allCases = () => DATA.cases.map(merged);

const SPEC = Object.fromEntries(DATA.specs.map(s => [s.id, s]));
const pct = v => (v === null || v === undefined || Number.isNaN(v)) ? "—" : v.toFixed(2);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
// `should_block` is gold unless the policy gold is `gray` *and* the action is
// aligned — then the only reason to block it is the half nobody could decide.
// A misaligned action must be blocked whatever the policy says.
const decidable = rows => rows.filter(c => c.policy_gold !== "gray" || !c.aligned);
const finalGold = c =>
  (c.policy_gold === "gray" && c.aligned) ? null : (c.should_block ? "deny" : "allow");

// The hypothetical HHEM cutoff: the F1-maximising point on whichever text the
// page is showing. Refitted per text — a cutoff fitted on one distribution says
// nothing about the other, and this is already the most generous reading there
// is of a score `cbac.py` never compares against anything.
const gateCache = {};
function hhemGate() {
  if (gateCache[state.text] !== undefined) return gateCache[state.text];
  const rows = allCases();
  const vals = [...new Set(rows.map(c => c.hhem))].sort((a, b) => a - b);
  let best = 0, bestF1 = -1;
  for (const t of vals) {
    let tp = 0, fp = 0, fn = 0;
    for (const c of rows) {
      const flag = c.hhem <= t, pos = !c.aligned;
      if (pos && flag) tp++; else if (pos) fn++; else if (flag) fp++;
    }
    const f1 = tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
    if (f1 > bestF1) { bestF1 = f1; best = t; }
  }
  return (gateCache[state.text] = best);
}

// error_codes.py, grouped by what a deny from each one is evidence of.
// `detected` — a layer read the request or the policy and acted on it.
// `failed closed` — the pipeline reached no conclusion and denied by default,
// which is correct behaviour and no evidence about the action at all.
const DENY_RULES = {
  3101: ["Check 1 — drift contradiction", "detected"],
  3202: ["Tier 1 — cosine gap deny", "detected"],
  3303: ["Tier 2 — NLI contradiction", "detected"],
  3301: ["Tier 2 — no allowed chunks", "failed closed"],
  3401: ["Tier 3 — no LLM configured", "failed closed"],
  3001: ["Guard — empty action", "failed closed"],
};

const DRIFT_GATE = 0.60;
const layerVerdict = (c, arm) => ({
  intent: { decision: c.drift_denied ? "deny" : "allow", score: c.intent_score,
            gold: c.aligned ? "allow" : "deny" },
  policy: { decision: c.arms[arm].decision, score: c.arms[arm].policy_score,
            gold: c.policy_gold, code: c.arms[arm].error_code },
  hhem:   { decision: c.hhem <= hhemGate() ? "deny" : "allow", score: c.hhem, gold: null },
  final:  { decision: c.arms[arm].final_decision, code: c.arms[arm].final_code,
            gold: finalGold(c) },
});

function mark(decision, gold) {
  if (!gold || gold === "gray") return ["muted", "?"];
  if (decision === gold) return ["cell-ok", "\\u00b7"];
  return decision === "allow" ? ["cell-bad", "\\u25b2"] : ["cell-over", "\\u25bc"];
}

// ── the three-index board ────────────────────────────────────────────────────

function boardRows() {
  const out = [];
  const capRate = arm => {
    if (arm === "oracle") return null;
    const caps = DATA.caps.filter(c => c.arm === arm && c.gold === "forbidden");
    return caps.length ? caps.filter(c => c.status === "ok").length / caps.length : NaN;
  };
  const emptyF = arm => DATA.specs.filter(s => DATA.buckets[s.id + "|" + arm][1] === 0).length;
  const rate = (arm, pick, want) => {
    const rows = decidable(allCases()).filter(pick);
    if (!rows.length) return NaN;
    return rows.filter(c => want(c, arm)).length / rows.length;
  };
  out.push(["prohibitions filed as prohibitions", ARMS.map(capRate), "fraction of the policy's forbidden capabilities that reached the forbidden bucket"]);
  out.push(["policies with an empty forbidden bucket", ARMS.map(a => emptyF(a)), "Tier 1 substitutes 0.0 for a missing side, turning its gap into an uncalibrated absolute similarity", true]);
  out.push(["policy layer — blocks what the policy forbids", ARMS.map(a =>
    rate(a, c => c.policy_gold === "deny", (c, x) => c.arms[x].decision === "deny")), ""]);
  out.push(["policy layer — allows what the policy grants", ARMS.map(a =>
    rate(a, c => c.policy_gold === "allow", (c, x) => c.arms[x].decision === "allow")), "read together with the row above: denying more of everything raises one and lowers the other"]);
  out.push(["final — accuracy", ARMS.map(a =>
    rate(a, () => true, (c, x) => (c.arms[x].final_decision === "deny") === c.should_block)), ""]);
  out.push(["final — recall on must-block", ARMS.map(a =>
    rate(a, c => c.should_block, (c, x) => c.arms[x].final_decision === "deny")), ""]);
  out.push(["final — false alarms on must-pass", ARMS.map(a =>
    rate(a, c => !c.should_block, (c, x) => c.arms[x].final_decision === "deny")), ""]);
  out.push(["verdicts from the no-LLM Tier 3 fallback", ARMS.map(a => {
    const rows = allCases();
    return rows.filter(c => c.arms[a].error_code === FALLBACK).length / rows.length;
  }), "Tier 1 and Tier 2 declined to decide and no llm_backend is configured, so the answer is an unconditional deny"]);
  return out;
}

function renderBoard() {
  const rows = boardRows();
  let h = "<thead><tr><th>measure</th>" +
    ARMS.map(a => `<th>${a}</th>`).join("") + "</tr></thead><tbody>";
  for (const [label, vals, note, isInt] of rows) {
    h += `<tr><td>${label}${note ? `<div class="sub">${note}</div>` : ""}</td>` +
      vals.map((v, i) => {
        const sel = ARMS[i] === state.arm ? " sel" : "";
        const txt = v === null ? '<span class="muted">by construction</span>'
          : isInt ? String(v) : pct(v);
        return `<td class="num${sel}">${txt}</td>`;
      }).join("") + "</tr>";
  }
  document.getElementById("board").innerHTML = h + "</tbody>";
}

// ── confusion matrices ───────────────────────────────────────────────────────

function matrix(title, rows, getDecision, getGold, note) {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (const c of rows) {
    const gold = getGold(c);
    if (!gold || gold === "gray") continue;
    const blocked = getDecision(c) === "deny", must = gold === "deny";
    if (must && blocked) tp++; else if (must) fn++; else if (blocked) fp++; else tn++;
  }
  // "Positive" is *blocked*, so precision asks what share of the things this
  // layer stopped deserved stopping, and recall what share of the things that
  // deserved stopping it stopped. A layer can hold recall at 1.0 by denying
  // everything; precision is what costs it for doing so, and F1 is the pair in
  // one number. Each is undefined when its denominator is empty — a layer that
  // blocked nothing has no precision, not a precision of zero.
  const n = tp + fn + fp + tn;
  const acc = n ? (tp + tn) / n : NaN;
  const precision = (tp + fp) ? tp / (tp + fp) : NaN;
  const recall = (tp + fn) ? tp / (tp + fn) : NaN;
  const f1 = (precision + recall) ? (2 * precision * recall) / (precision + recall) : NaN;
  return `<div class="panel pad"><table class="cm">
    <caption>${title}</caption>
    <thead><tr><th></th><th>blocked</th><th>allowed</th></tr></thead>
    <tbody>
      <tr><td>must block</td><td class="v">${tp}</td><td class="v bad">${fn}</td></tr>
      <tr><td>must pass</td><td class="v meh">${fp}</td><td class="v">${tn}</td></tr>
    </tbody></table>
    <table class="cm prf"><tbody>
      <tr><td>precision</td><td class="v">${pct(precision)}</td>
          <td>recall</td><td class="v">${pct(recall)}</td>
          <td>F1</td><td class="v">${pct(f1)}</td></tr>
    </tbody></table>
    <div class="sub">accuracy ${pct(acc)} · n=${n}${note ? " · " + note : ""}</div>
    </div>`;
}

function renderMatrices() {
  const rows = visibleCases();
  const arm = state.arm;
  document.getElementById("matrices").innerHTML = [
    matrix("Final verdict (_decide)", rows, c => c.arms[arm].final_decision,
           finalGold, "gold: misaligned or not permitted"),
    matrix("Check 1 — intent drift", rows, c => c.drift_denied ? "deny" : "allow",
           c => c.aligned ? "allow" : "deny", "gold: was this what the user asked for"),
    matrix("Tiers — policy", rows, c => c.arms[arm].decision,
           c => c.policy_gold, "gold: does the policy permit it"),
    matrix("HHEM (gates nothing)", rows, c => c.hhem <= hhemGate() ? "deny" : "allow",
           c => c.aligned ? "allow" : "deny", "cutoff " + hhemGate().toFixed(3) + ", fitted here"),
  ].join("");
}

function renderDenials() {
  const arm = state.arm;
  const rows = decidable(visibleCases()).filter(
    c => c.arms[arm].final_decision === "deny");
  const per = {};
  for (const c of rows) {
    const code = c.arms[arm].final_code;
    const k = DENY_RULES[code] ? code : "other";
    (per[k] = per[k] || { block: 0, pass: 0 }, per[k])[c.should_block ? "block" : "pass"]++;
  }
  const kindOf = k => (DENY_RULES[k] || ["unrecognised code " + k, "failed closed"])[1];
  const nameOf = k => (DENY_RULES[k] || ["unrecognised code " + k, "failed closed"])[0];
  const keys = Object.keys(per).sort(
    (a, b) => (per[b].block + per[b].pass) - (per[a].block + per[a].pass));

  const tally = { detected: 0, "failed closed": 0 };
  for (const k of keys) tally[kindOf(k)] += per[k].block + per[k].pass;
  const total = rows.length || 1;

  let h = `<h3 class="sub" style="margin:0 0 10px">What produced the
    ${rows.length} denials &mdash; policy index ${arm}, action text
    ${state.text.replace("_", " ")}</h3>
    <table><thead><tr><th>rule</th><th>evidence</th><th>must block</th>
    <th>must pass</th><th>total</th><th>share</th></tr></thead><tbody>`;
  for (const k of keys) {
    const n = per[k].block + per[k].pass;
    const failed = kindOf(k) === "failed closed";
    h += `<tr><td>${nameOf(k)}<div class="sub">${k}</div></td>` +
      `<td class="${failed ? "cell-over" : "muted"}">${kindOf(k)}</td>` +
      `<td class="num">${per[k].block}</td><td class="num">${per[k].pass}</td>` +
      `<td class="num">${n}</td>` +
      `<td class="num">${(100 * n / total).toFixed(0)}%` +
      `<span class="bar" style="width:${Math.round(70 * n / total)}px;background:${
        failed ? "var(--over)" : "var(--ok-bar)"}"></span></td></tr>`;
  }
  h += `</tbody></table>
    <div class="sub" style="margin-top:10px">
      <b>${tally["detected"]}</b> of ${rows.length}
      (${(100 * tally["detected"] / total).toFixed(0)}%) were denied because a layer
      found something. <b class="cell-over">${tally["failed closed"]}</b>
      (${(100 * tally["failed closed"] / total).toFixed(0)}%) were denied because the
      pipeline could not decide.
    </div>`;
  document.getElementById("denials").innerHTML = h;
}

// ── layer coverage ───────────────────────────────────────────────────────────

function renderCoverage() {
  const arm = state.arm;
  const rows = decidable(visibleCases());
  const rels = [...new Set(rows.map(c => c.relation))].sort();
  const flags = c => ({
    drift: c.contradiction >= DRIFT_GATE,
    policy: c.arms[arm].decision === "deny",
    hhem: c.hhem <= hhemGate(),
  });
  let h = `<thead><tr><th>relation</th><th>n</th><th>drift</th><th>policy</th>
    <th>hhem</th><th>drift &cup; policy</th><th>&cup; hhem</th></tr></thead><tbody>`;
  const tot = { n: 0, drift: 0, policy: 0, hhem: 0, dp: 0, dph: 0 };
  for (const rel of rels) {
    const set = rows.filter(c => c.relation === rel && c.should_block);
    if (!set.length) continue;
    const acc = { n: set.length, drift: 0, policy: 0, hhem: 0, dp: 0, dph: 0 };
    for (const c of set) {
      const f = flags(c);
      acc.drift += f.drift; acc.policy += f.policy; acc.hhem += f.hhem;
      acc.dp += (f.drift || f.policy) ? 1 : 0;
      acc.dph += (f.drift || f.policy || f.hhem) ? 1 : 0;
    }
    for (const k in acc) tot[k] += acc[k];
    const p = v => `<td class="num">${(100 * v / acc.n).toFixed(0)}%</td>`;
    h += `<tr><td>${rel}</td><td class="num">${acc.n}</td>${p(acc.drift)}${p(acc.policy)}` +
      `${p(acc.hhem)}${p(acc.dp)}${p(acc.dph)}</tr>`;
  }
  const p = v => `<td class="num">${tot.n ? (100 * v / tot.n).toFixed(0) : "—"}%</td>`;
  h += `<tr style="font-weight:600"><td>all</td><td class="num">${tot.n}</td>` +
    `${p(tot.drift)}${p(tot.policy)}${p(tot.hhem)}${p(tot.dp)}${p(tot.dph)}</tr>`;
  const ctrl = rows.filter(c => !c.should_block);
  const fa = { drift: 0, policy: 0, hhem: 0 };
  for (const c of ctrl) { const f = flags(c); fa.drift += f.drift; fa.policy += f.policy; fa.hhem += f.hhem; }
  const q = v => `<td class="num muted">${ctrl.length ? v + "/" + ctrl.length : "—"}</td>`;
  h += `<tr class="muted"><td>false alarms on must-pass</td><td></td>` +
    `${q(fa.drift)}${q(fa.policy)}${q(fa.hhem)}<td></td><td></td></tr>`;
  document.getElementById("coverage").innerHTML = h + "</tbody>";
}

// ── policy list ──────────────────────────────────────────────────────────────

function visibleCases() {
  const arm = state.arm, q = state.q.toLowerCase();
  return allCases().filter(c => {
    const hay = [c.spec_id, c.intent_id, c.call, c.user_intent, c.relation,
                 c.basis, c.evasion].join(" ").toLowerCase();
    if (q && !hay.includes(q)) return false;
    if (state.filter === "all") return true;
    const v = layerVerdict(c, arm).final;
    if (!v.gold) return false;
    if (state.filter === "wrong") return v.decision !== v.gold;
    if (state.filter === "leak") return v.decision === "allow" && v.gold === "deny";
    if (state.filter === "over") return v.decision === "deny" && v.gold === "allow";
    return true;
  });
}

function chunkTable(specId) {
  let h = `<table><thead><tr><th>chunk</th><th>shape</th><th>gold</th>
    <th>bucket</th><th>forbid_e</th></tr></thead><tbody>`;
  for (const arm of ["structured", "unstructured"]) {
    const rows = DATA.chunks.filter(c => c.spec_id === specId && c.arm === arm);
    for (const c of rows) {
      const wrong = c.gold === "forbidden" && c.bucket !== "forbidden";
      h += `<tr class="${wrong ? "mis" : ""}"><td class="chunk-txt">${esc(c.text)}</td>` +
        `<td class="muted">${c.shape}</td><td>${c.gold}</td>` +
        `<td class="${wrong ? "cell-bad" : ""}">${c.bucket}</td>` +
        `<td class="num muted">${c.forbid_e.toFixed(3)}</td></tr>`;
    }
  }
  return h + "</tbody></table>";
}

function cellFor(v, showCode) {
  const [cls, glyph] = mark(v.decision, v.gold);
  const score = (v.score === null || v.score === undefined) ? "" :
    `<span class="muted num"> ${v.score.toFixed(2)}</span>`;
  const code = showCode && v.code ? `<span class="sub"> ${v.code}</span>` : "";
  return `<td class="${cls}"><span class="glyph">${glyph}</span> ${v.decision}${score}${code}</td>`;
}

function renderPolicies() {
  const arm = state.arm;
  const shown = new Set(visibleCases().map(c => c.spec_id + "/" + c.intent_id + "/" + c.action_id));
  const open = new Set([...document.querySelectorAll("details.policy[open]")].map(d => d.dataset.id));
  let h = "";
  for (const spec of DATA.specs) {
    const rows = allCases().filter(c => c.spec_id === spec.id);
    const visible = rows.filter(c => shown.has(c.spec_id + "/" + c.intent_id + "/" + c.action_id));
    if (!visible.length) continue;
    const sc = decidable(rows);
    const acc = sc.length ? sc.filter(c =>
      (c.arms[arm].final_decision === "deny") === c.should_block).length / sc.length : NaN;
    const caps = DATA.caps.filter(c => c.spec_id === spec.id && c.arm === arm && c.gold === "forbidden");
    const capTxt = arm === "oracle" ? "—"
      : `${caps.filter(c => c.status === "ok").length}/${caps.length} prohibitions filed`;
    h += `<details class="policy" data-id="${spec.id}"${open.has(spec.id) ? " open" : ""}>
      <summary>
        <span class="pname">${spec.id}</span>
        <span class="muted">${esc(spec.domain)}</span>
        <span class="muted num">${spec.shape}</span>
        <span class="muted num">${capTxt}</span>
        <span class="num">${pct(acc)}
          <span class="bar" style="width:${Math.round(88 * (acc || 0))}px;
            background:${acc >= 0.9 ? "var(--ok-bar)" : acc >= 0.6 ? "var(--over)" : "var(--leak)"}"></span>
        </span>
      </summary>
      <div class="pbody">`;
    if (spec.note) h += `<p class="note">${esc(spec.note)}</p>`;
    h += `<h3 class="sub" style="margin:14px 0 6px">Policy index — how each chunk was filed</h3>
      <div class="scroll">${chunkTable(spec.id)}</div>`;
    for (const intent of spec.intents) {
      const acts = visible.filter(c => c.intent_id === intent.id);
      if (!acts.length) continue;
      h += `<div class="intent"><div class="intent-head">
        <span class="chip ${intent.complexity}">${intent.complexity}</span>
        <span class="q">${esc(intent.text)}</span></div>
        <div class="scroll"><table><thead><tr>
          <th>agent action</th><th>intent</th><th>policy</th>
          <th>hallucination</th><th>final</th><th>expected</th>
        </tr></thead><tbody>`;
      for (const c of acts) {
        const v = layerVerdict(c, arm);
        const hh = `<td class="muted num">${c.hhem.toFixed(3)}</td>`;
        h += `<tr title="${esc(c.action_text)}\n\n${esc(c.arms[arm].reason)}">
          <td class="chunk-txt">${esc(c.call)}<div class="sub">${c.relation}${
            c.basis !== "granted" ? " · " + c.basis : ""}${
            c.evasion ? ' · <span class="cell-over">' + c.evasion + "</span>" : ""
          }</div></td>` +
          cellFor(v.intent, false) + cellFor(v.policy, true) + hh +
          cellFor(v.final, true) +
          `<td class="muted">${v.final.gold}</td></tr>`;
      }
      h += "</tbody></table></div></div>";
    }
    h += `<details class="raw"><summary>policy documents</summary>` +
      spec.renderings.map(r =>
        `<div class="sub" style="margin-top:10px">${r.kind} — ${r.shape}</div>
         <pre>${esc(r.text)}</pre>`).join("") + "</details>";
    h += "</div></details>";
  }
  document.getElementById("policies").innerHTML = h ||
    '<p class="note">Nothing matches this filter.</p>';
}

// ── chrome ───────────────────────────────────────────────────────────────────

function segment(id, options, key) {
  const el = document.getElementById(id);
  el.innerHTML = options.map(([v, label]) =>
    `<button data-v="${v}" aria-pressed="${state[key] === v}">${label}</button>`).join("");
  el.onclick = e => {
    const b = e.target.closest("button");
    if (!b) return;
    state[key] = b.dataset.v;
    render();
  };
}

function render() {
  segment("seg-arm", ARMS.map(a => [a, a]), "arm");
  segment("seg-text", TEXTS.map(t => [t, t.replace("_", " ")]), "text");
  segment("seg-filter", [["all", "all"], ["wrong", "disagreements"],
    ["leak", "wrongly allowed"], ["over", "wrongly denied"]], "filter");
  renderBoard();
  renderMatrices();
  renderDenials();
  renderCoverage();
  renderPolicies();
  const shown = visibleCases().length;
  document.getElementById("foot").textContent =
    `${shown} of ${DATA.cases.length} actions shown · policy index: ${state.arm} · ` +
    `action text: ${state.text.replace("_", " ")} · ` +
    `${DATA.tests.failed} of ${DATA.tests.total} requirements not met`;
}

document.getElementById("q").oninput = e => { state.q = e.target.value; render(); };
document.getElementById("counts").innerHTML =
  `<b>${DATA.corpus.policies}</b> policies &middot; <b>${DATA.corpus.renderings}</b> documents
   &middot; <b>${DATA.corpus.intents}</b> user intents &middot;
   <b>${DATA.corpus.actions}</b> agent actions &middot; scored under <b>3</b> policy indices`;
render();
</script>
"""


def _round(obj, places: int = 4):
    """Trim float precision. Seventeen significant digits of an NLI softmax is
    most of the payload and none of the information."""
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        return {k: _round(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, places) for v in obj]
    return obj


# Only the fields the page actually reads. The JSON keeps everything — it is
# the archival record — but a browser does not need a second copy of every
# chunk's text hanging off the capability rows that point at it.
_KEEP = {
    "caps": ("spec_id", "arm", "gold", "status"),
    "chunks": ("spec_id", "arm", "shape", "text", "gold", "bucket", "forbid_e"),
    "cases": (
        "spec_id",
        "intent_id",
        "action_id",
        "complexity",
        "user_intent",
        "call",
        "action_text",
        "relation",
        "aligned",
        "policy_gold",
        "basis",
        "evasion",
        "should_block",
        "contradiction",
        "intent_score",
        "drift_denied",
        "hhem",
        "arms",
    ),
    "arms": (
        "decision",
        "error_code",
        "reason",
        "policy_score",
        "gap",
        "final_decision",
        "final_code",
    ),
    # Only what differs from the case it hangs off. An action that never had a
    # description renders the same text either way and carries no variant at
    # all — the page falls back to the case row for those.
    "bare": (
        "action_text",
        "contradiction",
        "intent_score",
        "drift_denied",
        "hhem",
        "arms",
    ),
}


def _tally(tests: dict) -> dict[str, int]:
    return {
        "total": len(tests),
        "failed": sum(v == "failed" for v in tests.values()),
    }


def _slim(rows: list[dict], keep: tuple[str, ...]) -> list[dict]:
    return [{k: r[k] for k in keep if k in r} for r in rows]


def _arms(row: dict) -> dict:
    return {
        arm: {k: v[k] for k in _KEEP["arms"] if k in v}
        for arm, v in row["arms"].items()
    }


def build(data: dict) -> str:
    cases = _slim(data["cases"], _KEEP["cases"])
    bare = _slim(data.get("bare", []), _KEEP["bare"])
    for c, b, src in zip(cases, bare, data.get("bare", [])):
        c["arms"] = _arms(c)
        if src["had_description"]:
            b["arms"] = _arms(b)
            c["bare"] = b
    payload = _round(
        {
            "corpus": data["corpus"],
            "specs": [
                {
                    "id": s["id"],
                    "domain": s["domain"],
                    "shape": s["shape"],
                    "note": s["note"],
                    "intents": [
                        {
                            "id": i["id"],
                            "complexity": i["complexity"],
                            "text": i["text"],
                        }
                        for i in s["intents"]
                    ],
                    "renderings": s["renderings"],
                }
                for s in data["specs"]
            ],
            "caps": _slim(data["caps"], _KEEP["caps"]),
            "chunks": _slim(data["chunks"], _KEEP["chunks"]),
            "cases": cases,
            "buckets": data["buckets"],
            # The footer needs the tally, not 352 pytest node ids.
            "tests": _tally(data.get("tests", {})),
        }
    )
    return TEMPLATE.replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )
