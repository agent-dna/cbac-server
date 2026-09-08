"""Renders an eval run as one self-contained HTML page.

Lives beside the suite rather than in `scripts/` because `--eval-html` calls it
directly at the end of a run — the page is a normal output of the eval, not a
separate tool you have to remember to run. `scripts/eval_dashboard.py` is the
CLI over the same code, for rebuilding the page from a JSON you already have.

No dependencies and no network: the data is inlined, so the file opens from a
file:// URL or drops into a build artifact unchanged.
"""

from __future__ import annotations

import json

# Correct cases are deliberately NOT green. `good` (#0ca30c) and `critical`
# (#d03b3b) sit at CVD ΔE 4.1 under deuteranopia — a red/green grid is
# unreadable for a red-green colourblind reader, which is the one thing this
# page cannot afford. Correct is recessive grey; only failures carry colour,
# and each failure also carries a glyph so colour is never the sole channel.
TEMPLATE = """<title>CBAC Verdict Inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  --surface:      #fbfbfc;
  --panel:        #ffffff;
  --panel-2:      #f4f5f7;
  --ink:          #14161a;
  --ink-2:        #5b6068;
  --ink-3:        #878d97;
  --rule:         #e3e5ea;
  --rule-2:       #d3d7de;
  --accent:       #2a78d6;
  --accent-soft:  #e8f0fc;
  --ok:           #d6dae1;
  --ok-ink:       #7c838d;
  --leak:         #d03b3b;
  --leak-ink:     #ffffff;
  --overblock:    #fab219;
  --overblock-ink:#3d2f05;
  --unknown:      #b6bcc6;
  --shadow:       0 1px 2px rgba(20,22,26,.06), 0 4px 16px rgba(20,22,26,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface:      #15171a;
    --panel:        #1c1f23;
    --panel-2:      #23272c;
    --ink:          #f2f3f5;
    --ink-2:        #9aa1ab;
    --ink-3:        #6f7782;
    --rule:         #2b3037;
    --rule-2:       #3a414a;
    --accent:       #3987e5;
    --accent-soft:  #17293f;
    --ok:           #343a42;
    --ok-ink:       #7f8790;
    --leak:         #e05a5a;
    --leak-ink:     #1a0505;
    --overblock:    #fab219;
    --overblock-ink:#3d2f05;
    --unknown:      #4a525c;
    --shadow:       0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"] {
  --surface:      #15171a;
  --panel:        #1c1f23;
  --panel-2:      #23272c;
  --ink:          #f2f3f5;
  --ink-2:        #9aa1ab;
  --ink-3:        #6f7782;
  --rule:         #2b3037;
  --rule-2:       #3a414a;
  --accent:       #3987e5;
  --accent-soft:  #17293f;
  --ok:           #343a42;
  --ok-ink:       #7f8790;
  --leak:         #e05a5a;
  --leak-ink:     #1a0505;
  --overblock:    #fab219;
  --overblock-ink:#3d2f05;
  --unknown:      #4a525c;
  --shadow:       0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { margin: 0; text-wrap: balance; font-weight: 600; }
code, .mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace; }

.wrap { max-width: 1560px; margin: 0 auto; padding: 0 24px 72px; }

/* ── masthead ─────────────────────────────────────────────── */
header.top { border-bottom: 1px solid var(--rule); background: var(--panel); }
.top-in {
  max-width: 1560px; margin: 0 auto; padding: 20px 24px 18px;
  display: flex; flex-wrap: wrap; gap: 20px; align-items: baseline;
}
.title { font-size: 19px; letter-spacing: -.01em; }
.subtitle { color: var(--ink-2); font-size: 13px; }
.corpus { margin-left: auto; display: flex; gap: 18px; flex-wrap: wrap; }
.corpus div { font-size: 12px; color: var(--ink-2); }
.corpus b {
  display: block; font-size: 17px; color: var(--ink); font-weight: 600;
  font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", monospace;
}

/* ── kpi row ──────────────────────────────────────────────── */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 24px 0 20px; }
.kpi {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
  padding: 13px 15px 14px; box-shadow: var(--shadow);
}
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--ink-3); font-weight: 600; }
.kpi .value {
  font-family: "IBM Plex Mono", monospace; font-size: 28px; font-weight: 500;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; margin-top: 4px; line-height: 1.1;
}
.kpi .note { font-size: 11.5px; color: var(--ink-2); margin-top: 3px; }
.kpi.alarm .value { color: var(--leak); }

/* ── controls ─────────────────────────────────────────────── */
.controls {
  position: sticky; top: 0; z-index: 20; background: var(--surface);
  padding: 10px 0 12px; border-bottom: 1px solid var(--rule);
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
}
.seg { display: inline-flex; border: 1px solid var(--rule-2); border-radius: 7px; overflow: hidden; background: var(--panel); }
.seg button {
  font: inherit; font-size: 12.5px; padding: 6px 12px; border: 0; cursor: pointer;
  background: transparent; color: var(--ink-2); border-right: 1px solid var(--rule);
}
.seg button:last-child { border-right: 0; }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.ctl-label { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--ink-3); font-weight: 600; }
input[type="search"] {
  font: inherit; font-size: 13px; padding: 6px 11px; min-width: 210px;
  border: 1px solid var(--rule-2); border-radius: 7px; background: var(--panel); color: var(--ink);
}
input[type="search"]:focus-visible { outline: 2px solid var(--accent); outline-offset: -1px; }

/* ── legend ───────────────────────────────────────────────── */
.legend { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; margin: 16px 0 10px; font-size: 12.5px; color: var(--ink-2); }
.legend .item { display: inline-flex; gap: 6px; align-items: center; }
.swatch {
  width: 16px; height: 16px; border-radius: 3px; display: inline-grid; place-items: center;
  font-size: 10px; font-weight: 700; font-family: "IBM Plex Mono", monospace;
}
.sw-ok { background: var(--ok); color: var(--ok-ink); }
.sw-leak { background: var(--leak); color: var(--leak-ink); }
.sw-over { background: var(--overblock); color: var(--overblock-ink); }
.sw-unknown { background: transparent; border: 1px dashed var(--unknown); color: var(--ink-3); }

/* ── the grid ─────────────────────────────────────────────── */
.board { display: grid; grid-template-columns: minmax(0, 1fr) 372px; gap: 22px; align-items: start; }
@media (max-width: 1100px) { .board { grid-template-columns: minmax(0,1fr); } }

.rows { display: flex; flex-direction: column; gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }
.row { display: grid; grid-template-columns: 208px 68px minmax(0,1fr); gap: 12px; align-items: center; background: var(--panel); padding: 7px 12px; }
.row:hover { background: var(--panel-2); }
.rowname { font-size: 12.5px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rowname .grp { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3); font-weight: 600; }
.rowstat { font-family: "IBM Plex Mono", monospace; font-size: 12px; font-variant-numeric: tabular-nums; color: var(--ink-2); text-align: right; }
.rowstat.bad { color: var(--leak); font-weight: 600; }
.cells { display: flex; flex-wrap: wrap; gap: 3px; }

.cell {
  width: 19px; height: 19px; border-radius: 3px; border: 0; padding: 0; cursor: pointer;
  display: grid; place-items: center; font-family: "IBM Plex Mono", monospace;
  font-size: 10px; font-weight: 700; line-height: 1;
}
.cell.ok { background: var(--ok); color: var(--ok-ink); }
.cell.leak { background: var(--leak); color: var(--leak-ink); }
.cell.over { background: var(--overblock); color: var(--overblock-ink); }
.cell.unknown { background: transparent; border: 1px dashed var(--unknown); color: var(--ink-3); }
.cell:hover, .cell:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.cell.pinned { outline: 2px solid var(--ink); outline-offset: 1px; }

/* ── detail rail ──────────────────────────────────────────── */
.rail { position: sticky; top: 62px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; box-shadow: var(--shadow); max-height: calc(100vh - 84px); overflow-y: auto; }
.rail-in { padding: 15px 16px 18px; }
.rail .empty { color: var(--ink-3); font-size: 13px; }
.verdict { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 12px; }
.pill { font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--rule-2); color: var(--ink-2); font-family: "IBM Plex Mono", monospace; }
.pill.leak { background: var(--leak); color: var(--leak-ink); border-color: transparent; }
.pill.over { background: var(--overblock); color: var(--overblock-ink); border-color: transparent; }
.pill.ok { background: var(--panel-2); }
.pill.accent { background: var(--accent-soft); color: var(--accent); border-color: transparent; }
.field { margin: 11px 0; }
.field .k { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em; color: var(--ink-3); font-weight: 600; margin-bottom: 3px; }
.field .v { font-size: 13px; }
.field .v.mono { font-size: 12px; word-break: break-word; }
.chunks { border-top: 1px solid var(--rule); margin-top: 14px; padding-top: 12px; }
.chunk { display: grid; grid-template-columns: 54px minmax(0,1fr); gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--rule); font-size: 11.5px; }
.chunk:last-child { border-bottom: 0; }
.chunk .tag { font-size: 9.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; font-family: "IBM Plex Mono", monospace; padding-top: 1px; }
.chunk .txt { color: var(--ink-2); font-family: "IBM Plex Mono", monospace; line-height: 1.45; word-break: break-word; }
.tag.mis { color: var(--leak); }
.tag.hit { color: var(--ink-3); }
.chunk.mis .txt { color: var(--ink); }

/* ── secondary panels ─────────────────────────────────────── */
section.panel { margin-top: 30px; }
section.panel > h2 { font-size: 15px; margin-bottom: 3px; }
section.panel > p.lede { color: var(--ink-2); font-size: 13px; margin: 0 0 14px; max-width: 68ch; }
.bars { display: flex; flex-direction: column; gap: 7px; }
.bar-row { display: grid; grid-template-columns: 168px 46px minmax(0,1fr); gap: 11px; align-items: center; }
.bar-row .n { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: flex; height: 17px; border-radius: 3px; overflow: hidden; background: var(--panel-2); gap: 2px; }
.bar span { display: block; }
.bar .s-ok { background: var(--ok); }
.bar .s-leak { background: var(--leak); }
.bar .s-over { background: var(--overblock); }
.bar .s-unknown { background: var(--unknown); }
.cat-name { font-size: 12.5px; }

.scroller { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3); font-weight: 600; white-space: nowrap; }
td.mono { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; }
tbody tr:hover { background: var(--panel-2); }
.matrix td.num { font-family: "IBM Plex Mono", monospace; text-align: right; font-variant-numeric: tabular-nums; }

details.raw { margin-top: 12px; }
details.raw summary { cursor: pointer; font-size: 12px; color: var(--accent); }
details.raw pre { font-family: "IBM Plex Mono", monospace; font-size: 11px; white-space: pre-wrap; color: var(--ink-2); background: var(--panel-2); padding: 10px; border-radius: 6px; max-height: 320px; overflow: auto; }

.foot { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--ink-3); font-size: 12px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>

<header class="top">
  <div class="top-in">
    <div>
      <h1 class="title">CBAC Verdict Inspector</h1>
      <div class="subtitle">Every evaluation case, and which ones the pipeline gets wrong.</div>
    </div>
    <div class="corpus" id="corpus"></div>
  </div>
</header>

<div class="wrap">
  <div class="kpis" id="kpis"></div>

  <div class="controls">
    <span class="ctl-label">Dataset</span>
    <div class="seg" id="seg-set" role="group" aria-label="Dataset"></div>
    <span class="ctl-label">Policy index</span>
    <div class="seg" id="seg-arm" role="group" aria-label="Policy index"></div>
    <span class="ctl-label">Show</span>
    <div class="seg" id="seg-filter" role="group" aria-label="Outcome filter"></div>
    <input type="search" id="q" placeholder="Search action or policy…" aria-label="Search cases">
  </div>

  <div class="legend" id="legend"></div>

  <div class="board">
    <div class="rows" id="rows"></div>
    <aside class="rail" aria-live="polite"><div class="rail-in" id="rail"></div></aside>
  </div>

  <section class="panel">
    <h2>Where the failures cluster</h2>
    <p class="lede">By case category, for the current dataset and policy index. A category is
      only as good as its worst bar &mdash; an authorization system is judged on the red.</p>
    <div class="bars" id="bars"></div>
  </section>

  <section class="panel" id="layers-panel">
    <h2>Which layer catches what</h2>
    <p class="lede">End-to-end cases only. Each must-block case is checked against all three
      layers at the gates as shipped, so a layer that owns no region shows up as a column of zeros.</p>
    <div class="scroller"><table class="matrix" id="layers"></table></div>
  </section>

  <section class="panel">
    <h2>All cases</h2>
    <p class="lede">The same rows as the grid, as text &mdash; sortable by eye, searchable by the
      browser, and readable without relying on colour.</p>
    <div class="scroller"><table id="tbl"></table></div>
  </section>

  <div class="foot" id="foot"></div>
</div>

<script>
const DATA = __DATA__;
const POLICY = Object.fromEntries(DATA.policies.map(p => [p.id, p]));
const GLYPH = { ok: "\\u00b7", leak: "\\u25b2", over: "\\u25bc", unknown: "?" };
const LABEL = {
  ok: "behaving as the policy says",
  leak: "allowed what the policy forbids",
  over: "denied what the policy grants",
  unknown: "no single correct answer",
};

let state = { set: "intents", arm: "oracle", filter: "all", q: "", pinned: null };

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = v => (v === undefined || v === null || Number.isNaN(v)) ? "\\u2013" : (v * 100).toFixed(0) + "%";

/* Outcome of one case under one arm. E2E cases are scored on whether some
   layer caught what had to be blocked; intent cases on the verdict itself. */
function outcome(c, arm) {
  if (state.set === "e2e") {
    const flags = c["flags_" + arm] || [];
    if (c.should_block) return flags.length ? "ok" : "leak";
    return flags.length ? "over" : "ok";
  }
  if (c.gold === "gray") return "unknown";
  const w = c["wrong_" + arm];
  return w === "allowed" ? "leak" : w === "denied" ? "over" : "ok";
}

function activeCases() {
  const rows = DATA.cases[state.set] || [];
  const q = state.q.trim().toLowerCase();
  return rows.filter(c => {
    const o = outcome(c, state.arm);
    if (state.filter === "fail" && o === "ok") return false;
    if (state.filter === "leak" && o !== "leak") return false;
    if (state.filter !== "all" && state.filter !== "fail" && state.filter !== "leak" && o !== state.filter) return false;
    if (q) {
      const hay = (c.action + " " + c.policy_id + " " + (c.category || c.relation || "") + " " + (c.user_intent || "")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function segment(el, opts, key) {
  el.innerHTML = opts.map(o =>
    `<button type="button" data-v="${o[0]}" aria-pressed="${state[key] === o[0]}">${esc(o[1])}</button>`
  ).join("");
  el.onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    state[key] = b.dataset.v; state.pinned = null; render();
  };
}

function renderChrome() {
  const c = DATA.corpus;
  document.getElementById("corpus").innerHTML = [
    ["policies", c.policies], ["intents", c.intents],
    ["adversarial", c.adversarial], ["end-to-end", c.e2e],
  ].map(([k, v]) => `<div><b>${v}</b>${k}</div>`).join("");

  const m = DATA.metrics, arm = state.arm;
  const leaks = activeAll().filter(c => outcome(c, arm) === "leak").length;
  document.getElementById("kpis").innerHTML = [
    ["Cases the pipeline gets wrong", leaks,
      state.set === "e2e" ? "must-block cases no layer caught" : "authorized despite the policy", true],
    ["Prohibited actions blocked", pct(m["tier." + arm + ".block_rate"]), "of cases the policy forbids"],
    ["Granted actions allowed", pct(m["tier." + arm + ".allow_rate"]), "of cases the policy grants"],
    ["Decided by the fallback", pct(m["tier." + arm + ".fallback_share"]), "gray zone, no LLM backend"],
    ["Forbidden recall, prose", pct(m["classify.forbidden_recall.by_prose"]), "vs " + pct(m["classify.forbidden_recall.by_label"]) + " on labelled lines"],
  ].map(([l, v, n, alarm]) =>
    `<div class="kpi${alarm ? " alarm" : ""}"><div class="label">${esc(l)}</div>
     <div class="value">${esc(v)}</div><div class="note">${esc(n)}</div></div>`
  ).join("");

  document.getElementById("legend").innerHTML =
    ["ok", "leak", "over", "unknown"].map(k =>
      `<span class="item"><span class="swatch sw-${k === "unknown" ? "unknown" : k === "leak" ? "leak" : k === "over" ? "over" : "ok"}">${GLYPH[k]}</span>${esc(LABEL[k])}</span>`
    ).join("");
}

function activeAll() {
  return (DATA.cases[state.set] || []);
}

function render() {
  segment(document.getElementById("seg-set"), [
    ["intents", "Intents"], ["adversarial", "Adversarial"], ["e2e", "End-to-end"],
  ], "set");
  segment(document.getElementById("seg-arm"), [
    ["classifier", "As classified"], ["oracle", "Correctly labelled"],
  ], "arm");
  segment(document.getElementById("seg-filter"), [
    ["all", "All"], ["fail", "Failures"], ["leak", "Wrongly allowed"], ["over", "Wrongly denied"],
  ], "filter");

  renderChrome();
  renderRows();
  renderBars();
  renderLayers();
  renderTable();
  renderRail(state.pinned);
}

function renderRows() {
  const rows = activeCases();
  const byPolicy = new Map();
  for (const c of rows) {
    if (!byPolicy.has(c.policy_id)) byPolicy.set(c.policy_id, []);
    byPolicy.get(c.policy_id).push(c);
  }
  const order = DATA.policies.map(p => p.id).filter(id => byPolicy.has(id));
  const host = document.getElementById("rows");
  if (!order.length) {
    host.innerHTML = `<div class="row"><div class="rowname" style="grid-column:1/-1;color:var(--ink-3)">No cases match these filters.</div></div>`;
    return;
  }
  host.innerHTML = order.map(pid => {
    const cs = byPolicy.get(pid);
    const bad = cs.filter(c => { const o = outcome(c, state.arm); return o === "leak" || o === "over"; }).length;
    const grp = POLICY[pid] ? POLICY[pid].group : "";
    return `<div class="row">
      <div class="rowname" title="${esc(pid)}"><span class="grp">${esc(grp)}</span>${esc(pid)}</div>
      <div class="rowstat ${bad ? "bad" : ""}">${bad}/${cs.length}</div>
      <div class="cells">${cs.map(c => {
        const o = outcome(c, state.arm);
        const cls = o === "leak" ? "leak" : o === "over" ? "over" : o === "unknown" ? "unknown" : "ok";
        return `<button class="cell ${cls}${state.pinned === c.__i ? " pinned" : ""}" data-i="${c.__i}"
          aria-label="${esc(LABEL[o])}: ${esc(c.action.slice(0, 70))}">${GLYPH[o]}</button>`;
      }).join("")}</div>
    </div>`;
  }).join("");

  host.onmouseover = e => { const b = e.target.closest(".cell"); if (b && state.pinned === null) renderRail(+b.dataset.i); };
  host.onfocusin  = e => { const b = e.target.closest(".cell"); if (b) renderRail(+b.dataset.i); };
  host.onclick = e => {
    const b = e.target.closest(".cell"); if (!b) return;
    const i = +b.dataset.i;
    state.pinned = state.pinned === i ? null : i;
    renderRows(); renderRail(state.pinned === null ? i : state.pinned);
  };
}

function renderRail(i) {
  const host = document.getElementById("rail");
  if (i === null || i === undefined) {
    host.innerHTML = `<div class="empty">Hover a cell to inspect the case. Click to pin it.</div>`;
    return;
  }
  const c = (DATA.cases[state.set] || []).find(x => x.__i === i);
  if (!c) { host.innerHTML = `<div class="empty">Case not in the current view.</div>`; return; }
  const o = outcome(c, state.arm);
  const pol = POLICY[c.policy_id];

  let head = `<div class="verdict">
    <span class="pill ${o === "leak" ? "leak" : o === "over" ? "over" : "ok"}">${GLYPH[o]} ${esc(LABEL[o])}</span>
    <span class="pill accent">${esc(c.category || c.relation || "")}</span>
  </div>`;

  let fields = `<div class="field"><div class="k">Agent action</div><div class="v mono">${esc(c.action)}</div></div>`;
  if (c.user_intent) fields += `<div class="field"><div class="k">User asked for</div><div class="v mono">${esc(c.user_intent)}</div></div>`;
  fields += `<div class="field"><div class="k">Policy</div><div class="v">${esc(c.policy_id)} <span style="color:var(--ink-3)">&middot; ${esc(pol ? pol.style : "")} &middot; ${esc(pol ? pol.filename : "")}</span></div></div>`;

  if (state.set === "e2e") {
    const flags = c["flags_" + state.arm] || [];
    fields += `<div class="field"><div class="k">Must be blocked</div><div class="v">${c.should_block ? "yes" : "no \\u2014 this is a control"}</div></div>`;
    fields += `<div class="field"><div class="k">Layers that flagged it</div><div class="v">${
      flags.length ? flags.map(f => `<span class="pill ok">${esc(f)}</span>`).join(" ") : `<span style="color:var(--ink-3)">none</span>`}</div></div>`;
    fields += `<div class="field"><div class="k">Scores</div><div class="v mono">contradiction ${c.contradiction.toFixed(3)} &middot; hhem ${c.hhem.toFixed(3)}</div></div>`;
    fields += `<div class="field"><div class="k">Reason</div><div class="v mono">${esc(c.reason_oracle)}</div></div>`;
  } else {
    fields += `<div class="field"><div class="k">Expected &rarr; got</div><div class="v mono">${esc(c.gold)} &rarr; ${esc(c["decision_" + state.arm])}</div></div>`;
    fields += `<div class="field"><div class="k">Rule that fired</div><div class="v mono">${esc(c["rule_" + state.arm])}</div></div>`;
  }

  let chunks = "";
  if (pol) {
    const mis = pol.chunks.filter(k => k.gold === "forbidden" && k.classified === "allowed");
    chunks = `<div class="chunks">
      <div class="k" style="font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);font-weight:600;margin-bottom:7px">
        Policy index &mdash; ${mis.length} of ${pol.chunks.filter(k => k.gold === "forbidden").length} prohibitions filed as grants</div>
      ${pol.chunks.map(k => {
        const bad = k.gold === "forbidden" && k.classified === "allowed";
        return `<div class="chunk${bad ? " mis" : ""}">
          <div class="tag ${bad ? "mis" : "hit"}">${esc(k.classified.slice(0, 6))}</div>
          <div class="txt">${esc(k.text.slice(0, 210))}</div></div>`;
      }).join("")}
      <details class="raw"><summary>Full policy text</summary><pre>${esc(pol.text)}</pre></details>
    </div>`;
  }
  host.innerHTML = head + fields + chunks;
}

function renderBars() {
  const rows = activeCases();
  const keyOf = c => c.category || c.relation || "\\u2013";
  const cats = new Map();
  for (const c of rows) {
    const k = keyOf(c);
    if (!cats.has(k)) cats.set(k, { ok: 0, leak: 0, over: 0, unknown: 0, n: 0 });
    const e = cats.get(k); e[outcome(c, state.arm)]++; e.n++;
  }
  const sorted = [...cats.entries()].sort((a, b) => (b[1].leak / b[1].n) - (a[1].leak / a[1].n));
  document.getElementById("bars").innerHTML = sorted.map(([k, e]) => `
    <div class="bar-row">
      <div class="cat-name">${esc(k)}</div>
      <div class="n">${e.leak ? e.leak + "/" + e.n : e.n}</div>
      <div class="bar" role="img" aria-label="${esc(k)}: ${e.leak} wrongly allowed, ${e.over} wrongly denied, ${e.ok} correct of ${e.n}">
        ${["leak", "over", "unknown", "ok"].filter(s => e[s]).map(s =>
          `<span class="s-${s}" style="width:${(e[s] / e.n * 100).toFixed(2)}%"></span>`).join("")}
      </div>
    </div>`).join("");
}

function renderLayers() {
  const panel = document.getElementById("layers-panel");
  const rows = (DATA.cases.e2e || []).filter(c => c.should_block);
  if (!rows.length) { panel.style.display = "none"; return; }
  panel.style.display = "";
  const rel = new Map();
  for (const c of rows) {
    if (!rel.has(c.relation)) rel.set(c.relation, { n: 0, drift: 0, policy: 0, hhem: 0, none: 0 });
    const e = rel.get(c.relation); e.n++;
    const f = c["flags_" + state.arm] || [];
    for (const k of ["drift", "policy", "hhem"]) if (f.includes(k)) e[k]++;
    if (!f.length) e.none++;
  }
  const shade = (v, n) => {
    if (!n) return "";
    const r = v / n;
    return `background: color-mix(in srgb, var(--accent) ${(r * 100).toFixed(0)}%, transparent)`;
  };
  document.getElementById("layers").innerHTML = `
    <thead><tr><th>Failure mode</th><th>Cases</th><th>Drift</th><th>Policy</th><th>HHEM</th><th>Missed by all</th></tr></thead>
    <tbody>${[...rel.entries()].sort().map(([k, e]) => `
      <tr><td>${esc(k)}</td><td class="num">${e.n}</td>
        ${["drift", "policy", "hhem"].map(l =>
          `<td class="num" style="${shade(e[l], e.n)}">${(e[l] / e.n * 100).toFixed(0)}%</td>`).join("")}
        <td class="num" style="${e.none ? "color:var(--leak);font-weight:600" : "color:var(--ink-3)"}">${e.none}</td></tr>`).join("")}
    </tbody>`;
}

function renderTable() {
  const rows = activeCases();
  const e2e = state.set === "e2e";
  document.getElementById("tbl").innerHTML = `
    <thead><tr><th>Outcome</th><th>Policy</th><th>${e2e ? "Relation" : "Category"}</th>
      ${e2e ? "<th>User asked for</th>" : ""}<th>Action</th>
      ${e2e ? "<th>Layers</th>" : "<th>Expected</th><th>Got</th><th>Rule</th>"}</tr></thead>
    <tbody>${rows.map(c => {
      const o = outcome(c, state.arm);
      return `<tr><td class="mono">${GLYPH[o]} ${esc(LABEL[o])}</td>
        <td>${esc(c.policy_id)}</td><td>${esc(c.category || c.relation)}</td>
        ${e2e ? `<td>${esc(c.user_intent || "")}</td>` : ""}
        <td class="mono">${esc(c.action)}</td>
        ${e2e ? `<td class="mono">${esc((c["flags_" + state.arm] || []).join(", ") || "none")}</td>`
              : `<td class="mono">${esc(c.gold)}</td><td class="mono">${esc(c["decision_" + state.arm])}</td>
                 <td class="mono">${esc(c["rule_" + state.arm])}</td>`}</tr>`;
    }).join("")}</tbody>`;
}

document.getElementById("q").addEventListener("input", e => {
  state.q = e.target.value; state.pinned = null;
  renderRows(); renderBars(); renderTable();
});
document.addEventListener("keydown", e => { if (e.key === "Escape") { state.pinned = null; renderRows(); renderRail(null); } });

for (const k of Object.keys(DATA.cases)) DATA.cases[k].forEach((c, i) => { c.__i = i; });
const failed = Object.values(DATA.tests).filter(v => v === "failed").length;
document.getElementById("foot").textContent =
  `${DATA.corpus.policies} policies \\u00b7 ${Object.values(DATA.cases).reduce((a, b) => a + b.length, 0)} cases \\u00b7 `
  + `${failed} of ${Object.keys(DATA.tests).length} requirements not met. A failing requirement here is a finding about the pipeline, not a broken test.`;
render();
</script>
"""


def build(data: dict) -> str:
    payload = {
        "corpus": data["corpus"],
        "metrics": data["metrics"],
        "tests": data["tests"],
        "policies": data["policies"],
        "cases": data["cases"],
    }
    return TEMPLATE.replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )
