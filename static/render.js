/* =====================================================================
 * render.js — PURE presentation layer for the Survey Auto-Tagger UI.
 *
 * Every function here takes (data, view-state) and returns an HTML string.
 * It performs NO fetches, touches NO DOM and holds NO mutable app state —
 * the single exception being the taxonomy cache inside ST.dims, which is
 * seeded once by app.js from GET /api/taxonomy.
 *
 * app.js owns state, fetching and DOM writes, and calls into this file.
 * Every interactive element is marked with data-action="..." (never an
 * inline onclick); app.js runs one delegated click listener keyed on it.
 *
 * Markup follows the UI design contract v1:
 *   .chip   — neutral micro-label   (--mono / --tint / --free)
 *   .status — semantic pill         (--ok/--warn/--danger/--info/--muted)
 *   .toggle — interactive filter chip (on|off)
 *   .src    — source label: mono abbreviation + coloured 6px dot
 *   data-conf="hi|mid|low|none" on .tag-card / .qtag drives the left rule
 * The old .pill / .ev-chip / .mv / .badge / .lowconf families are gone.
 *
 * Plain browser JS — no modules, no build step, no framework.
 * ===================================================================== */

window.ST = window.ST || {};

(function (ST) {
  "use strict";

  /* ==================================================================
   * ST.util — formatting & provenance primitives
   * ================================================================== */

  const ALL_SOURCES = ["deterministic", "statistical", "hybrid", "heuristic", "llm"];

  const SOURCE_METHOD_HINT = {
    deterministic: "value derived by a fixed rule from survey structure fields",
    statistical:   "value derived from response statistics (counts, cadence, distributions)",
    hybrid:        "value combined from multiple methods (rule + LLM/statistics)",
    heuristic:     "fallback / default applied because no stronger signal was available",
    llm:           "value inferred by the LLM from question & survey context",
  };

  /* ≤4 characters, per the contract: the label is a mono tag, not a sentence.
   * The tooltip carries the meaning. */
  const SOURCE_ABBR = {
    deterministic: "DET",
    statistical:   "STAT",
    hybrid:        "HYB",
    heuristic:     "HEUR",
    llm:           "LLM",
    unknown:       "UNK",
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[c]);
  }

  const labelFor = (k) => String(k).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  const sourceClass = (s) => ALL_SOURCES.includes(s) ? s : "unknown";

  const isLowConf = (tag) => (tag && typeof tag.confidence === "number" && tag.confidence < 0.6);

  const isEmpty = (v) => v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);

  /* The confidence rule that replaced `.lowconf`. A card whose tag carries no
   * numeric confidence still gets an attribute ("none") so CSS can draw a
   * transparent rule and keep every card in the grid optically aligned. */
  /* `dflt` is what an ABSENT confidence means at this level. Question tags omit
   * the field when it is exactly 1.0 (pipeline/assembly.py drops it to save
   * bytes), so there a missing value means *full* confidence — passing
   * dflt=1 keeps the rule green instead of blanking it on the surest tags.
   * Project tags always carry the field, so they pass nothing. */
  function confBucket(tag, dflt) {
    let c = tag && tag.confidence;
    if (typeof c !== "number" || !isFinite(c)) c = dflt;
    if (typeof c !== "number" || !isFinite(c)) return "none";
    if (c >= 0.75) return "hi";
    if (c >= 0.5) return "mid";
    return "low";
  }

  const fmtValue = (v) => {
    if (isEmpty(v)) return { html: '<span class="empty">— not set —</span>', empty: true };
    if (Array.isArray(v)) {
      return {
        html: `<div class="multi-list">${v.map(x => `<span class="chip">${escapeHtml(String(x))}</span>`).join("")}</div>`,
        empty: false
      };
    }
    return { html: escapeHtml(String(v)), empty: false };
  };

  function confBar(conf) {
    if (typeof conf !== "number" || !isFinite(conf)) return "";
    const pct = Math.max(0, Math.min(1, conf)) * 100;
    const cls = conf < 0.5 ? "low" : (conf < 0.75 ? "mid" : "");
    return `<span class="conf"><span class="conf-bar"><i class="${cls}" style="width:${pct}%"></i></span><span>${(conf * 100).toFixed(0)}%</span></span>`;
  }

  /* Source is provenance, not decoration: one 6px dot carries the hue and the
   * label stays muted mono. Six background fills competing with the value text
   * was the single loudest thing on the old cards. */
  function srcBadge(src) {
    const cls = sourceClass(src);
    const hint = SOURCE_METHOD_HINT[cls] || "unknown assignment method";
    const abbr = SOURCE_ABBR[cls] || "UNK";
    return `<span class="src src--${cls}" title="${escapeHtml(hint)}"><i class="src-dot"></i>${escapeHtml(abbr)}</span>`;
  }

  /* A non-default `status` is itself an explanation — it says the value is
   * provisional, or that the tag never got one. "assigned" is the default and
   * carries no information, so it renders nothing. */
  const STATUS_HINT = {
    pending_llm: "reserved by a rule; the LLM pass was meant to fill this in but did not",
    low_confidence_assigned: "the LLM was unsure — value kept and flagged for review",
    failed: "the tagger raised an error; see failure_reason",
    skipped: "the dimension does not apply to this question",
  };

  const STATUS_TONE = {
    pending_llm: "info",
    low_confidence_assigned: "warn",
    failed: "danger",
    skipped: "muted",
  };

  function statusBadge(tag) {
    const st = tag && tag.status;
    // Anything non-string is malformed input, not a status worth a badge —
    // rendering it would put String(value) in front of the user.
    if (typeof st !== "string" || !st || st === "assigned") return "";
    const hint = STATUS_HINT[st] || "non-standard tag status";
    const tone = STATUS_TONE[st] || "muted";
    const extra = tag.failure_reason ? ` — ${tag.failure_reason}` : "";
    return `<span class="status status--${tone}" title="${escapeHtml(hint + extra)}">${escapeHtml(st.replace(/_/g, " "))}</span>`;
  }

  /* ---------- evidence / provenance ----------
   * A tag's provenance comes in three shapes:
   *   legacy string  — evidence: "rs_type=3 (CES)"
   *   legacy LLM     — reasoning: "<model rationale>" (no evidence field)
   *   typed object   — evidence: {type, detail, rule_id, inputs, model, stage,
   *                    rationale, quote, measure, observed, threshold, components}
   * All three render: chips for the machine-readable bits, the detail line, and a
   * collapsible block for LLM rationale so long text doesn't swamp the card.
   */
  function evidenceParts(tag) {
    const ev = tag.evidence;
    const obj = (ev && typeof ev === "object" && !Array.isArray(ev)) ? ev : null;
    return {
      detail: obj ? (obj.detail || "") : (typeof ev === "string" ? ev : ""),
      reasoning: (obj && obj.rationale) || tag.reasoning || "",
      quote: (obj && obj.quote) || "",
      type: obj ? obj.type : null,
      ruleId: (obj && obj.rule_id) || tag.rule_id || null,
      model: (obj && obj.model) || tag.model || null,
      stage: obj ? obj.stage : null,
      inputs: (obj && obj.inputs && typeof obj.inputs === "object") ? obj.inputs : null,
      components: (obj && Array.isArray(obj.components)) ? obj.components : null,
      statistic: obj && obj.measure
        ? `${obj.measure}: ${obj.observed ?? "?"}${obj.threshold !== undefined ? ` (threshold ${obj.threshold})` : ""}`
        : "",
    };
  }

  function evidenceHtml(tag, cls) {
    if (!tag) return "";
    const p = evidenceParts(tag);
    const chips = [];
    // Only the evidence *type* is tinted — it is the one bit that classifies the
    // whole block. Everything else is a neutral mono chip.
    if (p.type)   chips.push(`<span class="chip chip--tint" title="evidence type">${escapeHtml(p.type)}</span>`);
    if (p.ruleId) chips.push(`<span class="chip chip--mono" title="deterministic rule id">${escapeHtml(p.ruleId)}</span>`);
    if (p.model)  chips.push(`<span class="chip chip--mono" title="LLM model">${escapeHtml(p.model)}</span>`);
    if (p.stage)  chips.push(`<span class="chip chip--mono" title="pipeline stage">${escapeHtml(p.stage)}</span>`);
    if (p.inputs) {
      for (const [k, v] of Object.entries(p.inputs)) {
        chips.push(`<span class="chip chip--mono" title="rule input">${escapeHtml(k)}=${escapeHtml(String(v))}</span>`);
      }
    }
    if (p.components) {
      for (const c of p.components) {
        chips.push(`<span class="chip chip--mono" title="hybrid component">${escapeHtml((c && (c.source || c.type)) || "?")}</span>`);
      }
    }
    const bits = [
      chips.length ? `<div class="ev-chips">${chips.join("")}</div>` : "",
      p.statistic ? `<div>${escapeHtml(p.statistic)}</div>` : "",
      p.detail ? `<div>${escapeHtml(p.detail)}</div>` : "",
      p.quote ? `<div class="ev-quote">&ldquo;${escapeHtml(p.quote)}&rdquo;</div>` : "",
      p.reasoning
        ? `<details class="ev-reasoning"><summary>${sourceClass(tag.source) === "llm" ? "LLM reasoning" : "Reasoning"}</summary><div>${escapeHtml(p.reasoning)}</div></details>`
        : "",
    ].filter(Boolean);
    return bits.length ? `<div class="${cls}">${bits.join("")}</div>` : "";
  }

  function evidenceTitle(tag) {  // plain-text summary for table-cell tooltips
    if (!tag) return "";
    const p = evidenceParts(tag);
    return [p.statistic, p.detail, p.reasoning].filter(Boolean).join(" — ");
  }

  /* ---------- small formatters ---------- */

  const pad2 = (n) => String(n).padStart(2, "0");

  function fmtMs(ms) {
    const n = Number(ms);
    if (!isFinite(n) || n < 0) return "";
    if (n < 1000) return `${Math.round(n)}ms`;
    if (n < 60000) return `${(n / 1000).toFixed(1)}s`;
    const m = Math.floor(n / 60000);
    const s = Math.round((n % 60000) / 1000);
    return `${m}m ${pad2(s)}s`;
  }

  // Never throws: a missing / unparseable timestamp renders as "".
  function fmtDate(iso) {
    if (iso === null || iso === undefined || iso === "") return "";
    try {
      const d = new Date(iso);
      const t = d.getTime();
      if (typeof t !== "number" || isNaN(t)) return "";
      return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} `
           + `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
    } catch (_e) {
      return "";
    }
  }

  function fmtElapsed(msSinceStart) {
    const n = Number(msSinceStart);
    if (!isFinite(n) || n < 0) return "0:00";
    const total = Math.floor(n / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h ? `${h}:${pad2(m)}:${pad2(s)}` : `${m}:${pad2(s)}`;
  }

  /* Integer percentages that always sum to exactly 100 (largest-remainder), so
   * a meter's segments tile its full width with no sliver of background
   * showing through at the right edge. An all-zero input stays all-zero. */
  function pctParts(counts) {
    const nums = (Array.isArray(counts) ? counts : []).map(n => (isFinite(n) && n > 0) ? n : 0);
    const total = nums.reduce((a, b) => a + b, 0);
    if (!total) return nums.map(() => 0);
    const out = nums.map(n => Math.floor(n / total * 100));
    let rem = 100 - out.reduce((a, b) => a + b, 0);
    const byRemainder = nums
      .map((n, i) => [(n / total * 100) - Math.floor(n / total * 100), i])
      .sort((a, b) => b[0] - a[0]);
    for (let i = 0; i < byRemainder.length && rem > 0; i++) {
      if (nums[byRemainder[i][1]] > 0) { out[byRemainder[i][1]]++; rem--; }
    }
    return out;
  }

  function passesFilter(tag, st) {
    if (!tag) return false;
    const s = st || {};
    const sf = s.sourceFilter;
    if (sf && typeof sf.has === "function" && !sf.has(sourceClass(tag.source))) return false;
    if (s.lowConfOnly && !isLowConf(tag)) return false;
    return true;
  }

  // Are any tag-level filters actually narrowing the result set?
  function filtersActive(st) {
    const s = st || {};
    const sf = s.sourceFilter;
    const narrowed = !!(sf && typeof sf.size === "number" && sf.size < ALL_SOURCES.length);
    return !!s.lowConfOnly || narrowed;
  }

  ST.util = {
    ALL_SOURCES,
    SOURCE_METHOD_HINT,
    SOURCE_ABBR,
    escapeHtml,
    labelFor,
    sourceClass,
    isLowConf,
    confBucket,
    isEmpty,
    fmtValue,
    confBar,
    srcBadge,
    statusBadge,
    evidenceParts,
    evidenceHtml,
    evidenceTitle,
    fmtMs,
    fmtDate,
    fmtElapsed,
    pctParts,
    passesFilter,
    filtersActive,
  };

  /* ==================================================================
   * ST.dims — taxonomy-driven dimension ordering.
   *
   * Nothing is hardcoded: the key order of the GET /api/taxonomy response
   * IS the canonical dimension order (it mirrors config/taxonomy.yaml).
   * The three module-level caches below are the ONLY mutable state in
   * this file.
   * ================================================================== */

  let _taxonomy = null;        // {dim: {level, description, allowed_values, multi_label, user_defined, canonical_values}}
  let _projectDims = [];       // level === "project", in taxonomy order
  let _questionDims = [];      // level === "question", in taxonomy order

  /* The handful of dimensions that answer "what is this survey / question?".
   * Everything else is detail and lives behind the disclosure. Ordered as
   * written — this list IS the reading order, not a filter over taxonomy order. */
  const PRIMARY_PROJECT_DIMS = [
    "project_type", "project_purpose", "audience_type",
    "relationship_type", "industry_vertical", "survey_sub_type",
  ];
  const PRIMARY_QUESTION_DIMS = [
    "topic_theme", "role_intent", "metric_type", "metric_name", "journey_stage",
  ];

  function setTaxonomy(taxonomy) {
    if (!taxonomy || typeof taxonomy !== "object" || Array.isArray(taxonomy)) {
      _taxonomy = null;
      _projectDims = [];
      _questionDims = [];
      return;
    }
    _taxonomy = taxonomy;
    _projectDims = [];
    _questionDims = [];
    for (const key of Object.keys(taxonomy)) {
      const dim = taxonomy[key] || {};
      if (dim.level === "project") _projectDims.push(key);
      else if (dim.level === "question") _questionDims.push(key);
    }
  }

  // Degrades gracefully: the 6 tenant-level dimensions are legitimately
  // absent from /api/taxonomy's project/question views, and app.js may call
  // this before the taxonomy has loaded. Never throws.
  function dimMeta(key) {
    const dim = (_taxonomy && _taxonomy[key]) || null;
    if (!dim) {
      return { label: labelFor(key), description: "", multi: false, free: false, allowed: [] };
    }
    return {
      label: labelFor(key),
      description: dim.description || "",
      multi: !!dim.multi_label,
      free: !!dim.user_defined,
      allowed: Array.isArray(dim.allowed_values) ? dim.allowed_values : [],
    };
  }

  function dimsForLevel(level) {
    if (level === "project") return _projectDims;
    if (level === "question") return _questionDims;
    return [];
  }

  // Known dims for the level that are present in tagsObj (taxonomy order),
  // then any unknown keys sorted alphabetically.
  function dimOrder(tagsObj, level) {
    const tags = tagsObj || {};
    const known = dimsForLevel(level);
    const knownSet = new Set(known);
    const head = known.filter(k => Object.prototype.hasOwnProperty.call(tags, k));
    const tail = Object.keys(tags).filter(k => !knownSet.has(k)).sort();
    return head.concat(tail);
  }

  /* Split an already-ordered, already-filtered key list into the primary block
   * (in the PRIMARY_* order, which is editorial) and the remainder (which keeps
   * dimOrder()'s taxonomy ordering). */
  function splitPrimary(keys, primaryList) {
    const present = new Set(Array.isArray(keys) ? keys : []);
    const primary = primaryList.filter(k => present.has(k));
    const primarySet = new Set(primary);
    const secondary = (Array.isArray(keys) ? keys : []).filter(k => !primarySet.has(k));
    return { primary, secondary };
  }

  // Question dims appearing on at least one question, in taxonomy order.
  function dimsPresent(questions) {
    const list = Array.isArray(questions) ? questions : [];
    const seen = new Set();
    for (const q of list) {
      const tags = (q && q.tags) || {};
      for (const k of Object.keys(tags)) seen.add(k);
    }
    const known = _questionDims.filter(k => seen.has(k));
    const knownSet = new Set(_questionDims);
    const extra = [...seen].filter(k => !knownSet.has(k)).sort();
    return known.concat(extra);
  }

  // Every question dim in the taxonomy, plus any off-taxonomy key seen on
  // the supplied questions (used by the table's "All columns" mode).
  function allQuestionCols(questions) {
    const known = _questionDims.slice();
    const knownSet = new Set(known);
    const extra = [];
    for (const q of (Array.isArray(questions) ? questions : [])) {
      for (const k of Object.keys((q && q.tags) || {})) {
        if (!knownSet.has(k)) { knownSet.add(k); extra.push(k); }
      }
    }
    return known.concat(extra.sort());
  }

  ST.dims = {
    set: setTaxonomy,
    raw: () => _taxonomy,
    meta: dimMeta,
    order: dimOrder,
    present: dimsPresent,
    projectDims: () => _projectDims.slice(),
    questionDims: () => _questionDims.slice(),
    allQuestionCols,
    PRIMARY_PROJECT_DIMS,
    PRIMARY_QUESTION_DIMS,
    splitPrimary,
  };

  /* ==================================================================
   * ST.render — every function returns an HTML string.
   * ================================================================== */

  /* ---------- shared chrome ---------- */

  function emptyState(title, body, actionsHtml) {
    return `<div class="empty-state">
      <h2>${escapeHtml(title || "")}</h2>
      ${body ? `<p>${escapeHtml(body)}</p>` : ""}
      ${actionsHtml ? `<div class="btn-row">${actionsHtml}</div>` : ""}
    </div>`;
  }

  function errorState(title, detail, retryAction) {
    const detailText = detail === null || detail === undefined ? "" : String(detail);
    return `<div class="lc-banner warn">
      <strong>${escapeHtml(title || "Something went wrong")}</strong>
      ${detailText
        ? `<details class="ev-reasoning"><summary>Details</summary><div class="mono">${escapeHtml(detailText)}</div></details>`
        : ""}
      ${retryAction
        ? `<div class="btn-row"><button class="btn sm" data-action="${escapeHtml(retryAction)}">Retry</button></div>`
        : ""}
    </div>`;
  }

  function emptyHintHtml(text) {
    return `<div class="empty-state"><p>${escapeHtml(text)}</p></div>`;
  }

  /* ---------- skeleton loaders ----------
   * A shimmering block of roughly the right shape beats the word "Loading…":
   * the layout does not jump when the real content lands. Pure CSS animation,
   * inert under prefers-reduced-motion (app.css owns that). */
  function skelLine(n) {
    const count = Math.max(1, Number(n) || 1);
    let out = "";
    for (let i = 0; i < count; i++) out += `<div class="skel skel--line"></div>`;
    return out;
  }

  function skelCard(n) {
    const count = Math.max(1, Number(n) || 1);
    let out = "";
    for (let i = 0; i < count; i++) out += `<div class="skel skel--card"></div>`;
    return out;
  }

  /* A whole-surface placeholder: header lines, then a grid of cards. */
  function skeleton(opts) {
    const o = opts || {};
    const lines = o.lines === undefined ? 2 : o.lines;
    const cards = o.cards === undefined ? 6 : o.cards;
    return `<div class="skel-wrap">
      ${skelLine(lines)}
      <div class="tag-grid">${skelCard(cards)}</div>
    </div>`;
  }

  function renderLegend() {
    return `<div class="legend">${ALL_SOURCES.map(s => srcBadge(s)).join("")}</div>`;
  }

  function kvSpan(label, value) {
    return `<span><strong>${escapeHtml(label)}</strong> ${escapeHtml(String(value))}</span>`;
  }

  function rawJson(obj) {
    let text;
    try {
      text = JSON.stringify(obj, null, 2);
    } catch (_e) {
      text = String(obj);
    }
    if (text === undefined) text = String(obj);
    return `<pre class="profile-json">${escapeHtml(text)}</pre>`;
  }

  /* ---------- survey header ---------- */

  function surveyHeader(survey) {
    const s = survey || {};
    const meta = s.metadata || {};
    const bits = [];

    if (!isEmpty(s.survey_no))      bits.push(kvSpan("Survey #", s.survey_no));
    if (!isEmpty(s.tenant_id))      bits.push(kvSpan("Tenant", s.tenant_id));
    if (!isEmpty(s.zarca_id))       bits.push(kvSpan("Zarca ID", s.zarca_id));
    if (!isEmpty(s.schema_version)) bits.push(kvSpan("Schema", s.schema_version));

    const gen = fmtDate(s.generated_at);
    if (gen) bits.push(kvSpan("Generated", gen));

    const qCount = Array.isArray(s.questions)
      ? s.questions.length
      : (Array.isArray(s.question_tags) ? s.question_tags.length : meta.total_questions);
    if (!isEmpty(qCount)) bits.push(kvSpan("Questions", qCount));

    if (!isEmpty(meta.llm_calls_made)) bits.push(kvSpan("LLM calls", meta.llm_calls_made));

    const took = fmtMs(meta.processing_time_ms);
    if (took) bits.push(kvSpan("Processing", took));

    const lcFlags = Array.isArray(meta.low_confidence_flags) ? meta.low_confidence_flags : [];

    return `<div class="survey-header">
      <h1>${escapeHtml(s.survey_name || "(untitled survey)")}</h1>
      ${bits.length ? `<div class="survey-meta">${bits.join("")}</div>` : ""}
    </div>
    ${lcFlags.length ? `<div class="lc-banner info">
      Low-confidence flags from the pipeline: ${lcFlags.map(f => `<span class="chip">${escapeHtml(String(f))}</span>`).join(" ")}
    </div>` : ""}`;
  }

  /* ---------- filter bar ---------- */

  /* Every toggle is keyboard-reachable; app.js binds Enter/Space to the same
   * delegated handler the click listener uses. */
  function toggleChip(on, action, dataAttrs, label, title, cls) {
    const attrs = Object.entries(dataAttrs || {})
      .map(([k, v]) => ` data-${escapeHtml(k)}="${escapeHtml(String(v))}"`).join("");
    return `<span class="toggle${cls ? ` ${cls}` : ""} ${on ? "on" : "off"}" role="button" tabindex="0"`
      + ` data-action="${escapeHtml(action)}"${attrs}`
      + `${title ? ` title="${escapeHtml(title)}"` : ""}>${label}</span>`;
  }

  function filterBar(st, opts) {
    const s = st || {};
    const o = opts || {};
    const sf = s.sourceFilter;
    const has = (x) => !sf || typeof sf.has !== "function" || sf.has(x);

    // `toggle--src`: these are subtractive filters that start all-on, so they
    // get the neutral/suppressed treatment rather than the accent tint.
    const srcChips = ALL_SOURCES.map(x =>
      toggleChip(has(x), "toggle-source", { source: x }, srcBadge(x),
                 SOURCE_METHOD_HINT[x], "toggle--src")
    ).join("");

    const searchHtml = o.questionControls
      ? `<input type="text" id="qSearch" placeholder="Search question text / number / tag value…" value="${escapeHtml(s.questionSearch || "")}" />`
      : "";

    const viewChips = o.questionControls
      ? `<div class="filter-group">
          <span class="filter-label">View</span>
          ${toggleChip((s.questionView || "cards") === "cards", "set-view", { view: "cards" }, "Cards")}
          ${toggleChip(s.questionView === "table", "set-view", { view: "table" }, "Table")}
          ${toggleChip(false, "expand-all", {}, "Expand all")}
          ${toggleChip(false, "collapse-all", {}, "Collapse all")}
        </div>`
      : "";

    const colChips = o.columnControls
      ? `<div class="filter-group">
          <span class="filter-label">Columns</span>
          ${toggleChip((s.tableCols || "present") === "present", "set-cols", { cols: "present" }, "Present")}
          ${toggleChip(s.tableCols === "all", "set-cols", { cols: "all" }, "All")}
        </div>`
      : "";

    const journeyChip = o.journeyToggle
      ? `<div class="filter-group">
          ${toggleChip(!!s.includeCandidates, "toggle-candidates", {}, "Show ranked candidates")}
        </div>`
      : "";

    return `<div class="filters">
      ${searchHtml}
      <div class="filter-group"><span class="filter-label">Source</span>${srcChips}</div>
      <div class="filter-group">
        ${toggleChip(!!s.lowConfOnly, "toggle-lowconf", {}, "Low confidence only")}
      </div>
      ${viewChips}
      ${colChips}
      ${journeyChip}
      ${renderLegend()}
    </div>`;
  }

  /* ---------- tag cards ---------- */

  function tagNameHtml(key, cls) {
    const meta = dimMeta(key);
    const title = meta.description ? ` title="${escapeHtml(meta.description)}"` : "";
    const freeChip = meta.free
      ? ` <span class="chip chip--free" title="user-defined dimension: values are not constrained to a taxonomy list">free text</span>`
      : "";
    return `<div class="${cls}"${title}>${escapeHtml(meta.label)}${freeChip}</div>`;
  }

  function projectTagCard(key, tag, primary) {
    const t = tag || {};
    const v = fmtValue(t.value);
    return `
      <div class="tag-card${primary ? " tag-card--primary" : ""}" data-conf="${confBucket(t)}" data-dim="${escapeHtml(String(key))}">
        ${tagNameHtml(key, "tag-name")}
        <div class="tag-value ${v.empty ? "empty" : ""}">${v.html}</div>
        <div class="tag-foot">
          ${srcBadge(t.source)}
          ${statusBadge(t)}
          ${confBar(t.confidence)}
        </div>
        ${evidenceHtml(t, "evidence")}
      </div>
    `;
  }

  function questionTagCard(key, tag, primary) {
    const t = tag || {};
    const v = fmtValue(t.value);
    return `
      <div class="qtag${primary ? " qtag--primary" : ""}" data-conf="${confBucket(t, 1)}" data-dim="${escapeHtml(String(key))}">
        ${tagNameHtml(key, "qtag-name")}
        <div class="qtag-value ${v.empty ? "empty" : ""}">${v.html}</div>
        <div class="qtag-meta">
          ${srcBadge(t.source)}
          ${statusBadge(t)}
          ${confBar(typeof t.confidence === "number" ? t.confidence : 1)}
        </div>
        ${evidenceHtml(t, "qtag-evidence")}
      </div>
    `;
  }

  // A card carrying a bare scalar/array (no source, no confidence) — used by
  // the tenant-profile summary, whose values are NOT tag objects.
  function plainCard(label, value) {
    const v = fmtValue(value);
    return `
      <div class="tag-card" data-conf="none">
        <div class="tag-name">${escapeHtml(label)}</div>
        <div class="tag-value ${v.empty ? "empty" : ""}">${v.html}</div>
      </div>
    `;
  }

  /* Primary cards up front; the long tail behind a disclosure whose label
   * states the real count. With nothing primary present there is nothing to
   * disclose *from*, so the whole set renders flat rather than hiding
   * everything behind a summary the user would always have to open. */
  function disclosedGrid(primaryHtml, secondaryHtml, secondaryCount, gridCls) {
    if (!secondaryCount) return `<div class="${gridCls}">${primaryHtml}</div>`;
    if (!primaryHtml) return `<div class="${gridCls}">${secondaryHtml}</div>`;
    return `<div class="${gridCls}">${primaryHtml}</div>
      <details class="dim-more">
        <summary>Show all ${secondaryCount} dimension${secondaryCount === 1 ? "" : "s"}</summary>
        <div class="${gridCls}">${secondaryHtml}</div>
      </details>`;
  }

  /* ---------- project tags ---------- */

  // journey_stage is tagged per-question; roll it up here so the project view
  // shows which journey stage(s) the survey belongs to. Not a pipeline tag, so
  // it ignores the source / low-confidence filters.
  function derivedStageCard(survey) {
    const qs = questionList(survey);
    const counts = new Map();
    let confSum = 0, confN = 0;
    for (const q of qs) {
      const t = ((q && q.tags) || {}).journey_stage;
      if (!t || isEmpty(t.value)) continue;
      counts.set(t.value, (counts.get(t.value) || 0) + 1);
      if (typeof t.confidence === "number") { confSum += t.confidence; confN++; }
    }
    if (!counts.size) return "";
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    const chips = [...counts.entries()].sort((a, b) => b[1] - a[1])
      .map(([name, n]) => `<span class="chip">${escapeHtml(String(name))}${counts.size > 1 ? ` <span class="micro">&times;${n}</span>` : ""}</span>`)
      .join("");
    return `
      <div class="tag-card tag-card--primary" data-conf="none">
        <div class="tag-name">Journey Stage <span class="micro">(derived)</span></div>
        <div class="tag-value"><div class="multi-list">${chips}</div></div>
        <div class="tag-foot">
          <span class="chip" title="rolled up from question-level journey_stage tags">derived</span>
          ${confN ? confBar(confSum / confN) : ""}
        </div>
        <div class="evidence">Rolled up from the journey_stage tag on ${total} of ${qs.length} question${qs.length === 1 ? "" : "s"}.</div>
      </div>
    `;
  }

  function projectTags(survey, st) {
    const s = survey || {};
    const tags = s.project_tags || {};
    const keys = dimOrder(tags, "project").filter(k => passesFilter(tags[k], st));
    const split = splitPrimary(keys, PRIMARY_PROJECT_DIMS);

    const stageCard = derivedStageCard(s);
    const primaryHtml = stageCard + split.primary.map(k => projectTagCard(k, tags[k], true)).join("");
    const secondaryHtml = split.secondary.map(k => projectTagCard(k, tags[k], false)).join("");

    if (!primaryHtml && !secondaryHtml) {
      return emptyState(
        "No project tags to show",
        "Either this survey has no project-level tags yet, or the current source / confidence filters exclude them all.",
        `<button class="btn sm ghost" data-action="reset-filters">Reset filters</button>`
      );
    }
    return disclosedGrid(primaryHtml, secondaryHtml, split.secondary.length, "tag-grid");
  }

  /* ---------- questions ---------- */

  function questionList(survey) {
    const s = survey || {};
    if (Array.isArray(s.questions)) return s.questions;
    if (Array.isArray(s.question_tags)) return s.question_tags;
    return [];
  }

  function questionText(q) {
    return (q && (q.question_text || q.question_title_preview)) || "";
  }

  function questionKey(q) {
    if (!q) return "";
    if (q.question_id !== null && q.question_id !== undefined) return String(q.question_id);
    return String(q.question_no === undefined ? "" : q.question_no);
  }

  function matchesSearch(q, needle) {
    if (!needle) return true;
    const hay = [
      questionText(q),
      String((q && q.question_no) ?? ""),
      String((q && q.question_id) ?? ""),
    ];
    const tags = (q && q.tags) || {};
    for (const k of Object.keys(tags)) {
      const t = tags[k];
      const v = t && t.value;
      if (isEmpty(v)) continue;
      hay.push(Array.isArray(v) ? v.join(" ") : String(v));
    }
    // NUL separator: no survey text can contain it, so two adjacent haystack
    // entries can never combine into a false match. Written as an escape so
    // the file stays plain text to grep/diff.
    return hay.join(" \u0000 ").toLowerCase().includes(needle);
  }

  function filterQuestions(survey, st) {
    const s = st || {};
    const needle = String(s.questionSearch || "").toLowerCase().trim();
    const active = filtersActive(s);
    return questionList(survey).filter(q => {
      if (!matchesSearch(q, needle)) return false;
      if (active && !q.is_content_message) {
        const anyMatch = Object.values((q && q.tags) || {}).some(t => passesFilter(t, s));
        if (!anyMatch) return false;
      }
      return true;
    });
  }

  function contentMessageCard(q) {
    const key = questionKey(q);
    return `
      <div class="question-card" data-q="${escapeHtml(key)}">
        <div class="question-head">
          <div class="qid">${escapeHtml(String(q.question_no ?? ""))}.</div>
          <div class="qbody">
            <div class="qtitle">${escapeHtml(questionText(q) || "(no title)")}</div>
            <div class="qsummary">
              <span class="status status--muted">content message</span>
              <span class="micro">not tagged — informational text shown to the respondent</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function questionCard(q, st) {
    if (q && q.is_content_message) return contentMessageCard(q);

    const s = st || {};
    const key = questionKey(q);
    // Always key on the stringified `key` — that is what `data-q` carries back
    // through the DOM, and dataset values are always strings.
    const expanded = !!(s.expandedQuestions && typeof s.expandedQuestions.has === "function"
      && s.expandedQuestions.has(key));
    const tags = (q && q.tags) || {};

    const summary = [
      tags.metric_name && !isEmpty(tags.metric_name.value)
        && `<span class="kv"><span>metric</span>${escapeHtml(String(tags.metric_name.value))}</span>`,
      tags.role_intent && !isEmpty(tags.role_intent.value)
        && `<span class="kv"><span>role</span>${escapeHtml(String(tags.role_intent.value))}</span>`,
      tags.flow_placement && !isEmpty(tags.flow_placement.value)
        && `<span class="kv"><span>flow</span>${escapeHtml(String(tags.flow_placement.value))}</span>`,
      tags.journey_stage && !isEmpty(tags.journey_stage.value)
        && `<span class="kv"><span>stage</span>${escapeHtml(String(tags.journey_stage.value))}</span>`,
    ].filter(Boolean).join("");

    const keys = dimOrder(tags, "question").filter(k => passesFilter(tags[k], s));
    const split = splitPrimary(keys, PRIMARY_QUESTION_DIMS);
    const primaryHtml = split.primary.map(k => questionTagCard(k, tags[k], true)).join("");
    const secondaryHtml = split.secondary.map(k => questionTagCard(k, tags[k], false)).join("");

    const body = (primaryHtml || secondaryHtml)
      ? disclosedGrid(primaryHtml, secondaryHtml, split.secondary.length, "qtag-grid")
      : `<div class="qtag-grid">${emptyHintHtml("No tags match the current filters.")}</div>`;

    return `
      <div class="question-card ${expanded ? "expanded" : ""}" data-q="${escapeHtml(key)}">
        <div class="question-head" data-action="toggle-q" data-q="${escapeHtml(key)}" role="button" tabindex="0">
          <div class="qid">${escapeHtml(String(q.question_no ?? ""))}.</div>
          <div class="qbody">
            <div class="qtitle">${escapeHtml(questionText(q) || "(no title)")}</div>
            <div class="qsummary">${summary}</div>
          </div>
          <span class="caret-q">&#9656;</span>
        </div>
        <div class="tags-wrap">${body}</div>
      </div>
    `;
  }

  function questionTable(questionList_, cols, st) {
    const list = Array.isArray(questionList_) ? questionList_ : [];
    const columns = Array.isArray(cols) ? cols : [];
    if (!list.length) return emptyHintHtml("No questions match.");

    const headHtml = ["#", "Question"].map(h => `<th>${escapeHtml(h)}</th>`).join("")
      + columns.map(k => {
          const meta = dimMeta(k);
          const title = meta.description ? ` title="${escapeHtml(meta.description)}"` : "";
          return `<th${title}>${escapeHtml(meta.label)}</th>`;
        }).join("");

    const rowsHtml = list.map(q => {
      const tags = (q && q.tags) || {};
      const cells = columns.map(k => {
        const t = tags[k];
        if (!t) return `<td><span class="empty">&mdash;</span></td>`;
        const v = fmtValue(t.value);
        // Cells are never hidden by the source/low-conf filters — the filters
        // pick which *rows* survive; dimming individual cells would make the
        // grid unreadable. `st` is kept in the signature for symmetry.
        const low = isLowConf(t) ? "qcell--low" : "";
        return `<td class="${low}" data-conf="${confBucket(t)}" title="${escapeHtml(evidenceTitle(t))}">
          <div>${v.html}</div>
          <div>${srcBadge(t.source)}</div>
        </td>`;
      }).join("");
      return `<tr>
        <td>${escapeHtml(String(q.question_no ?? ""))}</td>
        <td class="qcell-title">${escapeHtml(questionText(q))}${q.is_content_message ? ` <span class="status status--muted">content message</span>` : ""}</td>
        ${cells}
      </tr>`;
    }).join("");

    return `<div class="question-table-wrap"><table class="qtable">
      <thead><tr>${headHtml}</tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table></div>`;
  }

  function questions(survey, st) {
    const s = st || {};
    const filtered = filterQuestions(survey, s);

    if (!filtered.length) {
      return emptyState(
        "No questions match",
        "Clear the search box or widen the source / confidence filters.",
        `<button class="btn sm ghost" data-action="reset-filters">Reset filters</button>`
      );
    }

    if (s.questionView === "table") {
      const cols = (s.tableCols === "all")
        ? allQuestionCols(filtered)
        : dimsPresent(filtered);
      return questionTable(filtered, cols, s);
    }
    return filtered.map(q => questionCard(q, s)).join("");
  }

  /* ==================================================================
   * Summary tab — the one screen that answers "is this survey any good?"
   * without the reader opening a single card.
   *
   * Everything below reads only what the survey-view projection already
   * carries: project_tags, questions[].tags, metadata, survey_journey.
   * No new API surface.
   * ================================================================== */

  /* One flat list of every tag on the survey, project and question alike, each
   * carrying enough identity for the attention list to link back to it. */
  function collectTags(survey) {
    const s = survey || {};
    const out = [];
    const ptags = s.project_tags || {};
    for (const dim of Object.keys(ptags)) {
      out.push({ dim, tag: ptags[dim] || {}, level: "project", qkey: null, qno: null, qtext: "" });
    }
    // A question tagger that skips a dimension is dropped from the artifact
    // entirely (pipeline/assembly.py: `if tag.status == "skipped": continue`),
    // so an ABSENT taxonomy dimension *is* a skip and has to be synthesised
    // here. Without this the coverage meter could only ever report ~100%
    // assigned — the skipped segment would never once be drawn.
    // Degrades to the old behaviour when the taxonomy has not loaded yet.
    const qdims = dimsForLevel("question");
    for (const q of questionList(s)) {
      if (!q || q.is_content_message) continue;
      const tags = q.tags || {};
      const qkey = questionKey(q);
      const qno = q.question_no;
      const qtext = questionText(q);
      for (const dim of Object.keys(tags)) {
        out.push({ dim, tag: tags[dim] || {}, level: "question", qkey, qno, qtext });
      }
      for (const dim of qdims) {
        if (!Object.prototype.hasOwnProperty.call(tags, dim)) {
          out.push({ dim, tag: { status: "skipped" }, level: "question", qkey, qno, qtext });
        }
      }
    }
    return out;
  }

  /* A tag with no `status` at all is an assigned tag from an older artifact —
   * treating it as "pending" would slander every pre-status output. */
  const COVERAGE_KINDS = [
    ["assigned", "Assigned"],
    ["pending",  "Pending LLM"],
    ["skipped",  "Not applicable"],
    ["failed",   "Failed"],
  ];

  function coverageKind(tag) {
    const st = tag && tag.status;
    if (st === "skipped") return "skipped";
    if (st === "failed") return "failed";
    if (st === "pending_llm") return "pending";
    return "assigned";   // "assigned", "low_confidence_assigned", missing, unknown
  }

  function identitySection(survey) {
    const tags = (survey || {}).project_tags || {};
    const dims = ["project_type", "project_purpose", "audience_type", "industry_vertical"];
    const items = dims.map(k => {
      const t = tags[k];
      const v = t && t.value;
      if (isEmpty(v)) return "";
      const text = Array.isArray(v) ? v.join(", ") : String(v);
      return `<div class="identity-item">
        <span class="identity-label">${escapeHtml(dimMeta(k).label)}</span>
        <span class="identity-value">${escapeHtml(text)}</span>
      </div>`;
    }).filter(Boolean).join("");

    if (!items) {
      return `<section class="identity">${
        emptyHintHtml("No project-level identity tags yet — tag this survey to fill them in.")
      }</section>`;
    }
    return `<section class="identity">${items}</section>`;
  }

  function coverageSection(all) {
    const counts = COVERAGE_KINDS.map(([kind]) =>
      all.filter(e => coverageKind(e.tag) === kind).length);
    const total = counts.reduce((a, b) => a + b, 0);

    if (!total) {
      return `<section class="sum-card">
        <h3 class="sum-title">Tag coverage</h3>
        ${emptyHintHtml("No tags on this survey yet.")}
      </section>`;
    }

    const pcts = pctParts(counts);
    const segs = COVERAGE_KINDS.map(([kind], i) => pcts[i] > 0
      ? `<i class="meter-seg" data-kind="${kind}" style="width:${pcts[i]}%"></i>`
      : "").join("");
    const keys = COVERAGE_KINDS.map(([kind, label], i) => counts[i] > 0
      ? `<span class="meter-key" data-kind="${kind}">${escapeHtml(label)} ${counts[i]} <span class="micro">${pcts[i]}%</span></span>`
      : "").join("");

    return `<section class="sum-card">
      <h3 class="sum-title">Tag coverage <span class="micro">${total} tag${total === 1 ? "" : "s"} across project &amp; questions</span></h3>
      <div class="meter">${segs}</div>
      <div class="meter-legend">${keys}</div>
    </section>`;
  }

  function sourceMixSection(all) {
    // Only tags that actually got a value describe how the survey was decided;
    // skipped/failed ones carry a source that means nothing.
    const assigned = all.filter(e => coverageKind(e.tag) === "assigned");
    const order = ALL_SOURCES.concat(["unknown"]);
    const counts = order.map(src => assigned.filter(e => sourceClass(e.tag.source) === src).length);
    const total = counts.reduce((a, b) => a + b, 0);

    if (!total) {
      return `<section class="sum-card">
        <h3 class="sum-title">Source mix</h3>
        ${emptyHintHtml("Nothing has been assigned yet, so there is no method mix to show.")}
      </section>`;
    }

    const pcts = pctParts(counts);
    const segs = order.map((src, i) => pcts[i] > 0
      ? `<i class="mix-seg" data-src="${src}" title="${escapeHtml(`${src}: ${counts[i]} tag(s)`)}" style="width:${pcts[i]}%"></i>`
      : "").join("");
    const keys = order.map((src, i) => counts[i] > 0
      ? `<span class="chip">${srcBadge(src)} ${counts[i]} <span class="micro">${pcts[i]}%</span></span>`
      : "").join("");

    return `<section class="sum-card">
      <h3 class="sum-title">Source mix <span class="micro">how ${total} assigned tag${total === 1 ? " was" : "s were"} decided</span></h3>
      <div class="mix-bar">${segs}</div>
      <div class="meter-legend">${keys}</div>
    </section>`;
  }

  const ATTENTION_CAP = 25;

  function attentionReason(tag) {
    const bits = [];
    if (tag.status === "failed") {
      bits.push(tag.failure_reason ? `failed — ${tag.failure_reason}` : "the tagger failed");
    } else if (tag.status === "low_confidence_assigned") {
      bits.push("LLM was unsure — value kept for review");
    }
    if (typeof tag.confidence === "number" && isFinite(tag.confidence) && tag.confidence < 0.6) {
      bits.push(`confidence ${(tag.confidence * 100).toFixed(0)}%`);
    }
    return bits.join(" · ") || "flagged for review";
  }

  function attentionSection(all) {
    const rows = all.filter(e => {
      const t = e.tag || {};
      return isLowConf(t) || t.status === "low_confidence_assigned" || t.status === "failed";
    });

    if (!rows.length) {
      return `<section class="sum-card">
        <h3 class="sum-title">Needs attention</h3>
        <p class="micro">Nothing below the confidence threshold and nothing failed — every tag on this survey stands on its own.</p>
      </section>`;
    }

    // Worst first: failures, then the least confident. A hundred-question
    // survey can flag hundreds of tags, so the list is capped and the tail
    // becomes a count — the point is triage, not an export.
    const ranked = rows.slice().sort((a, b) => {
      const af = a.tag.status === "failed" ? 0 : 1;
      const bf = b.tag.status === "failed" ? 0 : 1;
      if (af !== bf) return af - bf;
      const ac = typeof a.tag.confidence === "number" ? a.tag.confidence : 1;
      const bc = typeof b.tag.confidence === "number" ? b.tag.confidence : 1;
      return ac - bc;
    });

    const shown = ranked.slice(0, ATTENTION_CAP);
    const rest = ranked.length - shown.length;

    const items = shown.map(e => {
      const where = e.level === "question"
        ? `Q${e.qno === null || e.qno === undefined ? "?" : e.qno}`
        : "Project";
      const qattr = e.qkey ? ` data-qkey="${escapeHtml(String(e.qkey))}"` : "";
      const title = e.level === "question" && e.qtext ? ` title="${escapeHtml(e.qtext)}"` : "";
      return `<li class="attention-row" role="button" tabindex="0" data-action="goto-tag"`
        + ` data-dim="${escapeHtml(String(e.dim))}"${qattr}${title}>
        <span class="attention-dim">${escapeHtml(where)} &middot; ${escapeHtml(dimMeta(e.dim).label)}</span>
        <span class="attention-why">${escapeHtml(attentionReason(e.tag || {}))}</span>
      </li>`;
    }).join("");

    const more = rest > 0
      ? `<li class="attention-row"><span class="attention-why">+${rest} more below the threshold — narrow the list with the Low-confidence filter on the Questions tab.</span></li>`
      : "";

    return `<section class="sum-card">
      <h3 class="sum-title">Needs attention <span class="micro">${ranked.length} tag${ranked.length === 1 ? "" : "s"}</span></h3>
      <ul class="attention-list">${items}${more}</ul>
    </section>`;
  }

  function journeySection(survey) {
    const s = survey || {};
    const j = s.survey_journey;
    const stages = (j && Array.isArray(j.stages_touched)) ? j.stages_touched.filter(n => !isEmpty(n)) : [];

    if (!stages.length) {
      // A survey with no metric questions legitimately has no journey at all.
      return `<section class="sum-card">
        <h3 class="sum-title">Journey</h3>
        <p class="micro">${escapeHtml(j
          ? "No journey stages were assigned — this survey has no journey-eligible metric questions."
          : "No journey mapping on this survey.")}</p>
      </section>`;
    }

    const counts = new Map();
    for (const q of questionList(s)) {
      const t = ((q && q.tags) || {}).journey_stage;
      if (!t || isEmpty(t.value)) continue;
      const vals = Array.isArray(t.value) ? t.value : [t.value];
      for (const v of vals) counts.set(String(v), (counts.get(String(v)) || 0) + 1);
    }

    const nodes = stages.map(n => {
      const name = String(n);
      const c = counts.get(name) || 0;
      return `<li class="journey-node" data-count="${c}" title="${escapeHtml(`${c} question${c === 1 ? "" : "s"} mapped to ${name}`)}">${escapeHtml(name)}</li>`;
    }).join("");

    const type = (j && j.journey_type) ? String(j.journey_type) : "";
    return `<section class="sum-card">
      <h3 class="sum-title">Journey${type ? ` <span class="micro">${escapeHtml(type)}</span>` : ""}</h3>
      <ol class="journey-strip">${nodes}</ol>
    </section>`;
  }

  /* The headline view. `st` is accepted for symmetry with every other renderer
   * (and so a future filter can narrow the summary) but the summary is
   * deliberately unfiltered: it describes the survey, not the current view. */
  function summary(survey, st) {
    const s = survey || {};
    const all = collectTags(s);
    const meta = s.metadata || {};
    const qs = questionList(s);
    const tagged = qs.filter(q => q && !q.is_content_message).length;

    const stat = (n, label) =>
      `<div class="stat"><strong>${escapeHtml(String(n))}</strong>${escapeHtml(label)}</div>`;

    const took = fmtMs(meta.processing_time_ms);
    const stats = [
      stat(qs.length, qs.length === 1 ? "question" : "questions"),
      tagged !== qs.length ? stat(tagged, "taggable") : "",
      stat(all.length, "tags"),
      isEmpty(meta.llm_calls_made) ? "" : stat(meta.llm_calls_made, "LLM calls"),
      took ? stat(took, "processing") : "",
    ].filter(Boolean).join("");

    return `<div class="summary">
      ${identitySection(s)}
      <div class="statrow">${stats}</div>
      ${coverageSection(all)}
      ${sourceMixSection(all)}
      ${attentionSection(all)}
      ${journeySection(s)}
    </div>`;
  }

  /* ---------- journey ---------- */

  function candidateName(c) {
    if (!c || typeof c !== "object") return String(c === undefined ? "" : c);
    return String(c.stage_name || c.sub_stage_name || c.name || c.stage || "");
  }

  function candidateChips(candidates) {
    const list = Array.isArray(candidates) ? candidates : [];
    return list.map((c, i) => {
      const name = candidateName(c);
      const score = (c && typeof c.score === "number") ? c.score : null;
      return `<span class="chip">
        <span class="micro">#${i + 1}</span>
        ${escapeHtml(name || "(unnamed)")}
        ${score === null ? "" : confBar(score)}
        ${(c && c.selected) ? `<span class="status status--ok">selected</span>` : ""}
      </span>`;
    }).join("");
  }

  function candidateBlocks(survey) {
    const dims = [["journey_stage", "Stage candidates"], ["sub_stage_name", "Sub-stage candidates"]];
    const cards = [];
    for (const q of questionList(survey)) {
      const tags = (q && q.tags) || {};
      const parts = [];
      for (const [dim, label] of dims) {
        const cov = tags[dim] && tags[dim].coverage_metadata;
        const cands = cov && Array.isArray(cov.candidates) ? cov.candidates : [];
        if (!cands.length) continue;
        parts.push(`
          <div class="micro">${escapeHtml(label)}${cov.confidence ? ` &middot; LLM confidence: ${escapeHtml(String(cov.confidence))}` : ""}</div>
          <div class="multi-list">${candidateChips(cands)}</div>
          ${cov.evidence ? `<div class="evidence">${escapeHtml(String(cov.evidence))}</div>` : ""}
        `);
      }
      if (!parts.length) continue;
      cards.push(`
        <div class="stage-card">
          <h4>${escapeHtml(String(q.question_no ?? ""))}. ${escapeHtml(questionText(q) || "(no title)")}</h4>
          ${parts.join("")}
        </div>
      `);
    }
    if (!cards.length) {
      return `<div class="section-head">Ranked candidates</div>
        ${emptyHintHtml("No candidate metadata on this survey — re-tag with journey candidates enabled, or the questions are not journey-eligible metrics.")}`;
    }
    return `<div class="section-head">Ranked candidates</div>
      <div class="stages-grid">${cards.join("")}</div>`;
  }

  function stagesGrid(names, emptyText) {
    const list = (Array.isArray(names) ? names : []).filter(n => !isEmpty(n));
    if (!list.length) return emptyHintHtml(emptyText);
    return `<div class="stages-grid">${
      list.map(n => `<div class="stage-card"><h4>${escapeHtml(String(n))}</h4></div>`).join("")
    }</div>`;
  }

  function journey(survey, st) {
    const s = survey || {};
    const j = s.survey_journey;
    if (!j) return "";
    const stages = Array.isArray(j.stages_touched) ? j.stages_touched : [];
    const subs = Array.isArray(j.sub_stages_touched) ? j.sub_stages_touched : [];
    const mapped = questionList(s).filter(q => {
      const t = ((q && q.tags) || {}).journey_stage;
      return !!(t && !isEmpty(t.value));
    }).length;

    const stat = (n, label) => `<div class="stat"><strong>${escapeHtml(String(n))}</strong>${escapeHtml(label)}</div>`;

    return `
      <div class="statrow">
        ${stat(j.journey_type || "—", "journey type")}
        ${stat(stages.length, "stages touched")}
        ${stat(subs.length, "sub-stages touched")}
        ${stat(mapped, "journey-mapped questions")}
      </div>
      <div class="section-head">Stages touched</div>
      ${stagesGrid(stages, "No journey stages assigned on this survey.")}
      <div class="section-head">Sub-stages touched</div>
      ${stagesGrid(subs, "No sub-stages assigned on this survey.")}
      ${(st && st.includeCandidates) ? candidateBlocks(s) : ""}
    `;
  }

  /* ---------- tenant tags ---------- */

  function boolBadge(label, on) {
    return `<span class="status status--${on ? "ok" : "muted"}">${escapeHtml(label)}${on ? "" : ": none"}</span>`;
  }

  function tenantTags(artifact, st) {
    const a = artifact || {};
    const tags = a.tags || {};
    const meta = a.metadata || {};

    const keys = Object.keys(tags);
    if (!keys.length) {
      return emptyState(
        "No tenant-level tags yet",
        "Build them from the tenant profile and corporate data.",
        `<button class="btn sm primary" data-action="tag-tenant">Build tenant tags</button>`
      );
    }

    // Tenant dimensions are absent from the project/question taxonomy levels,
    // so order() puts them all in the alphabetical tail — which is what we want.
    const ordered = dimOrder(tags, "tenant");
    const cards = ordered
      .filter(k => passesFilter(tags[k], st))
      .map(k => projectTagCard(k, tags[k], true))
      .join("");

    const gen = fmtDate(a.generated_at);
    const headBits = [
      a.schema_version ? `<span class="status status--info">${escapeHtml(`schema ${a.schema_version}`)}</span>` : "",
      gen ? `<span class="status status--muted">${escapeHtml(gen)}</span>` : "",
      boolBadge("org", !!meta.has_org),
      boolBadge("cx", !!meta.has_cx),
      boolBadge("ex", !!meta.has_ex),
    ].filter(Boolean).join(" ");

    return `<div class="tenant-panel">
      <div class="section-head">Tenant-level tags ${headBits}</div>
      <div class="tag-grid">${cards || emptyHintHtml("No tenant tags match the current filters.")}</div>
    </div>`;
  }

  /* ---------- tenant profile ---------- */

  const PROFILE_AGENTS = [["org", "Organization"], ["cx", "Customer (CX)"], ["ex", "Employee (EX)"]];

  function profileSummary(profile) {
    const p = profile || {};
    const summaryObj = p.summary || {};
    const paths = Array.isArray(p.artifact_paths) ? p.artifact_paths : [];

    const badges = [
      boolBadge("org", !!p.has_org),
      boolBadge("cx", !!p.has_cx),
      boolBadge("ex", !!p.has_ex),
    ].join(" ");

    const keys = Object.keys(summaryObj);
    const cards = keys.map(k => {
      let v = summaryObj[k];
      if ((k === "customer_types" || k === "employee_types") && Array.isArray(v)) {
        v = v.map(x => (x && typeof x === "object") ? String(x.type_name || "?") : String(x));
      }
      return plainCard(dimMeta(k).label, v);
    }).join("");

    return `<div class="tenant-panel">
      <div class="section-head">Tenant profile ${badges}</div>
      ${cards ? `<div class="tag-grid">${cards}</div>`
              : emptyHintHtml("The profile artifacts carry no summary fields.")}
      ${paths.length ? `<div class="kv-list">${
        paths.map(x => `<div class="mono">${escapeHtml(String(x))}</div>`).join("")
      }</div>` : ""}
    </div>`;
  }

  function agentBody(value, agent) {
    if (value === undefined || value === null) {
      // app.js fetches lazily when the <details> is opened.
      return `<div class="micro">Expand to load the raw envelope.</div>`;
    }
    if (value === "loading") {
      return `<div class="spinner"></div>${skelLine(4)}`;
    }
    if (value === "missing") {
      return `<div class="profile-missing">No ${escapeHtml(agent)} artifact on disk for this tenant.</div>`;
    }
    if (typeof value === "string" && value.indexOf("error") === 0) {
      const detail = value.slice(5).replace(/^[:\s]+/, "");
      return `<div class="lc-banner warn">
        <strong>Could not load the ${escapeHtml(agent)} artifact.</strong>
        ${detail ? `<details class="ev-reasoning"><summary>Details</summary><div class="mono">${escapeHtml(detail)}</div></details>` : ""}
        <div class="btn-row"><button class="btn sm" data-action="load-agent" data-agent="${escapeHtml(agent)}">Retry</button></div>
      </div>`;
    }
    if (typeof value === "string") {
      return `<pre class="profile-json">${escapeHtml(value)}</pre>`;
    }
    return rawJson(value);
  }

  function profileAgents(profileAgentsState) {
    const map = profileAgentsState || {};
    return PROFILE_AGENTS.map(([key, label]) => {
      const value = map[key];
      const loaded = value !== undefined && value !== null
        && value !== "loading" && value !== "missing"
        && !(typeof value === "string" && value.indexOf("error") === 0);
      const status = value === "loading" ? "loading"
        : value === "missing" ? "not on disk"
        : loaded ? "loaded" : "";
      return `<details class="profile-agent" data-agent="${key}">
        <summary>
          <span class="chip chip--mono">${escapeHtml(key)}</span> ${escapeHtml(label)}
          ${status ? `<span class="micro">${escapeHtml(status)}</span>` : ""}
        </summary>
        ${agentBody(value, key)}
      </details>`;
    }).join("");
  }

  /* ---------- taxonomy ---------- */

  function taxonomyTable(taxonomy, search) {
    const tax = (taxonomy && typeof taxonomy === "object") ? taxonomy : {};
    const needle = String(search || "").toLowerCase().trim();
    const keys = Object.keys(tax).filter(k => {
      if (!needle) return true;
      const dim = tax[k] || {};
      return k.toLowerCase().includes(needle)
        || String(dim.description || "").toLowerCase().includes(needle);
    });

    if (!keys.length) {
      return emptyState("No dimensions match", "Try a different search term.", "");
    }

    const rows = keys.map(k => {
      const dim = tax[k] || {};
      const allowed = Array.isArray(dim.allowed_values) ? dim.allowed_values : [];
      const chips = allowed.length
        ? `<div class="multi-list">${allowed.map(v => `<span class="chip">${escapeHtml(String(v))}</span>`).join("")}</div>`
        : `<span class="empty">${dim.user_defined ? "free text" : "&mdash;"}</span>`;
      return `<tr>
        <td class="mono" title="${escapeHtml(String(dim.description || ""))}">${escapeHtml(k)}</td>
        <td><span class="status status--info">${escapeHtml(String(dim.level || "—"))}</span></td>
        <td>${dim.multi_label ? `<span class="status status--ok">multi</span>` : `<span class="empty">&mdash;</span>`}</td>
        <td>${dim.user_defined ? `<span class="chip chip--free">free</span>` : `<span class="empty">&mdash;</span>`}</td>
        <td>${allowed.length}</td>
        <td>${chips}</td>
      </tr>`;
    }).join("");

    const headers = ["Dimension", "Level", "Multi-label", "User-defined", "#", "Allowed values"];
    return `<div class="question-table-wrap"><table class="qtable">
      <thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
  }

  /* ---------- batch tagging result ---------- */

  /* Per-survey outcome of a batch run. Deliberately NOT named statusBadge:
     that name belongs to the tag-status badge above, and two function
     declarations sharing a name in this scope silently shadow each other. */
  function runStatusBadge(status) {
    const s = String(status || "").toLowerCase();
    const tone = s === "success" ? "ok" : (s === "skipped" ? "muted" : "danger");
    return `<span class="status status--${tone}">${escapeHtml(status || "unknown")}</span>`;
  }

  function batchResult(result) {
    const r = result || {};
    const rows = Array.isArray(r.surveys) ? r.surveys : [];
    const stat = (n, label) => `<div class="stat"><strong>${escapeHtml(String(n ?? 0))}</strong>${escapeHtml(label)}</div>`;

    const body = rows.length
      ? `<div class="question-table-wrap"><table class="qtable">
          <thead><tr><th>Survey</th><th>Status</th><th>Detail</th></tr></thead>
          <tbody>${rows.map(row => {
            const rr = row || {};
            return `<tr>
              <td class="mono">${escapeHtml(String(rr.survey_no ?? "—"))}</td>
              <td>${runStatusBadge(rr.status)}</td>
              <td>${rr.error ? `<span class="mono">${escapeHtml(String(rr.error))}</span>` : `<span class="empty">&mdash;</span>`}</td>
            </tr>`;
          }).join("")}</tbody>
        </table></div>`
      : emptyHintHtml("The run returned no per-survey detail.");

    return `
      <div class="statrow">
        ${stat(r.processed, "processed")}
        ${stat(r.skipped, "skipped (unchanged)")}
        ${stat(r.failed, "failed")}
      </div>
      ${body}
    `;
  }

  /* ==================================================================
   * exports
   * ================================================================== */

  ST.render = {
    summary,
    surveyHeader,
    filterBar,
    projectTags,
    questions,
    questionTable,
    journey,
    rawJson,
    tenantTags,
    profileSummary,
    profileAgents,
    taxonomyTable,
    batchResult,
    emptyState,
    errorState,
    // reusable primitives (app.js may compose its own panels from these)
    projectTagCard,
    questionTagCard,
    plainCard,
    toggleChip,
    skeleton,
    skelCard,
    skelLine,
    legend: renderLegend,
    emptyHint: emptyHintHtml,
  };

})(window.ST);
