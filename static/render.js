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

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[c]);
  }

  const labelFor = (k) => String(k).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  const sourceClass = (s) => ALL_SOURCES.includes(s) ? s : "unknown";

  const isLowConf = (tag) => (tag && typeof tag.confidence === "number" && tag.confidence < 0.6);

  const isEmpty = (v) => v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);

  const fmtValue = (v) => {
    if (isEmpty(v)) return { html: '<span class="empty">— not set —</span>', empty: true };
    if (Array.isArray(v)) {
      return {
        html: `<div class="multi-list">${v.map(x => `<span class="mv">${escapeHtml(String(x))}</span>`).join("")}</div>`,
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

  function srcBadge(src) {
    const cls = sourceClass(src);
    const hint = SOURCE_METHOD_HINT[cls] || "unknown assignment method";
    return `<span class="src ${cls}" title="${escapeHtml(hint)}">${escapeHtml(src || "unknown")}</span>`;
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
    if (p.type)   chips.push(`<span class="ev-chip type">${escapeHtml(p.type)}</span>`);
    if (p.ruleId) chips.push(`<span class="ev-chip rule" title="deterministic rule id">${escapeHtml(p.ruleId)}</span>`);
    if (p.model)  chips.push(`<span class="ev-chip model" title="LLM model">${escapeHtml(p.model)}</span>`);
    if (p.stage)  chips.push(`<span class="ev-chip stage" title="pipeline stage">${escapeHtml(p.stage)}</span>`);
    if (p.inputs) {
      for (const [k, v] of Object.entries(p.inputs)) {
        chips.push(`<span class="ev-chip" title="rule input">${escapeHtml(k)}=${escapeHtml(String(v))}</span>`);
      }
    }
    if (p.components) {
      for (const c of p.components) {
        chips.push(`<span class="ev-chip" title="hybrid component">${escapeHtml((c && (c.source || c.type)) || "?")}</span>`);
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
    escapeHtml,
    labelFor,
    sourceClass,
    isLowConf,
    isEmpty,
    fmtValue,
    confBar,
    srcBadge,
    evidenceParts,
    evidenceHtml,
    evidenceTitle,
    fmtMs,
    fmtDate,
    fmtElapsed,
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

  function renderLegend() {
    return `<div class="legend">${ALL_SOURCES.map(s => `<span class="src ${s}">${s}</span>`).join("")}</div>`;
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
      Low-confidence flags from the pipeline: ${lcFlags.map(f => `<span class="pill">${escapeHtml(String(f))}</span>`).join(" ")}
    </div>` : ""}`;
  }

  /* ---------- filter bar ---------- */

  function filterBar(st, opts) {
    const s = st || {};
    const o = opts || {};
    const sf = s.sourceFilter;
    const has = (x) => !sf || typeof sf.has !== "function" || sf.has(x);

    const srcChips = ALL_SOURCES.map(x =>
      `<span class="chip src ${x} toggle ${has(x) ? "on" : "off"}" data-action="toggle-source" data-source="${x}">${x}</span>`
    ).join("");

    const searchHtml = o.questionControls
      ? `<input type="text" id="qSearch" placeholder="Search question text / number / tag value…" value="${escapeHtml(s.questionSearch || "")}" />`
      : "";

    const viewChips = o.questionControls
      ? `<div class="filter-group">
          <span class="filter-label">View</span>
          <span class="chip toggle ${(s.questionView || "cards") === "cards" ? "on" : "off"}" data-action="set-view" data-view="cards">Cards</span>
          <span class="chip toggle ${s.questionView === "table" ? "on" : "off"}" data-action="set-view" data-view="table">Table</span>
          <span class="chip toggle off" data-action="expand-all">Expand all</span>
          <span class="chip toggle off" data-action="collapse-all">Collapse all</span>
        </div>`
      : "";

    const colChips = o.columnControls
      ? `<div class="filter-group">
          <span class="filter-label">Columns</span>
          <span class="chip toggle ${(s.tableCols || "present") === "present" ? "on" : "off"}" data-action="set-cols" data-cols="present">Present</span>
          <span class="chip toggle ${s.tableCols === "all" ? "on" : "off"}" data-action="set-cols" data-cols="all">All</span>
        </div>`
      : "";

    const journeyChip = o.journeyToggle
      ? `<div class="filter-group">
          <span class="chip toggle ${s.includeCandidates ? "on" : "off"}" data-action="toggle-candidates">Show ranked candidates</span>
        </div>`
      : "";

    return `<div class="filters">
      ${searchHtml}
      <div class="filter-group"><span class="filter-label">Source</span>${srcChips}</div>
      <div class="filter-group">
        <span class="chip toggle ${s.lowConfOnly ? "on" : "off"}" data-action="toggle-lowconf">Low confidence only</span>
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
    const freeChip = meta.free ? ` <span class="ev-chip">free text</span>` : "";
    return `<div class="${cls}"${title}>${escapeHtml(meta.label)}${freeChip}</div>`;
  }

  function projectTagCard(key, tag) {
    const t = tag || {};
    const v = fmtValue(t.value);
    const lc = isLowConf(t) ? "lowconf" : "";
    return `
      <div class="tag-card ${lc}">
        ${tagNameHtml(key, "tag-name")}
        <div class="tag-value ${v.empty ? "empty" : ""}">${v.html}</div>
        <div class="tag-foot">
          ${srcBadge(t.source)}
          ${confBar(t.confidence)}
        </div>
        ${evidenceHtml(t, "evidence")}
      </div>
    `;
  }

  function questionTagCard(key, tag) {
    const t = tag || {};
    const v = fmtValue(t.value);
    const lc = isLowConf(t) ? "lowconf" : "";
    return `
      <div class="qtag ${lc}">
        ${tagNameHtml(key, "qtag-name")}
        <div class="qtag-value ${v.empty ? "empty" : ""}">${v.html}</div>
        <div class="qtag-meta">
          ${srcBadge(t.source)}
          ${confBar(t.confidence)}
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
      <div class="tag-card">
        <div class="tag-name">${escapeHtml(label)}</div>
        <div class="tag-value ${v.empty ? "empty" : ""}">${v.html}</div>
      </div>
    `;
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
      .map(([name, n]) => `<span class="mv">${escapeHtml(String(name))}${counts.size > 1 ? ` <span class="micro">&times;${n}</span>` : ""}</span>`)
      .join("");
    return `
      <div class="tag-card">
        <div class="tag-name">Journey Stage <span class="micro">(derived)</span></div>
        <div class="tag-value"><div class="multi-list">${chips}</div></div>
        <div class="tag-foot">
          <span class="src derived" title="rolled up from question-level journey_stage tags">derived</span>
          ${confN ? confBar(confSum / confN) : ""}
        </div>
        <div class="evidence">Rolled up from the journey_stage tag on ${total} of ${qs.length} question${qs.length === 1 ? "" : "s"}.</div>
      </div>
    `;
  }

  function projectTags(survey, st) {
    const s = survey || {};
    const tags = s.project_tags || {};
    const cards = dimOrder(tags, "project")
      .filter(k => passesFilter(tags[k], st))
      .map(k => projectTagCard(k, tags[k]))
      .join("");
    const stageCard = derivedStageCard(s);
    const body = stageCard + cards;
    if (!body) {
      return emptyState(
        "No project tags to show",
        "Either this survey has no project-level tags yet, or the current source / confidence filters exclude them all.",
        `<button class="btn sm ghost" data-action="reset-filters">Reset filters</button>`
      );
    }
    return `<div class="tag-grid">${body}</div>`;
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
    return hay.join("   ").toLowerCase().includes(needle);
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
          <div>
            <div class="qtitle">${escapeHtml(questionText(q) || "(no title)")}</div>
            <div class="qsummary">
              <span class="badge muted">content message</span>
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

    const qtagsHtml = dimOrder(tags, "question")
      .filter(k => passesFilter(tags[k], s))
      .map(k => questionTagCard(k, tags[k]))
      .join("");

    return `
      <div class="question-card ${expanded ? "expanded" : ""}" data-q="${escapeHtml(key)}">
        <div class="question-head" data-action="toggle-q" data-q="${escapeHtml(key)}">
          <div class="qid">${escapeHtml(String(q.question_no ?? ""))}.</div>
          <div>
            <div class="qtitle">${escapeHtml(questionText(q) || "(no title)")}</div>
            <div class="qsummary">${summary}</div>
          </div>
          <span class="caret-q">&#9656;</span>
        </div>
        <div class="tags-wrap">
          <div class="qtag-grid">${qtagsHtml || emptyHintHtml("No tags match the current filters.")}</div>
        </div>
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
        const lc = isLowConf(t) ? "lowconf" : "";
        return `<td class="${lc}" title="${escapeHtml(evidenceTitle(t))}">
          <div>${v.html}</div>
          <div>${srcBadge(t.source)}</div>
        </td>`;
      }).join("");
      return `<tr>
        <td>${escapeHtml(String(q.question_no ?? ""))}</td>
        <td class="qcell-title">${escapeHtml(questionText(q))}${q.is_content_message ? ` <span class="badge muted">content message</span>` : ""}</td>
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
      return `<span class="mv">
        <span class="micro">#${i + 1}</span>
        ${escapeHtml(name || "(unnamed)")}
        ${score === null ? "" : confBar(score)}
        ${(c && c.selected) ? `<span class="badge ok">selected</span>` : ""}
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
    return `<span class="badge ${on ? "ok" : "muted"}">${escapeHtml(label)}${on ? "" : ": none"}</span>`;
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
      .map(k => projectTagCard(k, tags[k]))
      .join("");

    const gen = fmtDate(a.generated_at);
    const headBits = [
      a.schema_version ? `<span class="badge">${escapeHtml(`schema ${a.schema_version}`)}</span>` : "",
      gen ? `<span class="badge muted">${escapeHtml(gen)}</span>` : "",
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
    const summary = p.summary || {};
    const paths = Array.isArray(p.artifact_paths) ? p.artifact_paths : [];

    const badges = [
      boolBadge("org", !!p.has_org),
      boolBadge("cx", !!p.has_cx),
      boolBadge("ex", !!p.has_ex),
    ].join(" ");

    const keys = Object.keys(summary);
    const cards = keys.map(k => {
      let v = summary[k];
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
      return `<div class="spinner"></div>`;
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
          <span class="agent-badge">${key}</span> ${escapeHtml(label)}
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
        ? `<div class="multi-list">${allowed.map(v => `<span class="mv">${escapeHtml(String(v))}</span>`).join("")}</div>`
        : `<span class="empty">${dim.user_defined ? "free text" : "&mdash;"}</span>`;
      return `<tr>
        <td class="mono" title="${escapeHtml(String(dim.description || ""))}">${escapeHtml(k)}</td>
        <td><span class="badge">${escapeHtml(String(dim.level || "—"))}</span></td>
        <td>${dim.multi_label ? `<span class="badge ok">multi</span>` : `<span class="empty">&mdash;</span>`}</td>
        <td>${dim.user_defined ? `<span class="badge warn">free</span>` : `<span class="empty">&mdash;</span>`}</td>
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

  function statusBadge(status) {
    const s = String(status || "").toLowerCase();
    const cls = s === "success" ? "ok" : (s === "skipped" ? "muted" : "err");
    return `<span class="badge ${cls}">${escapeHtml(status || "unknown")}</span>`;
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
              <td>${statusBadge(rr.status)}</td>
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
    legend: renderLegend,
    emptyHint: emptyHintHtml,
  };

})(window.ST);
