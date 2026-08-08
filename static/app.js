/* Survey Auto-Tagger — application shell.
 *
 * Owns state, fetching, DOM writes and event dispatch. All state -> HTML
 * rendering lives in render.js (ST.render / ST.util / ST.dims); nothing here
 * builds markup beyond small chrome fragments.
 *
 * Plain browser JS: no modules, no build step, no CDN. The deployment is an
 * internal network with no internet access.
 *
 * Data lives on a network share, so two things shape the design:
 *   - There is NO global catalog browse. A corp number is typed; loading it
 *     costs exactly one directory listing (GET /tenants/{t}/tag-surveys).
 *     GET /api/surveys walks the entire share and is never called.
 *   - A 404 usually means "not tagged yet", which is the normal path, while an
 *     unreachable share must look completely different. See isShareError().
 */
window.ST = window.ST || {};
(function (ST) {
  "use strict";

  var U = ST.util;
  var R = ST.render;
  var D = ST.dims;

  /* ==================================================================
   * 1. CONSTANTS
   * ================================================================== */

  var AGENTS = ["org", "cx", "ex"];

  var POLL_BATCH_MS = 6000;      // survey list re-poll while a batch run is in flight
  var POLL_PROFILE_MS = 30000;   // artifact re-poll while a background fetch runs
  var PROFILE_MAX_POLLS = 80;    // ~40 min ceiling; the server run can outlive it
  var SLOW_MS = 20000;           // "still waiting on the share" threshold
  var GET_TIMEOUT_MS = 30000;
  var TOAST_MS = 6000;

  var LS = { corp: "st.corp", topView: "st.topview", profileJob: "st.profilejob",
             tab: "st.tab", theme: "st.theme", density: "st.density" };
  /* Bump the version suffix whenever /api/taxonomy's response shape grows a
   * field the UI reads — a session that cached the old shape would otherwise
   * render blank columns until the tab is closed. v2 = the explanation layer
   * (explanation / derivation / strategy) plus tenant-level dimensions. */
  var SS = { taxonomy: "st.taxonomy.v2" };

  /* The platform shell's access token, written to same-origin localStorage on
   * login and on every silent renewal. We only ever READ it: the shell owns the
   * login and refresh lifecycle, and this app is one of several that ride along
   * on the same key. Attached to every API call so the server can (a) work out
   * which corp an embedded session belongs to and (b) forward the same token
   * outbound to apismx and friends. Absent (plain ops use at localhost) is a
   * normal state, not an error — the API is open. */
  var TOKEN_KEY = "access_token";

  var ROUTES = {
    taxonomy:        function ()      { return "/api/taxonomy"; },
    config:          function ()      { return "/api/config"; },
    me:              function ()      { return "/api/me"; },
    shareHealth:     function ()      { return "/api/health/share"; },
    surveyList:      function (t)     { return "/api/tenants/" + t + "/tag-surveys"; },
    batchTag:        function (t)     { return "/api/tenants/" + t + "/tag-surveys"; },
    tenantTags:      function (t)     { return "/api/tenants/" + t + "/tags"; },
    tenantTagsBuild: function (t)     { return "/api/tenants/" + t + "/tag"; },   // NOTE: singular
    surveyTags:      function (t,s,jc){ return "/api/tenants/" + t + "/surveys/" + s + "/tags" +
                                               (jc ? "?include_journey_candidates=true" : ""); },
    tagSurvey:       function (t,s)   { return "/api/tenants/" + t + "/surveys/" + s + "/tag"; },
    profile:         function (t)     { return "/api/tenants/" + t + "/profile"; },
    profileAgent:    function (t,a)   { return "/api/tenants/" + t + "/profile/" + a; },
    profileFetch:    function (t,bg)  { return "/api/tenants/" + t + "/profile/fetch" +
                                               (bg ? "?background=true" : ""); }
  };

  // A downed share surfaces as a Windows error code buried in a 500 detail, or
  // as an empty listing that is indistinguishable from "no surveys". These are
  // the codes SMB actually produces: 53 not found, 64 name deleted,
  // 67 bad net name, 1231 unreachable, 1326 bad credentials.
  var SHARE_ERR_RE = /\[WinError (53|64|67|1231|1326)\]|network path was not found|network name cannot be found|semaphore timeout|no such file or directory/i;

  /* ==================================================================
   * 1b. THEME + DENSITY
   *
   * Two attributes on <html>; app.css hangs every alias token off them.
   * Deliberately NOT part of the persist()/restore() app-state blob: restore()
   * runs inside boot(), which is one DOMContentLoaded away, and by then the
   * shell has already painted — a dark-mode user would see a white flash on
   * every reload. So these read their own keys and apply themselves at script
   * evaluation, before anything renders. Same localStorage convention as the
   * rest of the file (an `LS` key, writes wrapped in try/catch because private
   * mode throws), just on an earlier schedule.
   *
   * Manual toggle only: prefers-color-scheme is deliberately not consulted.
   * ================================================================== */

  var THEMES = ["light", "dark"];
  var DENSITIES = ["comfortable", "compact"];

  var ui = { theme: "light", density: "comfortable" };

  function readChoice(key, allowed, dflt) {
    try {
      var v = localStorage.getItem(key);
      return allowed.indexOf(v) >= 0 ? v : dflt;
    } catch (e) { return dflt; }        // private mode / storage disabled
  }

  function applyChrome() {
    var root = document.documentElement;
    root.setAttribute("data-theme", ui.theme);
    root.setAttribute("data-density", ui.density);
  }

  function persistChrome() {
    try {
      localStorage.setItem(LS.theme, ui.theme);
      localStorage.setItem(LS.density, ui.density);
    } catch (e) { /* private mode — the attributes still hold for this session */ }
  }

  /* The buttons live in the static topbar and are never re-rendered, so their
   * pressed state is pushed rather than pulled. */
  function syncChromeButtons() {
    var t = document.getElementById("themeBtn");
    var d = document.getElementById("densityBtn");
    if (t) t.setAttribute("aria-pressed", ui.theme === "dark" ? "true" : "false");
    if (d) d.setAttribute("aria-pressed", ui.density === "compact" ? "true" : "false");
  }

  ui.theme = readChoice(LS.theme, THEMES, "light");
  ui.density = readChoice(LS.density, DENSITIES, "comfortable");
  applyChrome();

  /* ==================================================================
   * 2. STATE
   * ================================================================== */

  var state = {
    corpNo: null,
    topView: "surveys",              // surveys | tenant

    surveys: [],                     // [{survey_no, tagged: bool|null, provisional?}]
    surveyNames: {},                 // {survey_no: title} — backfilled lazily
    surveysLoaded: false,
    surveysError: null,
    navSearch: "",
    navFilter: "all",                // all | tagged | untagged

    activeSurveyNo: null,
    survey: null,                    // normalized payload
    surveyError: null,
    etags: {},                       // {"<no>|jc": etag}
    activeTab: "summary",            // summary | project | questions | journey | raw
    includeCandidates: false,

    sourceFilter: new Set(U.ALL_SOURCES),
    lowConfOnly: false,
    questionSearch: "",
    questionView: "cards",
    tableCols: "present",
    expandedQuestions: new Set(),

    tenantSection: "tags",           // tags | profile | batch | taxonomy
    tenantTags: null, tenantTagsError: null,
    profile: null, profileError: null,
    profileAgents: {},               // {org: envelope|"loading"|"missing"|"error: …"}
    profileJob: null,                // {startedAt, website, agents, polls}
    profileTimer: null,
    batch: { running: false, startedAt: null, result: null,
             progress: null, timer: null },

    taxonomy: null,
    // Server config from /api/config. Until it loads we assume the Parallel
    // shape, which is the stricter of the two (needs a website), so the form
    // can never submit something the server would reject.
    config: { profile_source: "parallel", smx_allow_generate: true,
              smx_generate_wait_seconds: 90 },
    busy: 0,
    slowWarn: false,
    banner: null,                    // {kind, text, actionsHtml}
    shareDown: false,
    authExpired: false,              // a 401 came back — see showAuthExpired()
    tokenCorpNo: null,               // corp_no the JWT claims, from /api/me
    toasts: [],
    nextToastId: 1,
    ribbons: [],                     // [{id, text}] — stack; newest is on screen
    nextRibbonId: 1
  };

  function persist() {
    try {
      if (state.corpNo) localStorage.setItem(LS.corp, String(state.corpNo));
      localStorage.setItem(LS.topView, state.topView);
      localStorage.setItem(LS.tab, state.activeTab);
      if (state.profileJob) localStorage.setItem(LS.profileJob, JSON.stringify(state.profileJob));
      else localStorage.removeItem(LS.profileJob);
    } catch (e) { /* private mode / disabled storage — not worth surfacing */ }
  }

  function restore() {
    try {
      var v = localStorage.getItem(LS.topView);
      if (v === "surveys" || v === "tenant") state.topView = v;
      // A stored tab naming something SURVEY_TABS no longer defines (a renamed
      // or dropped tab from an older build) must land on Summary, not on a key
      // that renders nothing.
      var tab = localStorage.getItem(LS.tab);
      state.activeTab = SURVEY_TABS.some(function (t) { return t.key === tab; })
        ? tab : "summary";
      var job = localStorage.getItem(LS.profileJob);
      if (job) state.profileJob = JSON.parse(job);
      var tax = sessionStorage.getItem(SS.taxonomy);
      if (tax) { state.taxonomy = JSON.parse(tax); D.set(state.taxonomy); }
    } catch (e) { /* ignore malformed storage */ }
  }

  /* ==================================================================
   * 3. DOM HANDLES
   * ================================================================== */

  var el = {};
  function grabDom() {
    ["corpForm","corpInput","corpBtn","surveyForm","surveyInput","surveyBtn",
     "topStatus","themeBtn","densityBtn","busybar","ribbon","topTabs","navSearch",
     "navFilters","sidebarHeading",
     "nav","workspace","banner","content","toasts"].forEach(function (id) {
      el[id] = document.getElementById(id);
    });
  }

  /* ==================================================================
   * 4. API
   * ================================================================== */

  function detailOf(data, fallback) {
    if (!data) return fallback || "";
    var d = data.detail;
    if (typeof d === "string") return d;
    // FastAPI 422: [{loc, msg, type}, ...]
    if (Array.isArray(d)) {
      return d.map(function (x) {
        var loc = Array.isArray(x.loc) ? x.loc.join(".") : "";
        return (loc ? loc + ": " : "") + (x.msg || "");
      }).join("; ");
    }
    if (typeof data === "string") return data;
    return fallback || "";
  }

  /* Re-read on every call rather than caching: the shell rewrites the key on
   * each silent renewal, and a sibling tab's renewal has to be picked up too.
   * A blank/whitespace value is the same as missing — `Bearer ` with nothing
   * after it is never a token the server can do anything with. */
  function readToken() {
    try {
      var raw = localStorage.getItem(TOKEN_KEY);
      if (typeof raw !== "string") return null;
      raw = raw.trim();
      return raw.length ? raw : null;
    } catch (e) { return null; }   // private mode / storage disabled
  }

  function isShareError(res) {
    if (!res) return false;
    if (res.netError) return true;
    if (res.status >= 500 && SHARE_ERR_RE.test(String(res.detail || ""))) return true;
    return false;
  }

  /* Never throws on an HTTP status — the caller branches on `.status`.
   * `timeoutMs: null` disables the abort entirely; pass it for every
   * long-running POST (tag, batch, sync profile fetch),
   * otherwise the default would abort exactly the calls that matter. */
  function api(url, opts) {
    opts = opts || {};
    var method = opts.method || "GET";
    var timeoutMs = opts.timeoutMs === undefined ? GET_TIMEOUT_MS : opts.timeoutMs;

    var init = { method: method, headers: {} };
    if (opts.json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    }
    if (opts.form) init.body = opts.form;
    if (opts.etag) init.headers["If-None-Match"] = opts.etag;

    var token = readToken();
    if (token) init.headers["Authorization"] = "Bearer " + token;

    var ctrl = null;
    var timer = null;
    if (timeoutMs) {
      ctrl = new AbortController();
      init.signal = ctrl.signal;
      timer = setTimeout(function () { ctrl.abort(); }, timeoutMs);
    }
    var slowTimer = setTimeout(function () { setSlow(true); }, SLOW_MS);

    return fetch(url, init).then(function (r) {
      if (timer) clearTimeout(timer);
      clearTimeout(slowTimer); setSlow(false);
      var etag = r.headers.get("ETag");
      if (r.status === 304) return { ok: false, status: 304, data: null, etag: etag };
      if (r.status === 204) return { ok: true, status: 204, data: null, etag: etag };
      return r.text().then(function (text) {
        var data = null;
        if (text) { try { data = JSON.parse(text); } catch (e) { data = text; } }
        return {
          ok: r.ok, status: r.status, data: data, etag: etag,
          detail: r.ok ? "" : detailOf(data, r.statusText)
        };
      });
    }).catch(function (e) {
      if (timer) clearTimeout(timer);
      clearTimeout(slowTimer); setSlow(false);
      return {
        ok: false, status: 0, data: null, netError: true,
        detail: e && e.name === "AbortError"
          ? "Request timed out after " + Math.round(timeoutMs / 1000) + "s."
          : (e && e.message) || "Network error"
      };
    });
  }

  /* Wrap any call so share failures raise the persistent banner rather than
   * hiding inside a per-section error. */
  function guarded(res) {
    if (res && res.status === 401) showAuthExpired();
    else if (isShareError(res)) showShareDown(res.detail);
    else if (state.shareDown && res.ok) clearShareDown();
    return res;
  }

  /* ==================================================================
   * 5. NORMALIZERS
   * ================================================================== */

  /* Two payload shapes reach the UI:
   *   POST .../tag -> body.tagged = raw tagged_output.json
   *                         (question_tags[], question_title_preview, no survey_journey)
   *   GET  .../tags      -> the projection from projections/survey_view.py
   *                         (questions[], question_text, survey_journey)
   * Everything downstream sees one shape. */
  function normalizeTagged(p) {
    if (!p) return null;
    var raw = p.questions || p.question_tags || [];
    return {
      tenant_id: p.tenant_id,
      survey_no: p.survey_no,
      zarca_id: p.zarca_id,
      survey_name: p.survey_name || "",
      schema_version: p.schema_version || "",
      generated_at: p.generated_at || "",
      project_tags: p.project_tags || {},
      survey_journey: p.survey_journey || null,
      metadata: p.metadata || {},
      question_tags: raw.map(function (q) {
        return {
          question_id: q.question_id,
          question_no: q.question_no,
          question_title_preview: q.question_title_preview || q.question_text || "",
          question_text: q.question_text || q.question_title_preview || "",
          rs_type: q.rs_type == null ? 0 : q.rs_type,
          is_custom_metric: !!q.is_custom_metric,
          is_content_message: !!q.is_content_message,
          coverage_metadata: q.coverage_metadata || null,
          tags: q.tags || {}
        };
      })
    };
  }

  function rememberName(survey) {
    if (survey && survey.survey_no != null && survey.survey_name) {
      state.surveyNames[survey.survey_no] = survey.survey_name;
    }
  }

  /* ==================================================================
   * 6. CHROME (toasts, busybar, banner, top status)
   * ================================================================== */

  function toast(kind, text, sticky) {
    var t = { id: state.nextToastId++, kind: kind, text: text, sticky: !!sticky };
    state.toasts.push(t);
    renderToasts();
    if (!t.sticky) setTimeout(function () { dismissToast(t.id); }, TOAST_MS);
    return t.id;
  }

  function dismissToast(id) {
    state.toasts = state.toasts.filter(function (t) { return t.id !== id; });
    renderToasts();
  }

  function renderToasts() {
    el.toasts.innerHTML = state.toasts.map(function (t) {
      return '<div class="toast ' + t.kind + '">' +
             '<span>' + U.escapeHtml(t.text) + "</span>" +
             '<button class="x" data-action="toast-dismiss" data-toast-id="' + t.id +
             '" aria-label="Dismiss">&times;</button></div>';
    }).join("");
  }

  function setBusy(delta) {
    state.busy = Math.max(0, state.busy + delta);
    el.busybar.classList.toggle("on", state.busy > 0);
    if (state.busy === 0) setSlow(false);
  }

  function setSlow(on) {
    if (state.slowWarn === on) return;
    state.slowWarn = on;
    renderTopStatus();
    renderRibbon();
  }

  /* ---- floating activity ribbon ----
   * The 2px busybar says "something is happening"; the ribbon says what. Calls
   * nest, so it is a stack: the newest job is the one on screen and the ribbon
   * only disappears once every job that opened it has closed. Always pair
   * ribbonStart with ribbonStop in the same `.then` that clears setBusy. */
  function ribbonStart(text) {
    var job = { id: state.nextRibbonId++, text: text };
    state.ribbons.push(job);
    renderRibbon();
    return job.id;
  }

  function ribbonStop(id) {
    if (id == null) return;
    var before = state.ribbons.length;
    state.ribbons = state.ribbons.filter(function (r) { return r.id !== id; });
    if (state.ribbons.length !== before) renderRibbon();
  }

  function renderRibbon() {
    var top = state.ribbons[state.ribbons.length - 1];
    if (!top) {
      el.ribbon.hidden = true;
      el.ribbon.innerHTML = "";
      el.ribbon.className = "";
      return;
    }
    var extra = state.ribbons.length - 1;
    el.ribbon.className = state.slowWarn ? "slow" : "";
    el.ribbon.innerHTML =
      '<span class="spinner"></span>' +
      '<span class="ribbon-text">' + U.escapeHtml(top.text) +
      (state.slowWarn ? " — still waiting on the network share" : "") + "</span>" +
      (extra > 0 ? '<span class="ribbon-more">+' + extra + "</span>" : "");
    el.ribbon.hidden = false;
  }

  function showShareDown(detail) {
    state.shareDown = true;
    state.banner = {
      kind: "lc-banner",
      text: "Cannot reach the data share — the server may have lost its SMB " +
            "connection or its credentials. " + (detail || ""),
      actionsHtml: '<button class="btn sm" data-action="retry-share">Retry</button>'
    };
    renderBanner();
  }

  function clearShareDown() {
    if (!state.shareDown) return;
    state.shareDown = false;
    state.banner = null;
    renderBanner();
  }

  /* A 401 means the token in localStorage is gone, expired, or rejected. We do
   * not refresh it ourselves — the shell owns that lifecycle and rotates the
   * key we read — so the honest move is to say so and let the user reload,
   * which picks up whatever the shell has written since. Distinct from the
   * share banner: nothing is wrong with the server or the share. */
  function showAuthExpired() {
    if (state.authExpired) return;      // one banner, not one per in-flight call
    state.authExpired = true;
    state.banner = {
      kind: "lc-banner",
      text: "Your session has expired or is not recognized. Reload the page to " +
            "pick up a fresh sign-in.",
      actionsHtml: '<button class="btn sm" data-action="reload-page">Reload</button>'
    };
    renderBanner();
  }

  function renderBanner() {
    var parts = [];
    if (state.banner) {
      parts.push('<div class="' + state.banner.kind + '"><span>' +
                 U.escapeHtml(state.banner.text) + "</span>" +
                 (state.banner.actionsHtml || "") + "</div>");
    }
    if (state.batch.running) {
      var p = state.batch.progress;
      var pct = p && p.total ? Math.round((p.tagged / p.total) * 100) : 0;
      parts.push(
        '<div class="lc-banner info"><span>Tagging every survey for corp ' +
        U.escapeHtml(String(state.corpNo)) + " — " +
        (p ? p.tagged + " of " + p.total + " tagged" : "starting…") +
        ", elapsed " + U.fmtElapsed(Date.now() - state.batch.startedAt) + ".</span>" +
        '<button class="btn sm ghost" data-action="batch-stop-watch">Stop watching</button></div>' +
        '<div class="progress"><i style="width:' + pct + '%"></i></div>'
      );
    }
    if (state.profileJob) {
      parts.push(
        '<div class="lc-banner warn"><span>Profile fetch running for corp ' +
        U.escapeHtml(String(state.profileJob.corpNo)) + " — elapsed " +
        U.fmtElapsed(Date.now() - state.profileJob.startedAt) +
        ". Progress is inferred from artifacts appearing on disk; a server " +
        "restart loses the job.</span>" +
        '<button class="btn sm ghost" data-action="profile-stop-watch">Stop watching</button></div>'
      );
    }
    el.banner.innerHTML = parts.join("");
  }

  function renderTopStatus() {
    var bits = [];
    if (state.slowWarn) bits.push("Still waiting on the network share…");
    else if (state.corpNo) {
      bits.push("Corp <strong>" + U.escapeHtml(String(state.corpNo)) + "</strong>");
      if (state.surveysLoaded) {
        var tagged = state.surveys.filter(function (s) { return s.tagged; }).length;
        bits.push("<strong>" + tagged + "</strong>/" + state.surveys.length + " tagged");
      }
    } else bits.push("Enter a corp number to begin");
    el.topStatus.innerHTML = bits.join(" &nbsp;·&nbsp; ");
  }

  /* ==================================================================
   * 7. SIDEBAR
   * ================================================================== */

  function renderTopTabs() {
    Array.prototype.forEach.call(el.topTabs.children, function (node) {
      node.classList.toggle("active", node.dataset.topview === state.topView);
    });
  }

  var TENANT_SECTIONS = [
    { key: "tags",     label: "Tenant Tags" },
    { key: "profile",  label: "Tenant Profile" },
    { key: "batch",    label: "Batch Tagging" },
    { key: "taxonomy", label: "Taxonomy" }
  ];

  /* The rows the sidebar is actually showing, given the filter chips and the
   * search box. Split out of renderNav because j/k step through exactly this
   * set — two copies of the predicate would drift the moment either changes. */
  function filteredSurveys() {
    var needle = state.navSearch.toLowerCase().trim();
    return state.surveys.filter(function (s) {
      if (state.navFilter === "tagged" && !s.tagged) return false;
      if (state.navFilter === "untagged" && s.tagged) return false;
      if (!needle) return true;
      var name = state.surveyNames[s.survey_no] || "";
      return String(s.survey_no).indexOf(needle) >= 0 ||
             name.toLowerCase().indexOf(needle) >= 0;
    });
  }

  function renderNav() {
    var surveys = state.topView === "surveys";
    el.navSearch.hidden = !surveys;
    el.navFilters.hidden = !surveys;

    if (state.topView === "tenant") {
      el.sidebarHeading.textContent = "Tenant";
      el.nav.innerHTML = TENANT_SECTIONS.map(function (s) {
        return '<li class="survey-item' + (state.tenantSection === s.key ? " active" : "") +
               '" role="button" tabindex="0"' +
               ' data-action="set-tenant-section" data-section="' + s.key + '">' +
               '<span class="stitle">' + s.label + "</span></li>";
      }).join("");
      return;
    }

    // Surveys
    el.sidebarHeading.innerHTML = "Surveys" +
      (state.surveysLoaded
        ? ' <button class="btn sm ghost" data-action="reload-surveys">Refresh</button>'
        : "");
    el.navFilters.innerHTML =
      '<div class="filter-group" style="margin-bottom:10px">' +
      ["all", "tagged", "untagged"].map(function (f) {
        return '<span class="toggle ' + (state.navFilter === f ? "on" : "off") +
               '" role="button" tabindex="0"' +
               ' data-action="set-nav-filter" data-filter="' + f + '">' +
               f.charAt(0).toUpperCase() + f.slice(1) + "</span>";
      }).join("") + "</div>";

    if (!state.corpNo) {
      el.nav.innerHTML = "";
      return;
    }
    if (state.surveysError) {
      el.nav.innerHTML = '<li class="survey-item" style="cursor:default">' +
        '<span class="stitle">' + U.escapeHtml(state.surveysError) + "</span></li>";
      return;
    }

    var rows = filteredSurveys();

    if (!rows.length) {
      el.nav.innerHTML = '<li class="survey-item" style="cursor:default"><span class="stitle">' +
        (state.surveys.length ? "No surveys match this filter."
                              : "No surveys on disk for this corp.") +
        "</span></li>";
      return;
    }

    el.nav.innerHTML = rows.map(function (s) {
      var dot = s.tagged === true ? "tagged" : (s.tagged === false ? "untagged" : "unknown");
      var name = state.surveyNames[s.survey_no] || "";
      return '<li class="survey-item' +
        (state.activeSurveyNo === s.survey_no ? " active" : "") +
        '" role="button" tabindex="0"' +
        ' data-action="select-survey" data-survey-no="' + s.survey_no + '"' +
        (name ? ' title="' + U.escapeHtml(name) + '"' : "") + ">" +
        '<span class="dot ' + dot + '"></span>' +
        '<span class="sno">' + s.survey_no + "</span>" +
        '<span class="stitle">' + U.escapeHtml(name) + "</span></li>";
    }).join("");
  }

  /* ==================================================================
   * 8. WORKSPACE ROUTER
   * ================================================================== */

  function renderAll() {
    renderTopTabs();
    renderNav();
    renderBanner();
    renderTopStatus();
    renderWorkspace();
  }

  function renderWorkspace() {
    if (state.topView === "tenant") return renderTenant();
    return renderSurveys();
  }

  var SURVEY_TABS = [
    { key: "summary",   label: "Summary" },
    { key: "project",   label: "Project Tags" },
    { key: "questions", label: "Questions" },
    { key: "journey",   label: "Journey" },
    { key: "raw",       label: "Raw JSON" }
  ];

  function tabsHtml(tabs, active, action) {
    return '<div class="tabs">' + tabs.map(function (t) {
      return '<div class="tab' + (active === t.key ? " active" : "") +
             '" role="button" tabindex="0"' +
             ' data-action="' + action + '" data-tab="' + t.key + '">' +
             U.escapeHtml(t.label) + "</div>";
    }).join("") + "</div>";
  }

  /* Loading placeholders. A sentence tells you to wait; a skeleton tells you
   * what is coming, and holds the layout so nothing jumps when it lands.
   * `.skel` / `.skel--line` / `.skel--card` are the only two shapes the design
   * contract defines; the shimmer itself is pure CSS. */
  function skeletonHtml(lines, cards) {
    var out = [], i;
    for (i = 0; i < lines; i++) out.push('<div class="skel skel--line"></div>');
    if (cards) {
      var boxes = [];
      for (i = 0; i < cards; i++) boxes.push('<div class="skel skel--card"></div>');
      out.push('<div class="tag-grid">' + boxes.join("") + "</div>");
    }
    return out.join("");
  }

  function renderSurveys() {
    if (!state.corpNo) {
      el.content.innerHTML = R.emptyState(
        "Enter a corp number to begin",
        "Type a corp number in the topbar to list its surveys, or type a survey " +
        "number directly to load one.", "");
      return;
    }

    if (state.activeSurveyNo == null) {
      el.content.innerHTML = state.surveysLoaded
        ? R.emptyState("Corp " + state.corpNo,
            "Pick a survey from the sidebar, or type a survey number in the topbar.", "")
        : skeletonHtml(2, 6);
      return;
    }

    if (state.surveyError) {
      var no = state.activeSurveyNo;
      if (state.surveyError.status === 404) {
        // Not an error — this is the normal untagged path, so it gets the
        // primary call to action rather than a red toast.
        el.content.innerHTML = R.emptyState(
          "Survey " + no + " has no tagged output yet",
          "Nothing has been written to this survey's folder on the share.",
          '<div class="btn-row">' +
          '<button class="btn primary" data-action="tag-survey" data-survey-no="' + no + '">Tag survey</button>' +
          "</div>");
      } else {
        el.content.innerHTML = R.errorState(
          "Could not load survey " + no,
          state.surveyError.detail || "Unknown error", "retry-survey");
      }
      return;
    }

    if (!state.survey) {
      el.content.innerHTML = skeletonHtml(3, 8);
      return;
    }

    var tabs = SURVEY_TABS.filter(function (t) {
      return t.key !== "journey" || !!state.survey.survey_journey;
    });
    var active = tabs.some(function (t) { return t.key === state.activeTab; })
      ? state.activeTab : "summary";
    // Keep the state in step with what is on screen: renderTabBody() repaints
    // from state.activeTab, so leaving a dropped tab (journey, on a survey with
    // no journey) in there would make the next filter keystroke render a body
    // no tab is highlighting.
    state.activeTab = active;

    el.content.innerHTML =
      '<div class="section-head"><div class="survey-header">' +
      R.surveyHeader(state.survey) + "</div>" +
      '<div class="btn-row">' +
      '<button class="btn ghost" data-action="download-json" data-what="survey">Download JSON</button>' +
      '<button class="btn danger sm" data-action="delete-survey-tags" data-survey-no="' + state.activeSurveyNo + '">Delete tags</button>' +
      "</div></div>" +
      tabsHtml(tabs, active, "set-tab") +
      '<div id="tabBody">' + surveyTabBody(active) + "</div>";
  }

  function surveyTabBody(tab) {
    var s = state.survey;
    // Summary is a read-out of the whole survey, so it takes no filter bar —
    // filtering it would mean summarising a subset and calling it the total.
    if (tab === "summary") {
      return typeof R.summary === "function"
        ? R.summary(s, state)
        : R.emptyState("Summary unavailable", "This build of render.js has no summary view.", "");
    }
    if (tab === "raw") return R.rawJson(s);
    if (tab === "journey") {
      return R.filterBar(state, { journeyToggle: true }) + R.journey(s, state);
    }
    if (tab === "questions") {
      return R.filterBar(state, { questionControls: true, columnControls: true }) +
             R.questions(s, state);
    }
    return R.filterBar(state, {}) + R.projectTags(s, state);
  }

  /* The filter chips and expand/collapse act on the survey that is on screen. */
  function currentSurvey() {
    return state.survey || { question_tags: [] };
  }

  /* Repaint only the results region so the filter inputs keep focus and caret.
   * The workspace is the scroll container, and swapping innerHTML collapses its
   * content height for an instant, which makes the browser clamp scrollTop to
   * 0 — so the position is saved and restored around the swap. */
  function renderTabBody() {
    var body = document.getElementById("tabBody");
    if (!body) return renderWorkspace();
    var scroll = el.workspace.scrollTop;
    var qs = document.getElementById("qSearch");
    var caret = qs ? qs.selectionStart : null;
    body.innerHTML = surveyTabBody(state.activeTab);
    el.workspace.scrollTop = scroll;
    var qs2 = document.getElementById("qSearch");
    if (qs2 && caret != null) { qs2.focus(); try { qs2.setSelectionRange(caret, caret); } catch (e) {} }
  }

  /* Expand / collapse mutates the cards already on screen instead of
   * re-rendering the list. Nothing about which questions are *shown* changes,
   * so a rebuild would only cost a reflow and risk the scroll jumping. */
  function questionCardNodes() {
    var body = document.getElementById("tabBody");
    if (!body) return [];
    return Array.prototype.filter.call(
      body.querySelectorAll(".question-card"),
      function (card) { return !!card.querySelector(".tags-wrap"); });   // content messages have none
  }

  function applyExpansion(open) {
    var cards = questionCardNodes();
    if (!cards.length) { renderTabBody(); return; }
    cards.forEach(function (card) { card.classList.toggle("expanded", open); });
  }

  /* `e` is a single toggle, so it has to read the screen rather than a flag:
   * anything still closed means "open everything", otherwise close everything. */
  function toggleExpandAll() {
    var cards = questionCardNodes();
    if (!cards.length) return;
    var anyClosed = cards.some(function (c) { return !c.classList.contains("expanded"); });
    if (anyClosed) ACTIONS["expand-all"]();
    else ACTIONS["collapse-all"]();
  }

  /* Scroll a node to the top of the workspace by adjusting the container's own
   * scrollTop. scrollIntoView() would be shorter, but it also scrolls every
   * scrollable ancestor including the document, which on a narrow viewport
   * drags the topbar off screen. The sticky filter bar is measured rather than
   * assumed so the target never lands underneath it. */
  function scrollIntoWorkspace(node) {
    if (!node || !el.workspace) return;
    var bar = el.workspace.querySelector(".filters");
    var pad = (bar ? bar.getBoundingClientRect().height : 0) + 16;
    var top = node.getBoundingClientRect().top - el.workspace.getBoundingClientRect().top;
    el.workspace.scrollTop += top - pad;
  }

  /* Find the project tag card for a dimension. render.js does not promise a
   * data-dim on .tag-card, so the label text is the fallback route in — it is
   * the same string ST.dims produced, not a guess. */
  function projectCardFor(dim) {
    var body = document.getElementById("tabBody");
    if (!body || !dim) return null;
    var direct = body.querySelector('.tag-card[data-dim="' + dim + '"]');
    if (direct) return direct;
    var label = D.meta(dim).label;
    var cards = body.querySelectorAll(".tag-card");
    for (var i = 0; i < cards.length; i++) {
      var name = cards[i].querySelector(".tag-name");
      if (name && name.textContent.indexOf(label) === 0) return cards[i];
    }
    return null;
  }

  /* Secondary dimensions live inside <details class="dim-more">; a card the
   * user cannot see is not a destination. */
  function revealAncestors(node) {
    var d = node && node.parentNode;
    while (d && d !== document.body) {
      if (d.tagName === "DETAILS") d.open = true;
      d = d.parentNode;
    }
  }

  /* Summary → the tag itself. A question row switches to Questions, narrows the
   * search to that question and opens its card; a project row switches to
   * Project Tags. Either way the tab body is rebuilt first, then scrolled —
   * the node does not exist until the swap has happened. */
  function gotoTag(dim, qkey) {
    if (!state.survey) return;
    var node;

    if (qkey) {
      state.activeTab = "questions";
      state.questionSearch = String(qkey);
      state.questionView = "cards";        // the table has no card to expand
      state.expandedQuestions.add(String(qkey));
      persist();
      renderWorkspace();
      node = document.querySelector('.question-card[data-q="' +
             String(qkey).replace(/"/g, '\\"') + '"]');
      if (node) node.classList.add("expanded");
    } else {
      state.activeTab = "project";
      persist();
      renderWorkspace();
      node = projectCardFor(dim);
    }

    if (node) { revealAncestors(node); scrollIntoWorkspace(node); }
  }

  /* ==================================================================
   * 9. TENANT SECTIONS
   * ================================================================== */

  function renderTenant() {
    if (!state.corpNo) {
      el.content.innerHTML = R.emptyState(
        "Enter a corp number", "Tenant-level actions need a corp number.", "");
      return;
    }
    var fn = { tags: tenantTagsHtml, profile: profileHtml, batch: batchHtml,
               taxonomy: taxonomyHtml }[state.tenantSection];
    el.content.innerHTML = fn ? fn() : "";
  }

  function tenantTagsHtml() {
    var head = '<div class="section-head"><h2>Tenant Tags</h2><div class="btn-row">' +
      '<button class="btn primary" data-action="build-tenant-tags">Build</button>' +
      '<button class="btn" data-action="reload-tenant-tags">Reload</button>' +
      '<button class="btn ghost" data-action="download-json" data-what="tenant">Download JSON</button>' +
      '<button class="btn danger sm" data-action="delete-tenant-tags">Delete</button></div></div>';

    if (state.tenantTagsError) {
      if (state.tenantTagsError.status === 404) {
        return head + R.emptyState("No tenant tags yet",
          "Build them from the Parallel.ai profile and corporate signals on the share.",
          '<button class="btn primary" data-action="build-tenant-tags">Build tenant tags</button>');
      }
      if (state.tenantTagsError.status === 422) {
        // Expected on first run: no profile on disk to derive tenant tags from.
        return head + '<div class="lc-banner info"><span>' +
          U.escapeHtml(state.tenantTagsError.detail) + "</span>" +
          '<button class="btn sm" data-action="set-tenant-section" data-section="profile">' +
          "Open Tenant Profile</button></div>";
      }
      return head + R.errorState("Could not load tenant tags",
        state.tenantTagsError.detail, "reload-tenant-tags");
    }
    if (!state.tenantTags) return head + skeletonHtml(1, 6);
    return head + R.tenantTags(state.tenantTags, state);
  }

  function profileHtml() {
    var head = '<div class="section-head"><h2>Tenant Profile</h2><div class="btn-row">' +
      '<button class="btn" data-action="profile-reload">Reload</button>' +
      '<button class="btn ghost" data-action="download-json" data-what="profile">Download JSON</button>' +
      '<button class="btn danger sm" data-action="profile-delete">Delete artifacts</button></div></div>';

    var isSmx = state.config.profile_source === "smx";
    var agentBoxes =
      '<div class="field"><label>Agents</label><div class="btn-row">' +
      AGENTS.map(function (a) {
        return '<label class="checkline"><input type="checkbox" class="pfAgent" value="' +
               a + '" checked /> ' + a + "</label>";
      }).join("") + "</div></div>";
    var forceBox =
      '<div class="field"><label>Options</label>' +
      '<label class="checkline"><input type="checkbox" id="pfForce" /> ' +
      "Force refresh (ignore cached artifacts)</label>";

    var form;
    if (isSmx) {
      // SMX resolves in three steps and needs no website — the Research API
      // already knows the tenant's URL.
      form =
        '<div class="tenant-panel"><h3>Resolve from SoGo Research API</h3>' +
        '<ol class="micro" style="margin:4px 0 10px 18px">' +
        "<li>Artifacts already on the image server &rarr; used as-is</li>" +
        '<li>Otherwise read the generated profile from apismx — ' +
        '<span class="mono">GET /AIAccountProfile/Details</span></li>' +
        '<li>Otherwise trigger generation — ' +
        '<span class="mono">POST /AIAccountProfile/Generate</span> — then wait ~' +
        (state.config.smx_generate_wait_seconds || 90) + "s for it</li></ol>" +
        '<div class="form-grid">' + agentBoxes + forceBox +
        (state.config.smx_allow_generate
          ? '<label class="checkline"><input type="checkbox" id="pfGenerate" checked /> ' +
            "Generate if missing (starts research)</label>"
          : '<p class="micro">Generation is disabled server-side ' +
            "(SMX_ALLOW_GENERATE=false).</p>") +
        "</div></div>" +
        '<div class="btn-row" style="margin-top:12px">' +
        '<button class="btn primary" data-action="profile-fetch-sync"' +
        (state.profileJob ? " disabled" : "") + ">Resolve profile</button>" +
        '<button class="btn ghost" data-action="profile-fetch"' +
        (state.profileJob ? " disabled" : "") + ">Resolve in background</button>" +
        // Read-only step 2 on its own: same endpoint, generation forced off, so
        // it can never start (or be billed for) research.
        '<button class="btn ghost" data-action="profile-lookup"' +
        (state.profileJob ? " disabled" : "") +
        ' title="GET /AIAccountProfile/Details only — never generates">' +
        "Look up in apismx</button>" +
        '<p class="micro" style="margin:6px 0 0">' +
        "&ldquo;Look up&rdquo; checks the share and " +
        '<span class="mono">GET /AIAccountProfile/Details</span> only — it ' +
        "reports nothing found rather than starting research.</p>" +
        "</div></div>";
    } else {
      form =
        '<div class="tenant-panel"><h3>Fetch from Parallel.ai</h3>' +
        '<p class="micro" style="margin-top:0">Takes 10–30 minutes. Runs in the ' +
        'background by default; progress is inferred from artifacts appearing on disk.</p>' +
        '<div class="form-grid">' +
        '<div class="field"><label for="pfWebsite">Website</label>' +
        '<input type="text" id="pfWebsite" placeholder="https://acme.com" /></div>' +
        agentBoxes + forceBox + "</div>" +
        "</div>" +
        '<div class="btn-row" style="margin-top:12px">' +
        '<button class="btn primary" data-action="profile-fetch"' +
        (state.profileJob ? " disabled" : "") + ">Fetch in background</button>" +
        '<button class="btn ghost" data-action="profile-fetch-sync"' +
        (state.profileJob ? " disabled" : "") + ">Run synchronously (blocks up to 30 min)</button>" +
        "</div></div>";
    }

    var body;
    if (state.profileError && state.profileError.status === 404) {
      body = R.emptyState("No profile artifacts yet",
        "Nothing under this corp's tenant_profile folder on the share.", "");
    } else if (state.profileError) {
      body = R.errorState("Could not load profile", state.profileError.detail, "profile-reload");
    } else if (!state.profile) {
      body = skeletonHtml(1, 4);
    } else {
      body = R.profileSummary(state.profile) + R.profileAgents(state.profileAgents);
    }
    return head + form + body;
  }

  function batchHtml() {
    var running = state.batch.running;
    var head = '<div class="section-head"><h2>Batch Tagging</h2><div class="btn-row">' +
      '<button class="btn primary" data-action="batch-tag"' + (running ? " disabled" : "") +
      ">Tag all surveys</button></div></div>";

    var note = '<p class="micro">Incremental by default — surveys whose inputs are ' +
      'unchanged since the last run are skipped. The progress bar counts ' +
      'tagged_output.json files appearing on the share, so it advances while the ' +
      "run is still in flight.</p>";

    if (running) {
      var p = state.batch.progress;
      var pct = p && p.total ? Math.round((p.tagged / p.total) * 100) : 0;
      return head + note +
        '<div class="progress"><i style="width:' + pct + '%"></i></div>' +
        '<div class="progress-label">' +
        (p ? p.tagged + " of " + p.total + " tagged" : "starting…") +
        " · elapsed " + U.fmtElapsed(Date.now() - state.batch.startedAt) + "</div>";
    }
    if (state.batch.result) return head + note + R.batchResult(state.batch.result);
    return head + note + R.emptyState("No run yet",
      "Tag every survey under corp " + state.corpNo + ".", "");
  }

  function taxonomyHtml() {
    var head = '<div class="section-head"><h2>Taxonomy</h2></div>';
    if (!state.taxonomy) {
      return head + R.errorState("Taxonomy unavailable",
        "GET /api/taxonomy did not load; dimension labels fall back to their key names.",
        "reload-taxonomy");
    }
    return head + R.taxonomyTable(state.taxonomy, state.navSearch);
  }

  /* ==================================================================
   * 10. CONTROLLERS
   * ================================================================== */

  function validNumber(raw) {
    var v = String(raw == null ? "" : raw).trim();
    return /^\d+$/.test(v) ? Number(v) : null;
  }

  function loadConfig() {
    return api(ROUTES.config()).then(function (r) {
      // Degrade, don't break: the defaults in state.config describe the
      // Parallel workflow, which is the stricter form of the two.
      if (!r.ok || !r.data) return;
      state.config = {
        profile_source: r.data.profile_source || "parallel",
        smx_allow_generate: r.data.smx_allow_generate !== false,
        smx_generate_wait_seconds: r.data.smx_generate_wait_seconds || 90
      };
    });
  }

  /* Who does the server think we are, per the token we just sent? Only used to
   * answer "which corp?" when nothing else has. Never rejects: no token, or a
   * token with no corp claim, is the ordinary ops case where the number is
   * typed in by hand. */
  function loadMe() {
    return api(ROUTES.me()).then(function (r) {
      if (!r.ok || !r.data) return;
      state.tokenCorpNo = validNumber(r.data.corp_no);
    });
  }

  function loadTaxonomy() {
    return api(ROUTES.taxonomy()).then(function (r) {
      if (!r.ok) {
        // Degrade, don't break: dims fall back to labelFor(key).
        toast("info", "Taxonomy unavailable — dimension labels use their key names.");
        return;
      }
      state.taxonomy = r.data;
      D.set(r.data);
      try { sessionStorage.setItem(SS.taxonomy, JSON.stringify(r.data)); } catch (e) {}
    });
  }

  function loadTenant(corpNo) {
    state.corpNo = corpNo;
    state.surveys = [];
    state.surveysLoaded = false;
    state.surveysError = null;
    state.activeSurveyNo = null;
    state.survey = null;
    state.surveyError = null;
    state.etags = {};
    state.tenantTags = null; state.tenantTagsError = null;
    state.profile = null; state.profileError = null; state.profileAgents = {};
    state.batch.result = null; state.batch.progress = null;
    persist();
    renderAll();
    return loadSurveyList();
  }

  function loadSurveyList(opts) {
    opts = opts || {};
    if (!state.corpNo) return Promise.resolve();
    var rib = null;
    if (!opts.quiet) {
      setBusy(1);
      rib = ribbonStart("Fetching surveys for corp " + state.corpNo + "…");
    }
    return api(ROUTES.surveyList(state.corpNo)).then(guarded).then(function (r) {
      if (!opts.quiet) { setBusy(-1); ribbonStop(rib); }
      if (r.status === 404) {
        // Empty state, not an error — the corp may simply have no surveys.
        state.surveys = []; state.surveysLoaded = true;
        state.surveysError = "No surveys on disk for corp " + state.corpNo + ".";
      } else if (!r.ok) {
        state.surveysError = r.detail || "Could not list surveys.";
        state.surveysLoaded = false;
      } else {
        var listed = (r.data && r.data.surveys) || [];
        // Keep any provisional rows the user typed in that the listing lacks.
        var provisional = state.surveys.filter(function (s) {
          return s.provisional && !listed.some(function (x) { return x.survey_no === s.survey_no; });
        });
        state.surveys = listed.concat(provisional);
        state.surveysLoaded = true;
        state.surveysError = null;
      }
      renderNav(); renderTopStatus();
      if (state.topView === "surveys") renderWorkspace();
      return r;
    });
  }

  function selectSurvey(no) {
    if (no == null) return;
    state.topView = "surveys";
    state.activeSurveyNo = no;
    state.expandedQuestions = new Set();
    if (!state.surveys.some(function (s) { return s.survey_no === no; })) {
      // Typed directly and not in the listing — show it anyway so the sidebar
      // reflects what is on screen even when the listing 404s or lags.
      state.surveys = state.surveys.concat([{ survey_no: no, tagged: null, provisional: true }]);
    }
    persist();
    renderAll();
    return loadSurveyView(no);
  }

  function etagKey(no) { return no + (state.includeCandidates ? "|jc" : ""); }

  function loadSurveyView(no, opts) {
    opts = opts || {};
    if (no == null) return Promise.resolve();
    var key = etagKey(no);
    var etag = opts.bustEtag ? null : state.etags[key];
    setBusy(1);
    var rib = ribbonStart("Loading survey " + no + "…");
    return api(ROUTES.surveyTags(state.corpNo, no, state.includeCandidates), { etag: etag })
      .then(guarded).then(function (r) {
        setBusy(-1); ribbonStop(rib);
        if (r.status === 304) return r;          // cached view is current
        if (r.status === 404) {
          state.survey = null;
          state.surveyError = { status: 404, detail: r.detail };
          markTagged(no, false);
        } else if (!r.ok) {
          state.surveyError = { status: r.status, detail: r.detail };
        } else {
          state.survey = normalizeTagged(r.data);
          state.surveyError = null;
          if (r.etag) state.etags[key] = r.etag;
          rememberName(state.survey);
          markTagged(no, true);
        }
        renderNav(); renderTopStatus();
        if (state.topView === "surveys") renderWorkspace();
        return r;
      });
  }

  function markTagged(no, tagged) {
    state.surveys = state.surveys.map(function (s) {
      return s.survey_no === no ? { survey_no: no, tagged: tagged } : s;
    });
  }

  function tagSurvey(no) {
    if (no == null || !state.corpNo) return;
    setBusy(1);
    var rib = ribbonStart("Tagging survey " + no + " — this can take 30–90s…");
    var url = ROUTES.tagSurvey(state.corpNo, no);
    // No timeout: a full LLM pass takes 30-90s.
    return api(url, { method: "POST", timeoutMs: null }).then(guarded).then(function (r) {
      setBusy(-1); ribbonStop(rib);
      if (!r.ok) {
        toast("err", "Tag failed: " + (r.detail || r.status), true);
        state.surveyError = { status: r.status, detail: r.detail };
        renderWorkspace();
        return;
      }
      state.activeSurveyNo = no;
      delete state.etags[etagKey(no)];
      markTagged(no, true);
      toast("ok", "Survey " + no + " tagged" +
        (r.data && r.data.llm_enabled === false ? " (deterministic — LLM disabled)" : "") + ".");

      if (r.data && r.data.tagged) {
        // Render straight from the POST body; then refresh in the background
        // purely to pick up the ETag so the next visit can 304.
        state.survey = normalizeTagged(r.data.tagged);
        state.surveyError = null;
        rememberName(state.survey);
        renderAll();
        loadSurveyView(no, { bustEtag: true });
      } else {
        // orchestrator skipped and there was no prior file — fall back to GET.
        loadSurveyView(no, { bustEtag: true });
      }
    });
  }

  function deleteSurveyTags(no) {
    if (!window.confirm("Delete tagged_output.json for survey " + no + " on the share?")) return;
    setBusy(1);
    return api(ROUTES.surveyTags(state.corpNo, no, false), { method: "DELETE" })
      .then(guarded).then(function (r) {
        setBusy(-1);
        if (!r.ok && r.status !== 404) { toast("err", "Delete failed: " + r.detail, true); return; }
        toast("ok", r.status === 404 ? "Already gone." : "Deleted tags for survey " + no + ".");
        delete state.etags[etagKey(no)];
        state.survey = null;
        state.surveyError = { status: 404, detail: "deleted" };
        markTagged(no, false);
        renderAll();
      });
  }

  // ---- tenant tags ----

  function loadTenantTags() {
    setBusy(1);
    var rib = ribbonStart("Loading tenant tags for corp " + state.corpNo + "…");
    return api(ROUTES.tenantTags(state.corpNo)).then(guarded).then(function (r) {
      setBusy(-1); ribbonStop(rib);
      state.tenantTags = r.ok ? r.data : null;
      state.tenantTagsError = r.ok ? null : { status: r.status, detail: r.detail };
      renderTenant();
    });
  }

  function buildTenantTags() {
    setBusy(1);
    var rib = ribbonStart("Building tenant tags for corp " + state.corpNo + "…");
    return api(ROUTES.tenantTagsBuild(state.corpNo), { method: "POST", timeoutMs: null })
      .then(guarded).then(function (r) {
        setBusy(-1); ribbonStop(rib);
        if (!r.ok) {
          state.tenantTagsError = { status: r.status, detail: r.detail };
          state.tenantTags = null;
          if (r.status !== 422) toast("err", "Build failed: " + r.detail, true);
          renderTenant();
          return;
        }
        toast("ok", "Tenant tags built.");
        // The POST wraps the artifact ({tenant_id, tenant_tags}); re-read so
        // state always holds the same shape GET returns.
        state.tenantTagsError = null;
        return loadTenantTags();
      });
  }

  function deleteTenantTags() {
    if (!window.confirm("Delete tenant_tags.json for corp " + state.corpNo + " on the share?")) return;
    setBusy(1);
    return api(ROUTES.tenantTags(state.corpNo), { method: "DELETE" }).then(guarded).then(function (r) {
      setBusy(-1);
      if (!r.ok && r.status !== 404) { toast("err", "Delete failed: " + r.detail, true); return; }
      toast("ok", "Tenant tags deleted.");
      state.tenantTags = null;
      state.tenantTagsError = { status: 404, detail: "deleted" };
      renderTenant();
    });
  }

  // ---- tenant profile ----

  function loadProfile(opts) {
    opts = opts || {};
    var rib = null;
    if (!opts.quiet) {
      setBusy(1);
      rib = ribbonStart("Loading tenant profile for corp " + state.corpNo + "…");
    }
    return api(ROUTES.profile(state.corpNo)).then(guarded).then(function (r) {
      if (!opts.quiet) { setBusy(-1); ribbonStop(rib); }
      state.profile = r.ok ? r.data : null;
      state.profileError = r.ok ? null : { status: r.status, detail: r.detail };
      if (r.ok) {
        // Mark agents with no artifact so the accordion says "missing"
        // instead of firing a fetch that is guaranteed to 404.
        AGENTS.forEach(function (a) {
          if (r.data["has_" + a] === false && !state.profileAgents[a]) {
            state.profileAgents[a] = "missing";
          }
        });
      }
      if (!opts.quiet && state.topView === "tenant" && state.tenantSection === "profile") renderTenant();
      return r;
    });
  }

  function loadAgent(agent) {
    if (state.profileAgents[agent] && state.profileAgents[agent] !== "missing") return;
    state.profileAgents[agent] = "loading";
    renderTenant();
    return api(ROUTES.profileAgent(state.corpNo, agent)).then(guarded).then(function (r) {
      state.profileAgents[agent] = r.ok ? r.data
        : (r.status === 404 ? "missing" : "error: " + (r.detail || r.status));
      renderTenant();
    });
  }

  // Which step of the share -> apismx -> generate cascade produced the profile.
  // Worth surfacing: "found on the share" and "generated fresh research" cost
  // wildly different things, and the counts alone cannot tell them apart.
  var RESOLVED_VIA = {
    disk:      "already on the image server",
    smx:       "read from apismx",
    generated: "generated via apismx"
  };

  function profileOutcomeText(d, c) {
    var via = RESOLVED_VIA[d.resolved_via];
    return "Profile ready" + (via ? " — " + via : "") + ". " +
           "Fetched " + (c.fetched || 0) + ", cached " + (c.cache_hits || 0) +
           ", failed " + (c.failures || 0) + ".";
  }

  function collectProfileForm() {
    var website = (document.getElementById("pfWebsite") || {}).value || "";
    var agents = Array.prototype.slice
      .call(document.querySelectorAll(".pfAgent:checked"))
      .map(function (n) { return n.value; });
    var force = !!(document.getElementById("pfForce") || {}).checked;
    var genBox = document.getElementById("pfGenerate");
    return {
      website: website.trim(), agents: agents, force: force,
      // Absent checkbox (generation disabled server-side) must read as false,
      // not as the default-true the API would otherwise apply.
      allowGenerate: !!(genBox && genBox.checked)
    };
  }

  /* background: fire-and-forget + poll. lookupOnly: the read-only leg of the
   * cascade — share and GET /AIAccountProfile/Details, generation forced off
   * regardless of the checkbox, so a miss is a 404 and never a research run. */
  function startProfileFetch(background, lookupOnly) {
    var isSmx = state.config.profile_source === "smx";
    var f = collectProfileForm();
    if (!isSmx && f.website.length < 4) {
      toast("err", "Enter the tenant website first.", true); return;
    }
    if (!f.agents.length) { toast("err", "Pick at least one agent.", true); return; }
    if (!background && !lookupOnly && !isSmx && !window.confirm(
      "Run synchronously? This blocks for up to 30 minutes. Background mode is " +
      "usually what you want.")) return;
    if (!background && !lookupOnly && isSmx && f.allowGenerate && !window.confirm(
      "If this tenant has no profile on the share or in SMX, research will be " +
      "generated for it. Continue?")) return;

    var body = { website: f.website, agents: f.agents, force: f.force,
                 allow_generate: !lookupOnly && (isSmx ? f.allowGenerate : true) };

    if (!background) {
      setBusy(1);
      var msg = lookupOnly
        ? "Looking up the profile — share, then GET /AIAccountProfile/Details…"
        : isSmx
          ? "Resolving profile — share, then apismx, then generate…"
          : "Profile fetch running synchronously — this can take 30 minutes…";
      var rib = ribbonStart(msg);
      return api(ROUTES.profileFetch(state.corpNo, false),
                 { method: "POST", json: body, timeoutMs: null })
        .then(guarded).then(function (r) {
          setBusy(-1); ribbonStop(rib);
          if (!r.ok) {
            // A lookup miss is the expected negative answer, not a failure:
            // the server 404s precisely because it declined to generate.
            if (lookupOnly && r.status === 404) {
              toast("warn", "No profile for corp " + state.corpNo +
                            " on the share or in apismx. Nothing was generated.", true);
              return;
            }
            toast("err", (lookupOnly ? "Lookup failed: " : "Fetch failed: ") + r.detail, true);
            return;
          }
          var d = r.data || {};
          if (d.pending) {
            // Generation started but did not finish inside the server's wait
            // window. It is still running there, so hand off to the same poller
            // the background path uses instead of reporting a failure.
            toast("info", d.detail || "Generation still running — watching for artifacts…", true);
            state.profileJob = { corpNo: state.corpNo, startedAt: Date.now(),
                                 agents: f.agents, polls: 0 };
            persist();
            startProfilePoll();
            renderAll();
            return;
          }
          var c = d.counts || {};
          toast("ok", profileOutcomeText(d, c));
          state.profileAgents = {};
          return loadProfile();
        });
    }

    return api(ROUTES.profileFetch(state.corpNo, true), { method: "POST", json: body })
      .then(guarded).then(function (r) {
        if (!r.ok && r.status !== 202) { toast("err", "Fetch failed: " + r.detail, true); return; }
        state.profileJob = {
          corpNo: state.corpNo, startedAt: Date.now(),
          agents: f.agents, polls: 0
        };
        persist();
        toast("ok", "Profile fetch started in the background.");
        startProfilePoll();
        renderAll();
      });
  }

  function startProfilePoll() {
    stopProfilePoll(true);
    state.profileTimer = setInterval(pollProfile, POLL_PROFILE_MS);
    pollProfile();
  }

  function pollProfile() {
    var job = state.profileJob;
    if (!job) return stopProfilePoll();
    job.polls = (job.polls || 0) + 1;
    persist();
    renderBanner();

    if (job.polls > PROFILE_MAX_POLLS) {
      toast("warn", "Stopped watching the profile fetch after " +
                    Math.round(PROFILE_MAX_POLLS * POLL_PROFILE_MS / 60000) +
                    " minutes. It may still be running on the server — reload later.", true);
      stopProfilePoll();
      return;
    }

    loadProfile({ quiet: true }).then(function (r) {
      if (!r || !r.ok || !state.profileJob) return;
      var done = state.profileJob.agents.every(function (a) { return r.data["has_" + a]; });
      if (done) {
        toast("ok", "Profile fetch complete.");
        stopProfilePoll();
        state.profileAgents = {};
        renderAll();
      } else if (state.topView === "tenant" && state.tenantSection === "profile") {
        renderTenant();
      }
    });
  }

  function stopProfilePoll(keepJob) {
    if (state.profileTimer) { clearInterval(state.profileTimer); state.profileTimer = null; }
    if (!keepJob) { state.profileJob = null; persist(); renderAll(); }
  }

  function deleteProfile() {
    if (!window.confirm("Delete all Parallel.ai artifacts for corp " + state.corpNo + " on the share?")) return;
    setBusy(1);
    return api(ROUTES.profile(state.corpNo), { method: "DELETE" }).then(guarded).then(function (r) {
      setBusy(-1);
      if (!r.ok) { toast("err", "Delete failed: " + r.detail, true); return; }
      var n = ((r.data && r.data.removed) || []).length;
      toast("ok", n ? "Removed " + n + " artifact(s)." : "Nothing to remove.");
      state.profileAgents = {};
      return loadProfile();
    });
  }

  // ---- batch ----

  function runBatch() {
    if (state.batch.running) return;
    state.batch.running = true;
    state.batch.startedAt = Date.now();
    state.batch.result = null;
    state.batch.progress = { tagged: state.surveys.filter(function (s) { return s.tagged; }).length,
                             total: state.surveys.length };
    renderAll();

    // Poll the listing while the POST is in flight. tagged_output.json appears
    // per survey as the bounded-parallel run completes each one, so this is a
    // real determinate progress bar with no backend support needed.
    state.batch.timer = setInterval(pollBatchProgress, POLL_BATCH_MS);

    var url = ROUTES.batchTag(state.corpNo);
    return api(url, { method: "POST", timeoutMs: null }).then(guarded).then(function (r) {
      stopBatchPoll(true);
      state.batch.running = false;
      if (!r.ok) {
        toast("err", "Batch run failed: " + (r.detail || r.status), true);
      } else {
        state.batch.result = r.data;
        toast("ok", "Batch complete: " + (r.data.processed || 0) + " processed, " +
                    (r.data.skipped || 0) + " skipped, " + (r.data.failed || 0) + " failed.");
      }
      return loadSurveyList({ quiet: true }).then(function () { renderAll(); });
    });
  }

  function pollBatchProgress() {
    loadSurveyList({ quiet: true }).then(function () {
      if (!state.batch.running) return;
      state.batch.progress = {
        tagged: state.surveys.filter(function (s) { return s.tagged; }).length,
        total: state.surveys.length
      };
      renderBanner();
      if (state.topView === "tenant" && state.tenantSection === "batch") renderTenant();
    });
  }

  // Clears the poll only. It cannot cancel the server run — the label says so.
  function stopBatchPoll(silent) {
    if (state.batch.timer) { clearInterval(state.batch.timer); state.batch.timer = null; }
    if (!silent) {
      state.batch.running = false;
      toast("info", "Stopped watching. The run continues on the server.");
      renderAll();
    }
  }

  // ---- download ----

  function downloadJson(what) {
    var payload = null, name = "tagged.json";
    if (what === "survey" && state.survey) {
      payload = state.survey;
      name = "tagged_" + state.corpNo + "_" + state.activeSurveyNo + ".json";
    } else if (what === "tenant" && state.tenantTags) {
      payload = state.tenantTags; name = "tenant_tags_" + state.corpNo + ".json";
    } else if (what === "profile" && state.profile) {
      payload = state.profile; name = "tenant_profile_" + state.corpNo + ".json";
    }
    if (!payload) { toast("info", "Nothing to download yet."); return; }

    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function checkShare() {
    return api(ROUTES.shareHealth()).then(function (r) {
      if (r.ok && r.data && r.data.reachable === false) {
        showShareDown("Server reports: " + (r.data.error || "unreachable") +
                      " (" + r.data.root + ")");
        return false;
      }
      if (r.ok) clearShareDown();
      return true;
    });
  }

  /* ==================================================================
   * 11. DISPATCH
   * ================================================================== */

  var ACTIONS = {
    "set-topview": function (n) {
      state.topView = n.dataset.topview;
      persist();
      renderAll();
      if (state.topView === "tenant") loadTenantSection();
    },
    "set-tab": function (n) { state.activeTab = n.dataset.tab; persist(); renderWorkspace(); },
    "goto-tag": function (n) { gotoTag(n.dataset.dim, n.dataset.qkey); },

    "toggle-theme": function () {
      ui.theme = ui.theme === "dark" ? "light" : "dark";
      applyChrome(); persistChrome(); syncChromeButtons();
    },
    "toggle-density": function () {
      ui.density = ui.density === "compact" ? "comfortable" : "compact";
      applyChrome(); persistChrome(); syncChromeButtons();
    },

    "set-tenant-section": function (n) {
      state.topView = "tenant";
      state.tenantSection = n.dataset.section;
      renderAll();
      loadTenantSection();
    },
    "select-survey": function (n) { selectSurvey(Number(n.dataset.surveyNo)); },
    "set-nav-filter": function (n) { state.navFilter = n.dataset.filter; renderNav(); },
    "reload-surveys": function () { loadSurveyList(); },
    "retry-share": function () {
      checkShare().then(function (ok) { if (ok && state.corpNo) loadSurveyList(); });
    },
    "reload-page": function () { window.location.reload(); },
    "retry-survey": function () { loadSurveyView(state.activeSurveyNo, { bustEtag: true }); },
    "reload-taxonomy": function () { loadTaxonomy().then(renderAll); },

    "tag-survey": function (n) { tagSurvey(Number(n.dataset.surveyNo)); },
    "delete-survey-tags": function (n) { deleteSurveyTags(Number(n.dataset.surveyNo)); },
    "toggle-candidates": function () {
      state.includeCandidates = !state.includeCandidates;
      loadSurveyView(state.activeSurveyNo, { bustEtag: true });
    },

    "toggle-source": function (n) {
      var s = n.dataset.source;
      if (state.sourceFilter.has(s)) state.sourceFilter.delete(s);
      else state.sourceFilter.add(s);
      renderTabBody();
    },
    "toggle-lowconf": function () { state.lowConfOnly = !state.lowConfOnly; renderTabBody(); },
    "reset-filters": function () {
      state.sourceFilter = new Set(U.ALL_SOURCES);
      state.lowConfOnly = false;
      state.questionSearch = "";
      renderTabBody();
    },
    "set-view": function (n) { state.questionView = n.dataset.view; renderTabBody(); },
    "set-cols": function (n) { state.tableCols = n.dataset.cols; renderTabBody(); },
    "toggle-q": function (n) {
      var q = n.dataset.q;
      var open = !state.expandedQuestions.has(q);
      if (open) state.expandedQuestions.add(q);
      else state.expandedQuestions.delete(q);
      // Toggle the card in place. Re-rendering the whole list would rebuild
      // every sibling and drop the scroll position back to the top.
      var card = n.closest(".question-card");
      if (card) card.classList.toggle("expanded", open);
      else renderTabBody();
    },
    "expand-all": function () {
      var list = currentSurvey().question_tags || [];
      // Same string key render.js emits as data-q.
      list.forEach(function (q) {
        state.expandedQuestions.add(
          String(q.question_id != null ? q.question_id : (q.question_no == null ? "" : q.question_no)));
      });
      applyExpansion(true);
    },
    "collapse-all": function () { state.expandedQuestions.clear(); applyExpansion(false); },

    "build-tenant-tags": function () { buildTenantTags(); },
    "tag-tenant": function () { buildTenantTags(); },
    "reload-tenant-tags": function () { loadTenantTags(); },
    "delete-tenant-tags": function () { deleteTenantTags(); },

    "profile-reload": function () { state.profileAgents = {}; loadProfile(); },
    "profile-fetch": function () { startProfileFetch(true); },
    "profile-fetch-sync": function () { startProfileFetch(false); },
    "profile-lookup": function () { startProfileFetch(false, true); },
    "profile-stop-watch": function () {
      stopProfilePoll();
      toast("info", "Stopped watching. The fetch continues on the server.");
    },
    "profile-delete": function () { deleteProfile(); },
    "load-agent": function (n) { loadAgent(n.dataset.agent); },

    "batch-tag": function () { runBatch(); },
    "batch-stop-watch": function () { stopBatchPoll(false); },

    "download-json": function (n) { downloadJson(n.dataset.what); },
    "toast-dismiss": function (n) { dismissToast(Number(n.dataset.toastId)); }
  };

  function loadTenantSection() {
    if (!state.corpNo) return;
    var s = state.tenantSection;
    if (s === "tags" && !state.tenantTags && !state.tenantTagsError) loadTenantTags();
    else if (s === "profile" && !state.profile && !state.profileError) loadProfile();
  }

  /* ==================================================================
   * 11b. KEYBOARD
   * ================================================================== */

  function isTypingTarget(n) {
    if (!n || !n.tagName) return false;
    return n.tagName === "INPUT" || n.tagName === "TEXTAREA" ||
           n.tagName === "SELECT" || n.isContentEditable === true;
  }

  // The two boxes `/` and Esc act on. The corp / survey entry fields are not
  // filters — clearing what someone is halfway through typing is not a service.
  function isFilterInput(n) {
    return !!n && (n.id === "qSearch" || n === el.navSearch);
  }

  /* Whichever filter box is on screen. The question search only exists while
   * the Questions tab is rendered, so it wins when present. */
  function visibleFilterInput() {
    var qs = document.getElementById("qSearch");
    if (qs) return qs;
    if (el.navSearch && !el.navSearch.hidden) return el.navSearch;
    return null;
  }

  function clearFilterInput(box) {
    box.value = "";
    if (box.id === "qSearch") { state.questionSearch = ""; renderTabBody(); }
    else {
      state.navSearch = "";
      if (state.topView === "tenant" && state.tenantSection === "taxonomy") renderTenant();
      else renderNav();
    }
  }

  /* j / k walk the list the sidebar is actually showing — the filter chips and
   * the search box narrow it, and the keyboard must not teleport to a row that
   * is not on screen. Stops at both ends rather than wrapping: silently
   * jumping from the last survey to the first reads as a bug. */
  function stepSurvey(delta) {
    if (state.topView !== "surveys") return;
    var rows = filteredSurveys();
    if (!rows.length) return;
    var i = -1;
    for (var k = 0; k < rows.length; k++) {
      if (rows[k].survey_no === state.activeSurveyNo) { i = k; break; }
    }
    var next = i < 0 ? (delta > 0 ? 0 : rows.length - 1) : i + delta;
    if (next < 0 || next >= rows.length) return;
    selectSurvey(rows[next].survey_no);
    var li = el.nav.querySelector('.survey-item[data-survey-no="' + rows[next].survey_no + '"]');
    if (li && li.scrollIntoView) {
      try { li.scrollIntoView({ block: "nearest" }); } catch (e) { li.scrollIntoView(); }
    }
  }

  function onKeydown(e) {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    var typing = isTypingTarget(e.target);
    var key = e.key;

    // Esc is the one shortcut that works from inside a field — that is the
    // whole point of it.
    if (key === "Escape" || key === "Esc") {
      var box = isFilterInput(e.target) ? e.target : visibleFilterInput();
      if (box && box.value) { e.preventDefault(); clearFilterInput(box); }
      return;
    }

    // render.js marks .toggle / .tab / .top-tab / .survey-item as
    // role="button" tabindex="0"; a focusable control that ignores the keyboard
    // is worse than one that is not focusable at all. The action itself may sit
    // on the element or on an ancestor, so it is looked up the same way the
    // click handler does it.
    if (key === "Enter" || key === " " || key === "Spacebar") {
      if (typing) return;
      var hit = e.target && e.target.closest
        ? e.target.closest(".toggle,.tab,.top-tab,.survey-item")
        : null;
      if (!hit) return;
      e.preventDefault();          // Space would otherwise scroll the workspace
      var act = hit.closest("[data-action]");
      var fn = act && ACTIONS[act.dataset.action];
      if (fn) fn(act, e);
      return;
    }

    if (typing) return;

    if (key === "/") {
      var target = visibleFilterInput();
      if (!target) return;
      e.preventDefault();          // otherwise "/" lands in the box we just focused
      target.focus();
      try { target.select(); } catch (e2) {}
      return;
    }
    if (key === "j") { e.preventDefault(); stepSurvey(1); return; }
    if (key === "k") { e.preventDefault(); stepSurvey(-1); return; }
    if (key === "e") { e.preventDefault(); toggleExpandAll(); }
  }

  /* ==================================================================
   * 11c. WIRING
   * ================================================================== */

  function wire() {
    document.addEventListener("click", function (e) {
      var node = e.target.closest("[data-action]");
      if (!node) return;
      var fn = ACTIONS[node.dataset.action];
      if (!fn) return;
      e.preventDefault();
      fn(node, e);
    });

    document.addEventListener("keydown", onKeydown);

    el.corpForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var n = validNumber(el.corpInput.value);
      // Path params are typed `int` server-side; sending a typo just earns a 422.
      if (n == null) { toast("err", "Corp number must be digits only."); return; }
      loadTenant(n);
    });

    el.surveyForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var n = validNumber(el.surveyInput.value);
      if (n == null) { toast("err", "Survey number must be digits only."); return; }
      if (!state.corpNo) { toast("err", "Load a corp number first."); return; }
      selectSurvey(n);
    });

    el.navSearch.addEventListener("input", function () {
      state.navSearch = el.navSearch.value;
      if (state.topView === "tenant" && state.tenantSection === "taxonomy") renderTenant();
      else renderNav();
    });

    document.addEventListener("input", function (e) {
      if (e.target && e.target.id === "qSearch") {
        state.questionSearch = e.target.value;
        renderTabBody();
      }
    });

    // `toggle` does not bubble — capture phase is required. This is what makes
    // the profile accordions fetch their envelope lazily.
    document.addEventListener("toggle", function (e) {
      var d = e.target;
      if (d && d.tagName === "DETAILS" && d.classList.contains("profile-agent") && d.open) {
        loadAgent(d.dataset.agent);
      }
    }, true);

    window.addEventListener("beforeunload", function (e) {
      if (state.batch.running) { e.preventDefault(); e.returnValue = ""; }
    });
  }

  /* ==================================================================
   * 12. BOOT
   * ================================================================== */

  function boot() {
    grabDom();
    // The attributes were set at script evaluation (section 1b) so the first
    // paint is already correct; this only tells the buttons what they are
    // showing.
    syncChromeButtons();
    restore();
    wire();
    renderAll();

    // Config first: it decides the shape of the profile panel, and rendering
    // the Parallel form in an SMX deployment would ask for a website that is
    // ignored. It never rejects, so a failure just leaves the defaults.
    loadConfig()
      .then(loadMe)
      .then(loadTaxonomy)
      .then(checkShare)
      .then(function () {
        // Precedence: the corp last worked on, then whatever the token says.
        // Explicit beats implied — someone who typed 75885 into the box last
        // session means it, even when their own account is a different corp,
        // and share-only tenants have no account to be signed in as at all.
        var saved = null;
        try { saved = localStorage.getItem(LS.corp); } catch (e) {}
        var n = validNumber(saved);
        if (n == null) n = state.tokenCorpNo;
        if (n == null) { renderAll(); return; }
        el.corpInput.value = String(n);
        return loadTenant(n).then(function () {
          // A background profile fetch survives a reload.
          if (state.profileJob && state.profileJob.corpNo === n) startProfilePoll();
        });
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

})(window.ST);
