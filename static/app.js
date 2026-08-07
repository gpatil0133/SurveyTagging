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

  var LS = { corp: "st.corp", topView: "st.topview", profileJob: "st.profilejob" };
  var SS = { taxonomy: "st.taxonomy.v1" };

  var ROUTES = {
    taxonomy:        function ()      { return "/api/taxonomy"; },
    shareHealth:     function ()      { return "/api/health/share"; },
    surveyList:      function (t)     { return "/api/tenants/" + t + "/tag-surveys"; },
    batchTag:        function (t)     { return "/api/tenants/" + t + "/tag-surveys"; },
    batchRetag:      function (t)     { return "/api/tenants/" + t + "/retag-surveys"; },
    tenantTags:      function (t)     { return "/api/tenants/" + t + "/tags"; },
    tenantTagsBuild: function (t)     { return "/api/tenants/" + t + "/tag"; },   // NOTE: singular
    surveyTags:      function (t,s,jc){ return "/api/tenants/" + t + "/surveys/" + s + "/tags" +
                                               (jc ? "?include_journey_candidates=true" : ""); },
    tagSurvey:       function (t,s)   { return "/api/tenants/" + t + "/surveys/" + s + "/tag"; },
    retagSurvey:     function (t,s)   { return "/api/tenants/" + t + "/surveys/" + s + "/retag"; },
    adhoc:           function ()      { return "/api/tag"; },
    profile:         function (t)     { return "/api/tenants/" + t + "/profile"; },
    profileAgent:    function (t,a)   { return "/api/tenants/" + t + "/profile/" + a; },
    profileFetch:    function (t,bg)  { return "/api/tenants/" + t + "/profile/fetch" +
                                               (bg ? "?background=true" : ""); },
    autoretag:       function ()      { return "/api/admin/autoretag"; },
    autoretagRun:    function ()      { return "/api/admin/autoretag/run-now"; }
  };

  // A downed share surfaces as a Windows error code buried in a 500 detail, or
  // as an empty listing that is indistinguishable from "no surveys". These are
  // the codes SMB actually produces: 53 not found, 64 name deleted,
  // 67 bad net name, 1231 unreachable, 1326 bad credentials.
  var SHARE_ERR_RE = /\[WinError (53|64|67|1231|1326)\]|network path was not found|network name cannot be found|semaphore timeout|no such file or directory/i;

  /* ==================================================================
   * 2. STATE
   * ================================================================== */

  var state = {
    corpNo: null,
    topView: "surveys",              // surveys | tenant | adhoc

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
    activeTab: "project",            // project | questions | journey | raw
    includeCandidates: false,

    sourceFilter: new Set(U.ALL_SOURCES),
    lowConfOnly: false,
    questionSearch: "",
    questionView: "cards",
    tableCols: "present",
    expandedQuestions: new Set(),

    tenantSection: "tags",           // tags | profile | batch | scheduler | taxonomy
    tenantTags: null, tenantTagsError: null,
    profile: null, profileError: null,
    profileAgents: {},               // {org: envelope|"loading"|"missing"|"error: …"}
    profileJob: null,                // {startedAt, website, agents, polls}
    profileTimer: null,
    batch: { running: false, force: false, startedAt: null, result: null,
             progress: null, timer: null },
    scheduler: null, schedulerError: null,

    adhoc: { busy: false, result: null, error: null },

    taxonomy: null,
    busy: 0,
    slowWarn: false,
    banner: null,                    // {kind, text, actionsHtml}
    shareDown: false,
    toasts: [],
    nextToastId: 1
  };

  function persist() {
    try {
      if (state.corpNo) localStorage.setItem(LS.corp, String(state.corpNo));
      localStorage.setItem(LS.topView, state.topView);
      if (state.profileJob) localStorage.setItem(LS.profileJob, JSON.stringify(state.profileJob));
      else localStorage.removeItem(LS.profileJob);
    } catch (e) { /* private mode / disabled storage — not worth surfacing */ }
  }

  function restore() {
    try {
      var v = localStorage.getItem(LS.topView);
      if (v === "surveys" || v === "tenant" || v === "adhoc") state.topView = v;
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
     "topStatus","busybar","topTabs","navSearch","navFilters","sidebarHeading",
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

  function isShareError(res) {
    if (!res) return false;
    if (res.netError) return true;
    if (res.status >= 500 && SHARE_ERR_RE.test(String(res.detail || ""))) return true;
    return false;
  }

  /* Never throws on an HTTP status — the caller branches on `.status`.
   * `timeoutMs: null` disables the abort entirely; pass it for every
   * long-running POST (tag, retag, batch, sync profile fetch, run-now),
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
    if (isShareError(res)) showShareDown(res.detail);
    else if (state.shareDown && res.ok) clearShareDown();
    return res;
  }

  /* ==================================================================
   * 5. NORMALIZERS
   * ================================================================== */

  /* Two payload shapes reach the UI:
   *   POST .../tag|retag -> body.tagged = raw tagged_output.json
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
    { key: "scheduler",label: "Scheduler" },
    { key: "taxonomy", label: "Taxonomy" }
  ];

  function renderNav() {
    var surveys = state.topView === "surveys";
    el.navSearch.hidden = !surveys;
    el.navFilters.hidden = !surveys;

    if (state.topView === "tenant") {
      el.sidebarHeading.textContent = "Tenant";
      el.nav.innerHTML = TENANT_SECTIONS.map(function (s) {
        return '<li class="survey-item' + (state.tenantSection === s.key ? " active" : "") +
               '" data-action="set-tenant-section" data-section="' + s.key + '">' +
               '<span class="stitle">' + s.label + "</span></li>";
      }).join("");
      return;
    }

    if (state.topView === "adhoc") {
      el.sidebarHeading.textContent = "Ad-hoc";
      el.nav.innerHTML = '<li class="survey-item" style="cursor:default">' +
        '<span class="stitle">No corp needed — paste or upload a survey JSON.</span></li>';
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
        return '<span class="chip toggle ' + (state.navFilter === f ? "on" : "off") +
               '" data-action="set-nav-filter" data-filter="' + f + '">' +
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

    var needle = state.navSearch.toLowerCase().trim();
    var rows = state.surveys.filter(function (s) {
      if (state.navFilter === "tagged" && !s.tagged) return false;
      if (state.navFilter === "untagged" && s.tagged) return false;
      if (!needle) return true;
      var name = state.surveyNames[s.survey_no] || "";
      return String(s.survey_no).indexOf(needle) >= 0 ||
             name.toLowerCase().indexOf(needle) >= 0;
    });

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
        '" data-action="select-survey" data-survey-no="' + s.survey_no + '"' +
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
    if (state.topView === "adhoc") return renderAdhoc();
    if (state.topView === "tenant") return renderTenant();
    return renderSurveys();
  }

  var SURVEY_TABS = [
    { key: "project",   label: "Project Tags" },
    { key: "questions", label: "Questions" },
    { key: "journey",   label: "Journey" },
    { key: "raw",       label: "Raw JSON" }
  ];

  function tabsHtml(tabs, active, action) {
    return '<div class="tabs">' + tabs.map(function (t) {
      return '<div class="tab' + (active === t.key ? " active" : "") +
             '" data-action="' + action + '" data-tab="' + t.key + '">' +
             U.escapeHtml(t.label) + "</div>";
    }).join("") + "</div>";
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
      el.content.innerHTML = R.emptyState(
        "Corp " + state.corpNo,
        state.surveysLoaded
          ? "Pick a survey from the sidebar, or type a survey number in the topbar."
          : "Loading surveys…", "");
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
          '<button class="btn" data-action="retag-survey" data-survey-no="' + no + '">Force re-tag</button>' +
          "</div>");
      } else {
        el.content.innerHTML = R.errorState(
          "Could not load survey " + no,
          state.surveyError.detail || "Unknown error", "retry-survey");
      }
      return;
    }

    if (!state.survey) {
      el.content.innerHTML = R.emptyState("Loading survey " + state.activeSurveyNo + "…", "", "");
      return;
    }

    var tabs = SURVEY_TABS.filter(function (t) {
      return t.key !== "journey" || !!state.survey.survey_journey;
    });
    var active = tabs.some(function (t) { return t.key === state.activeTab; })
      ? state.activeTab : "project";

    el.content.innerHTML =
      '<div class="section-head"><div class="survey-header">' +
      R.surveyHeader(state.survey) + "</div>" +
      '<div class="btn-row">' +
      '<button class="btn" data-action="retag-survey" data-survey-no="' + state.activeSurveyNo + '">Force re-tag</button>' +
      '<button class="btn ghost" data-action="download-json" data-what="survey">Download JSON</button>' +
      '<button class="btn danger sm" data-action="delete-survey-tags" data-survey-no="' + state.activeSurveyNo + '">Delete tags</button>' +
      "</div></div>" +
      tabsHtml(tabs, active, "set-tab") +
      '<div id="tabBody">' + surveyTabBody(active) + "</div>";
  }

  function surveyTabBody(tab) {
    var s = state.survey;
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

  /* The filter chips and expand/collapse act on whichever survey is on screen —
   * the loaded one, or the ad-hoc result. */
  function currentSurvey() {
    if (state.topView === "adhoc") return state.adhoc.result || { question_tags: [] };
    return state.survey || { question_tags: [] };
  }

  // Repaint only the results region so the filter inputs keep focus and caret.
  function renderTabBody() {
    var body = document.getElementById("tabBody") || document.getElementById("adhocResult");
    if (!body) return renderWorkspace();
    var qs = document.getElementById("qSearch");
    var caret = qs ? qs.selectionStart : null;
    body.innerHTML = body.id === "adhocResult" ? adhocResultHtml() : surveyTabBody(state.activeTab);
    var qs2 = document.getElementById("qSearch");
    if (qs2 && caret != null) { qs2.focus(); try { qs2.setSelectionRange(caret, caret); } catch (e) {} }
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
               scheduler: schedulerHtml, taxonomy: taxonomyHtml }[state.tenantSection];
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
    if (!state.tenantTags) return head + R.emptyState("Loading…", "", "");
    return head + R.tenantTags(state.tenantTags, state);
  }

  function profileHtml() {
    var head = '<div class="section-head"><h2>Tenant Profile</h2><div class="btn-row">' +
      '<button class="btn" data-action="profile-reload">Reload</button>' +
      '<button class="btn ghost" data-action="download-json" data-what="profile">Download JSON</button>' +
      '<button class="btn danger sm" data-action="profile-delete">Delete artifacts</button></div></div>';

    var form =
      '<div class="tenant-panel"><h3>Fetch from Parallel.ai</h3>' +
      '<p class="micro" style="margin-top:0">Takes 10–30 minutes. Runs in the ' +
      'background by default; progress is inferred from artifacts appearing on disk.</p>' +
      '<div class="form-grid">' +
      '<div class="field"><label for="pfWebsite">Website</label>' +
      '<input type="text" id="pfWebsite" placeholder="https://acme.com" /></div>' +
      '<div class="field"><label>Agents</label><div class="btn-row">' +
      AGENTS.map(function (a) {
        return '<label class="checkline"><input type="checkbox" class="pfAgent" value="' +
               a + '" checked /> ' + a + "</label>";
      }).join("") + "</div></div>" +
      '<div class="field"><label>Options</label>' +
      '<label class="checkline"><input type="checkbox" id="pfForce" /> Force refresh (ignore cached artifacts)</label></div>' +
      "</div>" +
      '<div class="btn-row" style="margin-top:12px">' +
      '<button class="btn primary" data-action="profile-fetch"' +
      (state.profileJob ? " disabled" : "") + ">Fetch in background</button>" +
      '<button class="btn ghost" data-action="profile-fetch-sync"' +
      (state.profileJob ? " disabled" : "") + ">Run synchronously (blocks up to 30 min)</button>" +
      "</div></div>";

    var body;
    if (state.profileError && state.profileError.status === 404) {
      body = R.emptyState("No profile artifacts yet",
        "Nothing under this corp's tenant_profile folder on the share.", "");
    } else if (state.profileError) {
      body = R.errorState("Could not load profile", state.profileError.detail, "profile-reload");
    } else if (!state.profile) {
      body = R.emptyState("Loading…", "", "");
    } else {
      body = R.profileSummary(state.profile) + R.profileAgents(state.profileAgents);
    }
    return head + form + body;
  }

  function batchHtml() {
    var running = state.batch.running;
    var head = '<div class="section-head"><h2>Batch Tagging</h2><div class="btn-row">' +
      '<button class="btn primary" data-action="batch-tag"' + (running ? " disabled" : "") +
      ">Tag all surveys</button>" +
      '<button class="btn" data-action="batch-retag"' + (running ? " disabled" : "") +
      ">Force re-tag all</button></div></div>";

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

  function schedulerHtml() {
    var head = '<div class="section-head"><h2>Auto-retag Scheduler</h2><div class="btn-row">' +
      '<button class="btn" data-action="scheduler-refresh">Refresh</button>' +
      '<button class="btn danger" data-action="scheduler-run-now">Run scan now</button></div></div>';
    var warn = '<div class="lc-banner warn"><span>A scan walks <em>every</em> tenant ' +
      "on the share, not just this corp.</span></div>";
    if (state.schedulerError) {
      return head + warn + R.errorState("Scheduler unavailable",
        state.schedulerError.detail, "scheduler-refresh");
    }
    if (!state.scheduler) return head + warn + R.emptyState("Loading…", "", "");
    var s = state.scheduler;
    var rows = Object.keys(s).map(function (k) {
      var v = s[k];
      var txt = (v && typeof v === "object") ? JSON.stringify(v) : String(v);
      return "<dt>" + U.escapeHtml(U.labelFor(k)) + "</dt><dd>" + U.escapeHtml(txt) + "</dd>";
    }).join("");
    return head + warn + '<div class="tenant-panel"><dl class="kv-list">' + rows + "</dl></div>";
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
   * 10. AD-HOC
   * ================================================================== */

  function renderAdhoc() {
    el.content.innerHTML =
      '<div class="section-head"><h2>Ad-hoc tagging</h2></div>' +
      '<p class="micro">Tags a survey JSON in memory. Deterministic only — there ' +
      'is no tenant on disk, so no canon, no LLM, and nothing is persisted.</p>' +
      '<form id="adhocForm" class="tenant-panel">' +
      '<div class="field"><label for="adhocText">Paste survey_structure.json</label>' +
      '<textarea id="adhocText" class="code" placeholder="{ &quot;SurveyData&quot;: [ … ] }"></textarea></div>' +
      '<div class="field" style="margin-top:12px"><label for="adhocFile">…or upload a file</label>' +
      '<input type="file" id="adhocFile" accept=".json,application/json" /></div>' +
      '<div class="form-grid" style="margin-top:12px">' +
      ["industry", "company_name", "department", "purpose", "country"].map(function (f) {
        return '<div class="field"><label for="ad_' + f + '">' +
               U.escapeHtml(U.labelFor(f)) + " <span class=\"micro\">(optional)</span></label>" +
               '<input type="text" id="ad_' + f + '" /></div>';
      }).join("") + "</div>" +
      '<div class="btn-row" style="margin-top:14px">' +
      '<button type="submit" class="btn primary"' + (state.adhoc.busy ? " disabled" : "") + ">Tag</button>" +
      '<button type="button" class="btn ghost" data-action="adhoc-clear">Clear</button>' +
      (state.adhoc.result
        ? '<button type="button" class="btn ghost" data-action="download-json" data-what="adhoc">Download JSON</button>'
        : "") +
      "</div>" +
      (state.adhoc.error
        ? '<div class="lc-banner" style="margin-top:12px"><span>' +
          U.escapeHtml(state.adhoc.error) + "</span></div>"
        : "") +
      "</form>" +
      '<div id="adhocResult">' + adhocResultHtml() + "</div>";
  }

  function adhocResultHtml() {
    if (!state.adhoc.result) return "";
    var s = state.adhoc.result;
    return '<div class="survey-header">' + R.surveyHeader(s) + "</div>" +
      R.filterBar(state, { questionControls: true, columnControls: true }) +
      R.projectTags(s, state) +
      '<h3 style="margin-top:24px">Questions</h3>' +
      R.questions(s, state);
  }

  /* ==================================================================
   * 11. CONTROLLERS
   * ================================================================== */

  function validNumber(raw) {
    var v = String(raw == null ? "" : raw).trim();
    return /^\d+$/.test(v) ? Number(v) : null;
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
    if (!opts.quiet) setBusy(1);
    return api(ROUTES.surveyList(state.corpNo)).then(guarded).then(function (r) {
      if (!opts.quiet) setBusy(-1);
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
    return api(ROUTES.surveyTags(state.corpNo, no, state.includeCandidates), { etag: etag })
      .then(guarded).then(function (r) {
        setBusy(-1);
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

  function tagSurvey(no, force) {
    if (no == null || !state.corpNo) return;
    setBusy(1);
    var id = toast("info", (force ? "Re-tagging" : "Tagging") + " survey " + no + "…", true);
    var url = force ? ROUTES.retagSurvey(state.corpNo, no) : ROUTES.tagSurvey(state.corpNo, no);
    // No timeout: a full LLM pass takes 30-90s.
    return api(url, { method: "POST", timeoutMs: null }).then(guarded).then(function (r) {
      setBusy(-1); dismissToast(id);
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
    return api(ROUTES.tenantTags(state.corpNo)).then(guarded).then(function (r) {
      setBusy(-1);
      state.tenantTags = r.ok ? r.data : null;
      state.tenantTagsError = r.ok ? null : { status: r.status, detail: r.detail };
      renderTenant();
    });
  }

  function buildTenantTags() {
    setBusy(1);
    var id = toast("info", "Building tenant tags…", true);
    return api(ROUTES.tenantTagsBuild(state.corpNo), { method: "POST", timeoutMs: null })
      .then(guarded).then(function (r) {
        setBusy(-1); dismissToast(id);
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
    if (!opts.quiet) setBusy(1);
    return api(ROUTES.profile(state.corpNo)).then(guarded).then(function (r) {
      if (!opts.quiet) setBusy(-1);
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

  function collectProfileForm() {
    var website = (document.getElementById("pfWebsite") || {}).value || "";
    var agents = Array.prototype.slice
      .call(document.querySelectorAll(".pfAgent:checked"))
      .map(function (n) { return n.value; });
    var force = !!(document.getElementById("pfForce") || {}).checked;
    return { website: website.trim(), agents: agents, force: force };
  }

  function startProfileFetch(background) {
    var f = collectProfileForm();
    if (f.website.length < 4) { toast("err", "Enter the tenant website first.", true); return; }
    if (!f.agents.length) { toast("err", "Pick at least one agent.", true); return; }
    if (!background && !window.confirm(
      "Run synchronously? This blocks for up to 30 minutes. Background mode is " +
      "usually what you want.")) return;

    var body = { website: f.website, agents: f.agents, force: f.force };

    if (!background) {
      setBusy(1);
      var id = toast("info", "Profile fetch running synchronously — this can take 30 minutes…", true);
      return api(ROUTES.profileFetch(state.corpNo, false),
                 { method: "POST", json: body, timeoutMs: null })
        .then(guarded).then(function (r) {
          setBusy(-1); dismissToast(id);
          if (!r.ok) { toast("err", "Fetch failed: " + r.detail, true); return; }
          var c = (r.data && r.data.counts) || {};
          toast("ok", "Fetched " + (c.fetched || 0) + ", cached " + (c.cache_hits || 0) +
                      ", failed " + (c.failures || 0) + ".");
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

  function runBatch(force) {
    if (state.batch.running) return;
    state.batch.running = true;
    state.batch.force = force;
    state.batch.startedAt = Date.now();
    state.batch.result = null;
    state.batch.progress = { tagged: state.surveys.filter(function (s) { return s.tagged; }).length,
                             total: state.surveys.length };
    renderAll();

    // Poll the listing while the POST is in flight. tagged_output.json appears
    // per survey as the bounded-parallel run completes each one, so this is a
    // real determinate progress bar with no backend support needed.
    state.batch.timer = setInterval(pollBatchProgress, POLL_BATCH_MS);

    var url = force ? ROUTES.batchRetag(state.corpNo) : ROUTES.batchTag(state.corpNo);
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

  // ---- scheduler ----

  function loadScheduler() {
    setBusy(1);
    return api(ROUTES.autoretag()).then(function (r) {
      setBusy(-1);
      state.scheduler = r.ok ? r.data : null;
      state.schedulerError = r.ok ? null : { status: r.status, detail: r.detail };
      renderTenant();
    });
  }

  function runScanNow() {
    if (!window.confirm(
      "Run a change-scan now? This walks EVERY tenant on the share, not just " +
      "corp " + state.corpNo + ".")) return;
    setBusy(1);
    var id = toast("info", "Scanning…", true);
    return api(ROUTES.autoretagRun(), { method: "POST", timeoutMs: null })
      .then(guarded).then(function (r) {
        setBusy(-1); dismissToast(id);
        if (!r.ok) { toast(r.status === 503 ? "info" : "err", r.detail || "Scan failed", true); return; }
        toast("ok", "Scan complete.");
        state.scheduler = r.data;
        renderTenant();
      });
  }

  // ---- ad-hoc ----

  function submitAdhoc() {
    var text = (document.getElementById("adhocText") || {}).value || "";
    var fileInput = document.getElementById("adhocFile");
    var file = fileInput && fileInput.files && fileInput.files[0];

    if (!file && !text.trim()) {
      state.adhoc.error = "Paste JSON or choose a file first.";
      renderAdhoc(); return;
    }
    if (!file && text.trim()) {
      // Catch obvious syntax errors client-side rather than round-tripping.
      try { JSON.parse(text); }
      catch (e) { state.adhoc.error = "Invalid JSON: " + e.message; renderAdhoc(); return; }
    }

    var fd = new FormData();
    if (file) fd.append("survey_file", file);
    if (text.trim()) fd.append("survey_text", text);
    ["industry", "company_name", "department", "purpose", "country"].forEach(function (f) {
      var node = document.getElementById("ad_" + f);
      if (node && node.value.trim()) fd.append(f, node.value.trim());
    });

    state.adhoc.busy = true; state.adhoc.error = null;
    setBusy(1);
    return api(ROUTES.adhoc(), { method: "POST", form: fd, timeoutMs: null })
      .then(function (r) {
        setBusy(-1);
        state.adhoc.busy = false;
        if (!r.ok) {
          state.adhoc.error = r.detail || ("Request failed (" + r.status + ")");
          state.adhoc.result = null;
          toast("err", state.adhoc.error, true);
        } else {
          state.adhoc.result = normalizeTagged(r.data);
          state.adhoc.error = null;
          toast("ok", "Tagged.");
        }
        renderAdhoc();
      });
  }

  function resetAdhoc() {
    state.adhoc = { busy: false, result: null, error: null };
    renderAdhoc();
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
    } else if (what === "adhoc" && state.adhoc.result) {
      payload = state.adhoc.result; name = "adhoc_tagged.json";
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
   * 12. DISPATCH
   * ================================================================== */

  var ACTIONS = {
    "set-topview": function (n) {
      state.topView = n.dataset.topview;
      persist();
      renderAll();
      if (state.topView === "tenant") loadTenantSection();
    },
    "set-tab": function (n) { state.activeTab = n.dataset.tab; renderWorkspace(); },
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
    "retry-survey": function () { loadSurveyView(state.activeSurveyNo, { bustEtag: true }); },
    "reload-taxonomy": function () { loadTaxonomy().then(renderAll); },

    "tag-survey": function (n) { tagSurvey(Number(n.dataset.surveyNo), false); },
    "retag-survey": function (n) { tagSurvey(Number(n.dataset.surveyNo), true); },
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
      if (state.expandedQuestions.has(q)) state.expandedQuestions.delete(q);
      else state.expandedQuestions.add(q);
      renderTabBody();
    },
    "expand-all": function () {
      var list = currentSurvey().question_tags || [];
      // Same string key render.js emits as data-q.
      list.forEach(function (q) {
        state.expandedQuestions.add(
          String(q.question_id != null ? q.question_id : (q.question_no == null ? "" : q.question_no)));
      });
      renderTabBody();
    },
    "collapse-all": function () { state.expandedQuestions.clear(); renderTabBody(); },

    "build-tenant-tags": function () { buildTenantTags(); },
    "tag-tenant": function () { buildTenantTags(); },
    "reload-tenant-tags": function () { loadTenantTags(); },
    "delete-tenant-tags": function () { deleteTenantTags(); },

    "profile-reload": function () { state.profileAgents = {}; loadProfile(); },
    "profile-fetch": function () { startProfileFetch(true); },
    "profile-fetch-sync": function () { startProfileFetch(false); },
    "profile-stop-watch": function () {
      stopProfilePoll();
      toast("info", "Stopped watching. The fetch continues on the server.");
    },
    "profile-delete": function () { deleteProfile(); },
    "load-agent": function (n) { loadAgent(n.dataset.agent); },

    "batch-tag": function () { runBatch(false); },
    "batch-retag": function () { runBatch(true); },
    "batch-stop-watch": function () { stopBatchPoll(false); },

    "scheduler-refresh": function () { loadScheduler(); },
    "scheduler-run-now": function () { runScanNow(); },

    "download-json": function (n) { downloadJson(n.dataset.what); },
    "adhoc-clear": function () { resetAdhoc(); },
    "toast-dismiss": function (n) { dismissToast(Number(n.dataset.toastId)); }
  };

  function loadTenantSection() {
    if (!state.corpNo) return;
    var s = state.tenantSection;
    if (s === "tags" && !state.tenantTags && !state.tenantTagsError) loadTenantTags();
    else if (s === "profile" && !state.profile && !state.profileError) loadProfile();
    else if (s === "scheduler" && !state.scheduler && !state.schedulerError) loadScheduler();
  }

  function wire() {
    document.addEventListener("click", function (e) {
      var node = e.target.closest("[data-action]");
      if (!node) return;
      var fn = ACTIONS[node.dataset.action];
      if (!fn) return;
      e.preventDefault();
      fn(node, e);
    });

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

    document.addEventListener("submit", function (e) {
      if (e.target && e.target.id === "adhocForm") { e.preventDefault(); submitAdhoc(); }
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
   * 13. BOOT
   * ================================================================== */

  function boot() {
    grabDom();
    restore();
    wire();
    renderAll();

    loadTaxonomy().then(function () {
      checkShare().then(function () {
        var saved = null;
        try { saved = localStorage.getItem(LS.corp); } catch (e) {}
        var n = validNumber(saved);
        if (n != null) {
          el.corpInput.value = String(n);
          loadTenant(n).then(function () {
            // A background profile fetch survives a reload.
            if (state.profileJob && state.profileJob.corpNo === n) startProfilePoll();
          });
        } else {
          renderAll();
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

})(window.ST);
