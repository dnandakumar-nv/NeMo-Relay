/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

(() => {
  'use strict';

  let bootstrapFragment = window.location.hash;
  const requestedView = new URLSearchParams(window.location.search).get('view');
  const retainedSearch = /^(overview|pools|evidence|neighborhood|decisions|operations)$/.test(requestedView || '')
    ? `?view=${encodeURIComponent(requestedView)}`
    : '';
  window.history.replaceState(null, '', `${window.location.pathname}${retainedSearch}`);

  const POLL_MS = 5000;
  const PREVIEW_LIMIT = 256;
  const CONTENT_LIMIT = 16 * 1024;
  const INPUT_LIMIT = 1024 * 1024;
  const VALID_VIEWS = new Set(['overview', 'pools', 'evidence', 'neighborhood', 'decisions', 'operations']);
  const ICON_PATHS = {
    overview: ['M4 13h6V4H4v9Z', 'M14 20h6v-9h-6v9Z', 'M14 7h6V4h-6v3Z', 'M4 20h6v-3H4v3Z'],
    pools: ['M3 6h18', 'M6 3v18', 'M18 3v18', 'M3 18h18'],
    evidence: ['M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z', 'M14 2v6h6', 'M8 13h8', 'M8 17h6'],
    neighborhood: ['M12 2a10 10 0 1 0 10 10', 'M12 8a4 4 0 1 0 4 4', 'M21 3l-6 6', 'M15 3h6v6'],
    decisions: ['M4 4h16v5H4z', 'M4 15h10v5H4z', 'M18 14v7', 'M15 18h6'],
    operations: ['M4 6h16', 'M4 12h16', 'M4 18h16', 'M8 3v6', 'M16 9v6', 'M10 15v6'],
    refresh: ['M20 11a8.1 8.1 0 0 0-15.5-2M4 4v5h5', 'M4 13a8.1 8.1 0 0 0 15.5 2M20 20v-5h-5'],
    close: ['M18 6 6 18', 'M6 6l12 12'],
    previous: ['m15 18-6-6 6-6'],
    next: ['m9 18 6-6-6-6'],
    download: ['M12 3v12', 'm7 10 5 5 5-5', 'M5 21h14'],
    filter: ['M4 5h16', 'M7 12h10', 'M10 19h4'],
    search: ['m21 21-4.35-4.35', 'M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z'],
  };

  const state = {
    view: 'overview',
    cache: new Map(),
    etags: new Map(),
    stale: new Set(),
    incompatible: false,
    pollTimer: null,
    pools: null,
    overview: null,
    controls: null,
    health: null,
    migrations: null,
    evidence: { page: null, cursor: null, history: [], number: 1 },
    neighborhood: null,
    projectionPoints: [],
    decisions: new Map(),
    exposures: new Map(),
    decisionCursor: null,
    decisionSource: null,
    decisionFallback: null,
    decisionFilters: {},
    authenticated: false,
  };

  const refs = {};

  class ApiError extends Error {
    constructor(code, status) {
      super(code);
      this.code = code;
      this.status = status;
    }
  }

  class IncompatibleError extends Error {}

  function query(selector, root = document) {
    return root.querySelector(selector);
  }

  function queryAll(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function text(value, limit = CONTENT_LIMIT) {
    const source = value === null || value === undefined ? '-' : String(value);
    const points = Array.from(source);
    return points.length <= limit ? source : `${points.slice(0, limit).join('')}...`;
  }

  function jsonText(value, limit = CONTENT_LIMIT) {
    let rendered;
    try {
      rendered = JSON.stringify(value, null, 2);
    } catch (_) {
      rendered = 'Unavailable';
    }
    return text(rendered, limit);
  }

  function clear(node) {
    node.replaceChildren();
  }

  function element(tag, attributes = {}, children = []) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attributes)) {
      if (value === undefined || value === null || value === false) continue;
      if (name === 'className') node.className = value;
      else if (name === 'text') node.textContent = text(value);
      else if (name === 'hidden') node.hidden = Boolean(value);
      else if (name === 'tabIndex') node.tabIndex = value;
      else if (name.startsWith('aria'))
        node.setAttribute(
          name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`),
          String(value),
        );
      else node.setAttribute(name, String(value));
    }
    const values = Array.isArray(children) ? children : [children];
    for (const child of values) {
      if (child === null || child === undefined) continue;
      node.append(child instanceof Node ? child : document.createTextNode(text(child)));
    }
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'icon');
    for (const pathData of ICON_PATHS[name] || []) {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', pathData);
      svg.append(path);
    }
    return svg;
  }

  function installIcons() {
    queryAll('[data-icon]').forEach((slot) => slot.replaceChildren(icon(slot.dataset.icon)));
    refs.refreshButton.replaceChildren(icon('refresh'));
    refs.detailClose.replaceChildren(icon('close'));
    refs.evidenceBack.replaceChildren(icon('previous'));
    refs.evidenceNext.replaceChildren(icon('next'));
  }

  function badge(value, tone = '') {
    return element('span', { className: `badge ${tone}`.trim(), text: formatLabel(value) });
  }

  function toneFor(value) {
    const normalized = String(value || '').toLowerCase();
    if (
      ['clear', 'pass', 'passed', 'success', 'ready', 'active', 'verified', 'applied', 'current', 'true'].includes(
        normalized,
      )
    )
      return 'good';
    if (
      ['warning', 'paused', 'force_anchor', 'holdout', 'unlabeled', 'stale', 'rebuilding', 'no_op'].includes(normalized)
    )
      return 'warn';
    if (
      ['degraded', 'fail', 'failed', 'failure', 'unavailable', 'incompatible', 'false', 'conflict', 'newer'].includes(
        normalized,
      )
    )
      return 'bad';
    return 'info';
  }

  function formatLabel(value) {
    return text(value)
      .replaceAll('_', ' ')
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(Number(value));
  }

  function formatPercent(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
    return new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: digits }).format(Number(value));
  }

  function formatTime(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
    return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' }).format(
      new Date(Number(value)),
    );
  }

  function formatAge(value, snapshot = Date.now()) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'Never';
    const seconds = Math.max(0, Math.round((Number(snapshot) - Number(value)) / 1000));
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  function shortId(value) {
    if (!value) return '-';
    const source = String(value);
    return source.length > 18 ? `${source.slice(0, 8)}...${source.slice(-6)}` : source;
  }

  function metric(label, value, note) {
    const children = [element('span', { text: label }), element('strong', { text: value })];
    if (note) children.push(element('small', { text: note }));
    return element('div', { className: 'metric' }, children);
  }

  function keyValues(entries) {
    const list = element('dl', { className: 'key-value-list' });
    for (const [label, value] of entries) {
      list.append(element('dt', { text: label }));
      const cell = element('dd');
      cell.append(value instanceof Node ? value : document.createTextNode(text(value)));
      list.append(cell);
    }
    return list;
  }

  function dataState(target, message, visible = true) {
    target.textContent = text(message);
    target.hidden = !visible;
  }

  function announce(message) {
    refs.liveRegion.textContent = '';
    window.setTimeout(() => {
      refs.liveRegion.textContent = text(message, PREVIEW_LIMIT);
    }, 10);
  }

  function requireObject(value, name) {
    if (!value || typeof value !== 'object' || Array.isArray(value))
      throw new IncompatibleError(`${name} is not an object`);
    return value;
  }

  function requireSchema(value, schema) {
    requireObject(value, schema);
    if (value.schema !== schema) throw new IncompatibleError(`Unsupported schema: ${text(value.schema, 80)}`);
    return value;
  }

  function requireArray(value, name) {
    if (!Array.isArray(value)) throw new IncompatibleError(`${name} is not an array`);
    return value;
  }

  function requireFinite(value, name) {
    if (!Number.isFinite(value)) throw new IncompatibleError(`${name} is not finite`);
    return value;
  }

  function requirePage(value) {
    requireObject(value, 'page');
    requireArray(value.items, 'page items');
    requireFinite(value.snapshot_time_unix_ms, 'page snapshot');
    if (!value.items.every((item) => item && typeof item === 'object' && !Array.isArray(item)))
      throw new IncompatibleError('Unsupported page item');
    if (value.next !== null && value.next !== undefined && typeof value.next !== 'string')
      throw new IncompatibleError('Unsupported page cursor');
    return value;
  }

  function requireOverview(value) {
    requireSchema(value, 'nemo.relay.router.overview@1');
    requireFinite(value.window_start_unix_ms, 'overview window');
    requireFinite(value.snapshot_time_unix_ms, 'overview snapshot');
    requireObject(value.decisions, 'overview decisions');
    requireObject(value.decisions.final_reasons, 'overview reasons');
    requireObject(value.exposures, 'overview exposures');
    requireObject(value.outcomes, 'overview outcomes');
    requireStatus(value.status);
    return value;
  }

  function requireStatus(value) {
    requireSchema(value, 'nemo.relay.router.status@1');
    requireObject(value.database, 'status database');
    requireObject(value.queues, 'status queues');
    requireObject(value.leases, 'status leases');
    requireObject(value.vector_index, 'status vector index');
    requireObject(value.freshness, 'status freshness');
    requireObject(value.health, 'status health');
    return value;
  }

  function requirePoolDetail(value) {
    requireSchema(value, 'nemo.relay.router.pool-detail@1');
    const summary = requireObject(value.summary, 'pool summary');
    requireArray(summary.anchor_models, 'pool anchors');
    requireArray(summary.candidates, 'pool candidates');
    requireObject(value.selector, 'pool selector');
    requireObject(value.learning, 'pool learning');
    requireObject(value.support_prerequisites, 'pool support');
    return value;
  }

  function requireNeighborhood(value) {
    requireSchema(value, 'nemo.relay.router.neighborhood-report@1');
    requireArray(value.neighbors, 'neighborhood neighbors');
    requireObject(value.support, 'neighborhood support');
    requireObject(value.gates, 'neighborhood gates');
    requireObject(value.recommendation, 'neighborhood recommendation');
    return value;
  }

  function requireDecisionExposure(value) {
    requireSchema(value, 'nemo.relay.router.decision-exposure@1');
    if (value.active !== null) requireObject(value.active, 'active exposure');
    if (value.outcome !== null) requireObject(value.outcome, 'exposure outcome');
    return value;
  }

  async function responseError(response) {
    try {
      const body = await response.json();
      return text(body?.error?.code || `request_${response.status}`, 96);
    } catch (_) {
      return `request_${response.status}`;
    }
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set('Accept', 'application/json');
    if (options.etagKey && state.etags.has(options.etagKey))
      headers.set('If-None-Match', state.etags.get(options.etagKey));
    if (options.body !== undefined) headers.set('Content-Type', 'application/json');
    const response = await window.fetch(path, {
      method: options.method || 'GET',
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      credentials: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
    });
    if (response.status === 304 && options.etagKey && state.cache.has(options.etagKey))
      return state.cache.get(options.etagKey);
    if (!response.ok) throw new ApiError(await responseError(response), response.status);
    const value = await response.json();
    if (options.validator) options.validator(value);
    if (options.etagKey) {
      const etag = response.headers.get('ETag');
      if (etag) state.etags.set(options.etagKey, etag);
      state.cache.set(options.etagKey, value);
    }
    return value;
  }

  function setIncompatible(error) {
    state.incompatible = true;
    stopPolling();
    stopDecisionStream();
    queryAll('.view-panel').forEach((panel) => {
      const target = query('.data-state', panel);
      if (target) dataState(target, error.message || 'Inspection API is incompatible.');
      for (const child of panel.children) {
        if (!child.classList.contains('section-heading') && !child.classList.contains('data-state'))
          child.hidden = true;
      }
    });
    refs.staleIndicator.hidden = false;
    refs.staleIndicator.textContent = 'Incompatible';
    announce('Inspection API is incompatible');
  }

  function setStale(key, error) {
    state.stale.add(key);
    refs.staleIndicator.hidden = false;
    refs.staleIndicator.textContent = 'Stale';
    if (!state.cache.has(key)) {
      const target = key === 'overview' ? refs.overviewState : key === 'pools' ? refs.poolsState : refs.operationsState;
      if (target) dataState(target, `Unavailable: ${error.code || error.message}`);
    }
  }

  function clearStale(key) {
    state.stale.delete(key);
    refs.staleIndicator.hidden = state.stale.size === 0;
    if (state.stale.size === 0) refs.staleIndicator.textContent = 'Stale';
  }

  function updateHeader(status) {
    if (!status) return;
    refs.projectName.textContent = text(
      status.project_id || `Database: ${status.database?.state || 'unavailable'}`,
      96,
    );
    refs.configuredMode.replaceChildren(badge(status.configured_mode, toneFor(status.configured_mode)));
    refs.effectiveMode.replaceChildren(badge(status.effective_mode, toneFor(status.effective_mode)));
    refs.generationId.textContent = shortId(status.config_generation_id || status.cohort_generation_id);
    const control = status.controls?.effective;
    const controlLabel = control?.paused ? 'Paused' : control?.force_anchor ? 'Force anchor' : 'Normal';
    refs.controlState.replaceChildren(badge(controlLabel, toneFor(controlLabel.toLowerCase().replace(' ', '_'))));
    refs.healthState.replaceChildren(badge(status.health?.worst || 'unavailable', toneFor(status.health?.worst)));
  }

  async function refreshOverview() {
    try {
      const report = await api('/api/router/v1/overview', {
        etagKey: 'overview',
        validator: requireOverview,
      });
      state.overview = report;
      clearStale('overview');
      renderOverview(report);
      updateHeader(report.status);
    } catch (error) {
      if (error instanceof IncompatibleError) return setIncompatible(error);
      setStale('overview', error);
      try {
        const status = await api('/api/router/v1/status', { validator: requireStatus });
        updateHeader(status);
        if (!state.overview) {
          refs.overviewContent.hidden = true;
          dataState(refs.overviewState, `Overview unavailable: ${formatLabel(status.database?.state || error.code)}`);
        }
      } catch (statusError) {
        if (!state.overview)
          dataState(refs.overviewState, `Router unavailable: ${statusError.code || statusError.message}`);
      }
    }
  }

  async function refreshPools() {
    try {
      const page = await api('/api/router/v1/pools?limit=500', { etagKey: 'pools', validator: requirePage });
      state.pools = page;
      clearStale('pools');
      renderPools(page);
    } catch (error) {
      if (error instanceof IncompatibleError) return setIncompatible(error);
      setStale('pools', error);
    }
  }

  async function refreshOperationalPages() {
    const specs = [
      ['controls', '/api/router/v1/controls?limit=100'],
      ['health', '/api/router/v1/health?limit=100'],
      ['migrations', '/api/router/v1/migrations?limit=100'],
    ];
    await Promise.all(
      specs.map(async ([key, path]) => {
        try {
          const page = await api(path, { etagKey: key, validator: requirePage });
          state[key] = page;
          clearStale(key);
        } catch (error) {
          if (error instanceof IncompatibleError) return setIncompatible(error);
          setStale(key, error);
        }
      }),
    );
    renderOperations();
  }

  async function refreshPolled() {
    if (document.hidden || state.incompatible) return;
    await Promise.allSettled([refreshOverview(), refreshPools(), refreshOperationalPages()]);
  }

  function renderOverview(report) {
    refs.overviewContent.hidden = false;
    refs.overviewState.hidden = true;
    refs.overviewWindow.textContent = `Last 24 hours / ${formatTime(report.window_start_unix_ms)} to ${formatTime(report.snapshot_time_unix_ms)}`;
    refs.overviewMetrics.replaceChildren(
      metric('Decisions', formatNumber(report.decisions.total, 0), 'Last 24 hours'),
      metric('Candidate served', formatNumber(report.decisions.candidate_served, 0), 'Last 24 hours'),
      metric('Anchor served', formatNumber(report.decisions.anchor_served, 0), 'Last 24 hours'),
      metric('Fallbacks', formatNumber(report.decisions.fallback, 0), 'Last 24 hours'),
      metric('Treatment', formatNumber(report.exposures.candidate_treatment, 0), 'Assignments'),
      metric('Holdout', formatNumber(report.exposures.anchor_holdout, 0), 'Assignments'),
    );
    renderBars(refs.routingBreakdown, report.decisions.final_reasons, report.decisions.total);
    const outcomes = {};
    for (const [arm, counts] of Object.entries(report.outcomes || {})) outcomes[arm] = counts.total;
    renderBars(
      refs.outcomeBreakdown,
      outcomes,
      Object.values(outcomes).reduce((total, value) => total + Number(value || 0), 0),
    );
    const status = report.status;
    refs.runtimeStatus.replaceChildren(
      keyValues([
        ['Queues', `${formatNumber(status.queues?.pending, 0)} / ${formatNumber(status.queues?.capacity, 0)}`],
        ['Rejected', formatNumber(status.queues?.rejected, 0)],
        ['Active leases', formatNumber(status.leases?.active, 0)],
        ['Expired leases', formatNumber(status.leases?.expired, 0)],
        ['Vector spaces', formatNumber(status.vector_index?.active_spaces, 0)],
        ['Rebuilding', formatNumber(status.vector_index?.rebuilding_spaces, 0)],
        ['Degraded', formatNumber(status.vector_index?.degraded_spaces, 0)],
        ['Stale', formatNumber(status.vector_index?.stale_spaces, 0)],
      ]),
    );
    refs.freshnessStatus.replaceChildren(
      keyValues([
        ['Evidence', formatAge(status.freshness?.latest_evidence_unix_ms, report.snapshot_time_unix_ms)],
        ['Decision', formatAge(status.freshness?.latest_decision_unix_ms, report.snapshot_time_unix_ms)],
        ['Outcome', formatAge(status.freshness?.latest_outcome_unix_ms, report.snapshot_time_unix_ms)],
        ['Health event', formatAge(status.health?.latest_event_unix_ms, report.snapshot_time_unix_ms)],
        ['Cohort', shortId(status.cohort_generation_id)],
        ['Database', badge(status.database?.state, toneFor(status.database?.state))],
      ]),
    );
  }

  function renderBars(target, values, total) {
    const list = element('div', { className: 'bar-list' });
    const entries = Object.entries(values || {}).sort((left, right) => Number(right[1]) - Number(left[1]));
    if (!entries.length) {
      target.replaceChildren(element('p', { text: 'No records in this window.' }));
      return;
    }
    for (const [label, value] of entries) {
      const fraction = total > 0 ? Math.max(0, Math.min(1, Number(value) / total)) : 0;
      const fill = element('div', { className: 'bar-fill' });
      fill.style.width = `${fraction * 100}%`;
      list.append(
        element('div', { className: 'bar-row' }, [
          element('span', { text: formatLabel(label) }),
          element('div', { className: 'bar-track' }, fill),
          element('strong', { text: formatNumber(value, 0) }),
        ]),
      );
    }
    target.replaceChildren(list);
  }

  function renderPools(page) {
    refs.poolCount.textContent = `${page.items.length} ${page.items.length === 1 ? 'pool' : 'pools'}`;
    clear(refs.poolsBody);
    refs.poolsState.hidden = page.items.length > 0;
    if (!page.items.length) dataState(refs.poolsState, 'No configured pools.');
    for (const pool of page.items) {
      const control = pool.controls?.effective?.paused
        ? 'paused'
        : pool.controls?.effective?.force_anchor
          ? 'force_anchor'
          : 'normal';
      const row = selectableRow(
        () => openPool(pool.id),
        [
          element('span', { className: 'mono', text: pool.id }),
          formatLabel(pool.api_family),
          text(pool.anchor_models?.join(', ')),
          formatNumber(pool.candidates?.length, 0),
          formatNumber(pool.support?.ready_evidence, 0),
          formatPercent(pool.support?.coverage),
          badge(control, toneFor(control)),
          badge(pool.health?.worst, toneFor(pool.health?.worst)),
        ],
      );
      refs.poolsBody.append(row);
    }
  }

  function selectableRow(action, cells) {
    const row = element('tr', { 'data-selectable': 'true', tabIndex: 0 });
    for (const cell of cells) row.append(element('td', {}, cell));
    row.addEventListener('click', action);
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        action();
      }
    });
    return row;
  }

  async function openPool(poolId) {
    try {
      const detail = await api(`/api/router/v1/pools/${encodeURIComponent(poolId)}`, { validator: requirePoolDetail });
      const summary = detail.summary;
      const content = document.createDocumentFragment();
      content.append(
        detailBlock(
          'Identity',
          keyValues([
            ['Pool', summary.id],
            ['API family', formatLabel(summary.api_family)],
            ['Policy hash', summary.policy_version_id],
            ['Learning generation', summary.learning_generation_id],
            ['Anchor revision', summary.anchor_revision],
            ['Sampling probability', formatPercent(detail.sampling_probability)],
            ['Maximum candidates', formatNumber(detail.max_candidates_per_sample, 0)],
          ]),
        ),
      );
      content.append(
        detailBlock(
          'Models',
          simpleTable(
            ['Role', 'ID', 'Model', 'Revision', 'Cost rank'],
            [
              ...(summary.anchor_models || []).map((model) => ['Anchor', '-', model, summary.anchor_revision, '-']),
              ...(summary.candidates || []).map((candidate) => [
                'Candidate',
                candidate.id,
                candidate.model,
                candidate.model_revision,
                candidate.cost_rank,
              ]),
            ],
          ),
        ),
      );
      content.append(
        detailBlock(
          'Selector',
          keyValues([
            ['Tenants', detail.selector?.tenant_ids?.join(', ') || 'Any'],
            ['Agents', detail.selector?.agent_ids?.join(', ') || 'Any'],
            ['Owner scopes', detail.selector?.owner_scope_types?.join(', ') || 'Any'],
            ['Scope paths', detail.selector?.scope_path_patterns?.join(', ') || 'Any'],
            ['Metadata', jsonText(detail.selector?.metadata_equals)],
          ]),
        ),
      );
      content.append(
        detailBlock(
          'Learning',
          keyValues(
            Object.entries(detail.learning || {}).map(([key, value]) => [
              formatLabel(key),
              typeof value === 'number' ? formatNumber(value, 4) : value,
            ]),
          ),
        ),
      );
      content.append(
        detailBlock(
          'Support prerequisites',
          keyValues(
            Object.entries(detail.support_prerequisites || {}).map(([key, value]) => [
              formatLabel(key),
              badge(String(value), toneFor(String(value))),
            ]),
          ),
        ),
      );
      openDetail('Pool detail', summary.id, content);
    } catch (error) {
      announce(`Pool detail unavailable: ${error.code || error.message}`);
    }
  }

  function detailBlock(title, content) {
    return element('section', { className: 'detail-block' }, [element('h3', { text: title }), content]);
  }

  function simpleTable(headers, rows) {
    const head = element(
      'thead',
      {},
      element(
        'tr',
        {},
        headers.map((header) => element('th', { text: header })),
      ),
    );
    const body = element('tbody');
    for (const values of rows)
      body.append(
        element(
          'tr',
          {},
          values.map((value) => element('td', {}, value instanceof Node ? value : text(value))),
        ),
      );
    return element('div', { className: 'table-frame' }, element('table', {}, [head, body]));
  }

  function openDetail(eyebrow, title, content) {
    refs.detailEyebrow.textContent = text(eyebrow, 64);
    refs.detailTitle.textContent = text(title, 120);
    refs.detailContent.replaceChildren(content);
    if (!refs.detailDialog.open) refs.detailDialog.showModal();
  }

  function formValues(form) {
    const values = {};
    new FormData(form).forEach((value, key) => {
      const normalized = String(value).trim();
      if (normalized) values[key] = normalized;
    });
    return values;
  }

  function queryString(values) {
    const params = new URLSearchParams();
    Object.entries(values).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') params.set(key, value);
    });
    return params.toString();
  }

  async function loadEvidence(cursor = null, navigation = 'reset') {
    const filters = formValues(refs.evidenceFilters);
    const params = { limit: '100', ...filters };
    if (cursor) params.after = cursor;
    dataState(refs.evidenceState, 'Loading evidence...');
    try {
      const page = await api(`/api/router/v1/evidence?${queryString(params)}`, { validator: requirePage });
      if (navigation === 'next') {
        state.evidence.history.push(state.evidence.cursor);
        state.evidence.number += 1;
      } else if (navigation === 'back') {
        state.evidence.number = Math.max(1, state.evidence.number - 1);
      } else if (navigation !== 'same') {
        state.evidence.history = [];
        state.evidence.number = 1;
      }
      state.evidence.cursor = cursor;
      state.evidence.page = page;
      renderEvidence(page);
    } catch (error) {
      dataState(refs.evidenceState, `Evidence unavailable: ${error.code || error.message}`);
    }
  }

  function renderEvidence(page) {
    clear(refs.evidenceBody);
    refs.evidenceState.hidden = page.items.length > 0;
    if (!page.items.length) dataState(refs.evidenceState, 'No evidence matches these filters.');
    for (const evidence of page.items) {
      refs.evidenceBody.append(
        selectableRow(
          () => openEvidence(evidence.evidence_id),
          [
            formatTime(evidence.created_at_unix_ms),
            evidence.pool_id,
            evidence.candidate_id,
            badge(evidence.terminal_class, toneFor(evidence.terminal_class)),
            badge(evidence.quality_label || 'unlabeled', toneFor(evidence.quality_label)),
            badge(evidence.vector_state, toneFor(evidence.vector_state)),
            element('span', {
              className: 'preview',
              text: text(evidence.content?.preview || 'Redacted', PREVIEW_LIMIT),
            }),
          ],
        ),
      );
    }
    refs.evidenceBack.disabled = state.evidence.history.length === 0;
    refs.evidenceNext.disabled = !page.next;
    refs.evidencePageLabel.textContent = `Page ${state.evidence.number}`;
  }

  async function openEvidence(evidenceId) {
    try {
      const detail = await api(`/api/router/v1/evidence/${encodeURIComponent(evidenceId)}`, {
        validator: requireObject,
      });
      const summary = detail.summary;
      const content = document.createDocumentFragment();
      content.append(
        detailBlock(
          'Record',
          keyValues([
            ['Evidence', summary.evidence_id],
            ['Pool', summary.pool_id],
            ['Candidate', summary.candidate_id],
            ['Terminal', badge(summary.terminal_class, toneFor(summary.terminal_class))],
            ['Quality', badge(summary.quality_label || 'unlabeled', toneFor(summary.quality_label))],
            ['Vector', badge(summary.vector_state, toneFor(summary.vector_state))],
            ['Created', formatTime(summary.created_at_unix_ms)],
            ['Query hash', summary.canonical_query_hash],
            ['Generation', summary.learning_generation_id],
            ['Record hash', detail.record_hash],
          ]),
        ),
      );
      content.append(
        detailBlock(
          'Source',
          keyValues([
            ['Anchor', detail.anchor_id],
            ['Shadow attempt', detail.shadow_attempt_id],
            ['Evaluation', detail.evaluation_id || 'None'],
          ]),
        ),
      );
      content.append(detailBlock('Strict partition', element('pre', { text: jsonText(detail.partition) })));
      content.append(detailBlock('Content', contentDetail(summary.content)));
      openDetail('Evidence detail', shortId(evidenceId), content);
    } catch (error) {
      announce(`Evidence detail unavailable: ${error.code || error.message}`);
    }
  }

  function contentDetail(content) {
    const fragment = document.createDocumentFragment();
    fragment.append(
      keyValues([
        ['SHA-256', content?.sha256],
        ['Source bytes', formatNumber(content?.byte_length, 0)],
        ['Preview', text(content?.preview || 'Redacted', PREVIEW_LIMIT)],
      ]),
    );
    if (content?.value !== null && content?.value !== undefined) {
      const details = element('details');
      details.append(element('summary', { text: 'Full content' }));
      details.append(element('pre', { className: 'content-value', text: jsonText(content.value, CONTENT_LIMIT) }));
      fragment.append(details);
    }
    return fragment;
  }

  function evidenceFilter() {
    const values = formValues(refs.evidenceFilters);
    return {
      pool_id: values.pool_id || null,
      candidate_id: values.candidate_id || null,
      terminal_class: values.terminal_class || null,
      quality_label: values.quality_label || null,
      learning_generation_id: values.learning_generation_id || null,
    };
  }

  async function exportEvidence() {
    if (typeof window.showSaveFilePicker !== 'function') {
      dataState(refs.evidenceState, 'Streaming file export is unavailable in this browser.');
      return;
    }
    const format = refs.exportFormat.value;
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: `router-evidence.${format === 'csv' ? 'csv' : 'jsonl'}`,
        types: [
          {
            description: format.toUpperCase(),
            accept: {
              [format === 'csv' ? 'text/csv' : 'application/x-ndjson']: [format === 'csv' ? '.csv' : '.jsonl'],
            },
          },
        ],
      });
      const response = await window.fetch('/api/router/v1/evidence/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: format === 'csv' ? 'text/csv' : 'application/x-ndjson' },
        body: JSON.stringify({ filter: evidenceFilter(), format }),
        credentials: 'same-origin',
        cache: 'no-store',
        redirect: 'error',
      });
      if (!response.ok || !response.body) throw new ApiError(await responseError(response), response.status);
      const writable = await handle.createWritable();
      await response.body.pipeTo(writable);
      announce('Evidence export completed');
    } catch (error) {
      if (error?.name !== 'AbortError') dataState(refs.evidenceState, `Export failed: ${error.code || error.message}`);
    }
  }

  function setLookupMode(mode) {
    const evidence = mode === 'evidence';
    refs.evidenceLookupFields.hidden = !evidence;
    refs.queryLookupFields.hidden = evidence;
    refs.lookupEvidenceId.required = evidence;
    refs.lookupQueryHash.required = !evidence;
    refs.lookupPartition.required = !evidence;
    queryAll('[data-lookup-mode]').forEach((button) =>
      button.setAttribute('aria-pressed', String(button.dataset.lookupMode === mode)),
    );
  }

  async function inspectNeighborhood() {
    const mode = query("[data-lookup-mode][aria-pressed='true']").dataset.lookupMode;
    let body;
    if (mode === 'evidence') {
      const evidenceId = refs.lookupEvidenceId.value.trim();
      if (!evidenceId) return dataState(refs.neighborhoodState, 'Evidence ID is required.');
      body = { kind: 'evidence', evidence_id: evidenceId };
    } else {
      const hash = refs.lookupQueryHash.value.trim();
      const source = refs.lookupPartition.value;
      if (!hash || !source.trim())
        return dataState(refs.neighborhoodState, 'Query hash and exact partition are required.');
      if (new TextEncoder().encode(source).byteLength > INPUT_LIMIT)
        return dataState(refs.neighborhoodState, 'Partition JSON exceeds the input limit.');
      let partition;
      try {
        partition = JSON.parse(source);
      } catch (_) {
        return dataState(refs.neighborhoodState, 'Partition JSON is invalid.');
      }
      body = { kind: 'query_hash', canonical_query_hash: hash, partition };
    }
    dataState(refs.neighborhoodState, 'Inspecting neighborhood...');
    refs.neighborhoodContent.hidden = true;
    try {
      const report = await api('/api/router/v1/neighborhood', {
        method: 'POST',
        body,
        validator: requireNeighborhood,
      });
      state.neighborhood = report;
      renderNeighborhood(report);
    } catch (error) {
      if (error instanceof IncompatibleError) return setIncompatible(error);
      dataState(refs.neighborhoodState, `Neighborhood unavailable: ${error.code || error.message}`);
    }
  }

  function renderNeighborhood(report) {
    refs.neighborhoodState.hidden = true;
    refs.neighborhoodContent.hidden = false;
    refs.neighborhoodMetrics.replaceChildren(
      metric('Neighbors', formatNumber(report.support?.returned_neighbors, 0)),
      metric('Within radius', formatNumber(report.support?.within_radius, 0)),
      metric('Selected roots', formatNumber(report.support?.selected_roots, 0)),
      metric('Coverage', formatPercent(report.support?.coverage)),
      metric('Lower bound', formatPercent(report.credible_lower_bound)),
    );
    clear(refs.gateList);
    for (const [name, passed] of Object.entries(report.gates || {})) {
      refs.gateList.append(
        element('div', { className: 'gate-item' }, [
          element('span', { text: formatLabel(name) }),
          badge(passed ? 'Pass' : 'Fail', passed ? 'good' : 'bad'),
        ]),
      );
    }
    refs.recommendation.replaceChildren(
      keyValues([
        ['Candidate', report.recommendation?.candidate_id || 'Anchor'],
        ['Reason', formatLabel(report.recommendation?.reason)],
        [
          'Anchor fallback',
          badge(
            String(Boolean(report.recommendation?.anchor_fallback)),
            report.recommendation?.anchor_fallback ? 'warn' : 'good',
          ),
        ],
      ]),
    );
    clear(refs.neighborsBody);
    for (const neighbor of report.neighbors || []) {
      refs.neighborsBody.append(
        element(
          'tr',
          {},
          [
            neighbor.ordinal,
            element('span', { className: 'mono', text: shortId(neighbor.evidence_id) }),
            formatNumber(neighbor.distance, 4),
            `${formatNumber(neighbor.age_seconds, 0)}s`,
            formatNumber(neighbor.final_weight, 4),
            badge(neighbor.binary_label || 'unlabeled', toneFor(neighbor.binary_label)),
            formatLabel(neighbor.inclusion),
          ].map((value) => element('td', {}, value)),
        ),
      );
    }
    renderProjection(report.projection);
  }

  function validProjection(projection) {
    return (
      projection &&
      projection.algorithm === 'pca_2' &&
      projection.algorithm_version === 1 &&
      projection.diagnostic_only === true &&
      Array.isArray(projection.points) &&
      projection.points.every(
        (point) => Number.isFinite(point.x) && Number.isFinite(point.y) && ['query', 'evidence'].includes(point.kind),
      )
    );
  }

  function renderProjection(projection) {
    refs.projectionLegend.replaceChildren(
      element('span', { className: 'legend-item' }, [element('span', { className: 'legend-dot query' }), 'Query']),
      element('span', { className: 'legend-item' }, [element('span', { className: 'legend-dot' }), 'Evidence']),
    );
    if (!validProjection(projection)) {
      state.projectionPoints = [];
      dataState(refs.projectionEmpty, projection ? 'Unsupported projection.' : 'Projection unavailable.');
      return;
    }
    if (!projection.points.length) {
      state.projectionPoints = [];
      dataState(refs.projectionEmpty, 'Projection has no points.');
      return;
    }
    refs.projectionEmpty.hidden = true;
    drawProjection(projection);
  }

  function drawProjection(projection) {
    const canvas = refs.projectionCanvas;
    const rectangle = canvas.getBoundingClientRect();
    if (rectangle.width < 1 || rectangle.height < 1) return;
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(rectangle.width * ratio);
    canvas.height = Math.round(rectangle.height * ratio);
    const context = canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rectangle.width, rectangle.height);
    const margin = 34;
    const points = projection.points;
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);
    if (minX === maxX) {
      minX -= 1;
      maxX += 1;
    }
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const padX = (maxX - minX) * 0.15;
    const padY = (maxY - minY) * 0.15;
    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;
    const project = (point) => ({
      x: margin + ((point.x - minX) / (maxX - minX)) * (rectangle.width - margin * 2),
      y: rectangle.height - margin - ((point.y - minY) / (maxY - minY)) * (rectangle.height - margin * 2),
    });
    context.strokeStyle = '#d9ddd9';
    context.lineWidth = 1;
    for (let step = 0; step <= 4; step += 1) {
      const x = margin + ((rectangle.width - margin * 2) * step) / 4;
      const y = margin + ((rectangle.height - margin * 2) * step) / 4;
      context.beginPath();
      context.moveTo(x, margin);
      context.lineTo(x, rectangle.height - margin);
      context.stroke();
      context.beginPath();
      context.moveTo(margin, y);
      context.lineTo(rectangle.width - margin, y);
      context.stroke();
    }
    state.projectionPoints = points.map((point) => {
      const screen = project(point);
      return { ...point, sourceX: point.x, sourceY: point.y, screenX: screen.x, screenY: screen.y };
    });
    for (const point of state.projectionPoints) {
      context.fillStyle = point.kind === 'query' ? '#26734d' : '#176b78';
      context.strokeStyle = '#ffffff';
      context.lineWidth = 2;
      context.beginPath();
      if (point.kind === 'query') context.rect(point.screenX - 7, point.screenY - 7, 14, 14);
      else context.arc(point.screenX, point.screenY, 7, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }
    refs.projectionSelection.textContent = `Variance: ${(projection.explained_variance_ratio || []).map((value) => formatPercent(value)).join(' / ')}`;
  }

  function selectProjectionPoint(event) {
    const rectangle = refs.projectionCanvas.getBoundingClientRect();
    const x = event.clientX - rectangle.left;
    const y = event.clientY - rectangle.top;
    let selected = null;
    let distance = 16;
    for (const point of state.projectionPoints) {
      const current = Math.hypot(point.screenX - x, point.screenY - y);
      if (current < distance) {
        selected = point;
        distance = current;
      }
    }
    refs.projectionSelection.textContent = selected
      ? `${formatLabel(selected.kind)} / ${text(selected.record_id, 96)} / x ${formatNumber(selected.sourceX, 4)} / y ${formatNumber(selected.sourceY, 4)}`
      : 'No point selected';
  }

  function decisionQuery(cursor) {
    const values = { limit: '100', ...state.decisionFilters };
    if (cursor) values.after = cursor;
    return queryString(values);
  }

  async function loadDecisions() {
    dataState(refs.decisionsState, 'Loading decisions...');
    try {
      const page = await api(`/api/router/v1/decisions/tail?${decisionQuery(null)}`, { validator: requirePage });
      state.decisions.clear();
      state.exposures.clear();
      page.items.forEach((decision) => state.decisions.set(decision.decision_id, decision));
      state.decisionCursor = page.next;
      renderDecisions();
      await Promise.allSettled(page.items.map((decision) => loadExposure(decision.decision_id)));
      if (refs.decisionFollow.checked && !document.hidden && state.view === 'decisions') startDecisionStream();
    } catch (error) {
      dataState(refs.decisionsState, `Decisions unavailable: ${error.code || error.message}`);
      setStreamState('Unavailable', 'warn');
    }
  }

  async function loadExposure(decisionId) {
    try {
      const exposure = await api(`/api/router/v1/decisions/${encodeURIComponent(decisionId)}/exposure`, {
        validator: requireDecisionExposure,
      });
      state.exposures.set(decisionId, exposure);
      renderDecisions();
      return exposure;
    } catch (_) {
      return null;
    }
  }

  function renderDecisions() {
    clear(refs.decisionsBody);
    const decisions = Array.from(state.decisions.values())
      .sort((left, right) => right.created_at_unix_ms - left.created_at_unix_ms)
      .slice(0, 500);
    refs.decisionsState.hidden = decisions.length > 0;
    if (!decisions.length) dataState(refs.decisionsState, 'No decisions match these filters.');
    for (const decision of decisions) {
      const exposure = state.exposures.get(decision.decision_id);
      const active = exposure?.active;
      const outcome = exposure?.outcome;
      refs.decisionsBody.append(
        selectableRow(
          () => openDecision(decision.decision_id),
          [
            formatTime(decision.created_at_unix_ms),
            decision.pool_id,
            badge(decision.mode, toneFor(decision.mode)),
            decision.candidate_id || '-',
            decision.recommended_model,
            decision.served_model,
            formatLabel(active?.fallback_reason || decision.final_reason),
            badge(active?.assignment_arm || 'not exposed', toneFor(active?.assignment_arm)),
            badge(outcome?.label || (active ? 'unlabeled' : 'none'), toneFor(outcome?.label)),
          ],
        ),
      );
    }
  }

  async function openDecision(decisionId) {
    try {
      const [detail, exposure] = await Promise.all([
        api(`/api/router/v1/decisions/${encodeURIComponent(decisionId)}`, { validator: requireObject }),
        state.exposures.get(decisionId) || loadExposure(decisionId),
      ]);
      const summary = detail.summary;
      const content = document.createDocumentFragment();
      content.append(
        detailBlock(
          'Decision',
          keyValues([
            ['Decision', summary.decision_id],
            ['Created', formatTime(summary.created_at_unix_ms)],
            ['Pool', summary.pool_id],
            ['Mode', badge(summary.mode, toneFor(summary.mode))],
            ['Candidate', summary.candidate_id || 'None'],
            ['Recommended', summary.recommended_model],
            ['Served', summary.served_model],
            ['Final reason', formatLabel(summary.final_reason)],
            ['Cohort', summary.cohort_generation_id || 'None'],
            ['Query hash', summary.canonical_query_hash],
            ['Record hash', detail.record_hash],
          ]),
        ),
      );
      if (exposure?.active)
        content.append(
          detailBlock(
            'Exposure',
            keyValues(
              Object.entries(exposure.active).map(([key, value]) => [
                formatLabel(key),
                typeof value === 'number' ? formatNumber(value, 6) : (value ?? 'None'),
              ]),
            ),
          ),
        );
      else
        content.append(detailBlock('Exposure', element('p', { text: 'Recommend decision / no Active assignment.' })));
      if (exposure?.outcome)
        content.append(
          detailBlock(
            'Outcome',
            keyValues(Object.entries(exposure.outcome).map(([key, value]) => [formatLabel(key), value ?? 'Unlabeled'])),
          ),
        );
      content.append(
        detailBlock(
          'Candidate confidence',
          simpleTable(
            ['Candidate', 'Rank', 'Neighbors', 'ESS', 'Lower bound', 'Reason'],
            (detail.candidates || []).map((candidate) => [
              candidate.candidate_id,
              candidate.rank_ordinal,
              candidate.neighbor_count,
              formatNumber(candidate.effective_sample_size, 4),
              formatPercent(candidate.lower_bound),
              formatLabel(candidate.reason),
            ]),
          ),
        ),
      );
      content.append(
        detailBlock(
          'Neighbor audit',
          simpleTable(
            ['#', 'Candidate', 'Evidence', 'Distance', 'Weight', 'Label', 'Reason'],
            (detail.neighbors || []).map((neighbor) => [
              neighbor.neighbor_ordinal,
              neighbor.candidate_id,
              shortId(neighbor.evidence_id),
              formatNumber(neighbor.distance, 4),
              formatNumber(neighbor.final_weight, 4),
              neighbor.binary_label || '-',
              formatLabel(neighbor.exclusion_reason),
            ]),
          ),
        ),
      );
      openDetail('Decision detail', shortId(decisionId), content);
    } catch (error) {
      announce(`Decision detail unavailable: ${error.code || error.message}`);
    }
  }

  function startDecisionStream() {
    stopDecisionStream();
    if (!refs.decisionFollow.checked || document.hidden || state.view !== 'decisions') return;
    const params = new URLSearchParams(decisionQuery(state.decisionCursor));
    const source = new EventSource(`/api/router/v1/decisions/stream?${params.toString()}`, { withCredentials: true });
    state.decisionSource = source;
    setStreamState('Connecting', 'warn');
    source.addEventListener('open', () => setStreamState('Live', 'live'));
    source.addEventListener('decision', (event) => {
      try {
        const decision = JSON.parse(event.data);
        if (!decision?.decision_id) return;
        state.decisions.set(decision.decision_id, decision);
        renderDecisions();
        loadExposure(decision.decision_id);
      } catch (_) {
        setStreamState('Invalid event', 'warn');
      }
    });
    source.addEventListener('cursor', (event) => {
      try {
        const checkpoint = JSON.parse(event.data);
        if (checkpoint?.cursor) state.decisionCursor = checkpoint.cursor;
      } catch (_) {
        setStreamState('Invalid cursor', 'warn');
      }
    });
    source.addEventListener('error', () => {
      stopDecisionStream();
      setStreamState('Polling fallback', 'warn');
      startDecisionFallback();
    });
  }

  function stopDecisionStream() {
    if (state.decisionSource) state.decisionSource.close();
    state.decisionSource = null;
    if (state.decisionFallback) window.clearInterval(state.decisionFallback);
    state.decisionFallback = null;
  }

  function startDecisionFallback() {
    if (!refs.decisionFollow.checked || document.hidden || state.view !== 'decisions' || state.decisionFallback) return;
    pollDecisionTail();
    state.decisionFallback = window.setInterval(pollDecisionTail, POLL_MS);
  }

  async function pollDecisionTail() {
    try {
      const page = await api(`/api/router/v1/decisions/tail?${decisionQuery(state.decisionCursor)}`, {
        validator: requirePage,
      });
      for (const decision of page.items) {
        if (!state.decisions.has(decision.decision_id)) {
          state.decisions.set(decision.decision_id, decision);
          loadExposure(decision.decision_id);
        }
      }
      state.decisionCursor = page.next || state.decisionCursor;
      renderDecisions();
      setStreamState('Polling', 'warn');
    } catch (_) {
      setStreamState('Disconnected', 'warn');
      refs.streamResume.hidden = false;
    }
  }

  function setStreamState(label, tone = '') {
    refs.streamLabel.textContent = label;
    refs.streamIndicator.className = `status-dot ${tone}`.trim();
    refs.streamResume.hidden = !['Disconnected', 'Unavailable'].includes(label);
  }

  function renderOperations() {
    const controls = state.controls?.items || [];
    clear(refs.controlsBody);
    for (const entry of controls) {
      const scope = entry.scope?.kind === 'pool' ? entry.scope.pool_id : entry.scope?.kind || 'all';
      refs.controlsBody.append(
        selectableRow(
          () => openControl(entry),
          [
            formatTime(entry.created_at_unix_ms),
            formatLabel(entry.kind),
            scope,
            badge(entry.result, toneFor(entry.result)),
            entry.control_generation ?? '-',
            entry.actor,
            text(entry.reason, PREVIEW_LIMIT),
          ],
        ),
      );
    }
    renderEventList(refs.healthEvents, state.health?.items || [], (entry) => [
      badge(entry.severity, toneFor(entry.severity)),
      formatLabel(entry.subject_kind),
      formatLabel(entry.reason),
      formatTime(entry.created_at_unix_ms),
    ]);
    renderEventList(refs.migrationList, state.migrations?.items || [], (entry) => [
      badge(entry.verified ? 'Verified' : 'Mismatch', entry.verified ? 'good' : 'bad'),
      `v${entry.version} ${entry.name}`,
      formatTime(entry.applied_at_unix_ms),
    ]);
    const status = state.overview?.status;
    if (status) {
      refs.authorityStatus.replaceChildren(
        keyValues([
          ['Configured mode', badge(status.configured_mode, toneFor(status.configured_mode))],
          ['Effective mode', badge(status.effective_mode, toneFor(status.effective_mode))],
          ['Config generation', status.config_generation_id],
          ['Cohort generation', status.cohort_generation_id],
          ['Control generation', status.controls?.control_generation ?? '-'],
          [
            'Paused',
            badge(
              String(Boolean(status.controls?.effective?.paused)),
              status.controls?.effective?.paused ? 'warn' : 'good',
            ),
          ],
          [
            'Force anchor',
            badge(
              String(Boolean(status.controls?.effective?.force_anchor)),
              status.controls?.effective?.force_anchor ? 'warn' : 'good',
            ),
          ],
          ['Queue', `${status.queues?.pending ?? 0} / ${status.queues?.capacity ?? 0}`],
          ['Leases', `${status.leases?.active ?? 0} active / ${status.leases?.expired ?? 0} expired`],
        ]),
      );
    }
    refs.operationsState.hidden = Boolean(state.controls && state.health && state.migrations);
    if (!refs.operationsState.hidden) dataState(refs.operationsState, 'Some operational state is unavailable.');
  }

  function renderEventList(target, items, formatter) {
    const list = element('div', { className: 'event-list' });
    if (!items.length) list.append(element('p', { text: 'No records.' }));
    for (const item of items) {
      const values = formatter(item);
      const row = element('div', { className: 'event-item' });
      row.append(values[0] instanceof Node ? values[0] : element('strong', { text: values[0] }));
      values.slice(1).forEach((value) => row.append(element('p', { text: value })));
      list.append(row);
    }
    target.replaceChildren(list);
  }

  function openControl(entry) {
    const content = document.createDocumentFragment();
    content.append(
      detailBlock(
        'History entry',
        keyValues(
          Object.entries(entry).map(([key, value]) => [
            formatLabel(key),
            typeof value === 'object' ? jsonText(value) : (value ?? 'None'),
          ]),
        ),
      ),
    );
    openDetail('Control history', formatLabel(entry.kind), content);
  }

  function selectView(view, push = true, load = true) {
    if (!VALID_VIEWS.has(view)) view = 'overview';
    state.view = view;
    queryAll('[data-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.panel !== view;
    });
    queryAll('[data-view]').forEach((tab) => {
      const selected = tab.dataset.view === view;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    if (push) {
      const url = new URL(window.location.href);
      url.hash = '';
      url.search = '';
      url.searchParams.set('view', view);
      window.history.pushState({ view }, '', url);
    }
    if (load && view === 'evidence' && !state.evidence.page) loadEvidence();
    if (view === 'decisions' && load && state.decisions.size === 0) loadDecisions();
    else if (view === 'decisions' && load && refs.decisionFollow.checked) startDecisionStream();
    else stopDecisionStream();
    refs.mainContent.focus({ preventScroll: true });
  }

  function bindEvents() {
    const tabs = queryAll('[data-view]');
    tabs.forEach((button, index) => {
      button.addEventListener('click', () => selectView(button.dataset.view));
      button.addEventListener('keydown', (event) => {
        let target = null;
        if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') target = 0;
        if (event.key === 'End') target = tabs.length - 1;
        if (target === null) return;
        event.preventDefault();
        selectView(tabs[target].dataset.view);
        tabs[target].focus();
      });
    });
    window.addEventListener('popstate', (event) =>
      selectView(event.state?.view || new URLSearchParams(window.location.search).get('view') || 'overview', false),
    );
    refs.refreshButton.addEventListener('click', async () => {
      await refreshPolled();
      if (state.view === 'evidence') await loadEvidence(state.evidence.cursor, 'same');
      if (state.view === 'decisions') await loadDecisions();
      announce('Dashboard refreshed');
    });
    refs.detailClose.addEventListener('click', () => refs.detailDialog.close());
    refs.detailDialog.addEventListener('click', (event) => {
      if (event.target === refs.detailDialog) refs.detailDialog.close();
    });
    refs.evidenceFilters.addEventListener('submit', (event) => {
      event.preventDefault();
      loadEvidence();
    });
    refs.evidenceReset.addEventListener('click', () => {
      refs.evidenceFilters.reset();
      loadEvidence();
    });
    refs.evidenceNext.addEventListener('click', () => {
      if (state.evidence.page?.next) loadEvidence(state.evidence.page.next, 'next');
    });
    refs.evidenceBack.addEventListener('click', () => {
      const cursor = state.evidence.history.pop() || null;
      loadEvidence(cursor, 'back');
    });
    refs.exportButton.addEventListener('click', exportEvidence);
    queryAll('[data-lookup-mode]').forEach((button) =>
      button.addEventListener('click', () => setLookupMode(button.dataset.lookupMode)),
    );
    refs.neighborhoodForm.addEventListener('submit', (event) => {
      event.preventDefault();
      inspectNeighborhood();
    });
    refs.projectionCanvas.addEventListener('click', selectProjectionPoint);
    const redraw = () => {
      if (state.neighborhood?.projection && state.view === 'neighborhood')
        drawProjection(state.neighborhood.projection);
    };
    if (typeof ResizeObserver === 'function') new ResizeObserver(redraw).observe(refs.projectionCanvas);
    else window.addEventListener('resize', redraw);
    refs.decisionFilters.addEventListener('submit', (event) => {
      event.preventDefault();
      state.decisionFilters = formValues(refs.decisionFilters);
      loadDecisions();
    });
    refs.decisionReset.addEventListener('click', () => {
      refs.decisionFilters.reset();
      state.decisionFilters = {};
      loadDecisions();
    });
    refs.decisionFollow.addEventListener('change', () => {
      if (refs.decisionFollow.checked) startDecisionStream();
      else {
        stopDecisionStream();
        setStreamState('Paused');
      }
    });
    refs.streamResume.addEventListener('click', startDecisionStream);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopPolling();
        stopDecisionStream();
        if (state.view === 'decisions') setStreamState('Hidden / paused');
      } else {
        refreshPolled();
        startPolling();
        if (state.view === 'decisions' && refs.decisionFollow.checked) startDecisionStream();
      }
    });
  }

  function collectRefs() {
    const ids = [
      'project-name',
      'configured-mode',
      'effective-mode',
      'generation-id',
      'control-state',
      'health-state',
      'stale-indicator',
      'refresh-button',
      'main-content',
      'overview-window',
      'overview-state',
      'overview-content',
      'overview-metrics',
      'routing-breakdown',
      'outcome-breakdown',
      'runtime-status',
      'freshness-status',
      'pool-count',
      'pools-state',
      'pools-body',
      'evidence-filters',
      'evidence-reset',
      'evidence-state',
      'evidence-body',
      'evidence-back',
      'evidence-next',
      'evidence-page-label',
      'export-format',
      'export-button',
      'neighborhood-form',
      'evidence-lookup-fields',
      'query-lookup-fields',
      'lookup-evidence-id',
      'lookup-query-hash',
      'lookup-partition',
      'neighborhood-state',
      'neighborhood-content',
      'neighborhood-metrics',
      'gate-list',
      'recommendation',
      'neighbors-body',
      'projection-canvas',
      'projection-empty',
      'projection-legend',
      'projection-selection',
      'decision-filters',
      'decision-reset',
      'decision-follow',
      'decisions-state',
      'decisions-body',
      'stream-indicator',
      'stream-label',
      'stream-resume',
      'operations-state',
      'controls-body',
      'health-events',
      'migration-list',
      'authority-status',
      'detail-dialog',
      'detail-eyebrow',
      'detail-title',
      'detail-close',
      'detail-content',
      'live-region',
    ];
    for (const id of ids)
      refs[id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = document.getElementById(id);
  }

  function startPolling() {
    stopPolling();
    if (!document.hidden && !state.incompatible) state.pollTimer = window.setInterval(refreshPolled, POLL_MS);
  }

  function stopPolling() {
    if (state.pollTimer) window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }

  async function bootstrap() {
    let nonce = null;
    const fragment = bootstrapFragment;
    bootstrapFragment = null;
    if (fragment) {
      const match = /^#bootstrap=([A-Za-z0-9_-]{43})$/.exec(fragment);
      if (!match) throw new ApiError('invalid_bootstrap', 400);
      nonce = match[1];
      const response = await window.fetch('/auth/bootstrap', {
        method: 'POST',
        headers: { Authorization: `Bootstrap ${nonce}` },
        credentials: 'same-origin',
        cache: 'no-store',
        redirect: 'error',
      });
      nonce = null;
      if (!response.ok) throw new ApiError(await responseError(response), response.status);
    }
    state.authenticated = true;
  }

  async function initialize() {
    collectRefs();
    installIcons();
    bindEvents();
    try {
      await bootstrap();
      const requested = new URLSearchParams(window.location.search).get('view');
      const initialView = VALID_VIEWS.has(requested) ? requested : 'overview';
      selectView(initialView, false, false);
      await refreshPolled();
      if (initialView === 'evidence') await loadEvidence();
      if (initialView === 'decisions') await loadDecisions();
      startPolling();
    } catch (error) {
      refs.projectName.textContent = 'Session unavailable';
      refs.staleIndicator.hidden = false;
      refs.staleIndicator.textContent = 'Unauthenticated';
      dataState(refs.overviewState, `Dashboard session unavailable: ${error.code || error.message}`);
      refs.overviewContent.hidden = true;
    }
  }

  initialize();
})();
