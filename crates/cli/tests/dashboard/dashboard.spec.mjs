// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../../../..');
const ASSET_ROOT = path.join(REPO_ROOT, 'crates/cli/assets/router-dashboard');
const CONTRACT_PATH = path.join(REPO_ROOT, 'crates/cli/tests/fixtures/router-dashboard/contract-v1.json');
const BOOTSTRAP_NONCE = 'A'.repeat(43);
const VIEWS = ['overview', 'pools', 'evidence', 'neighborhood', 'decisions', 'operations'];

const [indexHtml, appJavaScript, stylesCss, contractSource] = await Promise.all([
  readFile(path.join(ASSET_ROOT, 'index.html')),
  readFile(path.join(ASSET_ROOT, 'app.js')),
  readFile(path.join(ASSET_ROOT, 'styles.css')),
  readFile(CONTRACT_PATH, 'utf8'),
]);
const contract = JSON.parse(contractSource);

function copy(value) {
  return structuredClone(value);
}

class FixtureHost {
  constructor() {
    this.server = null;
    this.origin = null;
    this.sockets = new Set();
    this.streams = new Set();
    this.reset();
  }

  reset() {
    this.requests = [];
    this.failPaths = new Set();
    this.incompatibleOverview = false;
    this.unsupportedProjection = false;
    this.closeDecisionStreams = false;
    this.streamConnections = 0;
  }

  async start() {
    this.server = http.createServer((request, response) => this.handle(request, response));
    this.server.on('connection', (socket) => {
      this.sockets.add(socket);
      socket.once('close', () => this.sockets.delete(socket));
    });
    await new Promise((resolve, reject) => {
      this.server.once('error', reject);
      this.server.listen(0, '127.0.0.1', resolve);
    });
    const address = this.server.address();
    this.origin = `http://127.0.0.1:${address.port}`;
  }

  async stop() {
    for (const response of this.streams) response.end();
    for (const socket of this.sockets) socket.destroy();
    if (this.server) await new Promise((resolve) => this.server.close(resolve));
  }

  asset(response, contentType, body) {
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Security-Policy':
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
      'Content-Type': contentType,
      'Referrer-Policy': 'no-referrer',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
    });
    response.end(body);
  }

  json(response, value, status = 200) {
    response.writeHead(status, { 'Cache-Control': 'no-store', 'Content-Type': 'application/json' });
    response.end(JSON.stringify(value));
  }

  page(items) {
    return {
      items: copy(items),
      next: null,
      snapshot_time_unix_ms: contract.overview.snapshot_time_unix_ms,
      content_policy: 'redacted',
    };
  }

  handle(request, response) {
    const url = new URL(request.url, this.origin || 'http://127.0.0.1');
    this.requests.push({ method: request.method, path: url.pathname, search: url.search });

    if (url.pathname === '/') return this.asset(response, 'text/html; charset=utf-8', indexHtml);
    if (url.pathname === '/app.js') return this.asset(response, 'text/javascript; charset=utf-8', appJavaScript);
    if (url.pathname === '/styles.css') return this.asset(response, 'text/css; charset=utf-8', stylesCss);
    if (url.pathname === '/favicon.ico') return response.writeHead(204).end();
    if (url.pathname === '/auth/bootstrap' && request.method === 'POST') {
      if (request.headers.authorization !== `Bootstrap ${BOOTSTRAP_NONCE}`)
        return this.json(response, { error: { code: 'unauthorized' } }, 401);
      response.writeHead(204, {
        'Cache-Control': 'no-store',
        'Set-Cookie': 'relay_dashboard_session=fixture; HttpOnly; SameSite=Strict; Path=/',
      });
      return response.end();
    }
    if (!String(request.headers.cookie || '').includes('relay_dashboard_session=fixture')) {
      return this.json(response, { error: { code: 'unauthorized' } }, 401);
    }
    if (this.failPaths.has(url.pathname)) return this.json(response, { error: { code: 'fixture_unavailable' } }, 503);

    if (url.pathname === '/api/router/v1/status') return this.json(response, contract.overview.status);
    if (url.pathname === '/api/router/v1/overview') {
      const overview = copy(contract.overview);
      if (this.incompatibleOverview) overview.schema = 'nemo.relay.router.overview@2';
      return this.json(response, overview);
    }
    if (url.pathname === '/api/router/v1/pools') return this.json(response, this.page([contract.pool_detail.summary]));
    if (url.pathname === '/api/router/v1/pools/pool-a') return this.json(response, contract.pool_detail);
    if (url.pathname === '/api/router/v1/evidence') {
      const page = copy(contract.evidence_page);
      page.items[0].content.preview = '<img src=x onerror=window.fixtureInjected=true> literal preview';
      return this.json(response, page);
    }
    if (url.pathname.startsWith('/api/router/v1/evidence/') && url.pathname !== '/api/router/v1/evidence/export') {
      return this.json(response, contract.evidence_detail);
    }
    if (url.pathname === '/api/router/v1/evidence/export' && request.method === 'POST') {
      response.writeHead(200, { 'Cache-Control': 'no-store', 'Content-Type': 'application/x-ndjson' });
      return response.end(`${JSON.stringify(contract.evidence_page.items[0])}\n`);
    }
    if (url.pathname === '/api/router/v1/neighborhood' && request.method === 'POST') {
      const report = copy(contract.neighborhood);
      if (this.unsupportedProjection) report.projection.algorithm_version = 2;
      return this.json(response, report);
    }
    if (url.pathname === '/api/router/v1/decisions/tail') return this.json(response, contract.decision_page);
    if (url.pathname === '/api/router/v1/decisions/stream') return this.decisionStream(request, response);
    if (/^\/api\/router\/v1\/decisions\/[^/]+\/exposure$/.test(url.pathname)) {
      const decisionId = url.pathname.split('/').at(-2);
      const exposure =
        contract.decision_exposures.find((item) => item.decision_id === decisionId) || contract.decision_exposures[0];
      return this.json(response, exposure);
    }
    if (/^\/api\/router\/v1\/decisions\/[^/]+$/.test(url.pathname))
      return this.json(response, contract.decision_detail);
    if (url.pathname === '/api/router/v1/controls') return this.json(response, contract.controls);
    if (url.pathname === '/api/router/v1/health') return this.json(response, contract.health);
    if (url.pathname === '/api/router/v1/migrations') return this.json(response, contract.migrations);
    return this.json(response, { error: { code: 'not_found' } }, 404);
  }

  decisionStream(request, response) {
    this.streamConnections += 1;
    this.streams.add(response);
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Type': 'text/event-stream',
      Connection: 'keep-alive',
    });
    response.write(`event: decision\ndata: ${JSON.stringify(contract.decision_page.items[0])}\n\n`);
    response.write(`id: fixture-cursor\nevent: cursor\ndata: {"cursor":"fixture-cursor"}\n\n`);
    const heartbeat = setInterval(() => response.write(': heartbeat\n\n'), 1_000);
    const cleanup = () => {
      clearInterval(heartbeat);
      this.streams.delete(response);
    };
    request.once('close', cleanup);
    response.once('close', cleanup);
    if (this.closeDecisionStreams) setTimeout(() => response.end(), 40);
  }
}

const host = new FixtureHost();

test.beforeAll(async () => host.start());
test.afterAll(async () => host.stop());
test.beforeEach(() => host.reset());

async function openDashboard(page, view = 'overview') {
  const externalRequests = [];
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.route('**/*', async (route) => {
    const target = new URL(route.request().url());
    if (target.origin === host.origin) return route.continue();
    externalRequests.push(target.href);
    return route.abort();
  });
  await page.goto(`${host.origin}/?view=${view}#bootstrap=${BOOTSTRAP_NONCE}`);
  await expect(page.locator('#project-name')).toHaveText('project-a');
  await expect(page).toHaveURL(`${host.origin}/?view=${view}`);
  expect(externalRequests).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(host.requests.filter((request) => request.path === '/auth/bootstrap')).toHaveLength(1);
  const browserState = await page.evaluate(async () => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
    databases: typeof indexedDB.databases === 'function' ? (await indexedDB.databases()).map((item) => item.name) : [],
  }));
  expect(browserState.local).toEqual([]);
  expect(browserState.session).toEqual([]);
  expect(browserState.databases).toEqual([]);
  return { externalRequests, pageErrors };
}

async function expectNoPageOverflow(page) {
  expect(
    await page.evaluate(() => ({
      document: document.documentElement.scrollWidth <= document.documentElement.clientWidth,
      body: document.body.scrollWidth <= document.body.clientWidth,
    })),
  ).toEqual({ document: true, body: true });
}

test('all six views render at desktop and mobile bounds', async ({ page }, testInfo) => {
  const { externalRequests, pageErrors } = await openDashboard(page);

  for (const view of VIEWS) {
    await page
      .getByRole('tab', { name: view === 'neighborhood' ? 'Neighborhoods' : new RegExp(`^${view}$`, 'i') })
      .click();
    await expect(page.locator(`[data-panel="${view}"]`)).toBeVisible();

    if (view === 'overview') {
      await expect(page.locator('#overview-metrics')).toContainText('12');
      await expect(page.locator('#overview-window')).toContainText('Last 24 hours');
      await expect(page.locator('#freshness-status')).toContainText('Never');
    }
    if (view === 'pools') {
      const row = page.locator('#pools-body tr').first();
      await expect(row).toContainText('pool-a');
      await row.focus();
      await page.keyboard.press('Enter');
      await expect(page.locator('#detail-dialog')).toBeVisible();
      await expect(page.locator('#detail-title')).toHaveText('pool-a');
      await page.locator('#detail-close').click();
    }
    if (view === 'evidence') {
      const row = page.locator('#evidence-body tr').first();
      await expect(row).toContainText('literal preview');
      expect(await page.evaluate(() => Boolean(window.fixtureInjected))).toBe(false);
      await row.focus();
      await page.keyboard.press('Enter');
      await expect(page.locator('#detail-dialog')).toBeVisible();
      await page.locator('#detail-close').click();
    }
    if (view === 'neighborhood') {
      await page.locator('#lookup-evidence-id').fill(contract.evidence_page.items[0].evidence_id);
      await page.locator('#neighborhood-form').getByRole('button', { name: 'Inspect' }).click();
      await expect(page.locator('#neighborhood-content')).toBeVisible();
      await expect(page.locator('#neighbors-body tr')).toHaveCount(contract.neighborhood.neighbors.length);
    }
    if (view === 'decisions') {
      await expect(page.locator('#decisions-body tr')).toHaveCount(1);
      await expect(page.locator('#stream-label')).toHaveText('Live');
    }
    if (view === 'operations') {
      await expect(page.locator('#controls-body tr')).toHaveCount(contract.controls.items.length);
      await expect(page.locator('#authority-status')).toContainText('Control generation');
      await expect(
        page.getByRole('button', { name: /pause router|force anchor|rotate generation|reset generation/i }),
      ).toHaveCount(0);
    }

    await expectNoPageOverflow(page);
    await page.screenshot({ path: testInfo.outputPath(`${testInfo.project.name}-${view}.png`), fullPage: false });
  }

  expect(externalRequests).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(host.requests.some((request) => request.path.startsWith('/api/router/v1/operations'))).toBe(false);
});

test('tab focus, stale preservation, and incompatible state are explicit', async ({ page }) => {
  await openDashboard(page);

  const overviewTab = page.getByRole('tab', { name: 'Overview' });
  await overviewTab.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Pools' })).toBeFocused();
  await expect(page.locator('#panel-pools')).toBeVisible();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Operations' })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(overviewTab).toBeFocused();

  host.failPaths.add('/api/router/v1/overview');
  await page.locator('#refresh-button').click();
  await expect(page.locator('#stale-indicator')).toHaveText('Stale');
  await expect(page.locator('#overview-metrics')).toContainText('12');

  host.failPaths.delete('/api/router/v1/overview');
  host.incompatibleOverview = true;
  await page.locator('#refresh-button').click();
  await expect(page.locator('#stale-indicator')).toHaveText('Incompatible');
  await expect(page.locator('#overview-state')).toContainText('Unsupported schema');
});

test('evidence export, projection validation, and live-tail fallback stay bounded', async ({ page }) => {
  await page.addInitScript(() => {
    window.__fixtureExportBytes = 0;
    window.showSaveFilePicker = async () => ({
      createWritable: async () =>
        new WritableStream({
          write(chunk) {
            window.__fixtureExportBytes += chunk.byteLength || chunk.length || 0;
          },
        }),
    });
  });
  await openDashboard(page, 'evidence');

  await page.locator('#export-button').click();
  await expect.poll(() => page.evaluate(() => window.__fixtureExportBytes)).toBeGreaterThan(0);
  expect(
    host.requests.some((request) => request.method === 'POST' && request.path === '/api/router/v1/evidence/export'),
  ).toBe(true);

  await page.getByRole('tab', { name: 'Neighborhoods' }).click();
  await page.locator('#lookup-evidence-id').fill(contract.evidence_page.items[0].evidence_id);
  await page.locator('#neighborhood-form').getByRole('button', { name: 'Inspect' }).click();
  await expect(page.locator('#neighborhood-content')).toBeVisible();
  const colorCount = await page.locator('#projection-canvas').evaluate((canvas) => {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    const colors = new Set();
    for (let index = 0; index < data.length; index += 4)
      colors.add(`${data[index]},${data[index + 1]},${data[index + 2]},${data[index + 3]}`);
    return colors.size;
  });
  expect(colorCount).toBeGreaterThan(3);

  host.unsupportedProjection = true;
  await page.locator('#neighborhood-form').getByRole('button', { name: 'Inspect' }).click();
  await expect(page.locator('#projection-empty')).toHaveText('Unsupported projection.');
  await expect(page.locator('#projection-empty')).toBeVisible();

  host.closeDecisionStreams = true;
  await page.getByRole('tab', { name: 'Decisions' }).click();
  await expect(page.locator('#stream-label')).toHaveText('Polling');
  await expect(page.locator('#decisions-body tr')).toHaveCount(1);
  await page.locator('#decision-follow').uncheck();
  await expect(page.locator('#stream-label')).toHaveText('Paused');
});

test('hidden pages pause polling and resume from memory', async ({ page }) => {
  await page.addInitScript(() => {
    const nativeSetInterval = window.setInterval.bind(window);
    window.setInterval = (callback, delay, ...args) => nativeSetInterval(callback, Math.min(delay, 80), ...args);
    let fixtureHidden = false;
    Object.defineProperty(Document.prototype, 'hidden', { configurable: true, get: () => fixtureHidden });
    window.__setFixtureHidden = (value) => {
      fixtureHidden = value;
      document.dispatchEvent(new Event('visibilitychange'));
    };
  });
  await openDashboard(page, 'decisions');
  await expect(page.locator('#stream-label')).toHaveText('Live');
  await page.waitForTimeout(120);

  await page.evaluate(() => window.__setFixtureHidden(true));
  await expect(page.locator('#stream-label')).toHaveText('Hidden / paused');
  const hiddenOverviewCount = host.requests.filter((request) => request.path === '/api/router/v1/overview').length;
  const hiddenStreamCount = host.streamConnections;
  await page.waitForTimeout(240);
  expect(host.requests.filter((request) => request.path === '/api/router/v1/overview')).toHaveLength(
    hiddenOverviewCount,
  );
  expect(host.streamConnections).toBe(hiddenStreamCount);

  await page.evaluate(() => window.__setFixtureHidden(false));
  await expect
    .poll(() => host.requests.filter((request) => request.path === '/api/router/v1/overview').length)
    .toBeGreaterThan(hiddenOverviewCount);
  await expect.poll(() => host.streamConnections).toBeGreaterThan(hiddenStreamCount);
});
