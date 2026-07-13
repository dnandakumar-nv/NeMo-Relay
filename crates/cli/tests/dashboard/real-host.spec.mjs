// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../../../..');
const BINARY = path.join(REPO_ROOT, 'target/debug', process.platform === 'win32' ? 'nemo-relay.exe' : 'nemo-relay');
const VIEWS = ['Overview', 'Pools', 'Evidence', 'Neighborhoods', 'Decisions', 'Operations'];

function pluginConfig(databasePath) {
  return `version = 1

[[components]]
kind = "router"
enabled = true

[components.config]
version = 1
mode = "shadow"
project_id = "dashboard-playwright-smoke"
database_path = ${JSON.stringify(databasePath)}

[[components.config.pools]]
id = "dashboard-smoke-pool"
api_family = "openai_chat_completions"
anchor_models = ["anchor-model"]
anchor_revision = "dashboard-smoke-v1"
sampling_probability = 1.0
max_candidates_per_sample = 1

[components.config.pools.selector]
tenant_ids = ["tenant-a"]

[components.config.pools.concurrency]
shadow = 1
judge = 1
max_pending = 2

[components.config.pools.judge]
version = 1
model = "judge-model"
model_revision = "dashboard-smoke-v1"
prompt_version = "pairwise-equivalence-v1"
rubric_version = "response-trajectory-equivalence-v1"
output_schema_version = 1
response_weight = 0.5
trajectory_weight = 0.5
response_floor = 0.8
trajectory_floor = 0.8
judge_confidence_floor = 0.7
pass_threshold = 0.85
max_rationale_bytes = 4096
base_cooloff_seconds = 10
max_cooloff_seconds = 300

[[components.config.pools.candidates]]
id = "candidate"
model = "candidate-model"
model_revision = "dashboard-smoke-v1"
cost_rank = 0
`;
}

function startProcess(arguments_, options) {
  const child = spawn(BINARY, arguments_, { ...options, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.output = '';
  child.errors = '';
  child.stdout.on('data', (chunk) => {
    child.output += chunk;
  });
  child.stderr.on('data', (chunk) => {
    child.errors += chunk;
  });
  return child;
}

async function processExit(child, timeoutMs = 20_000) {
  if (child.exitCode !== null) return child.exitCode;
  return Promise.race([
    new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code) => resolve(code));
    }),
    new Promise((_, reject) => setTimeout(() => reject(new Error(`process timeout: ${child.errors}`)), timeoutMs)),
  ]);
}

async function waitForToken(tokenPath) {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const value = (await readFile(tokenPath, 'utf8')).trim();
      if (value) return value;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error('dashboard token file was not created');
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

test('compiled CLI host bootstraps and remains read only', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'the real host smoke runs once');
  test.setTimeout(60_000);

  const temporary = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-dashboard-playwright-'));
  const databasePath = path.join(temporary, 'router.sqlite3');
  const tokenPath = path.join(temporary, 'dashboard.token');
  const pluginPath = path.join(temporary, 'plugins.toml');
  const environment = {
    ...process.env,
    HOME: temporary,
    XDG_CONFIG_HOME: path.join(temporary, 'xdg'),
    APPDATA: path.join(temporary, 'appdata'),
    LOCALAPPDATA: path.join(temporary, 'localappdata'),
    NEMO_RELAY_PLUGIN_CONFIG_PATH: pluginPath,
  };
  delete environment.OPENAI_API_KEY;
  delete environment.ANTHROPIC_API_KEY;
  delete environment.INFERENCE_API_KEY;
  await writeFile(pluginPath, pluginConfig(databasePath), 'utf8');

  let dashboard = null;
  try {
    const probe = startProcess(['router-package-probe'], { cwd: temporary, env: environment });
    expect(await processExit(probe)).toBe(0);
    const before = digest(await readFile(databasePath));

    dashboard = startProcess(['router', 'dashboard', '--no-open', '--token-file', tokenPath, '--idle-timeout', '60'], {
      cwd: temporary,
      env: environment,
    });
    const launchUrl = await waitForToken(tokenPath);
    const launchOrigin = new URL(launchUrl).origin;
    const externalRequests = [];
    await page.route('**/*', (route) => {
      const target = new URL(route.request().url());
      if (target.origin === launchOrigin) return route.continue();
      externalRequests.push(target.href);
      return route.abort();
    });

    await page.goto(launchUrl);
    await expect(page.locator('#project-name')).toHaveText('dashboard-playwright-smoke');
    expect(new URL(page.url()).hash).toBe('');
    expect(externalRequests).toEqual([]);
    await expect
      .poll(async () => {
        try {
          await stat(tokenPath);
          return false;
        } catch (error) {
          if (error.code === 'ENOENT') return true;
          throw error;
        }
      })
      .toBe(true);

    const cookies = await page.context().cookies(launchOrigin);
    const session = cookies.find((cookie) => cookie.name === 'nemo_relay_dashboard_session');
    expect(session).toMatchObject({ httpOnly: true, sameSite: 'Strict', secure: false });

    const routeStatuses = await page.evaluate(async () => {
      const paths = [
        '/api/router/v1/status',
        '/api/router/v1/overview',
        '/api/router/v1/pools?limit=10',
        '/api/router/v1/pools/dashboard-smoke-pool',
        '/api/router/v1/evidence?limit=10',
        '/api/router/v1/decisions?limit=10',
        '/api/router/v1/decisions/tail?limit=10',
        '/api/router/v1/outcomes?limit=10',
        '/api/router/v1/controls?limit=10',
        '/api/router/v1/health?limit=10',
        '/api/router/v1/migrations?limit=10',
      ];
      return Promise.all(
        paths.map(async (path_) => [path_, (await fetch(path_, { credentials: 'same-origin' })).status]),
      );
    });
    expect(routeStatuses).toEqual(routeStatuses.map(([route]) => [route, 200]));

    for (const view of VIEWS) {
      await page.getByRole('tab', { name: view }).click();
      await expect(
        page.getByRole('heading', { name: view === 'Neighborhoods' ? 'Neighborhoods' : view, level: 2 }),
      ).toBeVisible();
    }
    await page.getByRole('tab', { name: 'Decisions' }).click();
    await page.locator('#decision-follow').uncheck();
    await page.close();
    dashboard.kill('SIGINT');
    expect(await processExit(dashboard)).toBe(0);
    dashboard = null;
    expect(digest(await readFile(databasePath))).toBe(before);
  } finally {
    if (dashboard && dashboard.exitCode === null) {
      dashboard.kill('SIGINT');
      try {
        await processExit(dashboard, 5_000);
      } catch (_) {
        dashboard.kill('SIGKILL');
      }
    }
    await rm(temporary, { recursive: true, force: true });
  }
});
