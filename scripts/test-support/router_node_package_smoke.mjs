// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const sourceRoot = path.resolve(path.dirname(scriptPath), '..', '..');

function judgeConfig(router) {
  return router.judgeConfig({
    version: 1,
    model: 'judge-model',
    model_revision: 'package-smoke-v1',
    prompt_version: 'pairwise-equivalence-v1',
    rubric_version: 'response-trajectory-equivalence-v1',
    output_schema_version: 1,
    response_weight: 0.5,
    trajectory_weight: 0.5,
    response_floor: 0.8,
    trajectory_floor: 0.8,
    judge_confidence_floor: 0.7,
    pass_threshold: 0.85,
    max_rationale_bytes: 4096,
    base_cooloff_seconds: 10,
    max_cooloff_seconds: 300,
  });
}

async function installedSmoke(expectedSourceRoot) {
  const cleanRequire = createRequire(path.join(process.cwd(), 'router-package-smoke.cjs'));
  const resolved = cleanRequire.resolve('nemo-relay-node/router');
  assert.equal(path.resolve(resolved).startsWith(path.resolve(expectedSourceRoot)), false);
  const native = cleanRequire('nemo-relay-node');
  const plugin = cleanRequire('nemo-relay-node/plugin');
  const router = cleanRequire('nemo-relay-node/router');

  const nativeReport = native.__routerNativeVectorProbe();
  assert.equal(typeof nativeReport.sqlite_version, 'string');
  assert.equal(nativeReport.vector_version, 'v0.1.9');
  const databasePath = path.join(process.cwd(), 'router.sqlite3');
  const config = router.defaultConfig({
    mode: 'shadow',
    project_id: 'node-package-smoke',
    database_path: databasePath,
    pools: [
      router.poolConfig({
        id: 'node-package-pool',
        api_family: 'openai_chat_completions',
        anchor_models: ['anchor-model'],
        anchor_revision: 'package-smoke-v1',
        sampling_probability: 1,
        max_candidates_per_sample: 1,
        concurrency: router.concurrencyConfig({ shadow: 1, judge: 1, max_pending: 2 }),
        candidates: [
          router.candidateConfig({
            id: 'candidate',
            model: 'candidate-model',
            model_revision: 'package-smoke-v1',
            cost_rank: 0,
          }),
        ],
        judge: judgeConfig(router),
      }),
    ],
  });
  assert.equal(plugin.listKinds().includes(router.ROUTER_PLUGIN_KIND), true);
  assert.deepEqual(router.validateConfig(config).diagnostics, []);
  try {
    const report = await plugin.initialize({
      version: 1,
      components: [router.ComponentSpec(config)],
    });
    assert.deepEqual(report.diagnostics, []);
    assert.equal(fs.existsSync(databasePath), true);
  } finally {
    await plugin.clearAsync(5000);
  }
}

async function main() {
  if (process.argv[2] === '--installed') {
    await installedSmoke(process.argv[3]);
    return;
  }
  const packagePath = path.resolve(process.argv[2] ?? '');
  if (!fs.statSync(packagePath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`npm package does not exist: ${packagePath}`);
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-relay-router-node-package-'));
  try {
    fs.writeFileSync(
      path.join(temporary, 'package.json'),
      JSON.stringify({ name: 'router-package-smoke', private: true }),
    );
    const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    const install = spawnSync(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', packagePath], {
      cwd: temporary,
      encoding: 'utf8',
      env: { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false' },
    });
    assert.equal(install.status, 0, `${install.stdout}\n${install.stderr}`);
    const child = spawnSync(process.execPath, [scriptPath, '--installed', sourceRoot], {
      cwd: temporary,
      encoding: 'utf8',
      env: { ...process.env, NODE_PATH: '' },
    });
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

await main();
