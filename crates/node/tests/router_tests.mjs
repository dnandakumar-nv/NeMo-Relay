// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const router = require('../router.js');
const native = require('../index.js');

function judgeConfig() {
  return router.judgeConfig({
    version: 1,
    model: 'judge-model',
    model_revision: '2026-07-11',
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
    temperature: 1,
  });
}

function shadowConfig(databasePath) {
  return router.defaultConfig({
    mode: 'shadow',
    project_id: 'node-router-test',
    database_path: databasePath,
    embedders: [
      router.embedderConfig({
        id: 'embedding-main',
        base_url: 'https://127.0.0.1:9443/v1',
        model: 'embedding-model',
        provider_revision: '2026-07-11',
        dimensions: 3,
        timeout_ms: 1000,
      }),
    ],
    pools: [
      router.poolConfig({
        id: 'node-pool',
        api_family: 'openai_chat_completions',
        anchor_models: ['anchor-model'],
        anchor_revision: '2026-07-11',
        sampling_probability: 1,
        max_candidates_per_sample: 1,
        selector: router.selectorConfig({ tenant_ids: ['tenant-a'] }),
        concurrency: router.concurrencyConfig({ shadow: 1, judge: 1, max_pending: 2 }),
        candidates: [
          router.candidateConfig({
            id: 'candidate',
            model: 'candidate-model',
            model_revision: '2026-07-11',
            cost_rank: 0,
            capabilities: router.candidateCapabilities({ tools: true }),
          }),
        ],
        judge: judgeConfig(),
        learning: router.learningConfig({ embedder: 'embedding-main' }),
      }),
    ],
  });
}

describe('Router plugin helpers', () => {
  it('builds exact defaults and complete nested config', () => {
    assert.deepEqual(router.defaultConfig(), {
      version: 1,
      mode: 'off',
      database_path: '.nemo-relay/router/router.db',
      retention_days: 30,
      max_evidence_records: 100000,
      allow_remote_embedding_egress: false,
      embedders: [],
      pools: [],
      policy: {
        unknown_component: 'warn',
        unknown_field: 'warn',
        unsupported_value: 'error',
      },
    });

    const config = shadowConfig('/tmp/router-node-test.sqlite3');
    assert.equal(config.embedders[0].max_in_flight, 4);
    assert.equal(config.embedders[0].batch_size, 16);
    assert.equal(config.pools[0].lookahead.deadline_seconds, 300);
    assert.equal(config.pools[0].canonicalizer.max_context_messages, 8);
    assert.equal(config.pools[0].judge.temperature, 1);
    assert.equal(config.pools[0].concurrency.max_pending, 2);
    assert.equal(config.pools[0].candidates[0].capabilities.tools, true);
    assert.deepEqual(config.pools[0].learning, { version: 1, embedder: 'embedding-main' });
    assert.deepEqual(config.pools[0].outcome, {});
    assert.equal(Object.hasOwn(config.embedders[0], 'api_key_env'), false);

    const component = router.ComponentSpec(config, { enabled: false });
    assert.equal(component.kind, 'router');
    assert.equal(component.enabled, false);
    assert.deepEqual(component.config, config);
  });

  it('strips undefined optional fields recursively', () => {
    assert.deepEqual(
      router.selectorConfig({
        tenant_ids: undefined,
        metadata_equals: { keep: 'yes', remove: undefined },
      }),
      { metadata_equals: { keep: 'yes' } },
    );
  });

  it('passes the binding-private native vector artifact probe', () => {
    const report = native.__routerNativeVectorProbe();

    assert.equal(typeof report.sqlite_version, 'string');
    assert.equal(report.vector_version, 'v0.1.9');
  });

  it('auto-registers, validates, and activates Shadow without provider I/O', async () => {
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-relay-node-router-'));
    const databasePath = path.join(temporary, 'router.sqlite3');
    const config = shadowConfig(databasePath);

    try {
      assert.equal(plugin.listKinds().includes(router.ROUTER_PLUGIN_KIND), true);
      assert.deepEqual(router.validateConfig(config).diagnostics, []);
      const report = await plugin.initialize({
        version: 1,
        components: [router.ComponentSpec(config)],
      });
      assert.deepEqual(report.diagnostics, []);
      assert.equal(fs.existsSync(databasePath), true);
    } finally {
      await plugin.clearAsync(5000);
      fs.rmSync(temporary, { recursive: true, force: true });
    }
  });

  it('uses native mode-specific validation diagnostics', () => {
    const config = shadowConfig('/tmp/router-node-invalid.sqlite3');
    config.mode = 'active';
    const report = router.validateConfig(config);
    assert.equal(
      report.diagnostics.some((diagnostic) => diagnostic.level === 'error' && diagnostic.field?.includes('outcome')),
      true,
    );
  });

  it('type-checks the Router export and existing replay type references', () => {
    const tsc = require.resolve('typescript/bin/tsc');
    const fixture = fileURLToPath(new URL('types/router_types.ts', import.meta.url));
    const result = spawnSync(
      process.execPath,
      [
        tsc,
        '--noEmit',
        '--strict',
        '--target',
        'ES2022',
        '--module',
        'Node16',
        '--moduleResolution',
        'Node16',
        fixture,
      ],
      { encoding: 'utf8' },
    );
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  });
});
