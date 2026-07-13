// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const declarationsPath = fileURLToPath(new URL('../index.d.ts', import.meta.url));
const typeFixturePath = fileURLToPath(new URL('types/replay_types.ts', import.meta.url));
const declarations = readFileSync(declarationsPath, 'utf8');

function functionDeclaration(name) {
  return declarations.split(/\r?\n/).find((line) => line.startsWith(`export declare function ${name}(`));
}

describe('Node public API declarations', () => {
  it('preserves the exact V1 managed LLM call signatures', () => {
    assert.equal(
      functionDeclaration('llmCallExecute'),
      'export declare function llmCallExecute(name: string, request: Json, func: (arg: Json) => any, handle?: ScopeHandle | undefined | null, attributes?: number | undefined | null, data?: Json | undefined | null, metadata?: Json | undefined | null, modelName?: string | undefined | null, codecDecode?: (arg: Json) => any | undefined | null, codecEncode?: (arg: Json) => any | undefined | null, responseCodecDecode?: (arg: Json) => any | undefined | null): Promise<unknown>',
    );
    assert.equal(
      functionDeclaration('llmCallExecuteAsync'),
      'export declare function llmCallExecuteAsync(name: string, request: Json, func: (...args: any[]) => any, handle?: ScopeHandle | undefined | null, attributes?: number | undefined | null, data?: Json | undefined | null, metadata?: Json | undefined | null, modelName?: string | undefined | null, codecDecode?: (arg: Json) => any | undefined | null, codecEncode?: (arg: Json) => any | undefined | null, responseCodecDecode?: (arg: Json) => any | undefined | null): Promise<unknown>',
    );
  });

  it('exports reusable deep-readonly replay types', () => {
    for (const name of [
      'DeepReadonlyJson',
      'LlmApiFamily',
      'LlmCallRole',
      'LlmTrajectoryScope',
      'LlmExecutionContext',
      'LlmReplayInvocation',
      'LlmReplayCallable',
      'LlmReplayDescriptor',
      'LlmReplayFactory',
    ]) {
      assert.match(declarations, new RegExp(`export (?:type|interface) ${name}\\b`));
    }
    assert.match(functionDeclaration('llmCallExecuteV2'), /replayFactory\?: LlmReplayFactory/);
  });

  it('omits the binding-private replay test driver', () => {
    assert.equal(functionDeclaration('__testNodeReplayBridge'), undefined);
    assert.doesNotMatch(declarations, /Internal test helper for the Node replay-factory lifetime bridge/);
    assert.equal(functionDeclaration('__routerNativeVectorProbe'), undefined);
  });

  it('type-checks the replay surface and readonly mutation fixtures', () => {
    const tsc = require.resolve('typescript/bin/tsc');
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
        typeFixturePath,
      ],
      { encoding: 'utf8' },
    );
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  });
});
