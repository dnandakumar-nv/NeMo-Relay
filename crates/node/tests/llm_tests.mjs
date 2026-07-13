// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const nodePackageDir = fileURLToPath(new URL('..', import.meta.url));

const {
  pushScope,
  popScope,
  llmCall,
  llmCallEnd,
  llmCallExecute,
  llmCallExecuteAsync,
  llmCallExecuteV2,
  llmStreamCallExecute,
  __testNodeReplayBridge,
  llmRequestIntercepts,
  llmConditionalExecution,
  registerLlmSanitizeRequestGuardrail,
  deregisterLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail,
  deregisterLlmSanitizeResponseGuardrail,
  registerLlmConditionalExecutionGuardrail,
  deregisterLlmConditionalExecutionGuardrail,
  registerLlmRequestIntercept,
  deregisterLlmRequestIntercept,
  registerLlmExecutionIntercept,
  deregisterLlmExecutionIntercept,
  registerLlmStreamExecutionIntercept,
  deregisterLlmStreamExecutionIntercept,
  registerSubscriber,
  deregisterSubscriber,
  flushSubscribers,
  ScopeType,
} = lib;

const LLM_ATTR_STATELESS = 0b01;
const LLM_ATTR_STREAMING = 0b10;
const V2_API_FAMILIES = ['openai_responses', 'openai_chat_completions', 'anthropic_messages'];

function rejectWith(value) {
  return Promise.reject(value);
}

async function flushSubscriberCallbacks() {
  flushSubscribers();
  for (let i = 0; i < 10; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

function makeNative() {
  return {
    headers: {},
    content: {
      messages: [],
      model: 'test-model',
    },
  };
}

function assertRecursivelyFrozen(value) {
  if (value === null || typeof value !== 'object') return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

function assertReplayTestContext(context, apiFamily) {
  assert.equal(context.apiFamily, apiFamily);
  assert.equal(context.callRole, 'primary');
}

// ===========================================================================
// LLM lifecycle
// ===========================================================================

describe('LLM lifecycle', () => {
  it('llm call and end', () => {
    const native = makeNative();
    const handle = llmCall('test_llm', native, null, null, null, null, null);
    assert.equal(handle.name, 'test_llm');
    assert.ok(handle.uuid.length > 0);
    llmCallEnd(
      handle,
      {
        choices: [
          {
            text: 'hello',
          },
        ],
      },
      null,
      null,
    );
  });

  it('llm call with attributes', () => {
    const native = makeNative();
    const handle = llmCall('attr_llm', native, null, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING, null, null, null);
    assert.equal(handle.attributes, LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    llmCallEnd(handle, {}, null, null);
  });

  it('llm call with parent', () => {
    const scope = pushScope('llm_parent', ScopeType.Agent, null, null);
    const native = makeNative();
    const handle = llmCall('parented_llm', native, scope, null, null, null, null);
    assert.equal(handle.parentUuid, scope.uuid);
    llmCallEnd(handle, {}, null, null);
    popScope(scope);
  });

  it('llm call with data/metadata', () => {
    const native = makeNative();
    const handle = llmCall(
      'data_llm',
      native,
      null,
      null,
      {
        info: 'llm_test',
      },
      {
        version: '2.0',
      },
      null,
    );
    llmCallEnd(
      handle,
      {},
      {
        tokens: 100,
      },
      null,
    );
  });

  it('llm call generates events', async () => {
    const events = [];
    registerSubscriber('node_llm_evt_sub', (e) => events.push(e));
    try {
      const native = makeNative();
      const handle = llmCall('evt_llm', native, null, null, null, null, null);
      llmCallEnd(handle, {}, null, null);
      const deadline = Date.now() + 2000;
      while (events.length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }
      assert.ok(events.length >= 2, 'Expected at least 2 events');
    } finally {
      deregisterSubscriber('node_llm_evt_sub');
    }
  });
});

// ===========================================================================
// LLM execute
// ===========================================================================

describe('LLM execute', () => {
  it('basic execute', async () => {
    const native = makeNative();
    const result = await llmCallExecute(
      'exec_llm',
      native,
      (n) => ({
        response: 'hello from llm',
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.deepEqual(result, {
      response: 'hello from llm',
    });
  });

  it('treats implicit undefined llm results as null', async () => {
    const result = await llmCallExecute(
      'exec_llm_undefined',
      makeNative(),
      () => undefined,
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result, null);
  });

  it('execute records OTEL status metadata on end events', async () => {
    const events = [];
    registerSubscriber('node_llm_status_metadata_sub', (e) => events.push(e));
    try {
      const result = await llmCallExecute(
        'exec_status_ok_llm',
        makeNative(),
        () => ({
          response: 'ok',
        }),
        null,
        null,
        null,
        {
          caller: 'node-llm',
        },
        null,
      );
      assert.deepEqual(result, {
        response: 'ok',
      });

      await assert.rejects(
        () =>
          llmCallExecuteAsync(
            'exec_status_error_llm',
            makeNative(),
            async () => {
              throw new Error('llm status failure');
            },
            null,
            null,
            null,
            {
              caller: 'node-llm-error',
            },
            null,
          ),
        /llm status failure/,
      );

      await flushSubscriberCallbacks();
      const okEnd = events.find(
        (e) =>
          e.name === 'exec_status_ok_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      const errorEnd = events.find(
        (e) =>
          e.name === 'exec_status_error_llm' &&
          e.kind === 'scope' &&
          e.category === 'llm' &&
          e.scope_category === 'end',
      );
      assert.ok(okEnd, 'expected successful llm end event');
      assert.equal(okEnd.metadata.caller, 'node-llm');
      assert.equal(okEnd.metadata['otel.status_code'], 'OK');
      assert.ok(errorEnd, 'expected failed llm end event');
      assert.equal(errorEnd.metadata.caller, 'node-llm-error');
      assert.equal(errorEnd.metadata['otel.status_code'], 'ERROR');
      assert.match(errorEnd.metadata['otel.status_description'], /llm status failure/);
    } finally {
      deregisterSubscriber('node_llm_status_metadata_sub');
    }
  });

  it('async execute awaits Promise-returning callbacks', async () => {
    const result = await llmCallExecuteAsync(
      'exec_async_llm',
      makeNative(),
      async (request) => ({
        response: request.content.model,
      }),
      null,
      LLM_ATTR_STATELESS,
      {
        data: true,
      },
      {
        meta: true,
      },
      'async-model',
    );
    assert.deepEqual(result, {
      response: 'test-model',
    });
  });

  it('async execute surfaces plain string rejections', async () => {
    await assert.rejects(
      () =>
        llmCallExecuteAsync(
          'exec_async_llm_reject',
          makeNative(),
          async () => rejectWith('string llm error'),
          null,
          null,
          null,
          null,
          null,
        ),
      /string llm error/,
    );
  });
});

// ===========================================================================
// LLM execute V2 and replay factory
// ===========================================================================

describe('LLM execute V2', () => {
  it('requires explicit family, role, and sanitized metadata', async () => {
    let providerCalls = 0;
    const provider = async () => {
      providerCalls += 1;
      return { response: 'should not run' };
    };

    await assert.rejects(
      async () => llmCallExecuteV2('v2_invalid_family', makeNative(), provider, 'openai_unknown', 'primary', {}),
      /apiFamily/,
    );
    await assert.rejects(
      async () => llmCallExecuteV2('v2_invalid_role', makeNative(), provider, 'openai_responses', 'observer', {}),
      /callRole/,
    );
    await assert.rejects(
      async () => llmCallExecuteV2('v2_invalid_metadata', makeNative(), provider, 'openai_responses', 'primary', []),
      /sanitizedMetadata/,
    );
    await assert.rejects(
      async () =>
        llmCallExecuteV2(
          'v2_invalid_tenant',
          makeNative(),
          provider,
          'openai_responses',
          'primary',
          {},
          null,
          null,
          null,
          null,
          null,
          'Bearer tenant-secret',
        ),
      /tenant_id.*credential/,
    );
    assert.equal(providerCalls, 0);
  });

  it('passes an exact recursively frozen context to one synchronous factory for every family', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      let capturedContext;
      let factoryCalls = 0;
      let providerCalls = 0;
      let replayCalls = 0;
      const sanitizedMetadata = {
        region: 'us-east',
        routing: { family: apiFamily, fallbacks: ['none'] },
      };
      const response = await llmCallExecuteV2(
        `v2_context_${apiFamily}`,
        makeNative(),
        async () => {
          providerCalls += 1;
          return { response: 'anchor' };
        },
        apiFamily,
        'primary',
        sanitizedMetadata,
        null,
        null,
        null,
        null,
        'test-model',
        'Tenant-A',
        'Agent-A',
        (context) => {
          factoryCalls += 1;
          capturedContext = context;
          return {
            contractVersion: 1,
            apiFamily,
            transportIdentity: `node-test-transport-${apiFamily}`,
            replay(request) {
              replayCalls += 1;
              return { result: Promise.resolve(request), cancel() {} };
            },
          };
        },
      );

      assert.deepEqual(response, { response: 'anchor' });
      assertRecursivelyFrozen(capturedContext);
      assert.equal(capturedContext.apiFamily, apiFamily);
      assert.equal(capturedContext.callRole, 'primary');
      assert.equal(capturedContext.tenantId, 'Tenant-A');
      assert.equal(capturedContext.agentId, 'Agent-A');
      assert.deepEqual(capturedContext.sanitizedMetadata, sanitizedMetadata);
      assert.match(capturedContext.callUuid, /^[0-9a-f-]+$/);
      assert.equal(factoryCalls, 1);
      assert.equal(providerCalls, 1);
      assert.equal(replayCalls, 0);
    }
  });

  it('fails open once for every family when a factory throws or returns an invalid descriptor', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      const mismatchedFamily = V2_API_FAMILIES.find((family) => family !== apiFamily);
      const invalidFactories = [
        () => {
          throw new Error('factory secret must not escape');
        },
        () => ({
          contractVersion: 1,
          apiFamily,
          transportIdentity: 'node-test-transport',
        }),
        () => ({
          contractVersion: 99,
          apiFamily,
          transportIdentity: 'node-test-transport',
          replay() {},
        }),
        () => ({
          contractVersion: 1,
          apiFamily: mismatchedFamily,
          transportIdentity: 'node-test-transport',
          replay() {},
        }),
        () => ({
          contractVersion: 1,
          apiFamily,
          transportIdentity: 'Bearer must-not-be-a-transport-identity',
          replay() {},
        }),
        () => ({
          contractVersion: 1,
          apiFamily,
          transportIdentity: 'node-test-transport',
          endpoint: 'https://example.test',
          replay() {},
        }),
        async () => ({
          contractVersion: 1,
          apiFamily,
          transportIdentity: 'node-test-transport',
          replay() {},
        }),
      ];

      for (const invalidFactory of invalidFactories) {
        let capturedContext;
        let factoryCalls = 0;
        let providerCalls = 0;
        const response = await llmCallExecuteV2(
          `v2_factory_fail_open_${apiFamily}`,
          makeNative(),
          async () => {
            providerCalls += 1;
            return { response: 'anchor' };
          },
          apiFamily,
          'primary',
          {},
          null,
          null,
          null,
          null,
          null,
          null,
          null,
          (context) => {
            factoryCalls += 1;
            capturedContext = context;
            return invalidFactory();
          },
        );
        assert.deepEqual(response, { response: 'anchor' });
        assertRecursivelyFrozen(capturedContext);
        assert.equal(capturedContext.apiFamily, apiFamily);
        assert.equal(capturedContext.callRole, 'primary');
        assert.equal(factoryCalls, 1);
        assert.equal(providerCalls, 1);
      }
    }
  });

  it('rejects enumerable descriptor side-channel fields at the replay bridge', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      for (const extra of [
        { endpoint: 'https://example.test' },
        { authorization: 'Bearer must-not-cross-the-bridge' },
        { anchorResponse: { status: 200 } },
      ]) {
        await assert.rejects(
          () =>
            __testNodeReplayBridge(
              (context) => {
                assertReplayTestContext(context, apiFamily);
                return {
                  contractVersion: 1,
                  apiFamily,
                  transportIdentity: `node-side-channel-test-${apiFamily}`,
                  replay(request) {
                    return { result: Promise.resolve(request), cancel() {} };
                  },
                  ...extra,
                };
              },
              apiFamily,
              [{ headers: {}, content: {} }],
              false,
            ),
          /Node replay factory returned an invalid descriptor/,
        );
      }
    }
  });

  it('consumes rejected async factories under strict unhandled-rejection policy', () => {
    const script = `
      const { llmCallExecuteV2 } = require('./index.js');
      void (async () => {
        const response = await llmCallExecuteV2(
          'v2_rejected_async_factory',
          { headers: {}, content: {} },
          async () => ({ anchor: true }),
          'openai_responses',
          'primary',
          {},
          null,
          null,
          null,
          null,
          null,
          null,
          null,
          async () => { throw new Error('factory secret must not escape'); },
        );
        console.log(JSON.stringify(response));
        await new Promise((resolve) => setImmediate(resolve));
      })().catch((error) => {
        console.error(error);
        process.exitCode = 1;
      });
    `;
    const child = spawnSync(process.execPath, ['--unhandled-rejections=strict', '--eval', script], {
      cwd: nodePackageDir,
      encoding: 'utf8',
    });

    assert.equal(child.status, 0, child.stderr);
    assert.match(child.stdout, /\{"anchor":true\}/);
    assert.doesNotMatch(child.stderr, /factory secret must not escape/);
  });

  it('retains the original descriptor and private state until transport release', () => {
    const script = `
      const assert = require('node:assert/strict');
      const { __testNodeReplayBridge } = require('./index.js');
      void (async () => {
        const apiFamilies = ${JSON.stringify(V2_API_FAMILIES)};
        for (const apiFamily of apiFamilies) {
          let descriptorRef;
          let replayCalls = 0;
          const results = await __testNodeReplayBridge(
            (context) => {
              assert.equal(context.apiFamily, apiFamily);
              assert.equal(context.callRole, 'primary');
              const descriptor = {
                contractVersion: 1,
                apiFamily,
                transportIdentity: 'node-private-state-transport-' + apiFamily,
                replay(request) {
                  assert.equal(Object.isFrozen(this), true);
                  assert.equal(descriptorRef.deref(), this);
                  assert.equal(this.privateState.prefix, 'retained');
                  replayCalls += 1;
                  return {
                    result: Promise.resolve({ value: this.privateState.prefix + '-' + request.content.id }),
                    cancel() {},
                  };
                },
              };
              Object.defineProperty(descriptor, 'privateState', {
                value: { prefix: 'retained' },
                enumerable: false,
              });
              descriptorRef = new WeakRef(descriptor);
              return descriptor;
            },
            apiFamily,
            [
              { headers: {}, content: { id: 1 } },
              { headers: {}, content: { id: 2 } },
            ],
            false,
          );
          assert.deepEqual(results, [{ value: 'retained-1' }, { value: 'retained-2' }]);
          assert.equal(replayCalls, 2);

          let released = false;
          for (let attempt = 0; attempt < 100; attempt += 1) {
            await new Promise((resolve) => setImmediate(resolve));
            global.gc();
            await new Promise((resolve) => setImmediate(resolve));
            if (descriptorRef.deref() === undefined) {
              released = true;
              break;
            }
          }
          assert.equal(released, true, 'descriptor remained alive after replay transport release for ' + apiFamily);
        }
      })().catch((error) => {
        console.error(error);
        process.exitCode = 1;
      });
    `;
    const child = spawnSync(process.execPath, ['--expose-gc', '--eval', script], {
      cwd: nodePackageDir,
      encoding: 'utf8',
      timeout: 30_000,
    });

    assert.equal(child.status, 0, child.stderr || child.error?.message);
  });

  it('does not invoke a replay factory for stateful calls', async () => {
    let factoryCalls = 0;
    const response = await llmCallExecuteV2(
      'v2_stateful',
      makeNative(),
      async () => ({ response: 'anchor' }),
      'openai_responses',
      'primary',
      {},
      null,
      LLM_ATTR_STATELESS,
      null,
      null,
      null,
      null,
      null,
      () => {
        factoryCalls += 1;
        throw new Error('must not run');
      },
    );
    assert.deepEqual(response, { response: 'anchor' });
    assert.equal(factoryCalls, 0);
  });

  it('supports sequential and concurrent replay calls after the factory is released', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      const seen = [];
      let activeCalls = 0;
      let maxActiveCalls = 0;
      const replayFactory = (context) => {
        assertReplayTestContext(context, apiFamily);
        return {
          contractVersion: 1,
          apiFamily,
          transportIdentity: `node-repeated-transport-${apiFamily}`,
          replay(request) {
            assert.equal(Object.isFrozen(request), true);
            const id = request.content.id;
            seen.push(id);
            activeCalls += 1;
            maxActiveCalls = Math.max(maxActiveCalls, activeCalls);
            return {
              result: new Promise((resolve) =>
                setTimeout(
                  () => {
                    activeCalls -= 1;
                    resolve({ id, complete: true });
                  },
                  id === 2 ? 20 : 1,
                ),
              ),
              cancel() {},
            };
          },
        };
      };
      const results = await __testNodeReplayBridge(
        replayFactory,
        apiFamily,
        [
          { headers: {}, content: { id: 1 } },
          { headers: {}, content: { id: 2 } },
          { headers: {}, content: { id: 3 } },
        ],
        false,
      );
      assert.deepEqual(results, [
        { id: 1, complete: true },
        { id: 2, complete: true },
        { id: 3, complete: true },
      ]);
      assert.deepEqual(seen.sort(), [1, 2, 3]);
      assert.equal(maxActiveCalls, 2);
    }
  });

  it('cancels one dropped replay exactly once without affecting a sibling', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      let cancelled = 0;
      let completed = 0;
      let siblingCancelled = 0;
      const results = await __testNodeReplayBridge(
        (context) => {
          assertReplayTestContext(context, apiFamily);
          return {
            contractVersion: 1,
            apiFamily,
            transportIdentity: `node-cancel-transport-${apiFamily}`,
            replay(request) {
              if (request.content.pending) {
                return {
                  result: new Promise(() => {}),
                  cancel() {
                    cancelled += 1;
                  },
                };
              }
              return {
                result: Promise.resolve().then(() => {
                  completed += 1;
                  return { sibling: 'complete' };
                }),
                cancel() {
                  siblingCancelled += 1;
                },
              };
            },
          };
        },
        apiFamily,
        [
          { headers: {}, content: { pending: true } },
          { headers: {}, content: { pending: false } },
        ],
        true,
      );
      assert.deepEqual(results, [{ sibling: 'complete' }]);
      for (let i = 0; i < 20 && cancelled === 0; i += 1) {
        await new Promise((resolve) => setImmediate(resolve));
      }
      assert.equal(cancelled, 1);
      assert.equal(completed, 1);
      assert.equal(siblingCancelled, 0);
    }
  });

  it('maps replay rejection and malformed invocation to replay errors', async () => {
    for (const apiFamily of V2_API_FAMILIES) {
      await assert.rejects(
        () =>
          __testNodeReplayBridge(
            (context) => {
              assertReplayTestContext(context, apiFamily);
              return {
                contractVersion: 1,
                apiFamily,
                transportIdentity: `node-reject-transport-${apiFamily}`,
                replay() {
                  return { result: Promise.reject(new Error('replay failed')), cancel() {} };
                },
              };
            },
            apiFamily,
            [{ headers: {}, content: {} }],
            false,
          ),
        /replay failed/,
      );
      await assert.rejects(
        () =>
          __testNodeReplayBridge(
            (context) => {
              assertReplayTestContext(context, apiFamily);
              return {
                contractVersion: 1,
                apiFamily,
                transportIdentity: `node-invalid-invocation-${apiFamily}`,
                replay() {
                  return Promise.resolve({ invalid: true });
                },
              };
            },
            apiFamily,
            [{ headers: {}, content: {} }],
            false,
          ),
        /invalid invocation/,
      );

      for (const invalidResult of [undefined, 1n, Number.NaN, () => {}]) {
        await assert.rejects(
          () =>
            __testNodeReplayBridge(
              (context) => {
                assertReplayTestContext(context, apiFamily);
                return {
                  contractVersion: 1,
                  apiFamily,
                  transportIdentity: `node-invalid-result-${apiFamily}`,
                  replay() {
                    return { result: Promise.resolve(invalidResult), cancel() {} };
                  },
                };
              },
              apiFamily,
              [{ headers: {}, content: {} }],
              false,
            ),
          /Node replay result must be valid JSON/,
        );
      }
    }
  });
});

// ===========================================================================
// LLM guardrails
// ===========================================================================

describe('LLM guardrails', () => {
  it('sanitize request guardrail', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req', 10, (request) => {
      request.extra = 'sanitized';
      return request;
    });
    deregisterLlmSanitizeRequestGuardrail('node_llm_san_req');
  });

  it('sanitize request guardrail rewrites start event payload', async () => {
    const events = [];
    registerSubscriber('node_llm_san_req_evt', (e) => events.push(e));
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_evt_guard', 10, (request) => {
      request.headers = {
        ...request.headers,
        'X-Sanitized': 'yes',
      };
      return request;
    });

    try {
      const result = await llmCallExecute(
        'san_req_evt_llm',
        makeNative(),
        (request) => ({
          model: request.content.model,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        model: 'test-model',
      });
      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (e) =>
            e.name === 'san_req_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const start = events.find(
        (e) =>
          e.name === 'san_req_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
      );
      assert.deepEqual(start.data, {
        headers: {
          'X-Sanitized': 'yes',
        },
        content: {
          messages: [],
          model: 'test-model',
        },
      });
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_evt_guard');
      deregisterSubscriber('node_llm_san_req_evt');
    }
  });

  it('sanitize request guardrail falls back on malformed return', async () => {
    registerLlmSanitizeRequestGuardrail('node_llm_san_req_bad', 10, () => null);
    try {
      const result = await llmCallExecute(
        'san_req_bad_llm',
        makeNative(),
        (request) => ({
          model: request.content.model,
          headers: request.headers,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        model: 'test-model',
        headers: {},
      });
    } finally {
      deregisterLlmSanitizeRequestGuardrail('node_llm_san_req_bad');
    }
  });

  it('conditional guardrail rejects non-string return values', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_non_string', 10, () => ({
      blocked: true,
    }));
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'llm_cond_non_string',
            makeNative(),
            () => ({
              ok: true,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /expected string or null/i,
      );
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_non_string');
    }
  });

  it('sanitize response guardrail', () => {
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp', 10, (response) => {
      response.sanitized = true;
      return response;
    });
    deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp');
  });

  it('sanitize response guardrail rewrites end event payload', async () => {
    const events = [];
    registerSubscriber('node_llm_san_resp_evt', (e) => events.push(e));
    registerLlmSanitizeResponseGuardrail('node_llm_san_resp_evt_guard', 10, (response) => {
      response.sanitized = true;
      return response;
    });

    try {
      const result = await llmCallExecute(
        'san_resp_evt_llm',
        makeNative(),
        () => ({
          ok: true,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        ok: true,
      });
      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (e) =>
            e.name === 'san_resp_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((r) => setTimeout(r, 10));
      }
      const end = events.find(
        (e) =>
          e.name === 'san_resp_evt_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      assert.deepEqual(end.data, {
        ok: true,
        sanitized: true,
      });
    } finally {
      deregisterLlmSanitizeResponseGuardrail('node_llm_san_resp_evt_guard');
      deregisterSubscriber('node_llm_san_resp_evt');
    }
  });

  it('conditional guardrail (allow)', () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond', 10, (request) => null);
    deregisterLlmConditionalExecutionGuardrail('node_llm_cond');
  });

  it('conditional guardrail treats implicit undefined as allow', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_undefined', 10, () => undefined);
    try {
      const result = await llmCallExecute(
        'llm_cond_undefined',
        makeNative(),
        () => ({
          ok: true,
        }),
        null,
        null,
        null,
        null,
        null,
      );
      assert.deepEqual(result, {
        ok: true,
      });
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_undefined');
    }
  });

  it('conditional guardrail (block)', () => {
    registerLlmConditionalExecutionGuardrail('node_llm_block', 10, (request) => 'blocked');
    deregisterLlmConditionalExecutionGuardrail('node_llm_block');
  });

  it('duplicate guardrail fails', () => {
    registerLlmSanitizeRequestGuardrail('node_llm_dup_guard', 10, (r) => r);
    assert.throws(() => registerLlmSanitizeRequestGuardrail('node_llm_dup_guard', 20, (r) => r));
    deregisterLlmSanitizeRequestGuardrail('node_llm_dup_guard');
  });
});

// ===========================================================================
// LLM intercepts
// ===========================================================================

describe('LLM intercepts', () => {
  it('request intercept', () => {
    registerLlmRequestIntercept('node_llm_req_int', 10, false, ({ name, request, annotated }) => {
      request.intercepted = true;
      return {
        request,
        annotated,
      };
    });
    deregisterLlmRequestIntercept('node_llm_req_int');
  });

  it('execution intercept', () => {
    registerLlmExecutionIntercept('node_llm_exec_int', 10, async (native, next) => next(native));
    deregisterLlmExecutionIntercept('node_llm_exec_int');
  });

  it('stream execution intercept', () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec', 10, async (native, next) => next(native));
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec');
  });

  it('request intercept with break_chain', () => {
    registerLlmRequestIntercept('node_llm_break', 10, true, ({ name, request, annotated }) => ({
      request,
      annotated,
    }));
    deregisterLlmRequestIntercept('node_llm_break');
  });

  it('duplicate intercept fails', () => {
    registerLlmRequestIntercept('node_llm_dup_int', 10, false, ({ request, annotated }) => ({
      request,
      annotated,
    }));
    assert.throws(() =>
      registerLlmRequestIntercept('node_llm_dup_int', 20, false, ({ request, annotated }) => ({
        request,
        annotated,
      })),
    );
    deregisterLlmRequestIntercept('node_llm_dup_int');
  });

  it('request intercept modifies request', async () => {
    registerLlmRequestIntercept('node_llm_req_mod', 10, false, ({ request, annotated }) => {
      request.content.intercepted = true;
      return {
        request,
        annotated,
      };
    });
    const native = makeNative();
    const result = await llmCallExecute(
      'mod_llm',
      native,
      (n) => ({
        saw_intercepted: n.content.intercepted || false,
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.saw_intercepted, true);
    deregisterLlmRequestIntercept('node_llm_req_mod');
  });

  it('request intercept rejects malformed return values', async () => {
    registerLlmRequestIntercept('node_llm_req_bad', 10, false, () => null);
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'bad_req_llm',
            makeNative(),
            (n) => ({
              model: n.content.model,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /invalid JS LLM request intercept outcome/i,
      );
    } finally {
      deregisterLlmRequestIntercept('node_llm_req_bad');
    }
  });

  it('execution intercept composes with next', async () => {
    registerLlmExecutionIntercept('node_llm_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const result = await next(native);
      return {
        ...result,
        wrapped: true,
      };
    });
    const native = makeNative();
    const result = await llmCallExecute(
      'repl_llm',
      native,
      (n) => ({
        original: !n.content.intercepted,
      }),
      null,
      null,
      null,
      null,
      null,
    );
    assert.equal(result.original, false);
    assert.equal(result.wrapped, true);
    deregisterLlmExecutionIntercept('node_llm_exec_repl');
  });

  it('execution intercept rejects invalid next request payloads', async () => {
    registerLlmExecutionIntercept('node_llm_exec_invalid_next', 10, async (_native, next) => {
      return next({
        headers: 1,
        content: {
          model: 'broken',
        },
      });
    });
    await assert.rejects(
      () =>
        llmCallExecute(
          'invalid_next_llm',
          makeNative(),
          () => ({
            ok: true,
          }),
          null,
          null,
          null,
          null,
          null,
        ),
      /invalid LlmRequest from JS next/i,
    );
    deregisterLlmExecutionIntercept('node_llm_exec_invalid_next');
  });

  it('execution intercept propagates primitive rejection values as unknown error', async () => {
    registerLlmExecutionIntercept('node_llm_exec_unknown_err', 10, async () => {
      return rejectWith(42);
    });
    try {
      await assert.rejects(
        () =>
          llmCallExecute(
            'unknown_err_llm',
            makeNative(),
            () => ({
              ok: true,
            }),
            null,
            null,
            null,
            null,
            null,
          ),
        /unknown error/i,
      );
    } finally {
      deregisterLlmExecutionIntercept('node_llm_exec_unknown_err');
    }
  });

  it('async execute falls back to unknown error for primitive rejections', async () => {
    await assert.rejects(
      () =>
        llmCallExecuteAsync(
          'primitive_reject_llm',
          makeNative(),
          async () => rejectWith(42),
          null,
          null,
          null,
          null,
          null,
        ),
      /unknown error/i,
    );
  });

  it('stream execution intercept composes with next', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_exec_repl', 10, async (native, next) => {
      native.content.intercepted = true;
      const chunks = await next(native);
      return [
        ...chunks,
        {
          wrapped: native.content.intercepted,
        },
      ];
    });

    const native = makeNative();
    const seen = [];
    const stream = await llmStreamCallExecute(
      'stream_llm',
      native,
      (wrapper) => {
        lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
          chunk: wrapper.__nemo_relay_native.content.intercepted,
        });
        lib.endStream(wrapper.__nemo_relay_stream_id);
      },
      null,
      null,
      null,
      null,
      null,
      null,
      null,
    );

    for (;;) {
      const chunk = await stream.next();
      if (chunk === null) {
        break;
      }
      seen.push(chunk);
    }

    assert.deepEqual(seen, [
      {
        chunk: true,
      },
      {
        wrapped: true,
      },
    ]);
    deregisterLlmStreamExecutionIntercept('node_llm_stream_exec_repl');
  });

  it('stream execution intercept can return a single scalar chunk', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_scalar', 10, async () => ({
      scalar: true,
    }));

    const seen = [];
    const stream = await llmStreamCallExecute(
      'stream_scalar_llm',
      makeNative(),
      () => {
        throw new Error('downstream stream should not be called');
      },
      null,
      null,
      null,
      null,
      null,
      null,
      null,
    );

    for (;;) {
      const chunk = await stream.next();
      if (chunk === null) {
        break;
      }
      seen.push(chunk);
    }

    assert.deepEqual(seen, [
      {
        scalar: true,
      },
    ]);
    deregisterLlmStreamExecutionIntercept('node_llm_stream_scalar');
  });

  it('stream execution intercept rejects invalid next request payloads', async () => {
    registerLlmStreamExecutionIntercept('node_llm_stream_invalid_next', 10, async (_native, next) => {
      return next({
        headers: 1,
        content: {
          model: 'broken',
        },
      });
    });

    await assert.rejects(
      () =>
        llmStreamCallExecute(
          'stream_invalid_next_llm',
          makeNative(),
          (wrapper) => {
            lib.pushStreamChunk(wrapper.__nemo_relay_stream_id, {
              chunk: true,
            });
            lib.endStream(wrapper.__nemo_relay_stream_id);
          },
          null,
          null,
          null,
          null,
          null,
          null,
          null,
        ),
      /invalid LlmRequest from JS next/i,
    );

    deregisterLlmStreamExecutionIntercept('node_llm_stream_invalid_next');
  });

  it('standalone request intercepts helper applies intercept chain', async () => {
    const contributionFixture = JSON.parse(
      readFileSync(
        new URL('../../types/tests/fixtures/llm_optimization_contribution_v1.json', import.meta.url),
        'utf8',
      ),
    );
    registerLlmRequestIntercept('node_llm_req_helper', 10, false, ({ request, annotated }) => {
      request.content.helper = true;
      return {
        request,
        annotated,
        pendingMarks: [
          {
            name: 'request.first',
            categoryProfile: { subtype: 'optimizer.saved_tokens' },
            data: { order: 1 },
          },
          { name: 'request.second', metadata: { source: 'node' } },
        ],
        optimizationContributions: [contributionFixture],
      };
    });

    const result = await llmRequestIntercepts('helper_llm', makeNative());
    assert.equal(result.request.content.helper, true);
    assert.equal(result.annotated, null);
    assert.deepEqual(result.pendingMarks, [
      {
        name: 'request.first',
        category: null,
        categoryProfile: { subtype: 'optimizer.saved_tokens' },
        data: { order: 1 },
        metadata: null,
      },
      {
        name: 'request.second',
        category: null,
        categoryProfile: null,
        data: null,
        metadata: { source: 'node' },
      },
    ]);
    assert.deepEqual(result.optimizationContributions, [contributionFixture]);
    deregisterLlmRequestIntercept('node_llm_req_helper');
  });

  it('generated request-intercept declarations preserve the open optimization kind', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');
    const openKind = "kind: 'input_compression' | 'model_routing' | (string & {})";

    assert.equal(declarations.split(openKind).length - 1, 3);
  });

  it('standalone conditional execution helper throws on rejection', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_helper', 10, () => 'llm blocked by helper');
    try {
      await assert.rejects(() => llmConditionalExecution(makeNative()), /guardrail rejected/i);
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_helper');
    }
  });

  it('standalone conditional execution helper resolves when allowed', async () => {
    registerLlmConditionalExecutionGuardrail('node_llm_cond_allow', 10, () => null);
    try {
      await assert.doesNotReject(() => llmConditionalExecution(makeNative()));
    } finally {
      deregisterLlmConditionalExecutionGuardrail('node_llm_cond_allow');
    }
  });
});

describe('LLM event fields', () => {
  it('subscriber receives modelName and payload fields', async () => {
    const events = [];
    const scope = pushScope('llm_event_parent', ScopeType.Agent, null, null);
    registerSubscriber('node_llm_field_sub', (e) => events.push(e));
    try {
      const handle = llmCall(
        'field_llm',
        makeNative(),
        scope,
        LLM_ATTR_STATELESS,
        {
          start: true,
        },
        {
          meta: true,
        },
        'gpt-field-model',
      );
      assert.equal(handle.attributes, LLM_ATTR_STATELESS);
      assert.equal(handle.parentUuid, scope.uuid);
      llmCallEnd(
        handle,
        {
          ok: true,
        },
        {
          end: true,
        },
        {
          final: true,
        },
      );

      const deadline = Date.now() + 2000;
      while (events.filter((e) => e.name === 'field_llm').length < 2 && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const start = events.find(
        (e) => e.name === 'field_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'start',
      );
      const end = events.find(
        (e) => e.name === 'field_llm' && e.kind === 'scope' && e.category === 'llm' && e.scope_category === 'end',
      );
      assert.equal(start.category_profile.model_name, 'gpt-field-model');
      assert.equal(start.category_profile.call_role, 'primary');
      assert.deepEqual(start.data, {
        headers: {},
        content: {
          messages: [],
          model: 'test-model',
        },
      });
      assert.deepEqual(end.data, {
        ok: true,
      });
    } finally {
      deregisterSubscriber('node_llm_field_sub');
      popScope(scope);
    }
  });
});
