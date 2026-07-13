// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DeepReadonlyJson,
  LlmApiFamily,
  LlmCallRole,
  LlmExecutionContext,
  LlmReplayCallable,
  LlmReplayDescriptor,
  LlmReplayFactory,
  LlmReplayInvocation,
  LlmTrajectoryScope,
} from '../../index.js';

const family: LlmApiFamily = 'openai_responses';
const role: LlmCallRole = 'primary';
const trajectoryScope: LlmTrajectoryScope = {
  uuid: '00000000-0000-0000-0000-000000000001',
  name: 'root',
  scopeType: 'agent',
};
const context: LlmExecutionContext = {
  callUuid: '00000000-0000-0000-0000-000000000002',
  rootUuid: trajectoryScope.uuid,
  parentUuid: trajectoryScope.uuid,
  trajectoryOwnerUuid: trajectoryScope.uuid,
  trajectoryOwnerPath: [trajectoryScope],
  apiFamily: family,
  callRole: role,
  attributes: 0,
  tenantId: null,
  agentId: null,
  sanitizedMetadata: { routing: { region: 'us-east' } },
};
const invocation: LlmReplayInvocation = {
  result: Promise.resolve({ ok: true }),
  cancel() {},
};
const replay: LlmReplayCallable = (_request) => invocation;
const privateState = Symbol('replay-private-state');
const descriptor: LlmReplayDescriptor = {
  contractVersion: 1,
  apiFamily: family,
  transportIdentity: 'typed-transport',
  replay,
  [privateState]: { region: 'us-east' },
};
const factory: LlmReplayFactory = (_context) => descriptor;
const readonlyJson: { readonly [key: string]: DeepReadonlyJson } = {
  nested: { value: 1 },
};

// @ts-expect-error frozen context fields cannot be reassigned
context.callRole = 'judge';
// @ts-expect-error frozen trajectory paths cannot be extended
context.trajectoryOwnerPath.push(trajectoryScope);
// @ts-expect-error frozen metadata entries cannot be replaced
context.sanitizedMetadata.routing = null;
// @ts-expect-error deep-readonly JSON object entries cannot be replaced
readonlyJson.nested = { value: 2 };

void factory;
