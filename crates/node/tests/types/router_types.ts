// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ComponentSpec,
  candidateConfig,
  concurrencyConfig,
  defaultConfig,
  judgeConfig,
  poolConfig,
  type LlmReplayFactory,
} from 'nemo-relay-node/router';

const judge = judgeConfig({
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
});

const config = defaultConfig({
  mode: 'shadow',
  pools: [
    poolConfig({
      id: 'typed-pool',
      api_family: 'openai_chat_completions',
      anchor_models: ['anchor-model'],
      anchor_revision: '2026-07-11',
      sampling_probability: 1,
      max_candidates_per_sample: 1,
      concurrency: concurrencyConfig({ shadow: 1, judge: 1 }),
      candidates: [
        candidateConfig({
          id: 'candidate',
          model: 'candidate-model',
          model_revision: '2026-07-11',
          cost_rank: 0,
        }),
      ],
      judge,
    }),
  ],
});

const component = ComponentSpec(config);
const replayFactory: LlmReplayFactory | undefined = undefined;

void component;
void replayFactory;
