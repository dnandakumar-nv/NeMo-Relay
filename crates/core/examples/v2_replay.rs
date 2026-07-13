// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run a V2 anchor call and delayed replay through an in-memory transport.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nemo_relay::api::llm::{
    LlmApiFamily, LlmCallExecuteV2Params, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
    llm_call_execute_v2,
};
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, register_llm_execution_intercept_v2,
};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayFactory,
    LlmReplayTransport, create_scope_stack, set_thread_scope_stack,
};
use nemo_relay::error::Result;
use nemo_relay::json::Json;
use serde_json::{Map, json};
use tokio::sync::oneshot;

struct InMemoryFactory;

impl LlmReplayFactory for InMemoryFactory {
    fn build(&self, context: &LlmExecutionContextSnapshot) -> Result<Arc<dyn LlmReplayTransport>> {
        Ok(Arc::new(InMemoryTransport {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: context.api_family,
                transport_identity: "in-memory-example".to_string(),
            },
        }))
    }
}

struct InMemoryTransport {
    capability: LlmReplayCapability,
}

impl LlmReplayTransport for InMemoryTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> Result<LlmReplayCall> {
        Ok(LlmReplayCall::new(
            async move { Ok(json!({"replayed": request.content})) },
            || {},
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    set_thread_scope_stack(create_scope_stack());
    let (replay_tx, replay_rx) = oneshot::channel::<Result<Json>>();
    let replay_tx = Arc::new(Mutex::new(Some(replay_tx)));
    let replay_result = Arc::clone(&replay_tx);

    register_llm_execution_intercept_v2(
        "in-memory-example",
        0,
        Arc::new(move |_, _, request, replay, next| {
            let replay_result = Arc::clone(&replay_result);
            Box::pin(async move {
                let anchor = next(request.clone()).await?;
                let transport = replay.expect("the in-memory factory is replay-eligible");
                let sender = replay_result
                    .lock()
                    .expect("replay sender lock poisoned")
                    .take()
                    .expect("example intercept runs once");
                tokio::spawn(async move {
                    let result = match transport.start(request) {
                        Ok(call) => call.await,
                        Err(error) => Err(error),
                    };
                    let _ = sender.send(result);
                });
                Ok(anchor)
            })
        }),
    )?;

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "example-model", "input": "hello"}),
    };
    let anchor = llm_call_execute_v2(
        LlmCallExecuteV2Params::builder()
            .name("example-provider")
            .request(request)
            .func(Arc::new(|request| {
                Box::pin(async move { Ok(json!({"anchor": request.content})) })
            }))
            .api_family(LlmApiFamily::OpenAIResponses)
            .call_role(LlmCallRole::Primary)
            .sanitized_metadata(BTreeMap::from([("region".to_string(), json!("local"))]))
            .replay_factory(Arc::new(InMemoryFactory) as Arc<dyn LlmReplayFactory>)
            .build(),
    )
    .await?;
    let replay = replay_rx.await.expect("replay task must report a result")?;

    deregister_llm_execution_intercept("in-memory-example")?;
    println!("anchor: {anchor}");
    println!("replay: {replay}");
    Ok(())
}
