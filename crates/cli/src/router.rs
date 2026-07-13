// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded CLI adapter for the typed Router inspection service.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use nemo_relay::plugin::PluginConfig;
use nemo_relay_router::inspection::{
    CohortRotationRequestV1, DecisionFilterV1, DecisionSummaryV1, EvidenceExportFormatV1,
    EvidenceExportRequestV1, EvidenceFilterV1, EvidenceSummaryV1, INSPECTION_INPUT_MAX_BYTES,
    INSPECTION_PAGE_LIMIT_MAX, InspectionControlRequestV1, InspectionError, InspectionService,
    InspectionServiceOptions, LearningResetRequestV1, LearningResetScopeV1, NeighborhoodLookupV1,
    OperatorMutationReceiptV1, Page, PageRequest, RoutingInspectionInputV1, StatusReportV1,
};
use nemo_relay_router::{
    CONTROL_ACTOR_MAX_BYTES, CONTROL_REASON_MAX_BYTES, ControlOperation, ControlScope,
    ROUTER_PLUGIN_KIND, RouterConfig, RouterControlSnapshot, RoutingPartitionV1,
};
use serde::Serialize;
use serde_json::{Value, json};
use tempfile::NamedTempFile;
use uuid::Uuid;

use crate::config::{
    self, RouterCohortRotateCommand, RouterCohortSubcommand, RouterCommand, RouterControlCommand,
    RouterDecisionFilterArgs, RouterDecisionsSubcommand, RouterDecisionsTailCommand,
    RouterEvidenceExportCommand, RouterEvidenceExportFormat, RouterEvidenceFilterArgs,
    RouterEvidenceSubcommand, RouterForceAnchorSubcommand, RouterNeighborhoodInspectCommand,
    RouterNeighborhoodSubcommand, RouterPageArgs, RouterResetCommand, RouterSubcommand, ServerArgs,
};
use crate::error::CliError;

const ROUTER_CLI_SCHEMA_VERSION: u32 = 1;
const DECISION_FOLLOW_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) async fn run(command: RouterCommand, server: &ServerArgs) -> Result<ExitCode, CliError> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    run_with_io(command, server, &mut stdout, &mut stderr).await
}

async fn run_with_io(
    command: RouterCommand,
    server: &ServerArgs,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<ExitCode, CliError> {
    if let RouterSubcommand::Dashboard(command) = &command.command {
        return crate::router_dashboard::run_with_io(command.clone(), server, stdout, stderr).await;
    }
    let command_name = command_name(&command);
    let json = command_uses_json(&command);
    match execute(command, server, stdout).await {
        Ok(()) => Ok(ExitCode::SUCCESS),
        Err(RouterFailure::OutputClosed) => Ok(ExitCode::SUCCESS),
        Err(failure) => {
            let exit_code = failure.exit_code();
            let rendered = if json {
                write_failure_json(stdout, command_name, &failure)
            } else {
                write_text(stderr, format_args!("{failure}\n"))
            };
            match rendered {
                Ok(()) => Ok(exit_code),
                Err(RouterFailure::OutputClosed) => Ok(ExitCode::SUCCESS),
                Err(other) => Err(other.into_cli_error()),
            }
        }
    }
}

async fn execute(
    command: RouterCommand,
    server: &ServerArgs,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    let mutation = prepare_mutation_context(&command.command)?;
    let config = resolve_router_config(server)?;
    let service = InspectionService::open(
        config,
        InspectionServiceOptions {
            allow_operations: mutation.is_some(),
            ..InspectionServiceOptions::default()
        },
    )
    .await?;
    let mut staged_output = Vec::new();
    let command_output: &mut dyn Write = if mutation.is_some() {
        &mut staged_output
    } else {
        output
    };
    let result =
        execute_with_service(&service, command.command, mutation.as_ref(), command_output).await;
    let close = service.close().await;
    match (result, close) {
        (Err(error), _) => Err(error),
        (Ok(()), _) if mutation.is_some() => write_bytes(output, &staged_output),
        (Ok(()), Err(error)) => Err(error.into()),
        (Ok(()), Ok(())) => Ok(()),
    }
}

pub(crate) fn resolve_router_config(server: &ServerArgs) -> Result<RouterConfig, RouterFailure> {
    let resolved = config::resolve_router_command_config(server).map_err(RouterFailure::Cli)?;
    let value = resolved.gateway.plugin_config.ok_or_else(|| {
        RouterFailure::input(
            "router_not_configured",
            "no merged plugin configuration contains a Router component",
        )
    })?;
    let plugin: PluginConfig = serde_json::from_value(value).map_err(|error| {
        RouterFailure::input(
            "invalid_plugin_config",
            format!("merged plugin configuration is invalid: {error}"),
        )
    })?;
    let routers = plugin
        .components
        .into_iter()
        .filter(|component| component.kind == ROUTER_PLUGIN_KIND)
        .collect::<Vec<_>>();
    let [component] = routers.as_slice() else {
        return Err(if routers.is_empty() {
            RouterFailure::input(
                "router_not_configured",
                "merged plugin configuration does not contain a Router component",
            )
        } else {
            RouterFailure::input(
                "multiple_router_components",
                "merged plugin configuration must contain exactly one Router component",
            )
        });
    };
    if !component.enabled {
        return Err(RouterFailure::input(
            "router_disabled",
            "the configured Router component is disabled",
        ));
    }
    serde_json::from_value(Value::Object(component.config.clone())).map_err(|error| {
        RouterFailure::input(
            "invalid_router_config",
            format!("Router component configuration is invalid: {error}"),
        )
    })
}

#[derive(Debug)]
struct MutationContext {
    actor: String,
}

fn prepare_mutation_context(
    command: &RouterSubcommand,
) -> Result<Option<MutationContext>, RouterFailure> {
    let (actor, reason, pool, confirmation) = match command {
        RouterSubcommand::Pause(command) | RouterSubcommand::Resume(command) => (
            command.actor.as_deref(),
            command.reason.as_str(),
            command.pool.as_deref(),
            None,
        ),
        RouterSubcommand::ForceAnchor(command) => {
            let command = match &command.command {
                RouterForceAnchorSubcommand::Set(command)
                | RouterForceAnchorSubcommand::Clear(command) => command,
            };
            (
                command.actor.as_deref(),
                command.reason.as_str(),
                command.pool.as_deref(),
                None,
            )
        }
        RouterSubcommand::Reset(command) => (
            command.actor.as_deref(),
            command.reason.as_str(),
            command.pool.as_deref(),
            Some(command.confirm.as_str()),
        ),
        RouterSubcommand::Cohort(command) => {
            let RouterCohortSubcommand::Rotate(command) = &command.command;
            (
                command.actor.as_deref(),
                command.reason.as_str(),
                None,
                Some(command.confirm.as_str()),
            )
        }
        RouterSubcommand::Dashboard(_)
        | RouterSubcommand::Status(_)
        | RouterSubcommand::Pools(_)
        | RouterSubcommand::Evidence(_)
        | RouterSubcommand::Neighborhood(_)
        | RouterSubcommand::Decisions(_) => return Ok(None),
    };
    if !valid_operator_text(reason, CONTROL_REASON_MAX_BYTES) {
        return Err(RouterFailure::input(
            "invalid_reason",
            format!(
                "mutation reason must be nonblank, control-free, and at most {CONTROL_REASON_MAX_BYTES} bytes"
            ),
        ));
    }
    if pool.is_some_and(|pool| !valid_cli_id(pool, 128)) {
        return Err(RouterFailure::input(
            "invalid_pool",
            "pool must be a nonblank control-free identifier of at most 128 bytes",
        ));
    }
    if confirmation.is_some_and(|confirmation| !valid_cli_id(confirmation, 128)) {
        return Err(RouterFailure::input(
            "invalid_confirmation",
            "project confirmation must be a nonblank control-free identifier of at most 128 bytes",
        ));
    }
    let actor = match actor {
        Some(actor) => actor.to_owned(),
        None => crate::principal::authenticated_principal().map_err(|_| {
            RouterFailure::input(
                "actor_unavailable",
                "authenticated OS principal is unavailable; pass an explicit --actor",
            )
        })?,
    };
    if !valid_operator_text(&actor, CONTROL_ACTOR_MAX_BYTES) {
        return Err(RouterFailure::input(
            "invalid_actor",
            format!(
                "mutation actor must be nonblank, control-free, and at most {CONTROL_ACTOR_MAX_BYTES} bytes"
            ),
        ));
    }
    Ok(Some(MutationContext { actor }))
}

fn valid_operator_text(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= maximum
        && value.chars().all(|character| !character.is_control())
}

fn valid_cli_id(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.chars().all(|character| !character.is_control())
}

fn mutation_actor(mutation: Option<&MutationContext>) -> Result<&str, RouterFailure> {
    mutation
        .map(|mutation| mutation.actor.as_str())
        .ok_or(InspectionError::IntegrityError.into())
}

async fn execute_with_service(
    service: &InspectionService,
    command: RouterSubcommand,
    mutation: Option<&MutationContext>,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    match command {
        RouterSubcommand::Dashboard(_) => Err(RouterFailure::input(
            "invalid_dashboard_dispatch",
            "dashboard command bypassed its dedicated host",
        )),
        RouterSubcommand::Status(command) => {
            let report = service.status().await?;
            if command.json {
                write_success_json(output, "router status", &report)
            } else {
                write_status_text(output, &report)
            }
        }
        RouterSubcommand::Pools(command) => {
            let page = service.list_pools(page_request(&command.page)).await?;
            if command.json {
                write_success_json(output, "router pools", &page)
            } else {
                write_pools_text(output, &page)
            }
        }
        RouterSubcommand::Evidence(command) => match command.command {
            RouterEvidenceSubcommand::List(command) => {
                let page = service
                    .list_evidence(
                        evidence_filter(&command.filter),
                        page_request(&command.page),
                    )
                    .await?;
                if command.json {
                    write_success_json(output, "router evidence list", &page)
                } else {
                    write_evidence_text(output, &page)
                }
            }
            RouterEvidenceSubcommand::Show(command) => {
                let detail = service.get_evidence(command.evidence_id).await?;
                if command.json {
                    write_success_json(output, "router evidence show", &detail)
                } else {
                    write_evidence_detail_text(output, &detail)
                }
            }
            RouterEvidenceSubcommand::Export(command) => {
                export_evidence(service, command, output).await
            }
        },
        RouterSubcommand::Neighborhood(command) => match command.command {
            RouterNeighborhoodSubcommand::Inspect(command) => {
                let json = command.json;
                let report = service
                    .inspect_neighborhood(neighborhood_lookup(command)?)
                    .await?;
                if json {
                    write_success_json(output, "router neighborhood inspect", &report)
                } else {
                    write_neighborhood_text(output, &report)
                }
            }
        },
        RouterSubcommand::Decisions(command) => match command.command {
            RouterDecisionsSubcommand::Tail(command) => {
                tail_decisions(service, command, output).await
            }
        },
        RouterSubcommand::Pause(command) => {
            apply_control_command(
                service,
                command,
                mutation_actor(mutation)?,
                ControlOperation::SetPaused { value: true },
                "router pause",
                output,
            )
            .await
        }
        RouterSubcommand::Resume(command) => {
            apply_control_command(
                service,
                command,
                mutation_actor(mutation)?,
                ControlOperation::SetPaused { value: false },
                "router resume",
                output,
            )
            .await
        }
        RouterSubcommand::ForceAnchor(command) => match command.command {
            RouterForceAnchorSubcommand::Set(command) => {
                apply_control_command(
                    service,
                    command,
                    mutation_actor(mutation)?,
                    ControlOperation::SetForceAnchor { value: true },
                    "router force-anchor set",
                    output,
                )
                .await
            }
            RouterForceAnchorSubcommand::Clear(command) => {
                apply_control_command(
                    service,
                    command,
                    mutation_actor(mutation)?,
                    ControlOperation::SetForceAnchor { value: false },
                    "router force-anchor clear",
                    output,
                )
                .await
            }
        },
        RouterSubcommand::Reset(command) => {
            reset_learning(service, command, mutation_actor(mutation)?, output).await
        }
        RouterSubcommand::Cohort(command) => match command.command {
            RouterCohortSubcommand::Rotate(command) => {
                rotate_cohort(service, command, mutation_actor(mutation)?, output).await
            }
        },
    }
}

fn evidence_filter(args: &RouterEvidenceFilterArgs) -> EvidenceFilterV1 {
    EvidenceFilterV1 {
        pool_id: args.pool.clone(),
        candidate_id: args.candidate.clone(),
        terminal_class: args.terminal_class.clone(),
        quality_label: args.quality_label.clone(),
        learning_generation_id: args.learning_generation_id,
    }
}

fn decision_filter(args: &RouterDecisionFilterArgs) -> DecisionFilterV1 {
    DecisionFilterV1 {
        pool_id: args.pool.clone(),
        mode: args.mode.clone(),
        candidate_id: args.candidate.clone(),
        final_reason: args.final_reason.clone(),
    }
}

fn page_request(args: &RouterPageArgs) -> PageRequest {
    PageRequest {
        limit: args.limit,
        after: args.cursor.clone(),
    }
}

#[derive(Serialize)]
struct ControlCommandResult {
    mutation_id: Uuid,
    snapshot: RouterControlSnapshot,
}

async fn apply_control_command(
    service: &InspectionService,
    command: RouterControlCommand,
    actor: &str,
    operation: ControlOperation,
    command_name: &'static str,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    let status = service.status().await?;
    let expected_control_generation = status
        .controls
        .as_ref()
        .map(|controls| controls.control_generation)
        .ok_or(InspectionError::StorageUnavailable)?;
    let scope = command
        .pool
        .clone()
        .map_or(ControlScope::All, |pool_id| ControlScope::Pool { pool_id });
    let mutation_id = Uuid::now_v7();
    let snapshot = service
        .apply_control(InspectionControlRequestV1 {
            mutation_id,
            scope: scope.clone(),
            operation,
            expected_control_generation,
            actor: actor.to_owned(),
            reason: command.reason,
        })
        .await?;
    let result = ControlCommandResult {
        mutation_id,
        snapshot,
    };
    if command.json {
        write_success_json(output, command_name, &result)
    } else {
        write_control_result_text(output, &scope, &result)
    }
}

fn write_control_result_text(
    output: &mut dyn Write,
    scope: &ControlScope,
    result: &ControlCommandResult,
) -> Result<(), RouterFailure> {
    let (scope_name, state) = match scope {
        ControlScope::All => ("all".to_owned(), result.snapshot.all),
        ControlScope::Pool { pool_id } => {
            let state = result
                .snapshot
                .pools
                .get(pool_id)
                .map(|pool| pool.effective)
                .ok_or(InspectionError::IntegrityError)?;
            (format!("pool:{pool_id}"), state)
        }
    };
    write_text(
        output,
        format_args!(
            "Audit ID: {}\nControl generation: {}\nScope: {}\nPaused: {}\nForce anchor: {}\n",
            result.mutation_id,
            result.snapshot.control_generation,
            scope_name,
            state.paused,
            state.force_anchor,
        ),
    )
}

async fn reset_learning(
    service: &InspectionService,
    command: RouterResetCommand,
    actor: &str,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    let pools = service
        .list_pools(PageRequest {
            limit: INSPECTION_PAGE_LIMIT_MAX,
            after: None,
        })
        .await?;
    if pools.next.is_some() {
        return Err(InspectionError::CapacityExhausted.into());
    }
    let scope = match (command.pool.as_deref(), command.all) {
        (Some(pool_id), false) => {
            let generation = pools
                .items
                .iter()
                .find(|pool| pool.id == pool_id)
                .map(|pool| pool.learning_generation_id)
                .ok_or(InspectionError::NotFound)?;
            LearningResetScopeV1::Pool {
                pool_id: pool_id.to_owned(),
                expected_learning_generation_id: generation,
            }
        }
        (None, true) => LearningResetScopeV1::All {
            expected_learning_generation_ids: pools
                .items
                .iter()
                .map(|pool| (pool.id.clone(), pool.learning_generation_id))
                .collect::<BTreeMap<_, _>>(),
        },
        _ => {
            return Err(RouterFailure::input(
                "invalid_reset_scope",
                "select exactly one of --pool or --all",
            ));
        }
    };
    let receipt = service
        .reset(LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope,
            confirm_project_id: command.confirm,
            actor: actor.to_owned(),
            reason: command.reason,
        })
        .await?;
    if command.json {
        write_success_json(output, "router reset", &receipt)
    } else {
        write_operator_receipt_text(output, &receipt)
    }
}

async fn rotate_cohort(
    service: &InspectionService,
    command: RouterCohortRotateCommand,
    actor: &str,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    let status = service.status().await?;
    let expected_cohort_generation_id = status
        .cohort_generation_id
        .ok_or(InspectionError::StorageUnavailable)?;
    let receipt = service
        .rotate_cohort(CohortRotationRequestV1 {
            mutation_id: Uuid::now_v7(),
            expected_cohort_generation_id,
            confirm_project_id: command.confirm,
            actor: actor.to_owned(),
            reason: command.reason,
        })
        .await?;
    if command.json {
        write_success_json(output, "router cohort rotate", &receipt)
    } else {
        write_operator_receipt_text(output, &receipt)
    }
}

fn write_operator_receipt_text(
    output: &mut dyn Write,
    receipt: &OperatorMutationReceiptV1,
) -> Result<(), RouterFailure> {
    write_text(
        output,
        format_args!(
            "Audit ID: {}\nResult: {}\nCreated: {}\n",
            receipt.mutation_id,
            serialized_name(&receipt.result),
            receipt.created_at_unix_ms,
        ),
    )?;
    for (scope, generation) in &receipt.resulting_generations {
        write_text(output, format_args!("Generation {scope}: {generation}\n"))?;
    }
    Ok(())
}

fn neighborhood_lookup(
    command: RouterNeighborhoodInspectCommand,
) -> Result<NeighborhoodLookupV1, RouterFailure> {
    match (
        command.evidence_id,
        command.query_hash,
        command.partition_file,
        command.request_file,
    ) {
        (Some(evidence_id), None, None, None) => Ok(NeighborhoodLookupV1::Evidence { evidence_id }),
        (None, Some(canonical_query_hash), Some(path), None) => {
            let partition = read_bounded_json::<RoutingPartitionV1>(&path)?;
            Ok(NeighborhoodLookupV1::QueryHash {
                canonical_query_hash,
                partition: Box::new(partition),
            })
        }
        (None, None, None, Some(path)) => {
            let input = read_bounded_json::<RoutingInspectionInputV1>(&path)?;
            Ok(NeighborhoodLookupV1::Request {
                input: Box::new(input),
            })
        }
        _ => Err(RouterFailure::input(
            "invalid_lookup",
            "select exactly one complete neighborhood lookup form",
        )),
    }
}

fn read_bounded_json<T>(path: &Path) -> Result<T, RouterFailure>
where
    T: serde::de::DeserializeOwned,
{
    if path == Path::new("-") {
        let stdin = io::stdin();
        decode_bounded_json(stdin.lock())
    } else {
        let file = File::open(path).map_err(RouterFailure::Io)?;
        decode_bounded_json(file)
    }
}

fn decode_bounded_json<T>(reader: impl Read) -> Result<T, RouterFailure>
where
    T: serde::de::DeserializeOwned,
{
    let maximum = u64::try_from(INSPECTION_INPUT_MAX_BYTES)
        .map_err(|_| RouterFailure::input("input_too_large", "invalid input byte bound"))?;
    let mut bytes = Vec::new();
    reader
        .take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(RouterFailure::Io)?;
    if bytes.len() > INSPECTION_INPUT_MAX_BYTES {
        return Err(RouterFailure::input(
            "input_too_large",
            format!("input exceeds {INSPECTION_INPUT_MAX_BYTES} bytes"),
        ));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        RouterFailure::input("invalid_json_input", format!("invalid JSON input: {error}"))
    })
}

async fn export_evidence(
    service: &InspectionService,
    command: RouterEvidenceExportCommand,
    stdout: &mut dyn Write,
) -> Result<(), RouterFailure> {
    if command.output == Path::new("-") && command.force {
        return Err(RouterFailure::input(
            "invalid_export_target",
            "--force cannot be used when exporting to standard output",
        ));
    }
    let request = EvidenceExportRequestV1 {
        filter: evidence_filter(&command.filter),
        format: match command.format {
            RouterEvidenceExportFormat::Jsonl => EvidenceExportFormatV1::Jsonl,
            RouterEvidenceExportFormat::Csv => EvidenceExportFormatV1::Csv,
        },
    };
    if command.output == Path::new("-") {
        let mut stream = service.export_evidence(request).await?;
        return copy_export(&mut stream, stdout).await;
    }
    if command.force {
        let parent = nonempty_parent(&command.output);
        let mut temporary = NamedTempFile::new_in(parent).map_err(RouterFailure::Io)?;
        let mut stream = service.export_evidence(request).await?;
        copy_export(&mut stream, temporary.as_file_mut()).await?;
        temporary.as_file_mut().flush().map_err(RouterFailure::Io)?;
        temporary
            .persist(&command.output)
            .map_err(|error| RouterFailure::Io(error.error))?;
        Ok(())
    } else {
        let file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&command.output)
        {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                return Err(RouterFailure::input(
                    "output_exists",
                    "export destination already exists; pass --force to replace it",
                ));
            }
            Err(error) => return Err(RouterFailure::Io(error)),
        };
        let mut partial = PartialOutput::new(command.output, file);
        let mut stream = service.export_evidence(request).await?;
        copy_export(&mut stream, partial.writer()).await?;
        partial.commit()
    }
}

fn nonempty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

async fn copy_export(
    stream: &mut nemo_relay_router::inspection::EvidenceExportStream,
    writer: &mut dyn Write,
) -> Result<(), RouterFailure> {
    while let Some(chunk) = stream.next_chunk().await {
        write_bytes(writer, &chunk?)?;
    }
    writer.flush().map_err(map_output_error)
}

struct PartialOutput {
    path: PathBuf,
    file: File,
    committed: bool,
}

impl PartialOutput {
    fn new(path: PathBuf, file: File) -> Self {
        Self {
            path,
            file,
            committed: false,
        }
    }

    fn writer(&mut self) -> &mut File {
        &mut self.file
    }

    fn commit(mut self) -> Result<(), RouterFailure> {
        self.file.flush().map_err(RouterFailure::Io)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PartialOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

async fn tail_decisions(
    service: &InspectionService,
    command: RouterDecisionsTailCommand,
    output: &mut dyn Write,
) -> Result<(), RouterFailure> {
    let filter = decision_filter(&command.filter);
    let mut cursor = command.page.cursor.clone();
    let mut wrote_header = false;
    loop {
        let prior_cursor = cursor.clone();
        let page = service
            .tail_decisions(
                filter.clone(),
                PageRequest {
                    limit: command.page.limit,
                    after: cursor,
                },
            )
            .await?;
        if command.json && command.follow {
            write_decision_jsonl(output, &page, prior_cursor.as_deref())?;
        } else if command.json {
            return write_success_json(output, "router decisions tail", &page);
        } else {
            write_decisions_text(output, &page, !wrote_header, !command.follow)?;
            wrote_header = true;
        }
        let item_count = page.items.len();
        cursor = page.next;
        output.flush().map_err(map_output_error)?;
        if !command.follow {
            return Ok(());
        }
        if item_count == usize::from(command.page.limit) && cursor != prior_cursor {
            continue;
        }
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.map_err(RouterFailure::Io)?;
                return Ok(());
            }
            () = tokio::time::sleep(DECISION_FOLLOW_POLL_INTERVAL) => {}
        }
    }
}

#[derive(Serialize)]
struct SuccessEnvelope<'a, T> {
    schema_version: u32,
    ok: bool,
    command: &'static str,
    data: &'a T,
}

#[derive(Serialize)]
struct DecisionTailRecord<'a> {
    schema_version: u32,
    command: &'static str,
    cursor: Option<&'a str>,
    decision: Option<&'a DecisionSummaryV1>,
}

fn write_success_json<T: Serialize>(
    output: &mut dyn Write,
    command: &'static str,
    data: &T,
) -> Result<(), RouterFailure> {
    let bytes = serde_json::to_vec_pretty(&SuccessEnvelope {
        schema_version: ROUTER_CLI_SCHEMA_VERSION,
        ok: true,
        command,
        data,
    })
    .map_err(json_output_error)?;
    write_bytes(output, &bytes)?;
    write_bytes(output, b"\n")
}

fn write_failure_json(
    output: &mut dyn Write,
    command: &'static str,
    failure: &RouterFailure,
) -> Result<(), RouterFailure> {
    let bytes = serde_json::to_vec_pretty(&json!({
        "schema_version": ROUTER_CLI_SCHEMA_VERSION,
        "ok": false,
        "command": command,
        "error": {
            "code": failure.code(),
            "message": failure.to_string(),
        }
    }))
    .map_err(json_output_error)?;
    write_bytes(output, &bytes)?;
    write_bytes(output, b"\n")
}

fn write_decision_jsonl(
    output: &mut dyn Write,
    page: &Page<DecisionSummaryV1>,
    prior_cursor: Option<&str>,
) -> Result<(), RouterFailure> {
    let cursor = page.next.as_deref();
    for decision in &page.items {
        let bytes = serde_json::to_vec(&DecisionTailRecord {
            schema_version: ROUTER_CLI_SCHEMA_VERSION,
            command: "router decisions tail",
            cursor,
            decision: Some(decision),
        })
        .map_err(json_output_error)?;
        write_bytes(output, &bytes)?;
        write_bytes(output, b"\n")?;
    }
    if page.items.is_empty() && cursor != prior_cursor {
        let bytes = serde_json::to_vec(&DecisionTailRecord {
            schema_version: ROUTER_CLI_SCHEMA_VERSION,
            command: "router decisions tail",
            cursor,
            decision: None,
        })
        .map_err(json_output_error)?;
        write_bytes(output, &bytes)?;
        write_bytes(output, b"\n")?;
    }
    Ok(())
}

fn write_status_text(output: &mut dyn Write, report: &StatusReportV1) -> Result<(), RouterFailure> {
    write_text(
        output,
        format_args!(
            "Project: {}\nConfigured mode: {}\nEffective mode: {}\nDatabase: {}\nSchema: {}/{}\nQueues: {}/{} pending\nHealth: {}\nSnapshot: {}\n",
            report.project_id.as_deref().unwrap_or("unavailable"),
            serialized_name(&report.configured_mode),
            serialized_name(&report.effective_mode),
            serialized_name(&report.database.state),
            report
                .database
                .schema_version
                .map_or_else(|| "unavailable".into(), |value| value.to_string()),
            report.database.supported_schema_version,
            report.queues.pending,
            report.queues.capacity,
            report.health.worst,
            report.snapshot_time_unix_ms,
        ),
    )
}

fn write_pools_text<T: PoolText>(
    output: &mut dyn Write,
    page: &Page<T>,
) -> Result<(), RouterFailure> {
    write_bytes(
        output,
        b"POOL\tAPI\tANCHORS\tCANDIDATES\tREADY\tROOTS\tHEALTH\n",
    )?;
    for pool in &page.items {
        pool.write_row(output)?;
    }
    write_next_cursor(output, page.next.as_deref())
}

trait PoolText {
    fn write_row(&self, output: &mut dyn Write) -> Result<(), RouterFailure>;
}

impl PoolText for nemo_relay_router::inspection::PoolSummaryV1 {
    fn write_row(&self, output: &mut dyn Write) -> Result<(), RouterFailure> {
        write_text(
            output,
            format_args!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                self.id,
                serialized_name(&self.api_family),
                self.anchor_models.join(","),
                self.candidates.len(),
                self.support.ready_evidence,
                self.support.independent_roots,
                self.health.worst,
            ),
        )
    }
}

fn write_evidence_text(
    output: &mut dyn Write,
    page: &Page<EvidenceSummaryV1>,
) -> Result<(), RouterFailure> {
    write_bytes(
        output,
        b"EVIDENCE_ID\tPOOL\tCANDIDATE\tTERMINAL\tQUALITY\tCREATED_MS\n",
    )?;
    for item in &page.items {
        write_text(
            output,
            format_args!(
                "{}\t{}\t{}\t{}\t{}\t{}\n",
                item.evidence_id,
                item.pool_id,
                item.candidate_id,
                item.terminal_class,
                item.quality_label.as_deref().unwrap_or("-"),
                item.created_at_unix_ms,
            ),
        )?;
    }
    write_next_cursor(output, page.next.as_deref())
}

fn write_evidence_detail_text(
    output: &mut dyn Write,
    detail: &nemo_relay_router::inspection::EvidenceDetailV1,
) -> Result<(), RouterFailure> {
    write_text(
        output,
        format_args!(
            "Evidence: {}\nPool: {}\nCandidate: {}\nTerminal: {}\nQuality: {}\nQuery hash: {}\nLearning generation: {}\nVector state: {}\nCreated: {}\nContent bytes: {}\nContent SHA-256: {}\nPreview: {}\n",
            detail.summary.evidence_id,
            detail.summary.pool_id,
            detail.summary.candidate_id,
            detail.summary.terminal_class,
            detail.summary.quality_label.as_deref().unwrap_or("-"),
            detail.summary.canonical_query_hash,
            detail.summary.learning_generation_id,
            detail.summary.vector_state,
            detail.summary.created_at_unix_ms,
            detail.summary.content.byte_length,
            detail.summary.content.sha256,
            detail.summary.content.preview.as_deref().unwrap_or("-"),
        ),
    )
}

fn write_neighborhood_text(
    output: &mut dyn Write,
    report: &nemo_relay_router::inspection::NeighborhoodReportV1,
) -> Result<(), RouterFailure> {
    write_text(
        output,
        format_args!(
            "Pool: {}\nQuery hash: {}\nNeighbors: {}\nSelected roots: {}\nCoverage: {}\nEffective samples: {}\nLower bound: {}\nRecommendation: {}\nAnchor fallback: {}\nProjection: {}\nSnapshot: {}\n",
            report.pool_id,
            report.canonical_query_hash,
            report.support.returned_neighbors,
            report.support.selected_roots,
            optional_float(report.support.coverage),
            optional_float(report.support.effective_sample_size),
            optional_float(report.credible_lower_bound),
            report.recommendation.reason,
            report.recommendation.anchor_fallback,
            report
                .projection
                .as_ref()
                .map_or("none", |projection| projection.algorithm.as_str()),
            report.snapshot_time_unix_ms,
        ),
    )
}

fn write_decisions_text(
    output: &mut dyn Write,
    page: &Page<DecisionSummaryV1>,
    header: bool,
    cursor: bool,
) -> Result<(), RouterFailure> {
    if header {
        write_bytes(
            output,
            b"DECISION_ID\tPOOL\tMODE\tCANDIDATE\tSERVED\tREASON\tCREATED_MS\n",
        )?;
    }
    for item in &page.items {
        write_text(
            output,
            format_args!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                item.decision_id,
                item.pool_id,
                item.mode,
                item.candidate_id.as_deref().unwrap_or("-"),
                item.served_model,
                item.final_reason,
                item.created_at_unix_ms,
            ),
        )?;
    }
    if cursor {
        write_next_cursor(output, page.next.as_deref())?;
    }
    Ok(())
}

fn write_next_cursor(output: &mut dyn Write, cursor: Option<&str>) -> Result<(), RouterFailure> {
    if let Some(cursor) = cursor {
        write_text(output, format_args!("Next cursor: {cursor}\n"))?;
    }
    Ok(())
}

fn serialized_name(value: &impl Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".into())
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

fn write_text(
    output: &mut dyn Write,
    arguments: std::fmt::Arguments<'_>,
) -> Result<(), RouterFailure> {
    output.write_fmt(arguments).map_err(map_output_error)
}

fn write_bytes(output: &mut dyn Write, bytes: &[u8]) -> Result<(), RouterFailure> {
    output.write_all(bytes).map_err(map_output_error)
}

fn map_output_error(error: io::Error) -> RouterFailure {
    if error.kind() == io::ErrorKind::BrokenPipe {
        RouterFailure::OutputClosed
    } else {
        RouterFailure::Io(error)
    }
}

fn json_output_error(error: serde_json::Error) -> RouterFailure {
    RouterFailure::Io(io::Error::new(
        error.io_error_kind().unwrap_or(io::ErrorKind::InvalidData),
        error,
    ))
}

fn command_name(command: &RouterCommand) -> &'static str {
    match &command.command {
        RouterSubcommand::Dashboard(_) => "router dashboard",
        RouterSubcommand::Status(_) => "router status",
        RouterSubcommand::Pools(_) => "router pools",
        RouterSubcommand::Evidence(command) => match command.command {
            RouterEvidenceSubcommand::List(_) => "router evidence list",
            RouterEvidenceSubcommand::Show(_) => "router evidence show",
            RouterEvidenceSubcommand::Export(_) => "router evidence export",
        },
        RouterSubcommand::Neighborhood(_) => "router neighborhood inspect",
        RouterSubcommand::Decisions(_) => "router decisions tail",
        RouterSubcommand::Pause(_) => "router pause",
        RouterSubcommand::Resume(_) => "router resume",
        RouterSubcommand::ForceAnchor(command) => match command.command {
            RouterForceAnchorSubcommand::Set(_) => "router force-anchor set",
            RouterForceAnchorSubcommand::Clear(_) => "router force-anchor clear",
        },
        RouterSubcommand::Reset(_) => "router reset",
        RouterSubcommand::Cohort(_) => "router cohort rotate",
    }
}

fn command_uses_json(command: &RouterCommand) -> bool {
    match &command.command {
        RouterSubcommand::Dashboard(_) => false,
        RouterSubcommand::Status(command) => command.json,
        RouterSubcommand::Pools(command) => command.json,
        RouterSubcommand::Evidence(command) => match &command.command {
            RouterEvidenceSubcommand::List(command) => command.json,
            RouterEvidenceSubcommand::Show(command) => command.json,
            RouterEvidenceSubcommand::Export(_) => false,
        },
        RouterSubcommand::Neighborhood(command) => match &command.command {
            RouterNeighborhoodSubcommand::Inspect(command) => command.json,
        },
        RouterSubcommand::Decisions(command) => match &command.command {
            RouterDecisionsSubcommand::Tail(command) => command.json,
        },
        RouterSubcommand::Pause(command) | RouterSubcommand::Resume(command) => command.json,
        RouterSubcommand::ForceAnchor(command) => match &command.command {
            RouterForceAnchorSubcommand::Set(command)
            | RouterForceAnchorSubcommand::Clear(command) => command.json,
        },
        RouterSubcommand::Reset(command) => command.json,
        RouterSubcommand::Cohort(command) => match &command.command {
            RouterCohortSubcommand::Rotate(command) => command.json,
        },
    }
}

#[derive(Debug)]
pub(crate) enum RouterFailure {
    Input { code: &'static str, message: String },
    Inspection(InspectionError),
    Cli(CliError),
    Io(io::Error),
    OutputClosed,
}

impl RouterFailure {
    fn input(code: &'static str, message: impl Into<String>) -> Self {
        Self::Input {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::Input { code, .. } => code,
            Self::Inspection(error) => error.code(),
            Self::Cli(CliError::Config(_)) => "invalid_configuration",
            Self::Cli(_) => "command_failed",
            Self::Io(_) => "io_error",
            Self::OutputClosed => "output_closed",
        }
    }

    pub(crate) fn exit_code(&self) -> ExitCode {
        let refused = match self {
            Self::Input { .. } | Self::Cli(CliError::Config(_)) => true,
            Self::Inspection(error) => matches!(
                error,
                InspectionError::InvalidArgument
                    | InspectionError::InvalidCursor
                    | InspectionError::NotFound
                    | InspectionError::Unauthorized
                    | InspectionError::Forbidden
                    | InspectionError::Conflict
                    | InspectionError::MutationExpired
                    | InspectionError::EgressDenied
            ),
            Self::Cli(_) | Self::Io(_) | Self::OutputClosed => false,
        };
        if refused {
            ExitCode::from(2)
        } else {
            ExitCode::FAILURE
        }
    }

    fn into_cli_error(self) -> CliError {
        match self {
            Self::Cli(error) => error,
            Self::Io(error) => error.into(),
            other => CliError::Config(other.to_string()),
        }
    }
}

impl std::fmt::Display for RouterFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input { message, .. } => formatter.write_str(message),
            Self::Inspection(error) => {
                write!(formatter, "Router inspection failed: {}", error.code())
            }
            Self::Cli(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
            Self::OutputClosed => formatter.write_str("output closed"),
        }
    }
}

impl From<InspectionError> for RouterFailure {
    fn from(error: InspectionError) -> Self {
        Self::Inspection(error)
    }
}

#[cfg(test)]
#[path = "../tests/coverage/router_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/coverage/router_dashboard_contract_tests.rs"]
mod dashboard_contract_tests;
