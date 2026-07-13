// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded incremental evidence export.

use std::fmt;
use std::io::{self, Write};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::{
    EVIDENCE_EXPORT_RECORD_SCHEMA_V1, EvidenceExportFormatV1, EvidenceExportRecordV1,
    EvidenceExportRequestV1, EvidenceSummaryV1, InspectionError, InspectionService, PageRequest,
};

const CSV_HEADER: &[u8] = b"evidence_id,pool_id,candidate_id,terminal_class,quality_label,canonical_query_hash,learning_generation_id,vector_state,created_at_unix_ms,content_sha256,content_byte_length,content_preview\n";

/// Consumer-driven bounded byte stream for one evidence export.
pub struct EvidenceExportStream {
    receiver: mpsc::Receiver<Result<Vec<u8>, InspectionError>>,
    producer: JoinHandle<()>,
    format: EvidenceExportFormatV1,
}

impl EvidenceExportStream {
    /// Receive the next bounded export chunk.
    pub async fn next_chunk(&mut self) -> Option<Result<Vec<u8>, InspectionError>> {
        self.receiver.recv().await
    }

    /// Return the stable HTTP content type for this stream.
    pub const fn content_type(&self) -> &'static str {
        match self.format {
            EvidenceExportFormatV1::Jsonl => "application/x-ndjson",
            EvidenceExportFormatV1::Csv => "text/csv; charset=utf-8",
        }
    }

    /// Stop production and reject further chunks.
    pub fn abort(&mut self) {
        self.producer.abort();
        self.receiver.close();
    }
}

impl fmt::Debug for EvidenceExportStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvidenceExportStream")
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl Drop for EvidenceExportStream {
    fn drop(&mut self) {
        self.producer.abort();
    }
}

impl InspectionService {
    /// Start a bounded JSON Lines or scalar CSV evidence export.
    pub async fn export_evidence(
        &self,
        request: EvidenceExportRequestV1,
    ) -> Result<EvidenceExportStream, InspectionError> {
        request.validate()?;
        self.current()?;
        let chunk_bytes = self.export_chunk_bytes();
        let format = request.format;
        let service = self.clone();
        let (sender, receiver) = mpsc::channel(1);
        let producer = tokio::spawn(async move {
            produce_export(service, request, chunk_bytes, sender).await;
        });
        Ok(EvidenceExportStream {
            receiver,
            producer,
            format,
        })
    }
}

async fn produce_export(
    service: InspectionService,
    request: EvidenceExportRequestV1,
    chunk_bytes: usize,
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
) {
    if request.format == EvidenceExportFormatV1::Csv
        && !render_bytes(sender.clone(), chunk_bytes, CSV_HEADER.to_vec()).await
    {
        return;
    }
    let mut page_request = PageRequest {
        limit: 1,
        after: None,
    };
    loop {
        let page = match service
            .list_evidence(request.filter.clone(), page_request.clone())
            .await
        {
            Ok(page) => page,
            Err(error) => {
                let _ = sender.send(Err(error)).await;
                return;
            }
        };
        let Some(summary) = page.items.into_iter().next() else {
            return;
        };
        let rendered = match request.format {
            EvidenceExportFormatV1::Jsonl => {
                let detail = match service.get_evidence(summary.evidence_id).await {
                    Ok(detail) => detail,
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                };
                render_jsonl(sender.clone(), chunk_bytes, detail).await
            }
            EvidenceExportFormatV1::Csv => render_csv(sender.clone(), chunk_bytes, summary).await,
        };
        if !rendered {
            return;
        }
        let Some(next) = page.next else {
            return;
        };
        page_request.after = Some(next);
    }
}

async fn render_jsonl(
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
    chunk_bytes: usize,
    evidence: super::EvidenceDetailV1,
) -> bool {
    let worker_sender = sender.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut writer = ChunkWriter::new(worker_sender, chunk_bytes);
        let record = EvidenceExportRecordV1 {
            schema: EVIDENCE_EXPORT_RECORD_SCHEMA_V1.into(),
            evidence,
        };
        if let Err(error) = serde_json::to_writer(&mut writer, &record) {
            return Err(match error.io_error_kind() {
                Some(kind) => io::Error::from(kind),
                None => io::Error::new(io::ErrorKind::InvalidData, "JSON export failed"),
            });
        }
        writer.write_all(b"\n")?;
        writer.finish()
    })
    .await;
    handle_render_result(sender, result).await
}

async fn render_csv(
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
    chunk_bytes: usize,
    summary: EvidenceSummaryV1,
) -> bool {
    let worker_sender = sender.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut writer = ChunkWriter::new(worker_sender, chunk_bytes);
        write_csv_summary(&mut writer, &summary)?;
        writer.finish()
    })
    .await;
    handle_render_result(sender, result).await
}

async fn render_bytes(
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
    chunk_bytes: usize,
    bytes: Vec<u8>,
) -> bool {
    let worker_sender = sender.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut writer = ChunkWriter::new(worker_sender, chunk_bytes);
        writer.write_all(&bytes)?;
        writer.finish()
    })
    .await;
    handle_render_result(sender, result).await
}

async fn handle_render_result(
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
    result: Result<io::Result<()>, tokio::task::JoinError>,
) -> bool {
    match result {
        Ok(Ok(())) => true,
        Ok(Err(error)) if error.kind() == io::ErrorKind::BrokenPipe => false,
        Ok(Err(_)) | Err(_) => {
            let _ = sender.send(Err(InspectionError::IntegrityError)).await;
            false
        }
    }
}

struct ChunkWriter {
    sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>,
    buffer: Vec<u8>,
    maximum: usize,
}

impl ChunkWriter {
    fn new(sender: mpsc::Sender<Result<Vec<u8>, InspectionError>>, maximum: usize) -> Self {
        Self {
            sender,
            buffer: Vec::with_capacity(maximum),
            maximum,
        }
    }

    fn emit(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(&mut self.buffer, Vec::with_capacity(self.maximum));
        self.sender
            .blocking_send(Ok(chunk))
            .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))
    }

    fn finish(mut self) -> io::Result<()> {
        self.emit()
    }
}

impl Write for ChunkWriter {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let length = bytes.len();
        while !bytes.is_empty() {
            if self.buffer.len() == self.maximum {
                self.emit()?;
            }
            let available = self.maximum.saturating_sub(self.buffer.len());
            if available == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid export chunk bound",
                ));
            }
            let take = available.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
        }
        Ok(length)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.emit()
    }
}

fn write_csv_summary(writer: &mut impl Write, summary: &EvidenceSummaryV1) -> io::Result<()> {
    let evidence_id = summary.evidence_id.to_string();
    let learning_generation_id = summary.learning_generation_id.to_string();
    let created_at_unix_ms = summary.created_at_unix_ms.to_string();
    let byte_length = summary.content.byte_length.to_string();
    let fields = [
        evidence_id.as_str(),
        summary.pool_id.as_str(),
        summary.candidate_id.as_str(),
        summary.terminal_class.as_str(),
        summary.quality_label.as_deref().unwrap_or_default(),
        summary.canonical_query_hash.as_str(),
        learning_generation_id.as_str(),
        summary.vector_state.as_str(),
        created_at_unix_ms.as_str(),
        summary.content.sha256.as_str(),
        byte_length.as_str(),
        summary.content.preview.as_deref().unwrap_or_default(),
    ];
    for (index, field) in fields.into_iter().enumerate() {
        if index > 0 {
            writer.write_all(b",")?;
        }
        write_csv_field(writer, field)?;
    }
    writer.write_all(b"\n")
}

fn write_csv_field(writer: &mut impl Write, field: &str) -> io::Result<()> {
    if !field
        .bytes()
        .any(|byte| matches!(byte, b',' | b'"' | b'\r' | b'\n'))
    {
        return writer.write_all(field.as_bytes());
    }
    writer.write_all(b"\"")?;
    for byte in field.bytes() {
        if byte == b'"' {
            writer.write_all(b"\"\"")?;
        } else {
            writer.write_all(&[byte])?;
        }
    }
    writer.write_all(b"\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspection::{ContentPolicy, InspectionServiceOptions};
    use crate::ledger::repository::ready_evaluated_runtime_fixture;

    #[test]
    fn csv_field_quoting_is_deterministic() {
        let mut output = Vec::new();
        write_csv_field(&mut output, "plain").unwrap();
        assert_eq!(output, b"plain");
        output.clear();
        write_csv_field(&mut output, "a,\"b\"\n").unwrap();
        assert_eq!(output, b"\"a,\"\"b\"\"\n\"");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chunk_writer_never_exceeds_its_bound_and_stops_on_cancellation() {
        let (sender, mut receiver) = mpsc::channel(1);
        let worker = tokio::task::spawn_blocking(move || {
            let mut writer = ChunkWriter::new(sender, 4);
            writer.write_all(b"abcdefghij")?;
            writer.finish()
        });
        let mut output = Vec::new();
        while let Some(chunk) = receiver.recv().await {
            let chunk = chunk.unwrap();
            assert!(chunk.len() <= 4);
            output.extend(chunk);
        }
        worker.await.unwrap().unwrap();
        assert_eq!(output, b"abcdefghij");

        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let error = tokio::task::spawn_blocking(move || {
            let mut writer = ChunkWriter::new(sender, 4);
            writer.write_all(b"abcdefgh")
        })
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn service_exports_versioned_jsonl_and_deterministic_csv() {
        let (
            _temporary,
            config,
            activated,
            _vector_space_id,
            _vector_root,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_runtime_fixture();
        let options = InspectionServiceOptions {
            export_chunk_bytes: 4 * 1024,
            ..InspectionServiceOptions::default()
        };
        let service = InspectionService::open(config.clone(), options)
            .await
            .unwrap();

        let mut jsonl = service
            .export_evidence(EvidenceExportRequestV1::default())
            .await
            .unwrap();
        assert_eq!(jsonl.content_type(), "application/x-ndjson");
        let jsonl_bytes = collect_chunks(&mut jsonl, 4 * 1024).await;
        let lines = jsonl_bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 1);
        let record: EvidenceExportRecordV1 = serde_json::from_slice(lines[0]).unwrap();
        assert_eq!(record.schema, EVIDENCE_EXPORT_RECORD_SCHEMA_V1);
        assert_eq!(record.evidence.summary.evidence_id, evidence_id);
        assert!(record.evidence.summary.content.value.is_none());

        let mut csv = service
            .export_evidence(EvidenceExportRequestV1 {
                format: EvidenceExportFormatV1::Csv,
                ..EvidenceExportRequestV1::default()
            })
            .await
            .unwrap();
        assert_eq!(csv.content_type(), "text/csv; charset=utf-8");
        let csv_bytes = collect_chunks(&mut csv, 4 * 1024).await;
        let csv_text = String::from_utf8(csv_bytes).unwrap();
        assert!(csv_text.starts_with(std::str::from_utf8(CSV_HEADER).unwrap()));
        assert_eq!(csv_text.lines().count(), 2);
        assert!(
            csv_text
                .lines()
                .nth(1)
                .unwrap()
                .contains(&evidence_id.to_string())
        );

        let cancelled = service
            .export_evidence(EvidenceExportRequestV1::default())
            .await
            .unwrap();
        drop(cancelled);
        tokio::time::timeout(std::time::Duration::from_secs(2), service.close())
            .await
            .unwrap()
            .unwrap();

        let full = InspectionService::open(
            config,
            InspectionServiceOptions {
                content_policy: ContentPolicy::Full,
                export_chunk_bytes: 4 * 1024,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let mut export = full
            .export_evidence(EvidenceExportRequestV1::default())
            .await
            .unwrap();
        let bytes = collect_chunks(&mut export, 4 * 1024).await;
        let record: EvidenceExportRecordV1 = serde_json::from_slice(
            bytes
                .split(|byte| *byte == b'\n')
                .find(|line| !line.is_empty())
                .unwrap(),
        )
        .unwrap();
        assert!(record.evidence.summary.content.value.is_some());
        full.close().await.unwrap();
        drop(activated);
    }

    async fn collect_chunks(stream: &mut EvidenceExportStream, maximum: usize) -> Vec<u8> {
        let mut output = Vec::new();
        while let Some(chunk) = stream.next_chunk().await {
            let chunk = chunk.unwrap();
            assert!(!chunk.is_empty());
            assert!(chunk.len() <= maximum);
            output.extend(chunk);
        }
        output
    }
}
