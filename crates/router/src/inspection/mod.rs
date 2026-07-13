// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable typed contracts for Router inspection and operator surfaces.
//!
//! The service implementation opens an existing Router ledger through bounded
//! repositories. These DTOs never expose a SQLite connection, cohort salt,
//! credential, provider header, or executable replay capability.

pub(crate) mod content;
pub(crate) mod cursor;
mod export;
#[cfg(feature = "http")]
mod http;
pub(crate) mod projection;
pub(crate) mod request;
mod service;
mod types;

pub use export::EvidenceExportStream;
#[cfg(feature = "http")]
pub use http::{
    BearerTokenAuthGuard, InspectionHttpAuthContext, InspectionHttpAuthGuard, InspectionHttpMode,
    inspection_http_router,
};
pub use service::InspectionService;
pub use types::*;
