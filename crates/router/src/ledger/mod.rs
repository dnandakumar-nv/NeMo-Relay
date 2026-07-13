// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private SQLite ledger contracts and implementation.

#[allow(dead_code)] // Task 6 consumes the pure assignment facts during active admission.
pub(crate) mod cohort;
pub(crate) mod command;
pub(crate) mod fs;
pub(crate) mod migrations;
pub(crate) mod model;
pub(crate) mod read_pool;
pub(crate) mod repository;
pub(crate) mod writer;
