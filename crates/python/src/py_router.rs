// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binding-private Router native-package probe used by installed-wheel tests.

use pyo3::prelude::*;

#[pyfunction]
fn _router_native_vector_probe() -> PyResult<(String, String)> {
    nemo_relay_router::probe_native_vector_capability()
        .map(|report| (report.sqlite_version, report.vector_version))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.code()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_router_native_vector_probe, module)?)?;
    Ok(())
}
