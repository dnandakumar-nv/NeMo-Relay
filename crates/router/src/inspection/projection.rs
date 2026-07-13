// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, diagnostic-only two-dimensional PCA.

use std::cmp::Ordering;

use super::{
    DIAGNOSTIC_PROJECTION_MATRIX_CELLS_MAX, DIAGNOSTIC_PROJECTION_MULTIPLY_ADDS_MAX,
    DIAGNOSTIC_PROJECTION_POINTS_MAX, DiagnosticPointKindV1, DiagnosticPointV1,
    DiagnosticProjectionV1, InspectionError,
};
use crate::vector::NormalizedVector;

const JACOBI_SWEEPS: usize = 64;
const ZERO_TOLERANCE: f64 = 1.0e-14;

pub(crate) struct DiagnosticVector<'a> {
    pub(crate) record_id: String,
    pub(crate) kind: DiagnosticPointKindV1,
    pub(crate) vector: &'a NormalizedVector,
}

pub(crate) fn project_pca_2(
    vector_space_id: &str,
    mut vectors: Vec<DiagnosticVector<'_>>,
) -> Result<Option<DiagnosticProjectionV1>, InspectionError> {
    if vector_space_id.is_empty() || vectors.is_empty() {
        return Ok(None);
    }
    vectors.sort_by(|left, right| match (left.kind, right.kind) {
        (DiagnosticPointKindV1::Query, DiagnosticPointKindV1::Evidence) => Ordering::Less,
        (DiagnosticPointKindV1::Evidence, DiagnosticPointKindV1::Query) => Ordering::Greater,
        _ => left.record_id.cmp(&right.record_id),
    });
    if vectors.len() > DIAGNOSTIC_PROJECTION_POINTS_MAX
        || vectors.iter().any(|point| point.record_id.is_empty())
    {
        return Ok(None);
    }
    let dimensions = vectors[0].vector.dimensions().as_usize();
    if dimensions == 0
        || vectors
            .iter()
            .any(|point| point.vector.dimensions().as_usize() != dimensions)
    {
        return Err(InspectionError::IntegrityError);
    }
    let count = vectors.len();
    let matrix_cells = count
        .checked_mul(count)
        .ok_or(InspectionError::IntegrityError)?;
    let gram_operations = matrix_cells
        .checked_mul(dimensions)
        .ok_or(InspectionError::IntegrityError)?;
    let jacobi_operations = matrix_cells
        .checked_mul(count)
        .and_then(|value| value.checked_mul(JACOBI_SWEEPS))
        .ok_or(InspectionError::IntegrityError)?;
    let loading_operations = count
        .checked_mul(dimensions)
        .and_then(|value| value.checked_mul(2))
        .ok_or(InspectionError::IntegrityError)?;
    let total_operations = gram_operations
        .checked_add(jacobi_operations)
        .and_then(|value| value.checked_add(loading_operations))
        .ok_or(InspectionError::IntegrityError)?;
    if matrix_cells > DIAGNOSTIC_PROJECTION_MATRIX_CELLS_MAX
        || u64::try_from(total_operations)
            .ok()
            .is_none_or(|value| value > DIAGNOSTIC_PROJECTION_MULTIPLY_ADDS_MAX)
    {
        return Ok(None);
    }

    let mut centered = vectors
        .iter()
        .map(|point| {
            point
                .vector
                .values()
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for dimension in 0..dimensions {
        let mean = centered.iter().map(|point| point[dimension]).sum::<f64>() / count as f64;
        for point in &mut centered {
            point[dimension] -= mean;
        }
    }
    if count == 1 {
        return Ok(Some(zero_projection(vector_space_id, &vectors)));
    }

    let divisor = (count - 1) as f64;
    let mut gram = vec![vec![0.0; count]; count];
    for left in 0..count {
        for right in left..count {
            let value = centered[left]
                .iter()
                .zip(&centered[right])
                .map(|(left, right)| left * right)
                .sum::<f64>()
                / divisor;
            if !value.is_finite() {
                return Err(InspectionError::IntegrityError);
            }
            gram[left][right] = value;
            gram[right][left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = jacobi_eigen(gram)?;
    let total_variance = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > ZERO_TOLERANCE)
        .sum::<f64>();
    let mut components = (0..count)
        .map(|index| build_component(index, &eigenvalues, &eigenvectors, &centered, divisor))
        .collect::<Result<Vec<_>, InspectionError>>()?;
    components.sort_by(|left, right| {
        right
            .eigenvalue
            .total_cmp(&left.eigenvalue)
            .then_with(|| compare_loadings(&left.loading, &right.loading))
            .then_with(|| left.source_index.cmp(&right.source_index))
    });

    let first = components.first().filter(|value| value.is_nonzero());
    let second = components.get(1).filter(|value| value.is_nonzero());
    let mut points = Vec::with_capacity(count);
    for (index, vector) in vectors.iter().enumerate() {
        points.push(DiagnosticPointV1 {
            record_id: vector.record_id.clone(),
            kind: vector.kind,
            x: canonical_zero(first.map_or(0.0, |component| component.coordinates[index])),
            y: canonical_zero(second.map_or(0.0, |component| component.coordinates[index])),
        });
    }
    Ok(Some(DiagnosticProjectionV1 {
        algorithm: "pca_2".into(),
        algorithm_version: 1,
        diagnostic_only: true,
        vector_space_id: vector_space_id.into(),
        points,
        explained_variance_ratio: [
            variance_ratio(first, total_variance),
            variance_ratio(second, total_variance),
        ],
    }))
}

struct Component {
    source_index: usize,
    eigenvalue: f64,
    loading: Vec<f64>,
    coordinates: Vec<f64>,
}

impl Component {
    fn is_nonzero(&self) -> bool {
        self.eigenvalue > ZERO_TOLERANCE
            && self
                .coordinates
                .iter()
                .any(|value| value.abs() > ZERO_TOLERANCE)
    }
}

fn build_component(
    index: usize,
    eigenvalues: &[f64],
    eigenvectors: &[Vec<f64>],
    centered: &[Vec<f64>],
    divisor: f64,
) -> Result<Component, InspectionError> {
    let eigenvalue = eigenvalues[index].max(0.0);
    let dimensions = centered[0].len();
    let mut loading = vec![0.0; dimensions];
    if eigenvalue > ZERO_TOLERANCE {
        let scale = (divisor * eigenvalue).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(InspectionError::IntegrityError);
        }
        for (point_index, point) in centered.iter().enumerate() {
            let coefficient = eigenvectors[point_index][index] / scale;
            for dimension in 0..dimensions {
                loading[dimension] += point[dimension] * coefficient;
            }
        }
        fix_component_sign(&mut loading, &mut []);
    }
    let mut coordinates = centered
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(&loading)
                .map(|(value, loading)| value * loading)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    fix_component_sign(&mut loading, &mut coordinates);
    if loading
        .iter()
        .chain(&coordinates)
        .any(|value| !value.is_finite())
    {
        return Err(InspectionError::IntegrityError);
    }
    Ok(Component {
        source_index: index,
        eigenvalue,
        loading,
        coordinates,
    })
}

fn fix_component_sign(loading: &mut [f64], coordinates: &mut [f64]) {
    let pivot = loading
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.abs()
                .total_cmp(&right.abs())
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(_, value)| *value)
        .unwrap_or_default();
    if pivot.is_sign_negative() {
        for value in loading.iter_mut().chain(coordinates) {
            *value = -*value;
        }
    }
}

fn compare_loadings(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let order = right.abs().total_cmp(&left.abs());
            (!order.is_eq()).then_some(order)
        })
        .unwrap_or(Ordering::Equal)
}

fn variance_ratio(component: Option<&Component>, total: f64) -> f64 {
    if total <= ZERO_TOLERANCE {
        0.0
    } else {
        canonical_zero(component.map_or(0.0, |value| value.eigenvalue / total))
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value.abs() <= ZERO_TOLERANCE {
        0.0
    } else {
        value
    }
}

fn zero_projection(
    vector_space_id: &str,
    vectors: &[DiagnosticVector<'_>],
) -> DiagnosticProjectionV1 {
    DiagnosticProjectionV1 {
        algorithm: "pca_2".into(),
        algorithm_version: 1,
        diagnostic_only: true,
        vector_space_id: vector_space_id.into(),
        points: vectors
            .iter()
            .map(|vector| DiagnosticPointV1 {
                record_id: vector.record_id.clone(),
                kind: vector.kind,
                x: 0.0,
                y: 0.0,
            })
            .collect(),
        explained_variance_ratio: [0.0, 0.0],
    }
}

fn jacobi_eigen(mut matrix: Vec<Vec<f64>>) -> Result<(Vec<f64>, Vec<Vec<f64>>), InspectionError> {
    let size = matrix.len();
    let mut eigenvectors = vec![vec![0.0; size]; size];
    for (index, row) in eigenvectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..JACOBI_SWEEPS {
        for left in 0..size {
            for right in (left + 1)..size {
                let off_diagonal = matrix[left][right];
                let diagonal_scale = matrix[left][left]
                    .abs()
                    .max(matrix[right][right].abs())
                    .max(1.0);
                if off_diagonal.abs() <= f64::EPSILON * diagonal_scale * size.max(1) as f64 {
                    continue;
                }
                let tau = (matrix[right][right] - matrix[left][left]) / (2.0 * off_diagonal);
                let tangent = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = tangent * cosine;
                for index in 0..size {
                    if index != left && index != right {
                        let old_left = matrix[index][left];
                        let old_right = matrix[index][right];
                        matrix[index][left] = cosine * old_left - sine * old_right;
                        matrix[left][index] = matrix[index][left];
                        matrix[index][right] = sine * old_left + cosine * old_right;
                        matrix[right][index] = matrix[index][right];
                    }
                    let vector_left = eigenvectors[index][left];
                    let vector_right = eigenvectors[index][right];
                    eigenvectors[index][left] = cosine * vector_left - sine * vector_right;
                    eigenvectors[index][right] = sine * vector_left + cosine * vector_right;
                }
                matrix[left][left] -= tangent * off_diagonal;
                matrix[right][right] += tangent * off_diagonal;
                matrix[left][right] = 0.0;
                matrix[right][left] = 0.0;
            }
        }
    }
    let eigenvalues = (0..size).map(|index| matrix[index][index]).collect();
    if matrix
        .iter()
        .flatten()
        .chain(eigenvectors.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(InspectionError::IntegrityError);
    }
    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use serde_json::to_vec;

    use super::*;
    use crate::vector::VectorDimensions;

    fn vector(values: &[f64]) -> NormalizedVector {
        NormalizedVector::from_provider_f64(
            values,
            VectorDimensions::new(values.len() as u32).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn projection_is_order_independent_signed_and_byte_stable() {
        let query = vector(&[1.0, 0.0, 0.0]);
        let a = vector(&[0.0, 1.0, 0.0]);
        let b = vector(&[0.0, 0.0, 1.0]);
        let build = |reverse| {
            let mut evidence = vec![
                DiagnosticVector {
                    record_id: "b".into(),
                    kind: DiagnosticPointKindV1::Evidence,
                    vector: &b,
                },
                DiagnosticVector {
                    record_id: "a".into(),
                    kind: DiagnosticPointKindV1::Evidence,
                    vector: &a,
                },
            ];
            if reverse {
                evidence.reverse();
            }
            evidence.push(DiagnosticVector {
                record_id: "query".into(),
                kind: DiagnosticPointKindV1::Query,
                vector: &query,
            });
            project_pca_2("space", evidence).unwrap().unwrap()
        };
        let first = build(false);
        let second = build(true);
        assert_eq!(first, second);
        assert_eq!(first.points[0].kind, DiagnosticPointKindV1::Query);
        assert_eq!(first.points[1].record_id, "a");
        assert_eq!(first.points[2].record_id, "b");
        assert!((first.explained_variance_ratio[0] - 0.5).abs() < 1.0e-12);
        assert!((first.explained_variance_ratio[1] - 0.5).abs() < 1.0e-12);
        assert_eq!(to_vec(&first).unwrap(), to_vec(&build(false)).unwrap());
        assert!(
            first
                .points
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
    }

    #[test]
    fn sign_ties_use_the_lowest_loading_dimension() {
        let mut loading = [-1.0, 1.0];
        let mut coordinates = [2.0, -2.0];
        fix_component_sign(&mut loading, &mut coordinates);
        assert_eq!(loading, [1.0, -1.0]);
        assert_eq!(coordinates, [-2.0, 2.0]);
    }

    #[test]
    fn one_point_rank_deficiency_and_budget_omission_are_exact() {
        let query = vector(&[1.0, 0.0]);
        let one = project_pca_2(
            "space",
            vec![DiagnosticVector {
                record_id: "query".into(),
                kind: DiagnosticPointKindV1::Query,
                vector: &query,
            }],
        )
        .unwrap()
        .unwrap();
        assert_eq!(one.points[0].x, 0.0);
        assert_eq!(one.points[0].y, 0.0);
        assert_eq!(one.explained_variance_ratio, [0.0, 0.0]);
        assert_eq!(
            serde_json::to_string(&one).unwrap(),
            r#"{"algorithm":"pca_2","algorithm_version":1,"diagnostic_only":true,"vector_space_id":"space","points":[{"record_id":"query","kind":"query","x":0.0,"y":0.0}],"explained_variance_ratio":[0.0,0.0]}"#
        );

        assert!(project_pca_2("space", Vec::new()).unwrap().is_none());

        let orthogonal = vector(&[0.0, 1.0]);
        let rank_one = project_pca_2(
            "space",
            vec![
                DiagnosticVector {
                    record_id: "query".into(),
                    kind: DiagnosticPointKindV1::Query,
                    vector: &query,
                },
                DiagnosticVector {
                    record_id: "evidence".into(),
                    kind: DiagnosticPointKindV1::Evidence,
                    vector: &orthogonal,
                },
            ],
        )
        .unwrap()
        .unwrap();
        assert_eq!(rank_one.explained_variance_ratio, [1.0, 0.0]);
        assert!(rank_one.points.iter().all(|point| point.y == 0.0));
        assert!(rank_one.points[0].x > 0.0);
        assert!(rank_one.points[1].x < 0.0);

        let same = vector(&[1.0, 0.0]);
        let rank_zero = project_pca_2(
            "space",
            vec![
                DiagnosticVector {
                    record_id: "query".into(),
                    kind: DiagnosticPointKindV1::Query,
                    vector: &query,
                },
                DiagnosticVector {
                    record_id: "evidence".into(),
                    kind: DiagnosticPointKindV1::Evidence,
                    vector: &same,
                },
            ],
        )
        .unwrap()
        .unwrap();
        assert!(
            rank_zero
                .points
                .iter()
                .all(|point| point.x == 0.0 && point.y == 0.0)
        );

        let over = (0..=DIAGNOSTIC_PROJECTION_POINTS_MAX)
            .map(|index| DiagnosticVector {
                record_id: format!("{index:04}"),
                kind: DiagnosticPointKindV1::Evidence,
                vector: &query,
            })
            .collect();
        assert!(project_pca_2("space", over).unwrap().is_none());
    }
}
