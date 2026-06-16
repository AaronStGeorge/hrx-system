# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.gen.target.arch.amdgpu.lower.candidates import (
    amdgpu_atomic_candidates,
    amdgpu_compare_candidates,
)
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (
    dense_candidate_ranges,
)

_ATOMIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h"
_COMPARE_HEADER = "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"


def test_atomic_generator_emits_data_source_only() -> None:
    source = amdgpu_atomic_candidates._emit_source(public_header=_ATOMIC_HEADER)

    assert f'#include "{_ATOMIC_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuAtomicDescriptorCandidates[]" in source
    assert "kLoomAmdgpuAtomicDescriptorCandidateCount" in source
    assert "kLoomAmdgpuAtomicDescriptorCandidateRanges" in source
    assert ".memory_space" not in source
    assert ".address_form" not in source
    assert ".operation_kind" not in source
    assert ".atomic_kind" not in source
    assert ".value_kind" in source
    assert ".descriptor_ref" in source


def test_atomic_generator_builds_contiguous_candidate_ranges() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()
    ranges = amdgpu_atomic_candidates._candidate_ranges(candidates)

    assert ranges
    assert len(ranges) < len(candidates)
    covered_candidate_count = sum(candidate_count for _, _, candidate_count in ranges)
    assert covered_candidate_count == len(candidates)


def test_atomic_generator_rejects_missing_descriptor_ref() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()
    bad_candidate = replace(candidates[0], descriptor_key="amdgpu.missing")

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU atomic descriptor candidate table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_atomic_candidates._candidate_ranges((bad_candidate,))


def test_atomic_generator_rejects_noncontiguous_candidate_ranges() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()

    with pytest.raises(
        ValueError,
        match=("AMDGPU atomic descriptor candidate table must be contiguous by range key"),
    ):
        amdgpu_atomic_candidates._candidate_ranges((candidates[0], candidates[-1], candidates[0]))


def test_candidate_range_validation_rejects_uint16_overflow() -> None:
    with pytest.raises(ValueError, match="candidate index 65536"):
        dense_candidate_ranges(
            range(65537),
            lambda value: value,
            owner="AMDGPU test candidates",
        )


def test_compare_generator_emits_data_source_only() -> None:
    source = amdgpu_compare_candidates._emit_source(public_header=_COMPARE_HEADER)

    assert f'#include "{_COMPARE_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuVectorCmpiCompareDescriptorCandidates" in source
    assert "kLoomAmdgpuScalarCmpfCompareDescriptorCandidates" in source
    assert "kLoomAmdgpuVectorCmpfCompareDescriptorCandidates" in source
    assert ".op_kind" not in source
    assert ".predicate" not in source
    assert "[LOOM_VECTOR_CMPI_PREDICATE_EQ]" in source
    assert "[LOOM_SCALAR_CMPF_PREDICATE_OLT]" in source
    assert "[LOOM_VECTOR_CMPF_PREDICATE_OLT]" in source


def test_compare_generator_covers_predicate_rows() -> None:
    candidates = amdgpu_compare_candidates._compare_candidates()

    assert candidates
    assert any(family.source_op_name == "vector.cmpi" and predicate == "eq" for family, predicate, _ in candidates)
    assert any(family.source_op_name == "vector.cmpf" and predicate == "olt" for family, predicate, _ in candidates)


def test_compare_generator_rejects_missing_descriptor_ref() -> None:
    descriptor_ref_keys = tuple(key for key in amdgpu_compare_candidates.amdgpu_descriptor_ref_keys() if key != "amdgpu.v_cmp_eq_i32")
    original_descriptor_ref_keys = amdgpu_compare_candidates.amdgpu_descriptor_ref_keys
    try:
        amdgpu_compare_candidates.amdgpu_descriptor_ref_keys = lambda: descriptor_ref_keys
        with pytest.raises(
            ValueError,
            match=(
                r"AMDGPU compare candidate vector\.cmpi eq requires missing "
                r"descriptor refs: amdgpu\.v_cmp_eq_i32"
            ),
        ):
            amdgpu_compare_candidates._compare_candidates()
    finally:
        amdgpu_compare_candidates.amdgpu_descriptor_ref_keys = original_descriptor_ref_keys
