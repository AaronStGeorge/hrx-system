# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amdgpu.lower.candidates import (
    amdgpu_atomic_candidates,
    amdgpu_compare_candidates,
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


def test_compare_generator_emits_data_source_only() -> None:
    source = amdgpu_compare_candidates._emit_source(public_header=_COMPARE_HEADER)

    assert f'#include "{_COMPARE_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuCompareDescriptorCandidates[]" in source
    assert "kLoomAmdgpuCompareDescriptorCandidateCount" in source


def test_compare_generator_covers_predicate_rows() -> None:
    candidates = amdgpu_compare_candidates._compare_candidates()

    assert candidates
    assert any(family.source_op_name == "vector.cmpi" and predicate == "eq" for family, predicate, _ in candidates)
    assert any(family.source_op_name == "vector.cmpf" and predicate == "olt" for family, predicate, _ in candidates)
