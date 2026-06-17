# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amdgpu.refs import amdgpu_target_refs


def test_target_refs_header_is_constant_fragment() -> None:
    source = amdgpu_target_refs._emit_tables_header()

    assert "typedef " not in source
    assert "extern " not in source
    assert "loom_amdgpu_descriptor_ref_ordinal" not in source
    assert "loom/codegen/low/descriptors.h" not in source
    assert "#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32" in source
