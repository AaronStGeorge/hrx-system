# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.arch.amdgpu import amdgpu_config_tables


def test_encoding_field_ids_fragment_is_row_data_only() -> None:
    source = amdgpu_config_tables._emit_encoding_field_ids()

    assert "typedef " not in source
    assert "enum " not in source
    assert "#ifndef " not in source
    assert "#define " not in source
    assert "#include " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source

    lines = source.splitlines()
    assert lines[0] == "LOOM_AMDGPU_ENCODING_FIELD(LOOM_AMDGPU_ENCODING_FIELD_A16, 1)"
    assert lines[-1] == "LOOM_AMDGPU_ENCODING_FIELD(LOOM_AMDGPU_ENCODING_FIELD_MATRIX_B_SCALE_FMT, 195)"
