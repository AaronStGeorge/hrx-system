# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.error.type import ERR_TYPE_001
from loom.gen.c_errors import generate_error_tables_c


def test_generate_error_tables_keeps_definitions_private() -> None:
    tables_c = generate_error_tables_c([ERR_TYPE_001])

    assert "static const loom_error_def_t loom_err_type_001" in tables_c
    assert "extern const loom_error_def_t" not in tables_c
    assert "loom_error_def_lookup_ref" in tables_c
