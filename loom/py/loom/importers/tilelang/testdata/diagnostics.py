# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


# ====
# ERROR@+1: "no registered converter for `StringImm`" "unsupported_marker"
@tilelang_case(name="unsupported_string_evaluate", category="diagnostic")
def unsupported_string_evaluate(tir: Any) -> TileLangImportInput:
    source = tir.Var("source", "handle")
    source_buffer = tir.decl_buffer((1,), "float32", name="source")
    body = tir.Evaluate(tir.StringImm("unsupported_marker"))
    prim_func = tir.PrimFunc(
        [source],
        body,
        buffer_map={source: source_buffer},
    ).with_attr("global_symbol", "unsupported_string_evaluate")
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="unsupported_string_evaluate",
    )
