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
        target="hip -mcpu=gfx1100",
        name="unsupported_string_evaluate",
    )


# ====
# ERROR@+1: "replication `all` requires cross-thread allreduce import"
@tilelang_case(
    name="tileop_finalize_reducer_all",
    category="diagnostic",
    tags=("tileop", "finalize_reducer"),
)
def tileop_finalize_reducer_all(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_finalize_reducer_all_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(4,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=4):
            reducer = T.alloc_reducer(
                (4,),
                T.float32,
                "sum",
                replication="all",
            )
            T.fill(reducer, 0.0)
            for i in T.serial(0, 4):
                reducer[i] = src[i]
            T.finalize_reducer(reducer)
            T.copy(reducer, dst)

    return TileLangImportInput(
        source=tileop_finalize_reducer_all_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_finalize_reducer_all_kernel",
    )
