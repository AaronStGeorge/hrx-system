# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    GE,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Evaluate,
    FloatImm,
    IfThenElse,
    IntImm,
    PrimFunc,
    SeqStmt,
    Var,
)


# ====
@tilelang_case(name="thread_return_prefix_guard", category="op", tags=("control",))
def thread_return_prefix_guard() -> TileLangImportInput:
    n, i, src, dst = Var("n"), Var("i"), Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "float32")
    dst_buffer = Buffer("dst", (16,), "float32")
    body = SeqStmt(
        [
            IfThenElse(
                GE(i, n),
                Evaluate(Call("tir.thread_return", [], "void")),
                None,
            ),
            BufferStore(dst_buffer, BufferLoad(src_buffer, [i]), [i]),
        ]
    )
    prim_func = PrimFunc(
        [n, i, src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "thread_return_prefix_guard"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="thread_return_prefix_guard",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("thread_return_prefix_guard") workgroup_size(1, 1, 1) @thread_return_prefix_guard(%n: i32, %i: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf32>
#   %cmp = scalar.cmpi sge, %i, %n : i32
#   kernel.exit %cmp : i1
#   %i_idx = index.cast %i : i32 to index
#   %load = view.load %src[%i_idx] : view<16xf32> -> f32
#   view.store %load, %dst[%i_idx] : f32, view<16xf32>
#   kernel.return
# }


# ====
@tilelang_case(
    name="thread_return_prefix_guard_with_effects",
    category="op",
    tags=("control",),
)
def thread_return_prefix_guard_with_effects() -> TileLangImportInput:
    n, i, src, dst = Var("n"), Var("i"), Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "float32")
    dst_buffer = Buffer("dst", (16,), "float32")
    body = SeqStmt(
        [
            IfThenElse(
                GE(i, n),
                SeqStmt(
                    [
                        BufferStore(dst_buffer, FloatImm(0.0), [IntImm(0)]),
                        Evaluate(Call("tir.thread_return", [], "void")),
                    ]
                ),
                None,
            ),
            BufferStore(dst_buffer, BufferLoad(src_buffer, [i]), [i]),
        ]
    )
    prim_func = PrimFunc(
        [n, i, src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "thread_return_prefix_guard_with_effects"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="thread_return_prefix_guard_with_effects",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("thread_return_prefix_guard_with_effects") workgroup_size(1, 1, 1) @thread_return_prefix_guard_with_effects(%n: i32, %i: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf32>
#   %cmp = scalar.cmpi sge, %i, %n : i32
#   kernel.exit %cmp : i1 {
#     %const = scalar.constant 0.0 : f32
#     %c = index.constant 0 : index
#     view.store %const, %dst[%c] : f32, view<16xf32>
#   }
#   %i_idx = index.cast %i : i32 to index
#   %load = view.load %src[%i_idx] : view<16xf32> -> f32
#   view.store %load, %dst[%i_idx] : f32, view<16xf32>
#   kernel.return
# }
