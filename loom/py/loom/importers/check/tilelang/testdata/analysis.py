# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    EQ,
    LT,
    AttrStmt,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Evaluate,
    FloorMod,
    IntImm,
    PrimFunc,
    SeqStmt,
    Var,
)


# ====
@tilelang_case(name="scoped_tl_assume", category="op", tags=("analysis", "assume"))
def scoped_tl_assume() -> TileLangImportInput:
    n, src, dst = Var("n"), Var("src"), Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = AttrStmt(
        LT(IntImm(0), n),
        "tl.assume",
        "Buffer shape should be greater than 0",
        BufferStore(dst_buffer, BufferLoad(src_buffer, [IntImm(0)]), [IntImm(0)]),
    )
    prim_func = PrimFunc(
        [n, src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "scoped_tl_assume"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="scoped_tl_assume",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scoped_tl_assume") workgroup_size(1, 1, 1) @scoped_tl_assume(%n: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32>
#   %n_assumed = scalar.assume %n [lt(0, %n)] : i32
#   %c = index.constant 0 : index
#   %load = view.load %src[%c] : view<4xf32> -> f32
#   %c_2 = index.constant 0 : index
#   view.store %load, %dst[%c_2] : f32, view<4xf32>
#   kernel.return
# }


# ====
@tilelang_case(name="effect_tir_assume", category="op", tags=("analysis", "assume"))
def effect_tir_assume() -> TileLangImportInput:
    n, src, dst = Var("n"), Var("src"), Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    body = SeqStmt(
        [
            Evaluate(
                Call(
                    "tir.assume",
                    [EQ(FloorMod(n, IntImm(16)), IntImm(0))],
                    "bool",
                )
            ),
            BufferStore(dst_buffer, BufferLoad(src_buffer, [IntImm(0)]), [IntImm(0)]),
        ]
    )
    prim_func = PrimFunc(
        [n, src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "effect_tir_assume"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="effect_tir_assume",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("effect_tir_assume") workgroup_size(1, 1, 1) @effect_tir_assume(%n: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32>
#   %n_assumed = scalar.assume %n [mul(%n, 16)] : i32
#   %c = index.constant 0 : index
#   %load = view.load %src[%c] : view<4xf32> -> f32
#   %c_2 = index.constant 0 : index
#   view.store %load, %dst[%c_2] : f32, view<4xf32>
#   kernel.return
# }
