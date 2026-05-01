# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    Block,
    Buffer,
    BufferLoad,
    BufferStore,
    FloatImm,
    IntImm,
    PrimFunc,
    SeqStmt,
    Var,
)


# ====
@tilelang_case(name="shared_block_alloc", category="composition", tags=("memory",))
def shared_block_alloc() -> TileLangImportInput:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (4,), "float32")
    scratch = Buffer("scratch", (4,), "float32", scope="shared.dyn")
    body = Block(
        SeqStmt(
            [
                BufferStore(scratch, FloatImm(1.0), [IntImm(0)]),
                BufferStore(dst_buffer, BufferLoad(scratch, [IntImm(0)]), [IntImm(0)]),
            ]
        ),
        alloc_buffers=[scratch],
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "shared_block_alloc"},
    )
    return TileLangImportInput(
        source=prim_func, target="hip", name="shared_block_alloc"
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("shared_block_alloc") workgroup_size(1, 1, 1) @shared_block_alloc(%dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32>
#   %scratch_bytes = index.constant 16 : offset
#   %scratch_buffer = buffer.alloca %scratch_bytes {base_alignment = 4, memory_space = workgroup} : buffer
#   %scratch = buffer.view %scratch_buffer[%c0_bytes] : buffer -> view<4xf32>
#   %const = scalar.constant 1.0 : f32
#   %c = index.constant 0 : index
#   view.store %const, %scratch[%c] : f32, view<4xf32>
#   %c_2 = index.constant 0 : index
#   %load = view.load %scratch[%c_2] : view<4xf32> -> f32
#   %c_3 = index.constant 0 : index
#   view.store %load, %dst[%c_3] : f32, view<4xf32>
#   kernel.return
# }


# ====
@tilelang_case(name="private_block_alloc", category="composition", tags=("memory",))
def private_block_alloc() -> TileLangImportInput:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (4,), "int32")
    scratch = Buffer("scratch", (4,), "int32", scope="local")
    body = Block(
        SeqStmt(
            [
                BufferStore(scratch, IntImm(7), [IntImm(0)]),
                BufferStore(
                    dst_buffer, BufferLoad(scratch, [IntImm(0)], "int32"), [IntImm(0)]
                ),
            ]
        ),
        alloc_buffers=[scratch],
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "private_block_alloc"},
    )
    return TileLangImportInput(
        source=prim_func, target="hip", name="private_block_alloc"
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("private_block_alloc") workgroup_size(1, 1, 1) @private_block_alloc(%dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xi32>
#   %scratch_bytes = index.constant 16 : offset
#   %scratch_buffer = buffer.alloca %scratch_bytes {base_alignment = 4, memory_space = private} : buffer
#   %scratch = buffer.view %scratch_buffer[%c0_bytes] : buffer -> view<4xi32>
#   %const = scalar.constant 7 : i32
#   %c = index.constant 0 : index
#   view.store %const, %scratch[%c] : i32, view<4xi32>
#   %c_2 = index.constant 0 : index
#   %load = view.load %scratch[%c_2] : view<4xi32> -> i32
#   %c_3 = index.constant 0 : index
#   view.store %load, %dst[%c_3] : i32, view<4xi32>
#   kernel.return
# }
