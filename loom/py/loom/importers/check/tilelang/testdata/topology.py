# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    Add,
    AttrStmt,
    Buffer,
    BufferLoad,
    BufferStore,
    For,
    IntImm,
    PrimFunc,
    SeqStmt,
    ThreadAxis,
    Var,
)


# ====
@tilelang_case(name="launch_thread_attrs", category="op", tags=("topology",))
def launch_thread_attrs() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    bx = Var("bx")
    tx = Var("tx")
    src_buffer = Buffer("src", (1024,), "float32")
    dst_buffer = Buffer("dst", (1024,), "float32")
    index = Add(
        Add(
            Add(bx, bx, "int32"),
            Add(tx, tx, "int32"),
            "int32",
        ),
        IntImm(0),
        "int32",
    )
    body = AttrStmt(
        ThreadAxis("blockIdx.x", bx),
        "thread_extent",
        IntImm(8),
        AttrStmt(
            ThreadAxis("threadIdx.x", tx),
            "thread_extent",
            IntImm(64),
            BufferStore(
                dst_buffer,
                BufferLoad(src_buffer, [index]),
                [index],
            ),
        ),
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "launch_thread_attrs"},
    )
    return TileLangImportInput(
        source=prim_func, target="hip", name="launch_thread_attrs"
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("launch_thread_attrs") workgroup_size(64, 1, 1) @launch_thread_attrs(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<1024xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<1024xf32>
#   %bx = kernel.workgroup.id<x> : index
#   %tx = kernel.workitem.id<x> : index
#   %add = index.add %bx, %bx : index
#   %add_2 = index.add %tx, %tx : index
#   %add_3 = index.add %add, %add_2 : index
#   %c = index.constant 0 : index
#   %add_4 = index.add %add_3, %c : index
#   %load = view.load %src[%add_4] : view<1024xf32> -> f32
#   view.store %load, %dst[%add_4] : f32, view<1024xf32>
#   kernel.return
# }


# ====
@tilelang_case(name="thread_binding_loop", category="op", tags=("topology",))
def thread_binding_loop() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    tx = Var("tx")
    src_buffer = Buffer("src", (128,), "float32")
    dst_buffer = Buffer("dst", (128,), "float32")
    body = For(
        tx,
        IntImm(0),
        IntImm(128),
        BufferStore(dst_buffer, BufferLoad(src_buffer, [tx]), [tx]),
        thread_binding=ThreadAxis("threadIdx.x", tx),
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        SeqStmt([body]),
        attrs={"global_symbol": "thread_binding_loop"},
    )
    return TileLangImportInput(
        source=prim_func, target="hip", name="thread_binding_loop"
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("thread_binding_loop") workgroup_size(128, 1, 1) @thread_binding_loop(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<128xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<128xf32>
#   %tx = kernel.workitem.id<x> : index
#   %load = view.load %src[%tx] : view<128xf32> -> f32
#   view.store %load, %dst[%tx] : f32, view<128xf32>
#   kernel.return
# }
