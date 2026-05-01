# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Cast,
    IntImm,
    PrimFunc,
    Ramp,
    Var,
)


# ====
@tilelang_case(name="vector_float_trunc_store", category="op", tags=("vector",))
def vector_float_trunc_store() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "float32")
    dst_buffer = Buffer("dst", (16,), "float16")
    body = BufferStore(
        dst_buffer,
        Cast(
            BufferLoad(src_buffer, [Ramp(IntImm(0), IntImm(1), 4)], "float32x4"),
            "float16x4",
        ),
        [Ramp(IntImm(0), IntImm(1), 4)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "vector_float_trunc_store"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="vector_float_trunc_store",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("vector_float_trunc_store") workgroup_size(1, 1, 1) @vector_float_trunc_store(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf16>
#   %c = index.constant 0 : index
#   %load = vector.load %src[%c] : view<16xf32> -> vector<4xf32>
#   %fptrunc = vector.fptrunc %load : vector<4xf32> to vector<4xf16>
#   %c_2 = index.constant 0 : index
#   vector.store %fptrunc, %dst[%c_2] : vector<4xf16>, view<16xf16>
#   kernel.return
# }


# ====
@tilelang_case(
    name="scalar_reinterpret_store",
    category="op",
    tags=("call", "scalar"),
)
def scalar_reinterpret_store() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "uint32")
    body = BufferStore(
        dst_buffer,
        Call(
            "tir.reinterpret",
            [BufferLoad(src_buffer, [IntImm(0)], "float32")],
            "uint32",
        ),
        [IntImm(0)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "scalar_reinterpret_store"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="scalar_reinterpret_store",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scalar_reinterpret_store") workgroup_size(1, 1, 1) @scalar_reinterpret_store(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xi32>
#   %c = index.constant 0 : index
#   %load = view.load %src[%c] : view<4xf32> -> f32
#   %bitcast = scalar.bitcast %load : f32 to i32
#   %c_2 = index.constant 0 : index
#   view.store %bitcast, %dst[%c_2] : i32, view<4xi32>
#   kernel.return
# }


# ====
@tilelang_case(
    name="vector_reinterpret_store",
    category="op",
    tags=("call", "vector"),
)
def vector_reinterpret_store() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "float32")
    dst_buffer = Buffer("dst", (16,), "uint32")
    body = BufferStore(
        dst_buffer,
        Call(
            "tir.reinterpret",
            [BufferLoad(src_buffer, [Ramp(IntImm(0), IntImm(1), 4)], "float32x4")],
            "uint32x4",
        ),
        [Ramp(IntImm(0), IntImm(1), 4)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "vector_reinterpret_store"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="vector_reinterpret_store",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("vector_reinterpret_store") workgroup_size(1, 1, 1) @vector_reinterpret_store(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xi32>
#   %c = index.constant 0 : index
#   %load = vector.load %src[%c] : view<16xf32> -> vector<4xf32>
#   %bitcast = vector.bitcast %load : vector<4xf32> to vector<4xi32>
#   %c_2 = index.constant 0 : index
#   vector.store %bitcast, %dst[%c_2] : vector<4xi32>, view<16xi32>
#   kernel.return
# }


# ====
@tilelang_case(name="vector_int_to_float_store", category="op", tags=("vector",))
def vector_int_to_float_store() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "int32")
    dst_buffer = Buffer("dst", (16,), "float32")
    body = BufferStore(
        dst_buffer,
        Cast(
            BufferLoad(src_buffer, [Ramp(IntImm(0), IntImm(1), 4)], "int32x4"),
            "float32x4",
        ),
        [Ramp(IntImm(0), IntImm(1), 4)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "vector_int_to_float_store"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="vector_int_to_float_store",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("vector_int_to_float_store") workgroup_size(1, 1, 1) @vector_int_to_float_store(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xi32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf32>
#   %c = index.constant 0 : index
#   %load = vector.load %src[%c] : view<16xi32> -> vector<4xi32>
#   %sitofp = vector.sitofp %load : vector<4xi32> to vector<4xf32>
#   %c_2 = index.constant 0 : index
#   vector.store %sitofp, %dst[%c_2] : vector<4xf32>, view<16xf32>
#   kernel.return
# }
