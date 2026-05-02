# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _prim_func(
    tir: Any,
    *,
    name: str,
    params: list[Any],
    body: Any,
    buffer_map: dict[Any, Any],
) -> Any:
    return tir.PrimFunc(params, body, buffer_map=buffer_map).with_attr(
        "global_symbol", name
    )


def _ramp4(tir: Any) -> Any:
    return tir.Ramp(tir.IntImm("int32", 0), tir.IntImm("int32", 1), 4)


# ====
@tilelang_case(name="vector_float_trunc_store", category="op", tags=("vector", "cast"))
def vector_float_trunc_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float16", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.Cast("float16x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_float_trunc_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
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
@tilelang_case(name="scalar_reinterpret_store", category="op", tags=("cast",))
def scalar_reinterpret_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((4,), "float32", name="src")
    dst_buffer = tir.decl_buffer((4,), "uint32", name="dst")
    body = tir.BufferStore(
        dst_buffer,
        tir.reinterpret(
            "uint32",
            tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
        ),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="scalar_reinterpret_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
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
@tilelang_case(name="vector_reinterpret_store", category="op", tags=("vector", "cast"))
def vector_reinterpret_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "uint32", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.reinterpret("uint32x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_reinterpret_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
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
@tilelang_case(name="vector_int_to_float_store", category="op", tags=("vector", "cast"))
def vector_int_to_float_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "int32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float32", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.Cast("float32x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_int_to_float_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
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
