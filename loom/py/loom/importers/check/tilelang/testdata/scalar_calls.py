# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _buffer_pair(
    tir: Any,
    *,
    dtype: str = "float32",
    shape: tuple[int, ...] = (4,),
) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer(shape, dtype, name="src")
    dst_buffer = tir.decl_buffer(shape, dtype, name="dst")
    return src, dst, src_buffer, dst_buffer


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


# ====
@tilelang_case(name="scalar_calls", category="op", tags=("call", "scalar"))
def scalar_calls(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    load = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])
    body = tir.IfThenElse(
        tir.isfinite(load),
        tir.BufferStore(
            dst_buffer,
            tir.sqrt(tir.abs(load)) + tir.sigmoid(load),
            [tir.IntImm("int32", 0)],
        ),
        tir.BufferStore(dst_buffer, tir.exp(load), [tir.IntImm("int32", 0)]),
    )
    prim_func = _prim_func(
        tir,
        name="scalar_calls",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="scalar_calls")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scalar_calls") workgroup_size(1, 1, 1) @scalar_calls(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32>
#   %c = index.constant 0 : index
#   %load = view.load %src[%c] : view<4xf32> -> f32
#   %isfinitef = scalar.isfinitef %load : f32
#   scf.if %isfinitef {
#     %c_2 = index.constant 0 : index
#     %load_2 = view.load %src[%c_2] : view<4xf32> -> f32
#     %absf = scalar.absf %load_2 : f32
#     %sqrtf = scalar.sqrtf %absf : f32
#     %c_3 = index.constant 0 : index
#     %load_3 = view.load %src[%c_3] : view<4xf32> -> f32
#     %one = scalar.constant 1.0 : f32
#     %sigmoid_neg = scalar.negf %load_3 : f32
#     %sigmoid_exp = scalar.expf %sigmoid_neg : f32
#     %sigmoid_den = scalar.addf %one, %sigmoid_exp : f32
#     %sigmoid = scalar.divf %one, %sigmoid_den : f32
#     %addf = scalar.addf %sqrtf, %sigmoid : f32
#     %c_4 = index.constant 0 : index
#     view.store %addf, %dst[%c_4] : f32, view<4xf32>
#     scf.yield
#   } else {
#     %c_5 = index.constant 0 : index
#     %load_4 = view.load %src[%c_5] : view<4xf32> -> f32
#     %expf = scalar.expf %load_4 : f32
#     %c_6 = index.constant 0 : index
#     view.store %expf, %dst[%c_6] : f32, view<4xf32>
#     scf.yield
#   }
#   kernel.return
# }


# ====
@tilelang_case(name="bitwise_not", category="op", tags=("call", "integer"))
def bitwise_not(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, dtype="int32")
    body = tir.BufferStore(
        dst_buffer,
        tir.bitwise_not(tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="bitwise_not",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="bitwise_not")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("bitwise_not") workgroup_size(1, 1, 1) @bitwise_not(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xi32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xi32>
#   %c = index.constant 0 : index
#   %load = view.load %src[%c] : view<4xi32> -> i32
#   %all_ones = scalar.constant -1 : i32
#   %noti = scalar.xori %load, %all_ones : i32
#   %c_2 = index.constant 0 : index
#   view.store %noti, %dst[%c_2] : i32, view<4xi32>
#   kernel.return
# }


# ====
@tilelang_case(name="dynamic_loop_bound", category="op", tags=("index", "loop"))
def dynamic_loop_bound(tir: Any) -> TileLangImportInput:
    n = tir.Var("n", "int32")
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(8,))
    index = tir.Var("i", "int32")
    body = tir.For(
        index,
        tir.IntImm("int32", 0),
        n,
        tir.ForKind.SERIAL,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [index]), [index]),
    )
    prim_func = _prim_func(
        tir,
        name="dynamic_loop_bound",
        params=[n, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="dynamic_loop_bound",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("dynamic_loop_bound") workgroup_size(1, 1, 1) @dynamic_loop_bound(%n: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<8xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<8xf32>
#   %c = index.constant 0 : index
#   %n_idx = index.cast %n : i32 to index
#   %i_ub = index.add %c, %n_idx : index
#   %c1 = index.constant 1 : index
#   scf.for %i = [%c to %i_ub step %c1] {
#     %load = view.load %src[%i] : view<8xf32> -> f32
#     view.store %load, %dst[%i] : f32, view<8xf32>
#     scf.yield
#   }
#   kernel.return
# }


# ====
@tilelang_case(name="ceildiv_loop", category="op", tags=("call", "index"))
def ceildiv_loop(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(8,))
    index = tir.Var("i", "int32")
    body = tir.For(
        index,
        tir.IntImm("int32", 0),
        tir.ceildiv(tir.IntImm("int32", 7), tir.IntImm("int32", 4)),
        tir.ForKind.SERIAL,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [index]), [index]),
    )
    prim_func = _prim_func(
        tir,
        name="ceildiv_loop",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="ceildiv_loop")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("ceildiv_loop") workgroup_size(1, 1, 1) @ceildiv_loop(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<8xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<8xf32>
#   %c = index.constant 0 : index
#   %c_2 = index.constant 2 : index
#   %i_ub = index.add %c, %c_2 : index
#   %c1 = index.constant 1 : index
#   scf.for %i = [%c to %i_ub step %c1] {
#     %load = view.load %src[%i] : view<8xf32> -> f32
#     view.store %load, %dst[%i] : f32, view<8xf32>
#     scf.yield
#   }
#   kernel.return
# }
