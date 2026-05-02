# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _buffer_pair(tir: Any) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((4,), "float32", name="src")
    dst_buffer = tir.decl_buffer((4,), "float32", name="dst")
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
@tilelang_case(name="scoped_tl_assume", category="op", tags=("analysis",))
def scoped_tl_assume(tir: Any) -> TileLangImportInput:
    n = tir.Var("n", "int32")
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    index = tir.Var("i", "int32")
    body = tir.AttrStmt(
        tir.IntImm("int32", 0) < n,
        "tl.assume",
        tir.StringImm("n positive"),
        tir.For(
            index,
            tir.IntImm("int32", 0),
            n,
            tir.ForKind.SERIAL,
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [index]),
                [index],
            ),
        ),
    )
    prim_func = _prim_func(
        tir,
        name="scoped_tl_assume",
        params=[n, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="scoped_tl_assume")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scoped_tl_assume") workgroup_size(1, 1, 1) @scoped_tl_assume(%n: i32, %src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<4xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32>
#   %n_assumed = scalar.assume %n [lt(0, %n)] : i32
#   %c = index.constant 0 : index
#   %n_idx = index.cast %n_assumed : i32 to index
#   %i_ub = index.add %c, %n_idx : index
#   %c1 = index.constant 1 : index
#   scf.for %i = [%c to %i_ub step %c1] {
#     %load = view.load %src[%i] : view<4xf32> -> f32
#     view.store %load, %dst[%i] : f32, view<4xf32>
#     scf.yield
#   }
#   kernel.return
# }


# ====
@tilelang_case(name="effect_tir_assume", category="op", tags=("analysis",))
def effect_tir_assume(tir: Any) -> TileLangImportInput:
    n = tir.Var("n", "int32")
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    body = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_intrin(
                    "bool",
                    "tir.assume",
                    (n % tir.IntImm("int32", 16)) == tir.IntImm("int32", 0),
                )
            ),
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
                [tir.IntImm("int32", 0)],
            ),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="effect_tir_assume",
        params=[n, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="effect_tir_assume")


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
