# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _call(tir: Any, dtype: str, name: str, *args: Any) -> Any:
    return tir.Call(dtype, tir.op.Op.get(name), list(args))


def _access_ptr(tir: Any, dst_buffer: Any) -> Any:
    return _call(
        tir,
        "handle",
        "tl.access_ptr",
        tir.BufferLoad(dst_buffer, [tir.IntImm("int32", 0)]),
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 3),
    )


def _prim_func(tir: Any, *, name: str, body: Any, dst: Any, dst_buffer: Any) -> Any:
    return tir.PrimFunc([dst], body, buffer_map={dst: dst_buffer}).with_attr(
        "global_symbol", name
    )


# ====
@tilelang_case(name="scalar_atomic_add", category="op", tags=("memory",))
def scalar_atomic_add(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    body = tir.Evaluate(
        _call(
            tir,
            "handle",
            "tl.atomic_add_elem_op",
            _access_ptr(tir, dst_buffer),
            tir.IntImm("int32", 1),
        )
    )
    prim_func = _prim_func(
        tir,
        name="scalar_atomic_add",
        body=body,
        dst=dst,
        dst_buffer=dst_buffer,
    )
    return TileLangImportInput(source=prim_func, target="hip", name="scalar_atomic_add")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scalar_atomic_add") workgroup_size(1, 1, 1) @scalar_atomic_add(%dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xi32>
#   %const = scalar.constant 1 : i32
#   %c = index.constant 0 : index
#   view.atomic.reduce<addi> %const, %dst[%c] {ordering = relaxed, scope = device} : i32, view<4xi32>
#   kernel.return
# }


# ====
@tilelang_case(name="scalar_atomic_add_return", category="op", tags=("memory",))
def scalar_atomic_add_return(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    body = tir.BufferStore(
        dst_buffer,
        _call(
            tir,
            "int32",
            "tl.atomic_add_ret_elem_op",
            _access_ptr(tir, dst_buffer),
            tir.IntImm("int32", 1),
            tir.IntImm("int32", 2),
        ),
        [tir.IntImm("int32", 1)],
    )
    prim_func = _prim_func(
        tir,
        name="scalar_atomic_add_return",
        body=body,
        dst=dst,
        dst_buffer=dst_buffer,
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="scalar_atomic_add_return",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("scalar_atomic_add_return") workgroup_size(1, 1, 1) @scalar_atomic_add_return(%dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<4xi32>
#   %const = scalar.constant 1 : i32
#   %c = index.constant 0 : index
#   %atomic_add = view.atomic.rmw<addi> %const, %dst[%c] {ordering = acquire, scope = device} : i32, view<4xi32> -> i32
#   %c_2 = index.constant 1 : index
#   view.store %atomic_add, %dst[%c_2] : i32, view<4xi32>
#   kernel.return
# }
