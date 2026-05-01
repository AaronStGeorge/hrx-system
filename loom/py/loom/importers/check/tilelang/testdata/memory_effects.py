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
    Call,
    Evaluate,
    IntImm,
    PrimFunc,
    Var,
)


# ====
@tilelang_case(name="scalar_atomic_add", category="op", tags=("memory",))
def scalar_atomic_add() -> TileLangImportInput:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (4,), "int32")
    access = Call(
        "tl.access_ptr",
        [BufferLoad(dst_buffer, [IntImm(0)], "int32"), IntImm(1), IntImm(3)],
        "handle",
    )
    body = Evaluate(Call("tl.atomic_add_elem_op", [access, IntImm(1)], "handle"))
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "scalar_atomic_add"},
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
def scalar_atomic_add_return() -> TileLangImportInput:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (4,), "int32")
    access = Call(
        "tl.access_ptr",
        [BufferLoad(dst_buffer, [IntImm(0)], "int32"), IntImm(1), IntImm(3)],
        "handle",
    )
    body = Evaluate(
        Call("tl.atomic_add_ret_elem_op", [access, IntImm(1), IntImm(2)], "int32")
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "scalar_atomic_add_return"},
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
#   kernel.return
# }
