# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from collections.abc import Mapping, Sequence

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


class Var:
    def __init__(self, name: str, dtype: str = "int32") -> None:
        self.name = name
        self.dtype = dtype

    def __repr__(self) -> str:
        return self.name


class Buffer:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return self.name


class PrimFunc:
    def __init__(
        self,
        params: Sequence[Var],
        buffer_map: Mapping[Var, Buffer],
        body: object,
        attrs: Mapping[str, object] | None = None,
    ) -> None:
        self.params = params
        self.buffer_map = buffer_map
        self.body = body
        self.attrs = {} if attrs is None else attrs


class For:
    def __init__(
        self,
        loop_var: Var,
        minimum: object,
        extent: object,
        body: object,
    ) -> None:
        self.loop_var = loop_var
        self.min = minimum
        self.extent = extent
        self.body = body


class IfThenElse:
    def __init__(self, condition: object, then_case: object, else_case: object) -> None:
        self.condition = condition
        self.then_case = then_case
        self.else_case = else_case


class BufferLoad:
    def __init__(
        self,
        buffer: Buffer,
        indices: Sequence[object],
        dtype: str = "float32",
    ) -> None:
        self.buffer = buffer
        self.indices = tuple(indices)
        self.dtype = dtype


class BufferStore:
    def __init__(
        self,
        buffer: Buffer,
        value: object,
        indices: Sequence[object],
    ) -> None:
        self.buffer = buffer
        self.value = value
        self.indices = tuple(indices)


class IntImm:
    def __init__(self, value: int, dtype: str = "int32") -> None:
        self.value = value
        self.dtype = dtype


class Add:
    def __init__(self, lhs: object, rhs: object, dtype: str = "float32") -> None:
        self.a = lhs
        self.b = rhs
        self.dtype = dtype


class Op:
    def __init__(self, name: str) -> None:
        self.name = name


class Call:
    def __init__(
        self,
        op_name: str,
        args: Sequence[object],
        dtype: str = "float32",
        annotations: Mapping[str, object] | None = None,
    ) -> None:
        self.op = Op(op_name)
        self.args = tuple(args)
        self.dtype = dtype
        self.annotations = {} if annotations is None else dict(annotations)

    def __repr__(self) -> str:
        return f"{self.op.name}(...)"


# ====
@tilelang_case(name="scalar_calls", category="op", tags=("call", "scalar"))
def scalar_calls() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (4,), "float32")
    dst_buffer = Buffer("dst", (4,), "float32")
    load = BufferLoad(src_buffer, [IntImm(0)])
    body = IfThenElse(
        Call("tir.isfinite", [load], "bool"),
        BufferStore(
            dst_buffer,
            Add(
                Call("tir.sqrt", [Call("tir.abs", [load])]),
                Call("tir.sigmoid", [load]),
            ),
            [IntImm(0)],
        ),
        BufferStore(dst_buffer, Call("tir.exp", [load]), [IntImm(0)]),
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "scalar_calls"},
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
#     %absf = scalar.absf %load : f32
#     %sqrtf = scalar.sqrtf %absf : f32
#     %one = scalar.constant 1.0 : f32
#     %sigmoid_neg = scalar.negf %load : f32
#     %sigmoid_exp = scalar.expf %sigmoid_neg : f32
#     %sigmoid_den = scalar.addf %one, %sigmoid_exp : f32
#     %sigmoid = scalar.divf %one, %sigmoid_den : f32
#     %addf = scalar.addf %sqrtf, %sigmoid : f32
#     %c_2 = index.constant 0 : index
#     view.store %addf, %dst[%c_2] : f32, view<4xf32>
#     scf.yield
#   } else {
#     %expf = scalar.expf %load : f32
#     %c_3 = index.constant 0 : index
#     view.store %expf, %dst[%c_3] : f32, view<4xf32>
#     scf.yield
#   }
#   kernel.return
# }


# ====
@tilelang_case(name="ceildiv_loop", category="op", tags=("call", "index"))
def ceildiv_loop() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    index = Var("i", "int32")
    src_buffer = Buffer("src", (8,), "float32")
    dst_buffer = Buffer("dst", (8,), "float32")
    body = For(
        index,
        IntImm(0),
        Call("tir.ceildiv", [IntImm(7), IntImm(4)], "int32"),
        BufferStore(dst_buffer, BufferLoad(src_buffer, [index]), [index]),
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "ceildiv_loop"},
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
#   %c_2 = index.constant 7 : index
#   %c_3 = index.constant 4 : index
#   %c1 = index.constant 1 : index
#   %ceil_rhs_minus_one = index.sub %c_3, %c1 : index
#   %ceil_adjusted = index.add %c_2, %ceil_rhs_minus_one : index
#   %ceildiv = index.div %ceil_adjusted, %c_3 : index
#   %i_ub = index.add %c, %ceildiv : index
#   scf.for %i = [%c to %i_ub step %c1] {
#     %load = view.load %src[%i] : view<8xf32> -> f32
#     view.store %load, %dst[%i] : f32, view<8xf32>
#     scf.yield
#   }
#   kernel.return
# }
