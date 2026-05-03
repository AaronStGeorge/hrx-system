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
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="scoped_tl_assume"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scoped_tl_assume") @scoped_tl_assume(%n: i32, %src: buffer, %dst: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_view = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32, %layout>
  %n_assumed = scalar.assume %n [lt(0, %n)] : i32
  %c0 = index.constant 0 : index
  %n_idx = index.cast %n_assumed : i32 to index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %n_idx step %c1] {
    %load = view.load %src_view[%i] : view<4xf32, %layout> -> f32
    view.store %load, %dst_view[%i] : f32, view<4xf32, %layout>
    scf.yield
  }
  kernel.return
}
"""


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
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="effect_tir_assume"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("effect_tir_assume") @effect_tir_assume(%n: i32, %src: buffer, %dst: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_view = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32, %layout>
  %n_assumed = scalar.assume %n [mul(%n, 16)] : i32
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  view.store %load, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="mixed_address_scalar_assume",
    category="op",
    tags=("analysis", "tilelang"),
)
def mixed_address_scalar_assume(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_mixed_address_scalar_assume() -> Any:
        n = T.dynamic("n")

        @T.prim_func  # type: ignore[untyped-decorator]
        def mixed_address_scalar_assume(
            src: T.Tensor[(n,), T.float32],
            dst: T.Tensor[(n,), T.float32],
        ) -> None:
            with T.Kernel(n, threads=1) as (pid,):
                T.assume(pid < n)
                dst[pid] = src[pid]

        return mixed_address_scalar_assume

    return TileLangImportInput(
        source=get_mixed_address_scalar_assume,
        target="hip -mcpu=gfx1100",
        name="mixed_address_scalar_assume",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mixed_address_scalar_assume") @mixed_address_scalar_assume(%src_handle: buffer, %dst_handle: buffer, %n: i32) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<[%n]xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<[%n]xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %n_idx = index.cast %n : i32 to index
  %bx_assumed, %n_assumed = index.assume %bx, %n_idx [lt(%bx, %n_idx)] : index, index
  %load = view.load %src[%bx_assumed] : view<[%n]xf32, %layout> -> f32
  view.store %load, %dst[%bx_assumed] : f32, view<[%n]xf32, %layout>
  kernel.return
}
"""
