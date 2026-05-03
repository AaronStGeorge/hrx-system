# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


@tilelang_case(name="tileop_copy_1d", category="op", tags=("tileop", "copy"))
def tileop_copy_1d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(4,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst)

        return tileop_copy_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_kernel") @tileop_copy_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_fill_1d", category="op", tags=("tileop", "fill"))
def tileop_fill_1d(tilelang: Any, T: Any, tir: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_fill_kernel(dst: T.Tensor[(4,), T.float32]) -> None:
            with T.Kernel(1, threads=1):
                T.fill(dst, tir.FloatImm("float32", 3.0))

        return tileop_fill_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_fill_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_fill_kernel") @tileop_fill_kernel(%dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %const = scalar.constant 3.0 : f32
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    view.store %const, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_copy_2d", category="op", tags=("tileop", "copy"))
def tileop_copy_2d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_2d_kernel(
            src: T.Tensor[(2, 3), T.float16],
            dst: T.Tensor[(2, 3), T.float16],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst)

        return tileop_copy_2d_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_2d_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_2d_kernel") @tileop_copy_2d_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<2x3xf16, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<2x3xf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c3 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x3xf16, %layout> -> f16
      view.store %copy, %dst[%i0, %i1] : f16, view<2x3xf16, %layout>
    }
  }
  kernel.return
}
"""
