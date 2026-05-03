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
@tilelang_case(
    name="tileop_copy_disable_tma_1d",
    category="op",
    tags=("tileop", "copy", "annotation"),
)
def tileop_copy_disable_tma_1d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_disable_tma_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(4,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst, disable_tma=True)

        return tileop_copy_disable_tma_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_disable_tma_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_disable_tma_kernel") @tileop_copy_disable_tma_kernel(%src_handle: buffer, %dst_handle: buffer) {
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
@tilelang_case(
    name="tileop_fill_integer_zero_to_float",
    category="op",
    tags=("tileop", "fill", "constant"),
)
def tileop_fill_integer_zero_to_float(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_fill_integer_zero_kernel(dst: T.Tensor[(4,), T.float32]) -> None:
            with T.Kernel(1, threads=1):
                T.fill(dst, 0)

        return tileop_fill_integer_zero_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_fill_integer_zero_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_fill_integer_zero_kernel") @tileop_fill_integer_zero_kernel(%dst_handle: buffer) {
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
  %const = scalar.constant 0.0 : f32
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    view.store %const, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_reduce_sum_1d", category="op", tags=("tileop", "reduce"))
def tileop_reduce_sum_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_sum_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_sum(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_sum_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_sum_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_sum_kernel") @tileop_reduce_sum_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %local[%i0] : f32, view<4xf32, %layout>
  }
  %load = vector.load %local[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %identity = scalar.constant 0.0 : f32
  %reduce = vector.reduce<addf> %load, %identity : vector<4xf32>, f32
  view.store %reduce, %out[%c0] : f32, view<1xf32, %layout>
  %copy_2 = view.load %out[%c0] : view<1xf32, %layout> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_abssum_1d",
    category="op",
    tags=("tileop", "reduce"),
)
def tileop_reduce_abssum_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_abssum_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_abssum(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_abssum_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_abssum_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_abssum_kernel") @tileop_reduce_abssum_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %local[%i0] : f32, view<4xf32, %layout>
  }
  %load = vector.load %local[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %identity = scalar.constant 0.0 : f32
  %abs = vector.absf %load : vector<4xf32>
  %reduce = vector.reduce<addf> %abs, %identity : vector<4xf32>, f32
  view.store %reduce, %out[%c0] : f32, view<1xf32, %layout>
  %copy_2 = view.load %out[%c0] : view<1xf32, %layout> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_absmax_1d",
    category="op",
    tags=("tileop", "reduce"),
)
def tileop_reduce_absmax_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_absmax_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_absmax(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_absmax_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_absmax_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_absmax_kernel") @tileop_reduce_absmax_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %local[%i0] : f32, view<4xf32, %layout>
  }
  %load = vector.load %local[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %identity = scalar.constant 0.0 : f32
  %abs = vector.absf %load : vector<4xf32>
  %reduce = vector.reduce<maxnumf> %abs, %identity : vector<4xf32>, f32
  view.store %reduce, %out[%c0] : f32, view<1xf32, %layout>
  %copy_2 = view.load %out[%c0] : view<1xf32, %layout> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_reduce_sum_2d", category="op", tags=("tileop", "reduce"))
def tileop_reduce_sum_2d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_sum_2d_kernel(
        src: T.Tensor[(2, 4), T.float32],
        dst: T.Tensor[(2,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((2, 4), T.float32)
            out = T.alloc_fragment((2,), T.float32)
            T.copy(src, local)
            T.reduce_sum(local, out, dim=1)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_sum_2d_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_sum_2d_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_sum_2d_kernel") @tileop_reduce_sum_2d_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 32 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %out_bytes = index.constant 8 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<2xf32, %layout>
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c4 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x4xf32, %layout> -> f32
      view.store %copy, %local[%i0, %i1] : f32, view<2x4xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    %identity = scalar.constant 0.0 : f32
    %reduce = scf.for %r = [%c0 to %c4 step %c1](%acc = %identity : f32) -> (f32) {
      %reduce_value = view.load %local[%i0, %r] : view<2x4xf32, %layout> -> f32
      %addf = scalar.addf %acc, %reduce_value : f32
      scf.yield %addf : f32
    }
    view.store %reduce, %out[%i0] : f32, view<2xf32, %layout>
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    %copy_2 = view.load %out[%i0] : view<2xf32, %layout> -> f32
    view.store %copy_2, %dst[%i0] : f32, view<2xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_finalize_reducer_none",
    category="op",
    tags=("tileop", "finalize_reducer"),
)
def tileop_finalize_reducer_none(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_finalize_reducer_none_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(4,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            reducer = T.alloc_reducer(
                (4,),
                T.float32,
                "sum",
                replication="none",
            )
            T.fill(reducer, 0.0)
            for i in T.serial(0, 4):
                reducer[i] = src[i]
            T.finalize_reducer(reducer)
            T.copy(reducer, dst)

    return TileLangImportInput(
        source=tileop_finalize_reducer_none_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_finalize_reducer_none_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_finalize_reducer_none_kernel") @tileop_finalize_reducer_none_kernel(%src_handle: buffer, %dst_handle: buffer) {
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
  %reducer_bytes = index.constant 16 : offset
  %reducer_buffer = buffer.alloca %reducer_bytes {base_alignment = 4, memory_space = private} : buffer
  %reducer = buffer.view %reducer_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %const = scalar.constant 0.0 : f32
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    view.store %const, %reducer[%i0] : f32, view<4xf32, %layout>
  }
  scf.for %i = [%c0 to %c4 step %c1] {
    %load = view.load %src[%i] : view<4xf32, %layout> -> f32
    view.store %load, %reducer[%i] : f32, view<4xf32, %layout>
  }
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %reducer[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_finalize_reducer_all",
    category="op",
    tags=("tileop", "finalize_reducer", "workgroup"),
)
def tileop_finalize_reducer_all(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_finalize_reducer_all_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(4,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=4):
            thread_index = T.get_thread_binding()
            reducer = T.alloc_reducer(
                (1,),
                T.float32,
                "sum",
                replication="all",
            )
            reducer[0] = src[thread_index]
            T.finalize_reducer(reducer)
            dst[thread_index] = reducer[0]

    return TileLangImportInput(
        source=tileop_finalize_reducer_all_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_finalize_reducer_all_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_finalize_reducer_all_kernel") @tileop_finalize_reducer_all_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c4, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %reducer_bytes = index.constant 4 : offset
  %reducer_buffer = buffer.alloca %reducer_bytes {base_alignment = 4, memory_space = private} : buffer
  %reducer = buffer.view %reducer_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %load = view.load %src[%thread_index] : view<4xf32, %layout> -> f32
  %c0 = index.constant 0 : index
  view.store %load, %reducer[%c0] : f32, view<1xf32, %layout>
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c1 step %c1] {
    %reducer_value = view.load %reducer[%i0] : view<1xf32, %layout> -> f32
    %reducer_all = kernel.workgroup.reduce<addf> %reducer_value : f32
    view.store %reducer_all, %reducer[%i0] : f32, view<1xf32, %layout>
  }
  %load_2 = view.load %reducer[%c0] : view<1xf32, %layout> -> f32
  view.store %load_2, %dst[%thread_index] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_absmax_reshape_2d_to_3d",
    category="op",
    tags=("tileop", "reduce", "reshape"),
)
def tileop_reduce_absmax_reshape_2d_to_3d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_absmax_reshape_kernel(
        src: T.Tensor[(2, 4), T.float32],
        dst: T.Tensor[(2, 2), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((2, 4), T.float32)
            reshaped = T.reshape(local, (2, 2, 2))
            out = T.alloc_fragment((2, 2), T.float32)
            T.copy(src, local)
            T.reduce_absmax(reshaped, out, dim=2)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_absmax_reshape_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_absmax_reshape_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_absmax_reshape_kernel") @tileop_reduce_absmax_reshape_kernel(%src_handle: buffer, %dst_handle: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_handle[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %dst = buffer.view %dst_handle[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 32 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %out_bytes = index.constant 16 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c4 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x4xf32, %layout> -> f32
      view.store %copy, %local[%i0, %i1] : f32, view<2x4xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity = scalar.constant 0.0 : f32
      %reduce = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity : f32) -> (f32) {
        %mul = index.mul %i0, %c2 : index
        %add = index.add %mul, %i1 : index
        %mul_2 = index.mul %add, %c2 : index
        %add_2 = index.add %mul_2, %r : index
        %rem = index.rem %add_2, %c4 : index
        %div = index.div %add_2, %c4 : index
        %reduce_value = view.load %local[%div, %rem] : view<2x4xf32, %layout> -> f32
        %abs = scalar.absf %reduce_value : f32
        %maxnumf = scalar.maxnumf %acc, %abs : f32
        scf.yield %maxnumf : f32
      }
      view.store %reduce, %out[%i0, %i1] : f32, view<2x2xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy_2 = view.load %out[%i0, %i1] : view<2x2xf32, %layout> -> f32
      view.store %copy_2, %dst[%i0, %i1] : f32, view<2x2xf32, %layout>
    }
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
