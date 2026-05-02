# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _make_batched_transpose_kernel(tilelang: Any, T: Any) -> Any:
    def create_loop_layout_fn(block_x: int, num_threads: int = 256) -> Any:
        def loop_layout_fn(i: Any, j: Any) -> tuple[Any, Any]:
            elements = i * block_x + j
            forward_thread = (elements // 4) % num_threads
            forward_local = elements % 4 + elements // (num_threads * 4) * 4
            return forward_thread, forward_local

        return loop_layout_fn

    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_batched_transpose_kernel(
        shape_x_mod_128: int,
        shape_y_mod_128: int,
        dtype: T.dtype,
    ) -> Any:
        num_batches = T.dynamic("num_batches")
        shape_x = T.dynamic("shape_x")
        shape_y = T.dynamic("shape_y")
        stride_x = T.dynamic("stride_x")

        num_threads = 256
        block_x = 128 if shape_x_mod_128 == 0 else 64
        block_y = 128 if shape_y_mod_128 == 0 else 64
        block_k = 4
        num_threads_per_row = block_y // block_k

        loop_layout = T.Fragment(
            (block_y, block_x),
            forward_fn=create_loop_layout_fn(block_x, num_threads),
        )

        @T.prim_func  # type: ignore[untyped-decorator]
        def batched_transpose_kernel(
            x: T.StridedTensor[
                (num_batches, shape_x, shape_y),
                (shape_x * stride_x, stride_x, 1),
                dtype,
            ],
            out: T.Tensor[(num_batches, shape_y, shape_x), dtype],
        ) -> None:
            with T.Kernel(
                shape_y // block_y,
                shape_x // block_x,
                num_batches,
                threads=num_threads,
            ) as (pid_y, pid_x, pid_batch):
                out_shared = T.alloc_shared((block_y, block_x + block_k), dtype)
                thread_id = T.get_thread_binding()
                row = thread_id // num_threads_per_row
                col = thread_id % num_threads_per_row

                T.assume(shape_x % block_x == 0)
                T.assume(shape_y % block_y == 0)
                T.assume(stride_x % block_k == 0)

                tmp = T.alloc_local((block_k, block_k), dtype)
                tmp_row = T.alloc_local((block_k,), dtype)
                for i_outer in T.unroll(
                    block_x // block_k // (num_threads // num_threads_per_row)
                ):
                    i = i_outer * (num_threads // num_threads_per_row) + row
                    for j in T.unroll(block_k):
                        for k in T.vectorized(block_k):
                            tmp_row[k] = x[
                                pid_batch,
                                pid_x * block_x + i * block_k + j,
                                pid_y * block_y + col * block_k + k,
                            ]
                        for k in T.unroll(block_k):
                            tmp[k, j] = tmp_row[k]

                    for j in T.unroll(block_k):
                        swizzle_j = (j + thread_id // (8 // dtype.bytes)) % block_k
                        for k in T.vectorized(block_k):
                            out_shared[col * block_k + swizzle_j, i * block_k + k] = (
                                tmp[swizzle_j, k]
                            )

                T.sync_threads()
                for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
                    out[pid_batch, pid_y * block_y + i, pid_x * block_x + j] = (
                        out_shared[i, j]
                    )

        return batched_transpose_kernel

    return get_batched_transpose_kernel


def _batched_transpose_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_batched_transpose_kernel(tilelang, T),
        args=(0, 0, T.float16),
        target=target,
        name="batched_transpose_kernel",
    )


# ====
@tilelang_case(
    name="tilekernels_batched_transpose_gfx942",
    category="kernel",
    tags=("tilekernels", "transpose", "amdgpu"),
)
def tilekernels_batched_transpose_gfx942(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _batched_transpose_input(tilelang, T, target="hip -mcpu=gfx942")


# ----
r"""
target.generic<reference> @hip_mcpu_gfx942

kernel.def target(@hip_mcpu_gfx942) export("batched_transpose_kernel") workgroup_size(256, 1, 1) @batched_transpose_kernel(%x_handle: buffer, %out_handle: buffer, %num_batches: i32, %shape_x: i32, %shape_y: i32, %stride_x: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %x = buffer.view %x_handle[%c0_bytes] : buffer -> view<[%num_batches]x[%shape_x]x[%shape_y]xf16, %layout>
  %out = buffer.view %out_handle[%c0_bytes] : buffer -> view<[%num_batches]x[%shape_y]x[%shape_x]xf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %by = kernel.workgroup.id<y> : index
  %bz = kernel.workgroup.id<z> : index
  %thread_id = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %out_shared_bytes = index.constant 33792 : offset
  %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
  %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128x132xf16, %layout>
  %tmp_bytes = index.constant 32 : offset
  %tmp_buffer = buffer.alloca %tmp_bytes {base_alignment = 2, memory_space = private} : buffer
  %tmp = buffer.view %tmp_buffer[%c0_bytes] : buffer -> view<4x4xf16, %layout>
  %tmp_row_bytes = index.constant 8 : offset
  %tmp_row_buffer = buffer.alloca %tmp_row_bytes {base_alignment = 2, memory_space = private} : buffer
  %tmp_row = buffer.view %tmp_row_buffer[%c0_bytes] : buffer -> view<4xf16, %layout>
  %c32 = index.constant 32 : index
  %div = index.div %thread_id, %c32 : index
  %rem = index.rem %thread_id, %c32 : index
  %shape_x_assumed = scalar.assume %shape_x [mul(%shape_x, 128)] : i32
  %shape_y_assumed = scalar.assume %shape_y [mul(%shape_y, 128)] : i32
  %stride_x_assumed = scalar.assume %stride_x [mul(%stride_x, 4)] : i32
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %i_outer_ub = index.add %c0, %c4 : index
  %c1 = index.constant 1 : index
  scf.for %i_outer = [%c0 to %i_outer_ub step %c1] {
    %c8 = index.constant 8 : index
    %madd = index.madd %i_outer, %c8, %div : index
    %j_ub = index.add %c0, %c4 : index
    scf.for %j = [%c0 to %j_ub step %c1] {
      %k_ub = index.add %c0, %c4 : index
      scf.for %k = [%c0 to %k_ub step %c1] {
        %c128 = index.constant 128 : index
        %mul = index.mul %madd, %c4 : index
        %madd_2 = index.madd %by, %c128, %mul : index
        %add = index.add %madd_2, %j : index
        %mul_2 = index.mul %rem, %c4 : index
        %madd_3 = index.madd %bx, %c128, %mul_2 : index
        %add_2 = index.add %madd_3, %k : index
        %load = view.load %x[%bz, %add, %add_2] : view<[%num_batches]x[%shape_x]x[%shape_y]xf16, %layout> -> f16
        view.store %load, %tmp_row[%k] : f16, view<4xf16, %layout>
        scf.yield
      }
      %k_ub_2 = index.add %c0, %c4 : index
      scf.for %k = [%c0 to %k_ub_2 step %c1] {
        %load_2 = view.load %tmp_row[%k] : view<4xf16, %layout> -> f16
        view.store %load_2, %tmp[%k, %j] : f16, view<4x4xf16, %layout>
        scf.yield
      }
      scf.yield
    }
    %j_ub_2 = index.add %c0, %c4 : index
    scf.for %j = [%c0 to %j_ub_2 step %c1] {
      %div_2 = index.div %thread_id, %c4 : index
      %add_3 = index.add %j, %div_2 : index
      %rem_2 = index.rem %add_3, %c4 : index
      %k_ub_3 = index.add %c0, %c4 : index
      scf.for %k = [%c0 to %k_ub_3 step %c1] {
        %load_3 = view.load %tmp[%rem_2, %k] : view<4x4xf16, %layout> -> f16
        %madd_4 = index.madd %rem, %c4, %rem_2 : index
        %madd_5 = index.madd %madd, %c4, %k : index
        view.store %load_3, %out_shared[%madd_4, %madd_5] : f16, view<128x132xf16, %layout>
        scf.yield
      }
      scf.yield
    }
    scf.yield
  }
  kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
  %c128_2 = index.constant 128 : index
  %i_ub = index.add %c0, %c128_2 : index
  scf.for %i = [%c0 to %i_ub step %c1] {
    %j_ub_3 = index.add %c0, %c128_2 : index
    scf.for %j = [%c0 to %j_ub_3 step %c1] {
      %load_4 = view.load %out_shared[%i, %j] : view<128x132xf16, %layout> -> f16
      %madd_6 = index.madd %bx, %c128_2, %i : index
      %madd_7 = index.madd %by, %c128_2, %j : index
      view.store %load_4, %out[%bz, %madd_6, %madd_7] : f16, view<[%num_batches]x[%shape_y]x[%shape_x]xf16, %layout>
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_group_count_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_group_count_kernel(
        num_topk: int,
        num_groups: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_blocks = num_sms * 2
        num_tokens = T.dynamic("num_tokens")
        aligned_groups = _align_to(num_groups, num_threads)

        @T.prim_func  # type: ignore[untyped-decorator]
        def group_count_kernel(
            group_idx: T.Tensor[(num_tokens, num_topk), T.int64],
            out: T.Tensor[(num_groups,), T.int32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid * num_threads + thread_idx

                out_shared = T.alloc_shared((aligned_groups,), T.int32)
                for i in T.serial(thread_idx, aligned_groups, num_threads):
                    out_shared[i] = 0
                T.sync_threads()

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    num_blocks * num_threads,
                ):
                    for j in T.unroll(num_topk):
                        group = T.int32(group_idx[i, j])
                        T.assume(group < num_groups)
                        if group >= 0:
                            T.atomic_add(out_shared[group], 1)

                T.sync_threads()
                for i in T.serial(thread_idx, num_groups, num_threads):
                    if out_shared[i] > 0:
                        T.atomic_add(out[i], out_shared[i])

        return group_count_kernel

    return get_group_count_kernel


def _group_count_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_group_count_kernel(tilelang, T),
        args=(2, 8, 4),
        target=target,
        name="group_count_kernel",
    )


@tilelang_case(
    name="tilekernels_group_count_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "histogram", "amdgpu"),
)
def tilekernels_group_count_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _group_count_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("group_count_kernel") workgroup_size(128, 1, 1) @group_count_kernel(%group_idx_handle: buffer, %out_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %group_idx = buffer.view %group_idx_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xi64, %layout>
  %out = buffer.view %out_handle[%c0_bytes] : buffer -> view<8xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %out_shared_bytes = index.constant 512 : offset
  %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128xi32, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c0 = index.constant 0 : index
  %c255 = index.constant 255 : index
  %sub = index.sub %c255, %thread_idx : index
  %div = index.div %sub, %c128 : index
  %tmp_ub = index.add %c0, %div : index
  %c1 = index.constant 1 : index
  scf.for %tmp = [%c0 to %tmp_ub step %c1] {
    %madd_2 = index.madd %tmp, %c128, %thread_idx : index
    %const = scalar.constant 0 : i32
    view.store %const, %out_shared[%madd_2] : i32, view<128xi32, %layout>
    scf.yield
  }
  kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1023 = index.constant 1023 : index
  %add = index.add %num_tokens_idx, %c1023 : index
  %sub_2 = index.sub %add, %madd : index
  %c1024 = index.constant 1024 : index
  %div_2 = index.div %sub_2, %c1024 : index
  %tmp_ub_2 = index.add %c0, %div_2 : index
  scf.for %tmp = [%c0 to %tmp_ub_2 step %c1] {
    %madd_3 = index.madd %tmp, %c1024, %madd : index
    %c2 = index.constant 2 : index
    %j_ub = index.add %c0, %c2 : index
    scf.for %j = [%c0 to %j_ub step %c1] {
      %load = view.load %group_idx[%madd_3, %j] : view<[%num_tokens]x2xi64, %layout> -> i64
      %trunci = scalar.trunci %load : i64 to i32
      %group_assumed = scalar.assume %trunci [lt(%trunci, 8)] : i32
      %const_2 = scalar.constant 0 : i32
      %cmp = scalar.cmpi sge, %group_assumed, %const_2 : i32
      scf.if %cmp {
        %const_3 = scalar.constant 1 : i32
        %group_idx_2 = index.cast %group_assumed : i32 to index
        view.atomic.reduce<addi> %const_3, %out_shared[%group_idx_2] {ordering = relaxed, scope = workgroup} : i32, view<128xi32, %layout>
        scf.yield
      } else {
        scf.yield
      }
      scf.yield
    }
    scf.yield
  }
  kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
  %c135 = index.constant 135 : index
  %sub_3 = index.sub %c135, %thread_idx : index
  %div_3 = index.div %sub_3, %c128 : index
  %tmp_ub_3 = index.add %c0, %div_3 : index
  scf.for %tmp = [%c0 to %tmp_ub_3 step %c1] {
    %madd_4 = index.madd %tmp, %c128, %thread_idx : index
    %load_2 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
    %const_4 = scalar.constant 0 : i32
    %cmp_2 = scalar.cmpi sgt, %load_2, %const_4 : i32
    scf.if %cmp_2 {
      %load_3 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
      view.atomic.reduce<addi> %load_3, %out[%madd_4] {ordering = relaxed, scope = device} : i32, view<8xi32, %layout>
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""
