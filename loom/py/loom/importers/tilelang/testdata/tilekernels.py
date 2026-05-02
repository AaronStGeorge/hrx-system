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


# ====
def _make_mask_indices_by_tp_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_mask_indices_by_tp_kernel(num_topk: int, dtype: T.dtype) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = T.ceildiv(num_tokens * num_topk, num_threads)

        @T.prim_func  # type: ignore[untyped-decorator]
        def mask_indices_by_tp_kernel(
            indices: T.Tensor[(num_tokens, num_topk), dtype],
            masked_indices: T.Tensor[(num_tokens, num_topk), dtype],
            per_gpu: T.int32,
            per_dp: T.int32,
            num_tp_ranks: T.int32,
            tp_rank: T.int32,
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                indices_1d = T.reshape(indices, (num_tokens * num_topk,))
                masked_indices_1d = T.reshape(masked_indices, (num_tokens * num_topk,))
                thread_idx = T.get_thread_binding()
                index = pid * num_threads + thread_idx
                value = T.alloc_var(dtype)
                if index < num_tokens * num_topk:
                    value = indices_1d[index]
                    if (
                        value < 0
                        or T.truncmod(T.truncdiv(value, per_gpu), num_tp_ranks)
                        != tp_rank
                    ):
                        masked_indices_1d[index] = -1
                    else:
                        value -= tp_rank * per_gpu
                        dp_rank = T.truncdiv(value, per_dp)
                        value -= dp_rank * (per_dp - per_gpu)
                        masked_indices_1d[index] = T.Select(
                            value < 0,
                            T.int64(-1),
                            value,
                        )

        return mask_indices_by_tp_kernel

    return get_mask_indices_by_tp_kernel


def _mask_indices_by_tp_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_mask_indices_by_tp_kernel(tilelang, T),
        args=(2, T.int64),
        target=target,
        name="mask_indices_by_tp_kernel",
    )


@tilelang_case(
    name="tilekernels_mask_indices_by_tp_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "routing", "amdgpu"),
)
def tilekernels_mask_indices_by_tp_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _mask_indices_by_tp_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mask_indices_by_tp_kernel") workgroup_size(128, 1, 1) @mask_indices_by_tp_kernel(%indices_handle: buffer, %masked_indices_handle: buffer, %per_gpu: i32, %per_dp: i32, %num_tp_ranks: i32, %tp_rank: i32, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %indices = buffer.view %indices_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xi64, %layout>
  %masked_indices = buffer.view %masked_indices_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %value_bytes = index.constant 8 : offset
  %value_buffer = buffer.alloca %value_bytes {base_alignment = 8, memory_space = private} : buffer
  %value = buffer.view %value_buffer[%c0_bytes] : buffer -> view<1xi64, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c2 = index.constant 2 : index
  %mul = index.mul %num_tokens_idx, %c2 : index
  %cmp = index.cmp slt, %madd, %mul : index
  scf.if %cmp {
    %rem = index.rem %madd, %c2 : index
    %div = index.div %madd, %c2 : index
    %load = view.load %indices[%div, %rem] : view<[%num_tokens]x2xi64, %layout> -> i64
    %c0 = index.constant 0 : index
    view.store %load, %value[%c0] : i64, view<1xi64, %layout>
    %load_2 = view.load %value[%c0] : view<1xi64, %layout> -> i64
    %const = scalar.constant 0 : i64
    %cmp_2 = scalar.cmpi slt, %load_2, %const : i64
    %load_3 = view.load %value[%c0] : view<1xi64, %layout> -> i64
    %extsi = scalar.extsi %per_gpu : i32 to i64
    %divsi = scalar.divsi %load_3, %extsi : i64
    %extsi_2 = scalar.extsi %num_tp_ranks : i32 to i64
    %remsi = scalar.remsi %divsi, %extsi_2 : i64
    %extsi_3 = scalar.extsi %tp_rank : i32 to i64
    %cmp_3 = scalar.cmpi ne, %remsi, %extsi_3 : i64
    %ori = scalar.ori %cmp_2, %cmp_3 : i1
    scf.if %ori {
      %rem_2 = index.rem %madd, %c2 : index
      %div_2 = index.div %madd, %c2 : index
      %const_2 = scalar.constant -1 : i64
      view.store %const_2, %masked_indices[%div_2, %rem_2] : i64, view<[%num_tokens]x2xi64, %layout>
      scf.yield
    } else {
      %load_4 = view.load %value[%c0] : view<1xi64, %layout> -> i64
      %muli = scalar.muli %tp_rank, %per_gpu : i32
      %extsi_4 = scalar.extsi %muli : i32 to i64
      %subi = scalar.subi %load_4, %extsi_4 : i64
      view.store %subi, %value[%c0] : i64, view<1xi64, %layout>
      %load_5 = view.load %value[%c0] : view<1xi64, %layout> -> i64
      %extsi_5 = scalar.extsi %per_dp : i32 to i64
      %divsi_2 = scalar.divsi %load_5, %extsi_5 : i64
      %load_6 = view.load %value[%c0] : view<1xi64, %layout> -> i64
      %subi_2 = scalar.subi %per_dp, %per_gpu : i32
      %extsi_6 = scalar.extsi %subi_2 : i32 to i64
      %muli_2 = scalar.muli %divsi_2, %extsi_6 : i64
      %subi_3 = scalar.subi %load_6, %muli_2 : i64
      view.store %subi_3, %value[%c0] : i64, view<1xi64, %layout>
      %rem_3 = index.rem %madd, %c2 : index
      %div_3 = index.div %madd, %c2 : index
      %load_7 = view.load %value[%c0] : view<1xi64, %layout> -> i64
      %cmp_4 = scalar.cmpi slt, %load_7, %const : i64
      %const_3 = scalar.constant -1 : i64
      %load_8 = view.load %value[%c0] : view<1xi64, %layout> -> i64
      %select = scf.select %cmp_4, %const_3, %load_8 : i64
      view.store %select, %masked_indices[%div_3, %rem_3] : i64, view<[%num_tokens]x2xi64, %layout>
      scf.yield
    }
    scf.yield
  } else {
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_normalize_weight_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_normalize_weight_kernel(num_topk: int) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = T.ceildiv(num_tokens, 128)

        @T.prim_func  # type: ignore[untyped-decorator]
        def normalize_weight_kernel(
            topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
            denominator: T.Tensor[(num_tokens,), T.float32],
            normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                tid = T.get_thread_binding()
                weights_local = T.alloc_local((num_topk,), T.float32)
                row = pid * num_threads + tid
                if row < num_tokens:
                    total = T.alloc_var(T.float32, init=1e-20)
                    for i in T.vectorized(num_topk):
                        weights_local[i] = topk_weights[row, i]
                    for i in T.unroll(num_topk):
                        total += weights_local[i]
                    denominator[row] = total
                    for i in T.vectorized(num_topk):
                        normalized_weights[row, i] = weights_local[i] / total

        return normalize_weight_kernel

    return get_normalize_weight_kernel


def _normalize_weight_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_normalize_weight_kernel(tilelang, T),
        args=(2,),
        target=target,
        name="normalize_weight_kernel",
    )


@tilelang_case(
    name="tilekernels_normalize_weight_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "normalize", "amdgpu"),
)
def tilekernels_normalize_weight_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _normalize_weight_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("normalize_weight_kernel") workgroup_size(128, 1, 1) @normalize_weight_kernel(%topk_weights_handle: buffer, %denominator_handle: buffer, %normalized_weights_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %topk_weights = buffer.view %topk_weights_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xf32, %layout>
  %denominator = buffer.view %denominator_handle[%c0_bytes] : buffer -> view<[%num_tokens]xf32, %layout>
  %normalized_weights = buffer.view %normalized_weights_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tid = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %weights_local_bytes = index.constant 8 : offset
  %weights_local_buffer = buffer.alloca %weights_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %weights_local = buffer.view %weights_local_buffer[%c0_bytes] : buffer -> view<2xf32, %layout>
  %total_bytes = index.constant 4 : offset
  %total_buffer = buffer.alloca %total_bytes {base_alignment = 4, memory_space = private} : buffer
  %total = buffer.view %total_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %const = scalar.constant 9.9999999999999995e-21 : f32
  view.store %const, %total[%c0] : f32, view<1xf32, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %tid : index
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %cmp = index.cmp slt, %madd, %num_tokens_idx : index
  scf.if %cmp {
    %c2 = index.constant 2 : index
    %i_ub = index.add %c0, %c2 : index
    %c1 = index.constant 1 : index
    scf.for %i = [%c0 to %i_ub step %c1] {
      %load = view.load %topk_weights[%madd, %i] : view<[%num_tokens]x2xf32, %layout> -> f32
      view.store %load, %weights_local[%i] : f32, view<2xf32, %layout>
      scf.yield
    }
    %i_ub_2 = index.add %c0, %c2 : index
    scf.for %i = [%c0 to %i_ub_2 step %c1] {
      %load_2 = view.load %total[%c0] : view<1xf32, %layout> -> f32
      %load_3 = view.load %weights_local[%i] : view<2xf32, %layout> -> f32
      %addf = scalar.addf %load_2, %load_3 : f32
      view.store %addf, %total[%c0] : f32, view<1xf32, %layout>
      scf.yield
    }
    %load_4 = view.load %total[%c0] : view<1xf32, %layout> -> f32
    view.store %load_4, %denominator[%madd] : f32, view<[%num_tokens]xf32, %layout>
    %i_ub_3 = index.add %c0, %c2 : index
    scf.for %i = [%c0 to %i_ub_3 step %c1] {
      %load_5 = view.load %weights_local[%i] : view<2xf32, %layout> -> f32
      %load_6 = view.load %total[%c0] : view<1xf32, %layout> -> f32
      %divf = scalar.divf %load_5, %load_6 : f32
      view.store %divf, %normalized_weights[%madd, %i] : f32, view<[%num_tokens]x2xf32, %layout>
      scf.yield
    }
    scf.yield
  } else {
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_aux_fi_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_aux_fi_kernel(
        num_topk: int,
        num_experts: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = num_sms * 2

        @T.prim_func  # type: ignore[untyped-decorator]
        def aux_fi_kernel(
            topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
            out: T.Tensor[(num_experts,), T.float32],
            num_aux_topk: T.int32,
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid * num_threads + thread_idx
                out_shared = T.alloc_shared(
                    (_align_to(num_experts, num_threads),), T.int32
                )
                for i in T.serial(
                    thread_idx,
                    _align_to(num_experts, num_threads),
                    num_threads,
                ):
                    out_shared[i] = 0
                T.sync_threads()

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    num_blocks * num_threads,
                ):
                    for j in T.unroll(num_topk):
                        expert_idx = T.int32(topk_idx[i, j])
                        T.assume(expert_idx < num_experts)
                        if expert_idx >= 0:
                            T.atomic_add(out_shared[expert_idx], 1)

                T.sync_threads()
                for i in T.serial(thread_idx, num_experts, num_threads):
                    if out_shared[i] > 0:
                        numerator = T.cast(out_shared[i] * num_experts, T.float32)
                        denominator = T.cast(num_tokens * num_aux_topk, T.float32)
                        T.atomic_add(out[i], numerator / denominator)

        return aux_fi_kernel

    return get_aux_fi_kernel


def _aux_fi_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_aux_fi_kernel(tilelang, T),
        args=(2, 8, 4),
        target=target,
        name="aux_fi_kernel",
    )


@tilelang_case(
    name="tilekernels_aux_fi_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "atomic", "amdgpu"),
)
def tilekernels_aux_fi_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _aux_fi_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("aux_fi_kernel") workgroup_size(128, 1, 1) @aux_fi_kernel(%topk_idx_handle: buffer, %out_handle: buffer, %num_aux_topk: i32, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %topk_idx = buffer.view %topk_idx_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xi64, %layout>
  %out = buffer.view %out_handle[%c0_bytes] : buffer -> view<8xf32, %layout>
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
      %load = view.load %topk_idx[%madd_3, %j] : view<[%num_tokens]x2xi64, %layout> -> i64
      %trunci = scalar.trunci %load : i64 to i32
      %expert_idx_assumed = scalar.assume %trunci [lt(%trunci, 8)] : i32
      %const_2 = scalar.constant 0 : i32
      %cmp = scalar.cmpi sge, %expert_idx_assumed, %const_2 : i32
      scf.if %cmp {
        %const_3 = scalar.constant 1 : i32
        %expert_idx_idx = index.cast %expert_idx_assumed : i32 to index
        view.atomic.reduce<addi> %const_3, %out_shared[%expert_idx_idx] {ordering = relaxed, scope = workgroup} : i32, view<128xi32, %layout>
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
      %const_5 = scalar.constant 8 : i32
      %muli = scalar.muli %load_3, %const_5 : i32
      %sitofp = scalar.sitofp %muli : i32 to f32
      %muli_2 = scalar.muli %num_tokens, %num_aux_topk : i32
      %sitofp_2 = scalar.sitofp %muli_2 : i32 to f32
      %divf = scalar.divf %sitofp, %sitofp_2 : f32
      view.atomic.reduce<addf> %divf, %out[%madd_4] {ordering = relaxed, scope = device} : f32, view<8xf32, %layout>
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_inplace_unique_group_indices_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        },
    )
    def get_inplace_unique_group_indices_kernel(
        num_topk: int,
        num_groups_aligned: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        grid_x = num_sms * 2

        @T.prim_func  # type: ignore[untyped-decorator]
        def inplace_unique_group_indices_kernel(
            group_indices: T.Tensor[(num_tokens, num_topk), T.int64],
        ) -> None:
            with T.Kernel(grid_x, threads=num_threads) as (pid_token,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid_token * num_threads + thread_idx
                group_sel = T.alloc_local((2,), T.uint64)

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    grid_x * num_threads,
                ):
                    for j in T.unroll(num_groups_aligned // 64):
                        group_sel[j] = 0
                    for j in T.unroll(num_topk):
                        group_idx = group_indices[i, j]
                        T.assume(group_idx < num_groups_aligned)
                        mask = T.Select(
                            group_idx >= 0,
                            T.uint64(1) << (group_idx % 64),
                            T.uint64(0),
                        )
                        lo_mask = T.Select(group_idx < 64, mask, T.uint64(0))
                        hi_mask = T.Select(group_idx >= 64, mask, T.uint64(0))
                        found = (lo_mask & group_sel[0]) | (hi_mask & group_sel[1])
                        group_sel[0] |= lo_mask
                        group_sel[1] |= hi_mask
                        if found:
                            group_indices[i, j] = -1

        return inplace_unique_group_indices_kernel

    return get_inplace_unique_group_indices_kernel


def _inplace_unique_group_indices_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_inplace_unique_group_indices_kernel(tilelang, T),
        args=(4, _align_to(8, 64), 4),
        target=target,
        name="inplace_unique_group_indices_kernel",
    )


@tilelang_case(
    name="tilekernels_inplace_unique_group_indices_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "dedupe", "amdgpu"),
)
def tilekernels_inplace_unique_group_indices_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _inplace_unique_group_indices_input(
        tilelang,
        T,
        target="hip -mcpu=gfx1100",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("inplace_unique_group_indices_kernel") workgroup_size(128, 1, 1) @inplace_unique_group_indices_kernel(%group_indices_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %group_indices = buffer.view %group_indices_handle[%c0_bytes] : buffer -> view<[%num_tokens]x4xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %group_sel_bytes = index.constant 16 : offset
  %group_sel_buffer = buffer.alloca %group_sel_bytes {base_alignment = 8, memory_space = private} : buffer
  %group_sel = buffer.view %group_sel_buffer[%c0_bytes] : buffer -> view<2xi64, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c0 = index.constant 0 : index
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1023 = index.constant 1023 : index
  %add = index.add %num_tokens_idx, %c1023 : index
  %sub = index.sub %add, %madd : index
  %c1024 = index.constant 1024 : index
  %div = index.div %sub, %c1024 : index
  %tmp_ub = index.add %c0, %div : index
  %c1 = index.constant 1 : index
  scf.for %tmp = [%c0 to %tmp_ub step %c1] {
    %madd_2 = index.madd %tmp, %c1024, %madd : index
    %j_ub = index.add %c0, %c1 : index
    scf.for %j = [%c0 to %j_ub step %c1] {
      %const = scalar.constant 0 : i64
      view.store %const, %group_sel[%j] : i64, view<2xi64, %layout>
      scf.yield
    }
    %c4 = index.constant 4 : index
    %j_ub_2 = index.add %c0, %c4 : index
    scf.for %j = [%c0 to %j_ub_2 step %c1] {
      %load = view.load %group_indices[%madd_2, %j] : view<[%num_tokens]x4xi64, %layout> -> i64
      %group_idx_assumed = scalar.assume %load [lt(%load, 64)] : i64
      %const_2 = scalar.constant 0 : i64
      %cmp = scalar.cmpi sge, %group_idx_assumed, %const_2 : i64
      %const_3 = scalar.constant 1 : i64
      %const_4 = scalar.constant 64 : i64
      %remsi = scalar.remsi %group_idx_assumed, %const_4 : i64
      %shli = scalar.shli %const_3, %remsi : i64
      %const_5 = scalar.constant 0 : i64
      %select = scf.select %cmp, %shli, %const_5 : i64
      %cmp_2 = scalar.cmpi slt, %group_idx_assumed, %const_4 : i64
      %select_2 = scf.select %cmp_2, %select, %const_5 : i64
      %cmp_3 = scalar.cmpi sge, %group_idx_assumed, %const_4 : i64
      %select_3 = scf.select %cmp_3, %select, %const_5 : i64
      %load_2 = view.load %group_sel[%c0] : view<2xi64, %layout> -> i64
      %andi = scalar.andi %select_2, %load_2 : i64
      %load_3 = view.load %group_sel[%c1] : view<2xi64, %layout> -> i64
      %andi_2 = scalar.andi %select_3, %load_3 : i64
      %ori = scalar.ori %andi, %andi_2 : i64
      %load_4 = view.load %group_sel[%c0] : view<2xi64, %layout> -> i64
      %ori_2 = scalar.ori %load_4, %select_2 : i64
      view.store %ori_2, %group_sel[%c0] : i64, view<2xi64, %layout>
      %load_5 = view.load %group_sel[%c1] : view<2xi64, %layout> -> i64
      %ori_3 = scalar.ori %load_5, %select_3 : i64
      view.store %ori_3, %group_sel[%c1] : i64, view<2xi64, %layout>
      %const_6 = scalar.constant 0 : i64
      %cmp_4 = scalar.cmpi ne, %ori, %const_6 : i64
      scf.if %cmp_4 {
        %const_7 = scalar.constant -1 : i64
        view.store %const_7, %group_indices[%madd_2, %j] : i64, view<[%num_tokens]x4xi64, %layout>
        scf.yield
      } else {
        scf.yield
      }
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_engram_hash_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        },
    )
    def get_engram_hash_kernel(
        max_ngram_size: int,
        num_ngram_layers: int,
        num_embed_table_per_ngram: int,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")
        threads = 32

        @T.prim_func  # type: ignore[untyped-decorator]
        def engram_hash_kernel(
            ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
            multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
            vocab_sizes: T.Tensor[
                (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
                T.int32,
            ],
            offsets: T.Tensor[
                (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram),
                T.int32,
            ],
            output: T.Tensor[
                (
                    num_ngram_layers,
                    num_tokens,
                    (max_ngram_size - 1) * num_embed_table_per_ngram,
                ),
                T.int32,
            ],
        ) -> None:
            with T.Kernel(
                num_ngram_layers,
                T.ceildiv(num_tokens, threads),
                threads=threads,
            ) as (pid_h, pid_s):
                tid = T.get_thread_binding()
                token_idx = pid_s * threads + tid
                if token_idx >= num_tokens:
                    T.thread_return()

                hash_local = T.alloc_var(T.int64, init=0)
                for ngram_idx in T.unroll(max_ngram_size):
                    token = T.cast(ngram_token_ids[token_idx, ngram_idx], T.int64)
                    hash_local = T.bitwise_xor(
                        hash_local,
                        token * multipliers[pid_h, ngram_idx],
                    )
                    if ngram_idx > 0:
                        for j in T.unroll(num_embed_table_per_ngram):
                            col = (ngram_idx - 1) * num_embed_table_per_ngram + j
                            vocab = T.cast(
                                vocab_sizes[pid_h, ngram_idx - 1, j],
                                T.int64,
                            )
                            output[pid_h, token_idx, col] = (
                                T.cast(
                                    hash_local % vocab,
                                    T.int32,
                                )
                                + offsets[pid_h, col]
                            )

        return engram_hash_kernel

    return get_engram_hash_kernel


def _engram_hash_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_engram_hash_kernel(tilelang, T),
        args=(3, 2, 4),
        target=target,
        name="engram_hash_kernel",
    )


@tilelang_case(
    name="tilekernels_engram_hash_gfx1100",
    category="kernel",
    tags=("tilekernels", "engram", "hash", "amdgpu"),
)
def tilekernels_engram_hash_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _engram_hash_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("engram_hash_kernel") workgroup_size(32, 1, 1) @engram_hash_kernel(%ngram_token_ids_handle: buffer, %multipliers_handle: buffer, %vocab_sizes_handle: buffer, %offsets_handle: buffer, %output_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %ngram_token_ids = buffer.view %ngram_token_ids_handle[%c0_bytes] : buffer -> view<[%num_tokens]x3xi32, %layout>
  %multipliers = buffer.view %multipliers_handle[%c0_bytes] : buffer -> view<2x3xi64, %layout>
  %vocab_sizes = buffer.view %vocab_sizes_handle[%c0_bytes] : buffer -> view<2x2x4xi32, %layout>
  %offsets = buffer.view %offsets_handle[%c0_bytes] : buffer -> view<2x8xi32, %layout>
  %output = buffer.view %output_handle[%c0_bytes] : buffer -> view<2x[%num_tokens]x8xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %by = kernel.workgroup.id<y> : index
  %tid = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %hash_local_bytes = index.constant 8 : offset
  %hash_local_buffer = buffer.alloca %hash_local_bytes {base_alignment = 8, memory_space = private} : buffer
  %hash_local = buffer.view %hash_local_buffer[%c0_bytes] : buffer -> view<1xi64, %layout>
  %c0 = index.constant 0 : index
  %const = scalar.constant 0 : i64
  view.store %const, %hash_local[%c0] : i64, view<1xi64, %layout>
  %c32 = index.constant 32 : index
  %madd = index.madd %by, %c32, %tid : index
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %cmp = index.cmp sge, %madd, %num_tokens_idx : index
  kernel.exit %cmp : i1
  %c3 = index.constant 3 : index
  %ngram_idx_ub = index.add %c0, %c3 : index
  %c1 = index.constant 1 : index
  scf.for %ngram_idx = [%c0 to %ngram_idx_ub step %c1] {
    %load = view.load %ngram_token_ids[%madd, %ngram_idx] : view<[%num_tokens]x3xi32, %layout> -> i32
    %extsi = scalar.extsi %load : i32 to i64
    %load_2 = view.load %hash_local[%c0] : view<1xi64, %layout> -> i64
    %load_3 = view.load %multipliers[%bx, %ngram_idx] : view<2x3xi64, %layout> -> i64
    %muli = scalar.muli %extsi, %load_3 : i64
    %xori = scalar.xori %load_2, %muli : i64
    view.store %xori, %hash_local[%c0] : i64, view<1xi64, %layout>
    %cmp_2 = index.cmp sgt, %ngram_idx, %c0 : index
    scf.if %cmp_2 {
      %c4 = index.constant 4 : index
      %j_ub = index.add %c0, %c4 : index
      scf.for %j = [%c0 to %j_ub step %c1] {
        %sub = index.sub %ngram_idx, %c1 : index
        %madd_2 = index.madd %sub, %c4, %j : index
        %sub_2 = index.sub %ngram_idx, %c1 : index
        %load_4 = view.load %vocab_sizes[%bx, %sub_2, %j] : view<2x2x4xi32, %layout> -> i32
        %extsi_2 = scalar.extsi %load_4 : i32 to i64
        %load_5 = view.load %hash_local[%c0] : view<1xi64, %layout> -> i64
        %remsi = scalar.remsi %load_5, %extsi_2 : i64
        %trunci = scalar.trunci %remsi : i64 to i32
        %load_6 = view.load %offsets[%bx, %madd_2] : view<2x8xi32, %layout> -> i32
        %addi = scalar.addi %trunci, %load_6 : i32
        view.store %addi, %output[%bx, %madd, %madd_2] : i32, view<2x[%num_tokens]x8xi32, %layout>
        scf.yield
      }
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""


# ====
def _make_expand_to_fused_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_expand_to_fused_kernel(
        hidden: int,
        num_topk: int,
    ) -> Any:
        num_threads = 64
        num_tokens = T.dynamic("num_tokens")
        num_expanded_tokens = T.dynamic("num_expanded_tokens")
        num_blocks = T.max(num_tokens, num_expanded_tokens)

        @T.prim_func  # type: ignore[untyped-decorator]
        def expand_to_fused_kernel(
            x: T.Tensor[(num_tokens, hidden), T.float16],
            expanded_x: T.Tensor[(num_expanded_tokens, hidden), T.float16],
            token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
            pos_to_expert: T.Tensor[(num_expanded_tokens,), T.int32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid_token,):
                thread_idx = T.get_thread_binding()
                if pid_token < num_expanded_tokens:  # noqa: SIM102
                    if pos_to_expert[pid_token] < 0:
                        for i in T.serial(thread_idx, hidden, num_threads):
                            expanded_x[pid_token, i] = 0

                if pid_token >= num_tokens:
                    T.thread_return()
                T.assume(pid_token < num_tokens)

                for k in T.unroll(num_topk):
                    pos = token_topk_to_pos[pid_token, k]
                    T.assume(pos < num_expanded_tokens)
                    if pos >= 0:
                        for i in T.serial(thread_idx, hidden, num_threads):
                            expanded_x[pos, i] = x[pid_token, i]

        return expand_to_fused_kernel

    return get_expand_to_fused_kernel


def _expand_to_fused_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_expand_to_fused_kernel(tilelang, T),
        args=(64, 2),
        target=target,
        name="expand_to_fused_kernel",
    )


@tilelang_case(
    name="tilekernels_expand_to_fused_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "expand", "amdgpu"),
)
def tilekernels_expand_to_fused_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _expand_to_fused_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("expand_to_fused_kernel") workgroup_size(64, 1, 1) @expand_to_fused_kernel(%x_handle: buffer, %expanded_x_handle: buffer, %token_topk_to_pos_handle: buffer, %pos_to_expert_handle: buffer, %num_tokens: i32, %num_expanded_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %x = buffer.view %x_handle[%c0_bytes] : buffer -> view<[%num_tokens]x64xf16, %layout>
  %expanded_x = buffer.view %expanded_x_handle[%c0_bytes] : buffer -> view<[%num_expanded_tokens]x64xf16, %layout>
  %token_topk_to_pos = buffer.view %token_topk_to_pos_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xi32, %layout>
  %pos_to_expert = buffer.view %pos_to_expert_handle[%c0_bytes] : buffer -> view<[%num_expanded_tokens]xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %num_expanded_tokens_idx = index.cast %num_expanded_tokens : i32 to index
  %cmp = index.cmp slt, %bx, %num_expanded_tokens_idx : index
  scf.if %cmp {
    %load = view.load %pos_to_expert[%bx] : view<[%num_expanded_tokens]xi32, %layout> -> i32
    %const = scalar.constant 0 : i32
    %cmp_2 = scalar.cmpi slt, %load, %const : i32
    scf.if %cmp_2 {
      %c0 = index.constant 0 : index
      %c127 = index.constant 127 : index
      %sub = index.sub %c127, %thread_idx : index
      %c64 = index.constant 64 : index
      %div = index.div %sub, %c64 : index
      %tmp_ub = index.add %c0, %div : index
      %c1 = index.constant 1 : index
      scf.for %tmp = [%c0 to %tmp_ub step %c1] {
        %madd = index.madd %tmp, %c64, %thread_idx : index
        %const_2 = scalar.constant 0.0 : f16
        view.store %const_2, %expanded_x[%bx, %madd] : f16, view<[%num_expanded_tokens]x64xf16, %layout>
        scf.yield
      }
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  } else {
    scf.yield
  }
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %cmp_3 = index.cmp sge, %bx, %num_tokens_idx : index
  kernel.exit %cmp_3 : i1
  %bx_assumed, %num_tokens_assumed = index.assume %bx, %num_tokens_idx [lt(%bx, %num_tokens_idx)] : index, index
  %c0_2 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %k_ub = index.add %c0_2, %c2 : index
  %c1_2 = index.constant 1 : index
  scf.for %k = [%c0_2 to %k_ub step %c1_2] {
    %load_2 = view.load %token_topk_to_pos[%bx_assumed, %k] : view<[%num_tokens]x2xi32, %layout> -> i32
    %pos_assumed, %num_expanded_tokens_assumed = scalar.assume %load_2, %num_expanded_tokens [lt(%load_2, %num_expanded_tokens)] : i32, i32
    %const_3 = scalar.constant 0 : i32
    %cmp_4 = scalar.cmpi sge, %pos_assumed, %const_3 : i32
    scf.if %cmp_4 {
      %c127_2 = index.constant 127 : index
      %sub_2 = index.sub %c127_2, %thread_idx : index
      %c64_2 = index.constant 64 : index
      %div_2 = index.div %sub_2, %c64_2 : index
      %tmp_ub_2 = index.add %c0_2, %div_2 : index
      scf.for %tmp = [%c0_2 to %tmp_ub_2 step %c1_2] {
        %madd_2 = index.madd %tmp, %c64_2, %thread_idx : index
        %load_3 = view.load %x[%bx_assumed, %madd_2] : view<[%num_tokens]x64xf16, %layout> -> f16
        %pos_idx = index.cast %pos_assumed : i32 to index
        view.store %load_3, %expanded_x[%pos_idx, %madd_2] : f16, view<[%num_expanded_tokens]x64xf16, %layout>
        scf.yield
      }
      scf.yield
    } else {
      scf.yield
    }
    scf.yield
  }
  kernel.return
}
"""
