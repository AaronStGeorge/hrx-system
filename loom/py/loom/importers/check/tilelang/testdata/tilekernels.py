# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


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
# target.profile @hip_mcpu_gfx942 preset("hip -mcpu=gfx942")
#
# kernel.def target(@hip_mcpu_gfx942) export("batched_transpose_kernel") workgroup_size(256, 1, 1) @batched_transpose_kernel(%x_handle: buffer, %out_handle: buffer, %num_batches: i32, %shape_x: i32, %shape_y: i32, %stride_x: i32) {
#   %c0_bytes = index.constant 0 : offset
#   %x_handle = buffer.view %x_handle[%c0_bytes] : buffer -> view<?x?x?xf16>
#   %out_handle = buffer.view %out_handle[%c0_bytes] : buffer -> view<?x?x?xf16>
#   %bx = kernel.workgroup.id<x> : index
#   %by = kernel.workgroup.id<y> : index
#   %bz = kernel.workgroup.id<z> : index
#   %thread_id = kernel.workitem.id<x> : index
#   %ty = kernel.workitem.id<y> : index
#   %tz = kernel.workitem.id<z> : index
#   %out_shared_bytes = index.constant 33792 : offset
#   %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
#   %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128x132xf16>
#   %tmp_bytes = index.constant 32 : offset
#   %tmp_buffer = buffer.alloca %tmp_bytes {base_alignment = 2, memory_space = private} : buffer
#   %tmp = buffer.view %tmp_buffer[%c0_bytes] : buffer -> view<4x4xf16>
#   %tmp_row_bytes = index.constant 8 : offset
#   %tmp_row_buffer = buffer.alloca %tmp_row_bytes {base_alignment = 2, memory_space = private} : buffer
#   %tmp_row = buffer.view %tmp_row_buffer[%c0_bytes] : buffer -> view<4xf16>
#   %c = index.constant 32 : index
#   %div = index.div %thread_id, %c : index
#   %c_2 = index.constant 32 : index
#   %rem = index.rem %thread_id, %c_2 : index
#   %shape_x_assumed = scalar.assume %shape_x [mul(%shape_x, 128)] : i32
#   %shape_y_assumed = scalar.assume %shape_y [mul(%shape_y, 128)] : i32
#   %stride_x_assumed = scalar.assume %stride_x [mul(%stride_x, 4)] : i32
#   %c_3 = index.constant 0 : index
#   %c_4 = index.constant 4 : index
#   %i_outer_ub = index.add %c_3, %c_4 : index
#   %c1 = index.constant 1 : index
#   scf.for %i_outer = [%c_3 to %i_outer_ub step %c1] {
#     %c_5 = index.constant 8 : index
#     %mul = index.mul %i_outer, %c_5 : index
#     %add = index.add %mul, %div : index
#     %c_6 = index.constant 0 : index
#     %c_7 = index.constant 4 : index
#     %j_ub = index.add %c_6, %c_7 : index
#     scf.for %j = [%c_6 to %j_ub step %c1] {
#       %c_8 = index.constant 0 : index
#       %c_9 = index.constant 4 : index
#       %k_ub = index.add %c_8, %c_9 : index
#       scf.for %k = [%c_8 to %k_ub step %c1] {
#         %c_10 = index.constant 128 : index
#         %mul_2 = index.mul %by, %c_10 : index
#         %c_11 = index.constant 4 : index
#         %mul_3 = index.mul %add, %c_11 : index
#         %add_2 = index.add %mul_2, %mul_3 : index
#         %add_3 = index.add %add_2, %j : index
#         %c_12 = index.constant 128 : index
#         %mul_4 = index.mul %bx, %c_12 : index
#         %c_13 = index.constant 4 : index
#         %mul_5 = index.mul %rem, %c_13 : index
#         %add_4 = index.add %mul_4, %mul_5 : index
#         %add_5 = index.add %add_4, %k : index
#         %load = view.load %x_handle[%bz, %add_3, %add_5] : view<?x?x?xf16> -> f16
#         view.store %load, %tmp_row[%k] : f16, view<4xf16>
#         scf.yield
#       }
#       %c_14 = index.constant 0 : index
#       %c_15 = index.constant 4 : index
#       %k_ub_2 = index.add %c_14, %c_15 : index
#       scf.for %k = [%c_14 to %k_ub_2 step %c1] {
#         %load_2 = view.load %tmp_row[%k] : view<4xf16> -> f16
#         view.store %load_2, %tmp[%k, %j] : f16, view<4x4xf16>
#         scf.yield
#       }
#       scf.yield
#     }
#     %c_16 = index.constant 0 : index
#     %c_17 = index.constant 4 : index
#     %j_ub_2 = index.add %c_16, %c_17 : index
#     scf.for %j = [%c_16 to %j_ub_2 step %c1] {
#       %c_18 = index.constant 4 : index
#       %div_2 = index.div %thread_id, %c_18 : index
#       %add_6 = index.add %j, %div_2 : index
#       %c_19 = index.constant 4 : index
#       %rem_2 = index.rem %add_6, %c_19 : index
#       %c_20 = index.constant 0 : index
#       %c_21 = index.constant 4 : index
#       %k_ub_3 = index.add %c_20, %c_21 : index
#       scf.for %k = [%c_20 to %k_ub_3 step %c1] {
#         %load_3 = view.load %tmp[%rem_2, %k] : view<4x4xf16> -> f16
#         %c_22 = index.constant 4 : index
#         %mul_6 = index.mul %rem, %c_22 : index
#         %add_7 = index.add %mul_6, %rem_2 : index
#         %c_23 = index.constant 4 : index
#         %mul_7 = index.mul %add, %c_23 : index
#         %add_8 = index.add %mul_7, %k : index
#         view.store %load_3, %out_shared[%add_7, %add_8] : f16, view<128x132xf16>
#         scf.yield
#       }
#       scf.yield
#     }
#     scf.yield
#   }
#   kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
#   %c_24 = index.constant 0 : index
#   %c_25 = index.constant 128 : index
#   %i_ub = index.add %c_24, %c_25 : index
#   scf.for %i = [%c_24 to %i_ub step %c1] {
#     %c_26 = index.constant 0 : index
#     %c_27 = index.constant 128 : index
#     %j_ub_3 = index.add %c_26, %c_27 : index
#     scf.for %j = [%c_26 to %j_ub_3 step %c1] {
#       %load_4 = view.load %out_shared[%i, %j] : view<128x132xf16> -> f16
#       %c_28 = index.constant 128 : index
#       %mul_8 = index.mul %bx, %c_28 : index
#       %add_9 = index.add %mul_8, %i : index
#       %c_29 = index.constant 128 : index
#       %mul_9 = index.mul %by, %c_29 : index
#       %add_10 = index.add %mul_9, %j : index
#       view.store %load_4, %out_handle[%bz, %add_9, %add_10] : f16, view<?x?x?xf16>
#       scf.yield
#     }
#     scf.yield
#   }
#   kernel.return
# }


# ====
@tilelang_case(
    name="tilekernels_batched_transpose_gfx1100",
    category="kernel",
    tags=("tilekernels", "transpose", "amdgpu"),
)
def tilekernels_batched_transpose_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _batched_transpose_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
# target.profile @hip_mcpu_gfx1100 preset("hip -mcpu=gfx1100")
#
# kernel.def target(@hip_mcpu_gfx1100) export("batched_transpose_kernel") workgroup_size(256, 1, 1) @batched_transpose_kernel(%x_handle: buffer, %out_handle: buffer, %num_batches: i32, %shape_x: i32, %shape_y: i32, %stride_x: i32) {
#   %c0_bytes = index.constant 0 : offset
#   %x_handle = buffer.view %x_handle[%c0_bytes] : buffer -> view<?x?x?xf16>
#   %out_handle = buffer.view %out_handle[%c0_bytes] : buffer -> view<?x?x?xf16>
#   %bx = kernel.workgroup.id<x> : index
#   %by = kernel.workgroup.id<y> : index
#   %bz = kernel.workgroup.id<z> : index
#   %thread_id = kernel.workitem.id<x> : index
#   %ty = kernel.workitem.id<y> : index
#   %tz = kernel.workitem.id<z> : index
#   %out_shared_bytes = index.constant 33792 : offset
#   %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
#   %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128x132xf16>
#   %tmp_bytes = index.constant 32 : offset
#   %tmp_buffer = buffer.alloca %tmp_bytes {base_alignment = 2, memory_space = private} : buffer
#   %tmp = buffer.view %tmp_buffer[%c0_bytes] : buffer -> view<4x4xf16>
#   %tmp_row_bytes = index.constant 8 : offset
#   %tmp_row_buffer = buffer.alloca %tmp_row_bytes {base_alignment = 2, memory_space = private} : buffer
#   %tmp_row = buffer.view %tmp_row_buffer[%c0_bytes] : buffer -> view<4xf16>
#   %c = index.constant 32 : index
#   %div = index.div %thread_id, %c : index
#   %c_2 = index.constant 32 : index
#   %rem = index.rem %thread_id, %c_2 : index
#   %shape_x_assumed = scalar.assume %shape_x [mul(%shape_x, 128)] : i32
#   %shape_y_assumed = scalar.assume %shape_y [mul(%shape_y, 128)] : i32
#   %stride_x_assumed = scalar.assume %stride_x [mul(%stride_x, 4)] : i32
#   %c_3 = index.constant 0 : index
#   %c_4 = index.constant 4 : index
#   %i_outer_ub = index.add %c_3, %c_4 : index
#   %c1 = index.constant 1 : index
#   scf.for %i_outer = [%c_3 to %i_outer_ub step %c1] {
#     %c_5 = index.constant 8 : index
#     %mul = index.mul %i_outer, %c_5 : index
#     %add = index.add %mul, %div : index
#     %c_6 = index.constant 0 : index
#     %c_7 = index.constant 4 : index
#     %j_ub = index.add %c_6, %c_7 : index
#     scf.for %j = [%c_6 to %j_ub step %c1] {
#       %c_8 = index.constant 0 : index
#       %c_9 = index.constant 4 : index
#       %k_ub = index.add %c_8, %c_9 : index
#       scf.for %k = [%c_8 to %k_ub step %c1] {
#         %c_10 = index.constant 128 : index
#         %mul_2 = index.mul %by, %c_10 : index
#         %c_11 = index.constant 4 : index
#         %mul_3 = index.mul %add, %c_11 : index
#         %add_2 = index.add %mul_2, %mul_3 : index
#         %add_3 = index.add %add_2, %j : index
#         %c_12 = index.constant 128 : index
#         %mul_4 = index.mul %bx, %c_12 : index
#         %c_13 = index.constant 4 : index
#         %mul_5 = index.mul %rem, %c_13 : index
#         %add_4 = index.add %mul_4, %mul_5 : index
#         %add_5 = index.add %add_4, %k : index
#         %load = view.load %x_handle[%bz, %add_3, %add_5] : view<?x?x?xf16> -> f16
#         view.store %load, %tmp_row[%k] : f16, view<4xf16>
#         scf.yield
#       }
#       %c_14 = index.constant 0 : index
#       %c_15 = index.constant 4 : index
#       %k_ub_2 = index.add %c_14, %c_15 : index
#       scf.for %k = [%c_14 to %k_ub_2 step %c1] {
#         %load_2 = view.load %tmp_row[%k] : view<4xf16> -> f16
#         view.store %load_2, %tmp[%k, %j] : f16, view<4x4xf16>
#         scf.yield
#       }
#       scf.yield
#     }
#     %c_16 = index.constant 0 : index
#     %c_17 = index.constant 4 : index
#     %j_ub_2 = index.add %c_16, %c_17 : index
#     scf.for %j = [%c_16 to %j_ub_2 step %c1] {
#       %c_18 = index.constant 4 : index
#       %div_2 = index.div %thread_id, %c_18 : index
#       %add_6 = index.add %j, %div_2 : index
#       %c_19 = index.constant 4 : index
#       %rem_2 = index.rem %add_6, %c_19 : index
#       %c_20 = index.constant 0 : index
#       %c_21 = index.constant 4 : index
#       %k_ub_3 = index.add %c_20, %c_21 : index
#       scf.for %k = [%c_20 to %k_ub_3 step %c1] {
#         %load_3 = view.load %tmp[%rem_2, %k] : view<4x4xf16> -> f16
#         %c_22 = index.constant 4 : index
#         %mul_6 = index.mul %rem, %c_22 : index
#         %add_7 = index.add %mul_6, %rem_2 : index
#         %c_23 = index.constant 4 : index
#         %mul_7 = index.mul %add, %c_23 : index
#         %add_8 = index.add %mul_7, %k : index
#         view.store %load_3, %out_shared[%add_7, %add_8] : f16, view<128x132xf16>
#         scf.yield
#       }
#       scf.yield
#     }
#     scf.yield
#   }
#   kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
#   %c_24 = index.constant 0 : index
#   %c_25 = index.constant 128 : index
#   %i_ub = index.add %c_24, %c_25 : index
#   scf.for %i = [%c_24 to %i_ub step %c1] {
#     %c_26 = index.constant 0 : index
#     %c_27 = index.constant 128 : index
#     %j_ub_3 = index.add %c_26, %c_27 : index
#     scf.for %j = [%c_26 to %j_ub_3 step %c1] {
#       %load_4 = view.load %out_shared[%i, %j] : view<128x132xf16> -> f16
#       %c_28 = index.constant 128 : index
#       %mul_8 = index.mul %bx, %c_28 : index
#       %add_9 = index.add %mul_8, %i : index
#       %c_29 = index.constant 128 : index
#       %mul_9 = index.mul %by, %c_29 : index
#       %add_10 = index.add %mul_9, %j : index
#       view.store %load_4, %out_handle[%bz, %add_9, %add_10] : f16, view<?x?x?xf16>
#       scf.yield
#     }
#     scf.yield
#   }
#   kernel.return
# }
