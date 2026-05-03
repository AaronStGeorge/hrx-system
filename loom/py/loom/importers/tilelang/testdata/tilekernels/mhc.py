# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _make_head_compute_mix_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_head_compute_mix_kernel(
        mhc_mult: int,
        mhc_pre_eps: float,
        token_block_size: int,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func  # type: ignore[untyped-decorator]
        def mhc_head_compute_mix_fwd_kernel(
            input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
            mhc_scale: T.Tensor[(1,), T.float32],
            mhc_base: T.Tensor[(mhc_mult,), T.float32],
            output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        ) -> None:
            with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as (pid,):
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        output_mix[i, j] = (
                            T.sigmoid(
                                input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                            )
                            + mhc_pre_eps
                        )

        return mhc_head_compute_mix_fwd_kernel

    return get_head_compute_mix_kernel


def _make_sinkhorn_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_sinkhorn_kernel(
        hidden_size: int,
        token_block_size: int,
        repeat: int,
        eps: float,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func  # type: ignore[untyped-decorator]
        def mhc_sinkhorn_kernel(
            comb_res_mix: T.Tensor[
                (num_tokens, hidden_size, hidden_size),
                T.float32,
            ],
            comb_res_mix_out: T.Tensor[
                (num_tokens, hidden_size, hidden_size),
                T.float32,
            ],
        ) -> None:
            with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as (pid_x,):
                comb_frag = T.alloc_fragment(
                    (token_block_size, hidden_size, hidden_size),
                    T.float32,
                )
                row_sum = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )
                col_sum = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )

                T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

                row_max = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )
                T.reduce_max(comb_frag, row_max, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                for _ in T.serial(repeat - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=2)
                    for i, j, k in T.Parallel(
                        token_block_size,
                        hidden_size,
                        hidden_size,
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

                    T.reduce_sum(comb_frag, col_sum, dim=1)
                    for i, j, k in T.Parallel(
                        token_block_size,
                        hidden_size,
                        hidden_size,
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                T.copy(
                    comb_frag,
                    comb_res_mix_out[pid_x * token_block_size, 0, 0],
                )

        return mhc_sinkhorn_kernel

    return get_sinkhorn_kernel


@tilelang_case(
    name="tilekernels_mhc_head_compute_mix_fwd_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "sigmoid", "amdgpu"),
)
def tilekernels_mhc_head_compute_mix_fwd_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_head_compute_mix_kernel(tilelang, T),
        args=(2, 1e-5, 2),
        target="hip -mcpu=gfx1100",
        name="mhc_head_compute_mix_fwd_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mhc_head_compute_mix_fwd_kernel") @mhc_head_compute_mix_fwd_kernel(%input_mix_handle: buffer, %mhc_scale_handle: buffer, %mhc_base_handle: buffer, %output_mix_handle: buffer, %num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c2 = index.constant 2 : index
  %add = index.add %num_tokens_idx, %c2 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c2 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %input_mix = buffer.view %input_mix_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xf32, %layout>
  %mhc_scale = buffer.view %mhc_scale_handle[%c0_bytes] : buffer -> view<1xf32, %layout>
  %mhc_base = buffer.view %mhc_base_handle[%c0_bytes] : buffer -> view<2xf32, %layout>
  %output_mix = buffer.view %output_mix_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c1 = index.constant 1 : index
  scf.for %i1 = [%c0 to %c2 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      %madd = index.madd %bx, %c2, %i1 : index
      %num_tokens_idx = index.cast %num_tokens : i32 to index
      %cmp = index.cmp slt, %madd, %num_tokens_idx : index
      scf.if %cmp {
        %load = view.load %input_mix[%madd, %j] : view<[%num_tokens]x2xf32, %layout> -> f32
        %load_2 = view.load %mhc_scale[%c0] : view<1xf32, %layout> -> f32
        %mulf = scalar.mulf %load, %load_2 : f32
        %load_3 = view.load %mhc_base[%j] : view<2xf32, %layout> -> f32
        %addf = scalar.addf %mulf, %load_3 : f32
        %one = scalar.constant 1.0 : f32
        %sigmoid_neg = scalar.negf %addf : f32
        %sigmoid_exp = scalar.expf %sigmoid_neg : f32
        %sigmoid_den = scalar.addf %one, %sigmoid_exp : f32
        %sigmoid = scalar.divf %one, %sigmoid_den : f32
        %const = scalar.constant 1.0000000000000001e-05 : f32
        %addf_2 = scalar.addf %sigmoid, %const : f32
        view.store %addf_2, %output_mix[%madd, %j] : f32, view<[%num_tokens]x2xf32, %layout>
      }
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tilekernels_mhc_sinkhorn_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "sinkhorn", "amdgpu"),
)
def tilekernels_mhc_sinkhorn_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_sinkhorn_kernel(tilelang, T),
        args=(2, 1, 2, 1e-6),
        target="hip -mcpu=gfx1100",
        name="mhc_sinkhorn_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mhc_sinkhorn_kernel") @mhc_sinkhorn_kernel(%comb_res_mix_handle: buffer, %comb_res_mix_out_handle: buffer, %num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1 = index.constant 1 : index
  %add = index.add %num_tokens_idx, %c1 : index
  %sub = index.sub %add, %c1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%sub, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %comb_res_mix = buffer.view %comb_res_mix_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2x2xf32, %layout>
  %comb_res_mix_out = buffer.view %comb_res_mix_out_handle[%c0_bytes] : buffer -> view<[%num_tokens]x2x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %comb_frag_bytes = index.constant 16 : offset
  %comb_frag_buffer = buffer.alloca %comb_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %comb_frag = buffer.view %comb_frag_buffer[%c0_bytes] : buffer -> view<1x2x2xf32, %layout>
  %row_sum_bytes = index.constant 8 : offset
  %row_sum_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %row_sum = buffer.view %row_sum_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %col_sum_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %col_sum = buffer.view %col_sum_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %row_max_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %row_max = buffer.view %row_max_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy = view.load %comb_res_mix[%bx, %i0, %i1] : view<[%num_tokens]x2x2xf32, %layout> -> f32
      view.store %copy, %comb_frag[%c0, %i0, %i1] : f32, view<1x2x2xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity = scalar.constant -inf : f32
      %reduce = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity : f32) -> (f32) {
        %reduce_value = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
        %maxnumf = scalar.maxnumf %acc, %reduce_value : f32
        scf.yield %maxnumf : f32
      }
      view.store %reduce, %row_max[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_2 = view.load %row_max[%i, %j] : view<1x2xf32, %layout> -> f32
        %subf = scalar.subf %load, %load_2 : f32
        %expf = scalar.expf %subf : f32
        view.store %expf, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity_2 = scalar.constant 0 : f32
      %reduce_2 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_2 : f32) -> (f32) {
        %reduce_value_2 = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
        %addf = scalar.addf %acc, %reduce_value_2 : f32
        scf.yield %addf : f32
      }
      view.store %reduce_2, %row_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load_3 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_4 = view.load %row_sum[%i, %j] : view<1x2xf32, %layout> -> f32
        %divf = scalar.divf %load_3, %load_4 : f32
        %const = scalar.constant 9.9999999999999995e-07 : f32
        %addf_2 = scalar.addf %divf, %const : f32
        view.store %addf_2, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity_3 = scalar.constant 0 : f32
      %reduce_3 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_3 : f32) -> (f32) {
        %reduce_value_3 = view.load %comb_frag[%i0, %r, %i1] : view<1x2x2xf32, %layout> -> f32
        %addf_3 = scalar.addf %acc, %reduce_value_3 : f32
        scf.yield %addf_3 : f32
      }
      view.store %reduce_3, %col_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load_5 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_6 = view.load %col_sum[%i, %k] : view<1x2xf32, %layout> -> f32
        %const_2 = scalar.constant 9.9999999999999995e-07 : f32
        %addf_4 = scalar.addf %load_6, %const_2 : f32
        %divf_2 = scalar.divf %load_5, %addf_4 : f32
        view.store %divf_2, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %v = [%c0 to %c1 step %c1] {
    scf.for %i0 = [%c0 to %c1 step %c1] {
      scf.for %i1 = [%c0 to %c2 step %c1] {
        %identity_4 = scalar.constant 0 : f32
        %reduce_4 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_4 : f32) -> (f32) {
          %reduce_value_4 = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
          %addf_5 = scalar.addf %acc, %reduce_value_4 : f32
          scf.yield %addf_5 : f32
        }
        view.store %reduce_4, %row_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
      }
    }
    scf.for %i = [%c0 to %c1 step %c1] {
      scf.for %j = [%c0 to %c2 step %c1] {
        scf.for %k = [%c0 to %c2 step %c1] {
          %load_7 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
          %load_8 = view.load %row_sum[%i, %j] : view<1x2xf32, %layout> -> f32
          %const_3 = scalar.constant 9.9999999999999995e-07 : f32
          %addf_6 = scalar.addf %load_8, %const_3 : f32
          %divf_3 = scalar.divf %load_7, %addf_6 : f32
          view.store %divf_3, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
        }
      }
    }
    scf.for %i0 = [%c0 to %c1 step %c1] {
      scf.for %i1 = [%c0 to %c2 step %c1] {
        %identity_5 = scalar.constant 0 : f32
        %reduce_5 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_5 : f32) -> (f32) {
          %reduce_value_5 = view.load %comb_frag[%i0, %r, %i1] : view<1x2x2xf32, %layout> -> f32
          %addf_7 = scalar.addf %acc, %reduce_value_5 : f32
          scf.yield %addf_7 : f32
        }
        view.store %reduce_5, %col_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
      }
    }
    scf.for %i = [%c0 to %c1 step %c1] {
      scf.for %j = [%c0 to %c2 step %c1] {
        scf.for %k = [%c0 to %c2 step %c1] {
          %load_9 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
          %load_10 = view.load %col_sum[%i, %k] : view<1x2xf32, %layout> -> f32
          %const_4 = scalar.constant 9.9999999999999995e-07 : f32
          %addf_8 = scalar.addf %load_10, %const_4 : f32
          %divf_4 = scalar.divf %load_9, %addf_8 : f32
          view.store %divf_4, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
        }
      }
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy_2 = view.load %comb_frag[%c0, %i0, %i1] : view<1x2x2xf32, %layout> -> f32
      view.store %copy_2, %comb_res_mix_out[%bx, %i0, %i1] : f32, view<[%num_tokens]x2x2xf32, %layout>
    }
  }
  kernel.return
}
"""
