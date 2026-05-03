# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


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

kernel.def target(@hip_mcpu_gfx1100) export("expand_to_fused_kernel") @expand_to_fused_kernel(%x_handle: buffer, %expanded_x_handle: buffer, %token_topk_to_pos_handle: buffer, %pos_to_expert_handle: buffer, %num_tokens: i32, %num_expanded_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %num_expanded_tokens_idx = index.cast %num_expanded_tokens : i32 to index
  %max = index.max %num_tokens_idx, %num_expanded_tokens_idx : index
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%max, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch {
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
      %c1 = index.constant 1 : index
      scf.for %tmp = [%c0 to %div step %c1] {
        %madd = index.madd %tmp, %c64, %thread_idx : index
        %const_2 = scalar.constant 0.0 : f16
        view.store %const_2, %expanded_x[%bx, %madd] : f16, view<[%num_expanded_tokens]x64xf16, %layout>
      }
    }
  }
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %cmp_3 = index.cmp sge, %bx, %num_tokens_idx : index
  kernel.exit %cmp_3 : i1
  %bx_assumed, %num_tokens_assumed = index.assume %bx, %num_tokens_idx [lt(%bx, %num_tokens_idx)] : index, index
  %c0_2 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c1_2 = index.constant 1 : index
  scf.for %k = [%c0_2 to %c2 step %c1_2] {
    %load_2 = view.load %token_topk_to_pos[%bx_assumed, %k] : view<[%num_tokens]x2xi32, %layout> -> i32
    %pos_assumed, %num_expanded_tokens_assumed = scalar.assume %load_2, %num_expanded_tokens [lt(%load_2, %num_expanded_tokens)] : i32, i32
    %const_3 = scalar.constant 0 : i32
    %cmp_4 = scalar.cmpi sge, %pos_assumed, %const_3 : i32
    scf.if %cmp_4 {
      %c127_2 = index.constant 127 : index
      %sub_2 = index.sub %c127_2, %thread_idx : index
      %c64_2 = index.constant 64 : index
      %div_2 = index.div %sub_2, %c64_2 : index
      scf.for %tmp = [%c0_2 to %div_2 step %c1_2] {
        %madd_2 = index.madd %tmp, %c64_2, %thread_idx : index
        %load_3 = view.load %x[%bx_assumed, %madd_2] : view<[%num_tokens]x64xf16, %layout> -> f16
        %pos_idx = index.cast %pos_assumed : i32 to index
        view.store %load_3, %expanded_x[%pos_idx, %madd_2] : f16, view<[%num_expanded_tokens]x64xf16, %layout>
      }
    }
  }
  kernel.return
}
"""
