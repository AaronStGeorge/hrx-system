# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _buffer_pair(
    tir: Any,
    *,
    shape: tuple[int, ...],
) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer(shape, "float32", name="src")
    dst_buffer = tir.decl_buffer(shape, "float32", name="dst")
    return src, dst, src_buffer, dst_buffer


def _call(tir: Any, dtype: str, name: str, *args: Any) -> Any:
    return tir.call_intrin(dtype, name, *args)


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
@tilelang_case(name="launch_thread_attrs", category="op", tags=("topology",))
def launch_thread_attrs(tir: Any, tvm: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(1024,))
    bx = tvm.te.thread_axis("blockIdx.x")
    tx = tvm.te.thread_axis("threadIdx.x")
    index = bx.var + bx.var + tx.var + tx.var
    body = tir.AttrStmt(
        bx,
        "thread_extent",
        tir.IntImm("int32", 8),
        tir.AttrStmt(
            tx,
            "thread_extent",
            tir.IntImm("int32", 64),
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [index]),
                [index],
            ),
        ),
    )
    prim_func = _prim_func(
        tir,
        name="launch_thread_attrs",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="launch_thread_attrs"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("launch_thread_attrs") @launch_thread_attrs(%src: buffer, %dst: buffer) {
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src[%c0_bytes] : buffer -> view<1024xf32, %layout>
  %dst_view = buffer.view %dst[%c0_bytes] : buffer -> view<1024xf32, %layout>
  %blockIdx_x = kernel.workgroup.id<x> : index
  %threadIdx_x = kernel.workitem.id<x> : index
  %add = index.add %blockIdx_x, %blockIdx_x : index
  %add_2 = index.add %add, %threadIdx_x : index
  %add_3 = index.add %add_2, %threadIdx_x : index
  %load = view.load %src_view[%add_3] : view<1024xf32, %layout> -> f32
  %add_4 = index.add %blockIdx_x, %blockIdx_x : index
  %add_5 = index.add %add_4, %threadIdx_x : index
  %add_6 = index.add %add_5, %threadIdx_x : index
  view.store %load, %dst_view[%add_6] : f32, view<1024xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="shared_storage_sync", category="op", tags=("topology", "memory"))
def shared_storage_sync(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(4,))
    body = tir.SeqStmt(
        [
            tir.Evaluate(_call(tir, "int32", "tir.tvm_storage_sync", "shared")),
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
                [tir.IntImm("int32", 0)],
            ),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="shared_storage_sync",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="shared_storage_sync"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("shared_storage_sync") @shared_storage_sync(%src: buffer, %dst: buffer) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_view = buffer.view %dst[%c0_bytes] : buffer -> view<4xf32, %layout>
  kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  view.store %load, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="thread_binding_loop", category="op", tags=("topology",))
def thread_binding_loop(tir: Any, tvm: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(128,))
    tx = tir.Var("tx", "int32")
    body = tir.For(
        tx,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 128),
        tir.ForKind.THREAD_BINDING,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [tx]), [tx]),
        thread_binding=tvm.te.thread_axis("threadIdx.x"),
    )
    prim_func = _prim_func(
        tir,
        name="thread_binding_loop",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="thread_binding_loop"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("thread_binding_loop") @thread_binding_loop(%src: buffer, %dst: buffer) {
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch {
  %c0_bytes = index.constant 0 : offset
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src[%c0_bytes] : buffer -> view<128xf32, %layout>
  %dst_view = buffer.view %dst[%c0_bytes] : buffer -> view<128xf32, %layout>
  %tx = kernel.workitem.id<x> : index
  %load = view.load %src_view[%tx] : view<128xf32, %layout> -> f32
  view.store %load, %dst_view[%tx] : f32, view<128xf32, %layout>
  kernel.return
}
"""
