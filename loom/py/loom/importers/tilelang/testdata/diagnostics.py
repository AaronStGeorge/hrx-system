# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


# ====
# ERROR@+1: "no registered converter for `StringImm`" "unsupported_marker"
@tilelang_case(name="unsupported_string_evaluate", category="diagnostic")
def unsupported_string_evaluate(tir: Any) -> TileLangImportInput:
    source = tir.Var("source", "handle")
    source_buffer = tir.decl_buffer((1,), "float32", name="source")
    body = tir.Evaluate(tir.StringImm("unsupported_marker"))
    prim_func = tir.PrimFunc(
        [source],
        body,
        buffer_map={source: source_buffer},
    ).with_attr("global_symbol", "unsupported_string_evaluate")
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="unsupported_string_evaluate",
    )


# ====
# ERROR@+1: "target-independent kernel assert/trap op"
@tilelang_case(
    name="device_assert_with_msg",
    category="diagnostic",
    tags=("assert", "kernel"),
)
def device_assert_with_msg(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def device_assert_with_msg_kernel(
            src: T.Tensor[(1,), T.float32],
            dst: T.Tensor[(1,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                value = src[0]
                T.device_assert(T.isfinite(value), msg="finite input required")
                dst[0] = value

        return device_assert_with_msg_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="device_assert_with_msg_kernel",
    )


# ====
# ERROR@+1: "copy annotation `coalesced_width` needs scheduling import"
@tilelang_case(
    name="tileop_copy_coalesced_width",
    category="diagnostic",
    tags=("tileop", "copy", "annotation"),
)
def tileop_copy_coalesced_width(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_coalesced_width_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(4,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst, coalesced_width=4)

        return tileop_copy_coalesced_width_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_coalesced_width_kernel",
    )


# ====
# ERROR@+1: "tl.tileop.gemm" "vector.fragment" "m=16" "a=a_shared"
@tilelang_case(
    name="tileop_gemm_deferred",
    category="diagnostic",
    tags=("tileop", "gemm", "matrix"),
)
def tileop_gemm_deferred(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_gemm_deferred_kernel(
        a: T.Tensor[(16, 16), T.float16],
        b: T.Tensor[(16, 16), T.float16],
        c: T.Tensor[(16, 16), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            a_shared = T.alloc_shared((16, 16), T.float16)
            b_shared = T.alloc_shared((16, 16), T.float16)
            c_local = T.alloc_fragment((16, 16), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.fill(c_local, 0.0)
            T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, c)

    return TileLangImportInput(
        source=tileop_gemm_deferred_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_gemm_deferred_kernel",
    )


# ====
# ERROR@+1: "call `tl.shfl_xor_sync` mask must be the full warp mask"
@tilelang_case(
    name="masked_warp_shuffle",
    category="diagnostic",
    tags=("topology", "subgroup", "shuffle"),
)
def masked_warp_shuffle(tir: Any, tvm: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    src_buffer = tir.decl_buffer((32,), "float32", name="src")
    thread_index = tvm.te.thread_axis("threadIdx.x")
    body = tir.AttrStmt(
        thread_index,
        "thread_extent",
        tir.IntImm("int32", 32),
        tir.Evaluate(
            tir.call_intrin(
                "float32",
                tir.op.Op.get("tl.shfl_xor_sync"),
                tir.IntImm("uint32", 0xFFFF),
                tir.BufferLoad(src_buffer, [thread_index.var]),
                tir.IntImm("int32", 1),
                tir.IntImm("int32", 32),
            )
        ),
    )
    prim_func = tir.PrimFunc(
        [src],
        body,
        buffer_map={src: src_buffer},
    ).with_attr("global_symbol", "masked_warp_shuffle_kernel")
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="masked_warp_shuffle_kernel",
    )


# ====
# ERROR@+1: "call `tl.sync_warp` mask must be the full warp mask"
@tilelang_case(
    name="masked_warp_sync",
    category="diagnostic",
    tags=("topology", "subgroup", "barrier"),
)
def masked_warp_sync(tir: Any, tvm: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    src_buffer = tir.decl_buffer((32,), "float32", name="src")
    thread_index = tvm.te.thread_axis("threadIdx.x")
    body = tir.AttrStmt(
        thread_index,
        "thread_extent",
        tir.IntImm("int32", 32),
        tir.Evaluate(
            tir.call_intrin(
                "void",
                tir.op.Op.get("tl.sync_warp"),
                tir.IntImm("uint32", 0xFFFF),
            )
        ),
    )
    prim_func = tir.PrimFunc(
        [src],
        body,
        buffer_map={src: src_buffer},
    ).with_attr("global_symbol", "masked_warp_sync_kernel")
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="masked_warp_sync_kernel",
    )
