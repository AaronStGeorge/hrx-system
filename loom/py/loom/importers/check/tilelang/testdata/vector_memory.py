# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501, ERA001

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case
from loom.importers.check.tilelang.testdata.tir_fakes import (
    Broadcast,
    Buffer,
    BufferLoad,
    BufferStore,
    FloatImm,
    IntImm,
    PrimFunc,
    Ramp,
    Var,
)


# ====
@tilelang_case(name="broadcast_vector_store", category="op", tags=("vector",))
def broadcast_vector_store() -> TileLangImportInput:
    dst = Var("dst")
    dst_buffer = Buffer("dst", (16,), "float32")
    body = BufferStore(
        dst_buffer,
        Broadcast(FloatImm(0.0), 4),
        [Ramp(IntImm(4), IntImm(1), 4)],
    )
    prim_func = PrimFunc(
        [dst],
        {dst: dst_buffer},
        body,
        attrs={"global_symbol": "broadcast_vector_store"},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip",
        name="broadcast_vector_store",
    )


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("broadcast_vector_store") workgroup_size(1, 1, 1) @broadcast_vector_store(%dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf32>
#   %const = scalar.constant 0.0 : f32
#   %splat = vector.splat %const : vector<4xf32>
#   %c = index.constant 4 : index
#   vector.store %splat, %dst[%c] : vector<4xf32>, view<16xf32>
#   kernel.return
# }


# ====
@tilelang_case(name="ramp_vector_load", category="op", tags=("vector",))
def ramp_vector_load() -> TileLangImportInput:
    src, dst = Var("src"), Var("dst")
    src_buffer = Buffer("src", (16,), "float32")
    dst_buffer = Buffer("dst", (16,), "float32")
    body = BufferStore(
        dst_buffer,
        BufferLoad(src_buffer, [Ramp(IntImm(8), IntImm(1), 4)], "float32x4"),
        [Ramp(IntImm(0), IntImm(1), 4)],
    )
    prim_func = PrimFunc(
        [src, dst],
        {src: src_buffer, dst: dst_buffer},
        body,
        attrs={"global_symbol": "ramp_vector_load"},
    )
    return TileLangImportInput(source=prim_func, target="hip", name="ramp_vector_load")


# ----
# target.profile @hip preset("hip")
#
# kernel.def target(@hip) export("ramp_vector_load") workgroup_size(1, 1, 1) @ramp_vector_load(%src: buffer, %dst: buffer) {
#   %c0_bytes = index.constant 0 : offset
#   %src = buffer.view %src[%c0_bytes] : buffer -> view<16xf32>
#   %dst = buffer.view %dst[%c0_bytes] : buffer -> view<16xf32>
#   %c = index.constant 8 : index
#   %load = vector.load %src[%c] : view<16xf32> -> vector<4xf32>
#   %c_2 = index.constant 0 : index
#   vector.store %load, %dst[%c_2] : vector<4xf32>, view<16xf32>
#   kernel.return
# }
