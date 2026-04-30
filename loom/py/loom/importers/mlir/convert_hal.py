# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for IREE HAL interface ops."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp

AXIS_BY_DIM = {0: "x", 1: "y", 2: "z"}


def register(registry: ConverterRegistry) -> None:
    registry.register("hal.interface.binding.subspan", convert_binding)
    registry.register("hal.interface.workgroup.id", convert_workgroup_id)


def convert_binding(op: SourceOp, context: MlirConversionContext) -> bool:
    binding_attr = op.attr("binding")
    if binding_attr is None:
        context.record_blocked(op.text, "binding subspan has no binding attribute")
        return True
    mapped = context.mapped(op.result())
    if mapped is None:
        context.record_blocked(
            op.text,
            "binding subspan reached body conversion before kernel ABI prelude",
        )
        return True
    context.record_converted(op.text, f"{context.ssa(mapped)} = buffer.view prelude")
    return True


def convert_workgroup_id(op: SourceOp, context: MlirConversionContext) -> bool:
    dim = context.attr_decoder.integer(op.attr("dimension"))
    if dim is None:
        context.record_blocked(op.text, "workgroup id has no dimension attribute")
        return True
    axis = AXIS_BY_DIM.get(dim)
    if axis is None:
        context.record_blocked(
            op.text,
            f"workgroup dimension {dim} has no Loom axis spelling",
        )
        return True
    mapped = context.mapped(op.result())
    if mapped is None:
        mapped = context.builder.kernel.workgroup_id(
            dimension=axis,
            results=[context.type("index")],
            name=f"wg{axis}",
        )
        context.map_result(op.result(), mapped, "index")
    context.record_converted(
        op.text,
        f"{context.ssa(mapped)} = kernel.workgroup.id<{axis}> prelude",
    )
    return True
