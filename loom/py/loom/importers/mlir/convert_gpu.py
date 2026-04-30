# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR gpu ops at the kernel import boundary."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("gpu.barrier", convert_barrier)
    registry.register("gpu.thread_id", convert_thread_id)
    registry.register("gpu.shuffle", convert_shuffle)


def convert_thread_id(op: SourceOp, context: MlirConversionContext) -> bool:
    axis = gpu_dimension_axis(op.attr("dimension"))
    if axis is None:
        context.record_blocked(
            op.text, "gpu.thread_id dimension attr is not a known axis"
        )
        return True
    mapped = context.mapped(op.result())
    if mapped is None:
        mapped = context.builder.kernel.workitem_id(
            dimension=axis,
            results=[context.type("index")],
            name={"x": "tidx", "y": "tidy", "z": "tidz"}[axis],
        )
        context.map_result(op.result(), mapped, "index")
    context.record_converted(
        op.text,
        f"{context.ssa(mapped)} = kernel.workitem.id<{axis}> prelude",
    )
    return True


def convert_barrier(op: SourceOp, context: MlirConversionContext) -> bool:
    context.builder.kernel.barrier(
        memory_space="workgroup",
        ordering="acq_rel",
        scope="workgroup",
    )
    context.record_converted(op.text, "kernel.barrier")
    return True


def convert_shuffle(op: SourceOp, context: MlirConversionContext) -> bool:
    mode = gpu_shuffle_mode(op.attr("mode")) or "unknown"
    context.record_blocked(
        op.text,
        f"kernel.subgroup.shuffle<{mode}> once subgroup ops are available",
    )
    return True


def gpu_dimension_axis(attr: object) -> str | None:
    text = str(attr)
    prefix = "#gpu<dim "
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    axis = text[len(prefix) : -1]
    return axis if axis in {"x", "y", "z"} else None


def gpu_shuffle_mode(attr: object) -> str | None:
    text = str(attr)
    prefix = "#gpu<shuffle_mode "
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    return text[len(prefix) : -1]
