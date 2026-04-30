# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Registry and dispatch for MLIR-to-Loom op converters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
from typing import Any

from loom.importers.core import ImportBodyReport
from loom.importers.mlir.model import MlirConversionContext, SourceOp

Handler = Callable[[SourceOp, MlirConversionContext], bool]


class ConverterRegistry:
    """Maps MLIR operation names to conversion handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, op_name: str, handler: Handler) -> None:
        if op_name in self._handlers:
            raise ValueError(f"converter already registered for {op_name}")
        self._handlers[op_name] = handler

    def convert(self, op: SourceOp, context: MlirConversionContext) -> None:
        handler = self._handlers.get(op.op_name)
        if handler is None:
            context.record_unsupported_op(op)
            return
        with context.source_location(op):
            if not handler(op, context):
                context.record_unsupported_op(op)


def build_default_registry() -> ConverterRegistry:
    import loom.importers.mlir.convert_affine as convert_affine
    import loom.importers.mlir.convert_arith as convert_arith
    import loom.importers.mlir.convert_func as convert_func
    import loom.importers.mlir.convert_gpu as convert_gpu
    import loom.importers.mlir.convert_hal as convert_hal
    import loom.importers.mlir.convert_scf as convert_scf
    import loom.importers.mlir.convert_vector as convert_vector

    registry = ConverterRegistry()
    convert_affine.register(registry)
    convert_arith.register(registry)
    convert_func.register(registry)
    convert_gpu.register(registry)
    convert_hal.register(registry)
    convert_scf.register(registry)
    convert_vector.register(registry)
    return registry


def convert_function_body(
    function_operation: Any,
    context: MlirConversionContext,
    registry: ConverterRegistry | None = None,
) -> ImportBodyReport:
    registry = registry or build_default_registry()
    context.registry = registry
    convert_immediate_region_ops(function_operation, context)
    return context.finish()


def convert_immediate_region_ops(
    operation: Any,
    context: MlirConversionContext,
) -> None:
    registry = context.registry
    if not isinstance(registry, ConverterRegistry):
        raise ValueError("conversion context has no registry")
    for op in immediate_region_ops(operation, depth=1):
        registry.convert(op, context)


def immediate_region_ops(
    operation: Any,
    depth: int,
    *,
    region: Any | None = None,
) -> Iterator[SourceOp]:
    """Yields direct child ops using generic MLIR operations, not OpViews."""
    parent = generic_operation(operation)
    for child in walk_operations(parent):
        if child == parent or child.parent != parent:
            continue
        if region is not None and child.block.region != region:
            continue
        yield SourceOp(child, depth)


def walk_operations(operation: Any) -> Iterator[Any]:
    """Walks generic MLIR operations without constructing dialect OpViews."""
    root = generic_operation(operation)
    operations: list[Any] = []
    failure: BaseException | None = None

    # Keep the optional IREE binding import at traversal time so importing this
    # Python package does not require MLIR.
    ir = import_module("iree.compiler.ir")

    def callback(walked_operation: Any) -> Any:
        nonlocal failure
        try:
            operations.append(walked_operation)
        except BaseException as exc:
            failure = exc
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    root.walk(callback, ir.WalkOrder.PRE_ORDER)
    if failure is not None:
        raise failure
    yield from operations


def generic_operation(value: Any) -> Any:
    return getattr(value, "operation", value)
