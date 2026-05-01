# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer load/store TileLang/TIR node converters."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_kind, node_text
from loom.importers.tilelang.ops.topology import integer_value
from loom.importers.tilelang.ops.vector import vector_lanes
from loom.ir import ShapedType, TypeKind


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("BufferStore", convert_buffer_store)
    registry.register_expression("BufferLoad", convert_buffer_load)


def convert_buffer_store(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR buffer store as view.store."""

    buffer = getattr(stmt, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(node_text(stmt), "buffer store target is not mapped")
        return
    value = converter.convert_expr(getattr(stmt, "value", None), context)
    source_indices = tuple(getattr(stmt, "indices", ()))
    if value is None:
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    has_ramp_index = _has_ramp_index(stmt)
    if has_ramp_index and not is_vector_type(value.type):
        context.record_blocked(
            node_text(stmt), "vector store value is not vector-typed"
        )
        return
    if is_vector_type(value.type) or has_ramp_index:
        vector_indices = _vector_memory_indices(
            source_indices,
            context,
            converter,
        )
        if vector_indices is None:
            context.record_blocked(
                node_text(stmt), "vector store indices are not mapped"
            )
            return
        context.builder.vector.store(
            value=value,
            view=view,
            indices=vector_indices,
        )
        context.record_converted(node_text(stmt), "vector.store")
        return
    indices = [
        converter.convert_expr(index, context, index_like=True)
        for index in source_indices
    ]
    if any(index is None for index in indices):
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    mapped_indices: list[int | ValueRef] = [
        index for index in indices if index is not None
    ]
    context.builder.view.store(value=value, view=view, indices=mapped_indices)
    context.record_converted(node_text(stmt), "view.store")


def convert_buffer_load(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR buffer load as view.load."""

    buffer = getattr(expr, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(node_text(expr), "buffer load source is not mapped")
        return None
    source_indices = tuple(getattr(expr, "indices", ()))
    result_type = context.type_converter.map_dtype(dtype(expr))
    if is_vector_type(result_type) or _has_ramp_index(expr):
        vector_indices = _vector_memory_indices(source_indices, context, converter)
        if vector_indices is None:
            context.record_blocked(
                node_text(expr), "vector load indices are not mapped"
            )
            return None
        if not is_vector_type(result_type):
            lanes = _single_ramp_lanes(source_indices)
            if lanes is None:
                context.record_blocked(
                    node_text(expr), "vector load lanes are not mapped"
                )
                return None
            result_type = context.type_converter.vector_type(dtype(expr), lanes)
        result = context.builder.vector.load(
            view=view,
            indices=vector_indices,
            results=[result_type],
            name=context.fresh_name("load"),
        )
        context.map_value(expr, result, str(result_type))
        context.record_converted(
            node_text(expr), f"{context.ssa(result)} = vector.load"
        )
        return result
    indices = [
        converter.convert_expr(index, context, index_like=True)
        for index in source_indices
    ]
    if any(index is None for index in indices):
        context.record_blocked(node_text(expr), "buffer load indices are not mapped")
        return None
    result = context.builder.view.load(
        view=view,
        indices=[index for index in indices if index is not None],
        results=[result_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = view.load")
    return result


def is_vector_type(value_type: object) -> bool:
    return (
        isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VECTOR
    )


def _has_ramp_index(expr: object) -> bool:
    return any(
        node_kind(index) == "Ramp" for index in tuple(getattr(expr, "indices", ()))
    )


def _vector_memory_indices(
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> list[int | ValueRef] | None:
    mapped_indices: list[int | ValueRef] = []
    ramp_count = 0
    for source_index in source_indices:
        if node_kind(source_index) == "Ramp":
            ramp_count += 1
            if ramp_count > 1:
                return None
            stride = integer_value(getattr(source_index, "stride", None))
            if stride != 1:
                return None
            base = converter.convert_expr(
                getattr(source_index, "base", None),
                context,
                index_like=True,
            )
            if base is None:
                return None
            mapped_indices.append(base)
            continue
        mapped = converter.convert_expr(source_index, context, index_like=True)
        if mapped is None:
            return None
        mapped_indices.append(mapped)
    return mapped_indices


def _single_ramp_lanes(source_indices: tuple[object, ...]) -> int | None:
    lanes: int | None = None
    for source_index in source_indices:
        if node_kind(source_index) != "Ramp":
            continue
        if lanes is not None:
            return None
        lanes = vector_lanes(source_index)
    return lanes
