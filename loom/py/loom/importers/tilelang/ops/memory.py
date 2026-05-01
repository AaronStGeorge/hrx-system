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
from loom.importers.tilelang.nodes import dtype, node_text


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
    indices = [
        converter.convert_expr(index, context, index_like=True)
        for index in tuple(getattr(stmt, "indices", ()))
    ]
    if value is None or any(index is None for index in indices):
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    context.builder.view.store(
        value=value,
        view=view,
        indices=[index for index in indices if index is not None],
    )
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
    indices = [
        converter.convert_expr(index, context, index_like=True)
        for index in tuple(getattr(expr, "indices", ()))
    ]
    if any(index is None for index in indices):
        context.record_blocked(node_text(expr), "buffer load indices are not mapped")
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    result = context.builder.view.load(
        view=view,
        indices=[index for index in indices if index is not None],
        results=[result_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = view.load")
    return result
