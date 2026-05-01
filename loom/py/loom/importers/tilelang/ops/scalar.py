# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar expression TileLang/TIR node converters."""

from __future__ import annotations

from typing import cast

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_text
from loom.ir import INDEX


def register(registry: TileLangConverterRegistry) -> None:
    for node_name in ("IntImm", "FloatImm"):
        registry.register_expression(node_name, convert_immediate)
    registry.register_expression("Var", convert_var)
    for node_name in sorted(
        set(_BINARY_INDEX_OPS) | set(_BINARY_INTEGER_OPS) | set(_BINARY_FLOAT_OPS)
    ):
        registry.register_expression(node_name, convert_binary_expr)


def convert_immediate(
    expr: object,
    context: TileLangConversionContext,
    _converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import an integer or floating-point immediate."""

    value = getattr(expr, "value", expr)
    value_type = "index" if options.index_like else dtype(expr)
    name = context.fresh_name("c" if options.index_like else "const")
    result = context.build_constant(value, value_type, name)
    context.map_value(expr, result, value_type)
    return result


def convert_var(
    expr: object,
    context: TileLangConversionContext,
    _converter: TileLangConverter,
    _options: ExpressionOptions,
) -> ValueRef | None:
    """Resolve a mapped TIR variable."""

    mapped_var = context.mapped(expr)
    if mapped_var is None:
        context.record_blocked(node_text(expr), "variable is not mapped")
    return mapped_var


def convert_binary_expr(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a scalar or index binary expression."""

    lhs = converter.convert_expr(
        getattr(expr, "a", None),
        context,
        index_like=options.index_like,
    )
    rhs = converter.convert_expr(
        getattr(expr, "b", None),
        context,
        index_like=options.index_like,
    )
    if lhs is None or rhs is None:
        context.record_blocked(node_text(expr), "binary operands are not mapped")
        return None
    kind = type(expr).__name__
    if options.index_like:
        builder_name = _BINARY_INDEX_OPS.get(kind)
        if builder_name is None:
            context.record_blocked(node_text(expr), f"no index builder for {kind}")
            return None
        builder = getattr(context.builder.index, builder_name)
        result = cast(
            ValueRef,
            builder(
                lhs=lhs,
                rhs=rhs,
                results=[INDEX],
                name=context.fresh_name(builder_name),
            ),
        )
        context.map_value(expr, result, "index")
        return result
    result_type = context.type_converter.map_dtype(dtype(expr))
    if str(result_type).startswith("f"):
        scalar_builder_name = _BINARY_FLOAT_OPS.get(kind)
    else:
        scalar_builder_name = _BINARY_INTEGER_OPS.get(kind)
    if scalar_builder_name is None:
        context.record_blocked(node_text(expr), f"no scalar builder for {kind}")
        return None
    builder = getattr(context.builder.scalar, scalar_builder_name)
    result = cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[result_type],
            name=context.fresh_name(scalar_builder_name),
        ),
    )
    context.map_value(expr, result, str(result_type))
    return result


_BINARY_INDEX_OPS = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "FloorDiv": "div",
    "FloorMod": "rem",
}

_BINARY_INTEGER_OPS = {
    "Add": "addi",
    "Sub": "subi",
    "Mul": "muli",
    "FloorDiv": "floordivsi",
    "FloorMod": "remsi",
    "Div": "divsi",
}

_BINARY_FLOAT_OPS = {
    "Add": "addf",
    "Sub": "subf",
    "Mul": "mulf",
    "Div": "divf",
}
