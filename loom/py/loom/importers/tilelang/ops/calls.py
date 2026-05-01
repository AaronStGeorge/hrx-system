# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR call expression converters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.coverage import OpCoverage, coverage_by_name
from loom.importers.tilelang.nodes import dtype, mapping_items, node_kind, node_text
from loom.importers.tilelang.ops.assumptions import convert_assume_call
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import I1, INDEX, ScalarType, ShapedType, StaticDim, Type, TypeKind


@dataclass(frozen=True, slots=True)
class UnaryCallSpec:
    """One-operand scalar call mapping."""

    builder_name: str
    result_kind: str = "source"


@dataclass(frozen=True, slots=True)
class BinaryCallSpec:
    """Two-operand scalar call mapping."""

    builder_name: str


@dataclass(frozen=True, slots=True)
class TernaryCallSpec:
    """Three-operand scalar call mapping."""

    builder_name: str


@dataclass(frozen=True, slots=True)
class AccessPtrValue:
    """Decoded TileLang pointer metadata for element-level memory effects."""

    view: ValueRef
    indices: tuple[int | ValueRef, ...]
    buffer: object
    rw_mask: int


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_expression("Call", convert_call)


def convert_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR call by dispatching on its registered op name."""

    op_name = call_op_name(expr)
    if op_name is None:
        context.record_blocked(node_text(expr), "call has no structured op name")
        return None
    if _annotations(expr):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` carries annotations that are not imported",
        )
        return None
    if op_name in _UNARY_FLOAT_CALLS:
        return _convert_unary_call(
            expr,
            context,
            converter,
            op_name,
            _UNARY_FLOAT_CALLS[op_name],
        )
    if op_name in _BINARY_FLOAT_CALLS:
        return _convert_binary_call(
            expr,
            context,
            converter,
            op_name,
            _BINARY_FLOAT_CALLS[op_name],
        )
    if op_name in _TERNARY_FLOAT_CALLS:
        return _convert_ternary_call(
            expr,
            context,
            converter,
            op_name,
            _TERNARY_FLOAT_CALLS[op_name],
        )
    if op_name in _FLOAT_PREDICATE_CALLS:
        return _convert_float_predicate_call(
            expr,
            context,
            converter,
            op_name,
            _FLOAT_PREDICATE_CALLS[op_name],
        )
    if op_name in _ABS_CALLS:
        return _convert_abs_call(expr, context, converter, op_name)
    if op_name in _UNARY_INTEGER_CALLS:
        return _convert_unary_integer_call(
            expr,
            context,
            converter,
            op_name,
            _UNARY_INTEGER_CALLS[op_name],
        )
    if op_name in _BITWISE_NOT_CALLS:
        return _convert_bitwise_not_call(expr, context, converter, op_name)
    if op_name in _BINARY_INTEGER_CALLS:
        return _convert_binary_integer_call(
            expr,
            context,
            converter,
            op_name,
            _BINARY_INTEGER_CALLS[op_name],
            options=options,
        )
    if op_name in _CEILDIV_CALLS:
        return _convert_ceildiv_call(expr, context, converter, op_name, options=options)
    if op_name in _REINTERPRET_CALLS:
        return _convert_reinterpret_call(expr, context, converter, op_name)
    if op_name in _ATOMIC_ADD_CALLS:
        return _convert_atomic_add_call(
            expr,
            context,
            converter,
            op_name,
            options=options,
        )
    if op_name in _EFFECT_CALLS:
        return _convert_effect_call(
            expr,
            context,
            op_name,
            options=options,
        )
    if op_name in _SIGMOID_CALLS:
        return _convert_sigmoid_call(expr, context, converter, op_name)
    if op_name in _IDENTITY_CALLS:
        return _convert_identity_call(expr, context, converter, op_name)
    _record_unsupported_call(expr, context, op_name)
    return None


def call_op_name(call: object) -> str | None:
    """Returns the structured TVM op name for a TIR call."""

    op = getattr(call, "op", None)
    if op is None:
        return None
    name = getattr(op, "name", None)
    if name:
        return str(name)
    get_name = getattr(op, "get_name", None)
    if get_name is not None:
        resolved_name = get_name()
        if resolved_name:
            return str(resolved_name)
    if isinstance(op, str):
        return op
    text = str(op)
    return text if text else None


def _convert_unary_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    spec: UnaryCallSpec,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    result_type = _call_result_type(expr, context, spec.result_kind)
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, spec.builder_name)(
            input=input_value,
            results=[result_type],
            name=context.fresh_name(spec.builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_binary_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    spec: BinaryCallSpec,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 2 operands")
        return None
    lhs = converter.convert_expr(args[0], context)
    rhs = converter.convert_expr(args[1], context)
    if lhs is None or rhs is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` operands are not mapped",
        )
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, spec.builder_name)(
            lhs=lhs,
            rhs=rhs,
            results=[result_type],
            name=context.fresh_name(spec.builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_ternary_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    spec: TernaryCallSpec,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 3:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 3 operands")
        return None
    a = converter.convert_expr(args[0], context)
    b = converter.convert_expr(args[1], context)
    c = converter.convert_expr(args[2], context)
    if a is None or b is None or c is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` operands are not mapped",
        )
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, spec.builder_name)(
            a=a,
            b=b,
            c=c,
            results=[result_type],
            name=context.fresh_name(spec.builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_float_predicate_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    builder_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, builder_name)(
            input=input_value,
            results=[I1],
            name=context.fresh_name(builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_abs_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    input_type = str(input_value.type)
    if _is_unsigned_dtype(dtype(args[0])):
        context.map_value(expr, input_value, input_type)
        context.record_converted(node_text(expr), f"{op_name} normalized to identity")
        return input_value
    builder_name = "absf" if _is_float_type(input_type) else "absi"
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, builder_name)(
            input=input_value,
            results=[context.type_converter.map_dtype(dtype(expr))],
            name=context.fresh_name(builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_unary_integer_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    builder_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    index_like = str(input_value.type) == "index"
    dialect = context.builder.index if index_like else context.builder.scalar
    result_type = INDEX if index_like else context.type_converter.map_dtype(dtype(expr))
    result = cast(
        ValueRef,
        getattr(dialect, builder_name)(
            input=input_value,
            results=[result_type],
            name=context.fresh_name(builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_bitwise_not_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    if _is_vector_type(input_value.type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` vector operands are not imported by the scalar path",
        )
        return None
    input_type = str(input_value.type)
    if input_type in ("index", "offset"):
        all_ones = context.ensure_constant("-1", input_type, "all_ones")
        result = context.builder.index.xori(
            lhs=input_value,
            rhs=all_ones,
            results=[input_value.type],
            name=context.fresh_name("noti"),
        )
        _map_call_result(expr, context, result, op_name, value_type=input_type)
        return result
    result_type = context.type_converter.map_dtype(dtype(expr))
    if not _is_integer_type(str(result_type)):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` result type {result_type} is not integer",
        )
        return None
    all_ones = _ensure_integer_all_ones(context, dtype(expr))
    result = context.builder.scalar.xori(
        lhs=input_value,
        rhs=all_ones,
        results=[result_type],
        name=context.fresh_name("noti"),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_binary_integer_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    builder_name: str,
    *,
    options: ExpressionOptions,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 2 operands")
        return None
    index_like = (
        options.index_like
        or op_name in _FORCED_INDEX_CALLS
        or any(_source_is_index_like(arg, context) for arg in args)
    )
    lhs = converter.convert_expr(args[0], context, index_like=index_like)
    rhs = converter.convert_expr(args[1], context, index_like=index_like)
    if lhs is None or rhs is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` operands are not mapped",
        )
        return None
    if index_like:
        index_builder_name = "shrui" if builder_name == "shr" else builder_name
        result = cast(
            ValueRef,
            getattr(context.builder.index, index_builder_name)(
                lhs=lhs,
                rhs=rhs,
                results=[INDEX],
                name=context.fresh_name(index_builder_name),
            ),
        )
        _map_call_result(expr, context, result, op_name, value_type="index")
        return result
    scalar_builder_name = _scalar_integer_builder(builder_name, dtype(expr))
    result = cast(
        ValueRef,
        getattr(context.builder.scalar, scalar_builder_name)(
            lhs=lhs,
            rhs=rhs,
            results=[context.type_converter.map_dtype(dtype(expr))],
            name=context.fresh_name(scalar_builder_name),
        ),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_ceildiv_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    options: ExpressionOptions,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 2 operands")
        return None
    index_like = options.index_like or any(
        _source_is_index_like(arg, context) for arg in args
    )
    lhs = converter.convert_expr(args[0], context, index_like=index_like)
    rhs = converter.convert_expr(args[1], context, index_like=index_like)
    if lhs is None or rhs is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` operands are not mapped",
        )
        return None
    if not index_like:
        builder_name = "ceildivui" if _is_unsigned_dtype(dtype(expr)) else "ceildivsi"
        result = cast(
            ValueRef,
            getattr(context.builder.scalar, builder_name)(
                lhs=lhs,
                rhs=rhs,
                results=[context.type_converter.map_dtype(dtype(expr))],
                name=context.fresh_name(builder_name),
            ),
        )
        _map_call_result(expr, context, result, op_name)
        return result
    one = context.ensure_constant("1", "index", "c1")
    rhs_minus_one = context.builder.index.sub(
        lhs=rhs,
        rhs=one,
        results=[INDEX],
        name=context.fresh_name("ceil_rhs_minus_one"),
    )
    adjusted = context.builder.index.add(
        lhs=lhs,
        rhs=rhs_minus_one,
        results=[INDEX],
        name=context.fresh_name("ceil_adjusted"),
    )
    result = context.builder.index.div(
        lhs=adjusted,
        rhs=rhs,
        results=[INDEX],
        name=context.fresh_name("ceildiv"),
    )
    _map_call_result(expr, context, result, op_name, value_type="index")
    return result


def _convert_sigmoid_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    one = _ensure_float_one(context, dtype(expr))
    neg = context.builder.scalar.negf(
        input=input_value,
        results=[result_type],
        name=context.fresh_name("sigmoid_neg"),
    )
    exp = context.builder.scalar.expf(
        input=neg,
        results=[result_type],
        name=context.fresh_name("sigmoid_exp"),
    )
    denominator = context.builder.scalar.addf(
        lhs=one,
        rhs=exp,
        results=[result_type],
        name=context.fresh_name("sigmoid_den"),
    )
    result = context.builder.scalar.divf(
        lhs=one,
        rhs=denominator,
        results=[result_type],
        name=context.fresh_name("sigmoid"),
    )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_identity_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    result = converter.convert_expr(args[0], context)
    if result is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    context.map_value(expr, result, str(result.type))
    context.record_converted(node_text(expr), f"{op_name} normalized to identity")
    return result


def _convert_reinterpret_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
) -> ValueRef | None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 1 operand")
        return None
    input_value = converter.convert_expr(args[0], context)
    if input_value is None:
        context.record_blocked(
            node_text(expr), f"call `{op_name}` operand is not mapped"
        )
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    if str(input_value.type) == str(result_type):
        context.map_value(expr, input_value, str(result_type))
        context.record_converted(node_text(expr), f"{op_name} normalized to identity")
        return input_value
    input_is_vector = _is_vector_type(input_value.type)
    result_is_vector = _is_vector_type(result_type)
    if input_is_vector != result_is_vector:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` cannot cross scalar/vector type boundaries",
        )
        return None
    input_bit_width = _total_bit_width(input_value.type)
    result_bit_width = _total_bit_width(result_type)
    if input_bit_width is None or result_bit_width is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` requires statically known bit widths",
        )
        return None
    if input_bit_width != result_bit_width:
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` bit widths differ: "
                f"{input_value.type} has {input_bit_width} bits, "
                f"{result_type} has {result_bit_width} bits"
            ),
        )
        return None
    if input_is_vector:
        result = context.builder.vector.bitcast(
            input=input_value,
            results=[result_type],
            name=context.fresh_name("bitcast"),
        )
    else:
        result = context.builder.scalar.bitcast(
            input=input_value,
            results=[result_type],
            name=context.fresh_name("bitcast"),
        )
    _map_call_result(expr, context, result, op_name)
    return result


def _convert_effect_call(
    expr: object,
    context: TileLangConversionContext,
    op_name: str,
    *,
    options: ExpressionOptions,
) -> ValueRef | None:
    if not options.effect:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` is effect-only and must appear under tir.Evaluate",
        )
        return None
    if op_name in _ASSUME_CALLS:
        convert_assume_call(expr, context)
        return None
    if op_name in _STORAGE_SYNC_CALLS:
        _convert_storage_sync_call(expr, context, op_name)
        return None
    _record_unsupported_call(expr, context, op_name)
    return None


def _convert_atomic_add_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    options: ExpressionOptions,
) -> ValueRef | None:
    returns_old_value = op_name in _ATOMIC_ADD_RETURNING_CALLS
    if not returns_old_value and not options.effect:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` is effect-only and must appear under tir.Evaluate",
        )
        return None
    args = _args(expr)
    if len(args) not in (2, 3):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects destination, value, and optional ordering",
        )
        return None
    value = converter.convert_expr(args[1], context)
    if value is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` value operand is not mapped",
        )
        return None
    if _is_vector_type(value.type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` vector atomics are not imported by the scalar path",
        )
        return None
    access = _decode_tilelang_access_ptr(args[0], expr, context, converter)
    if access is None:
        return None
    if access.rw_mask != 3:
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` requires an rw access pointer, "
                f"got mask {access.rw_mask}"
            ),
        )
        return None
    ordering = _atomic_ordering(args, expr, context, op_name)
    if ordering is None:
        return None
    scope = _atomic_scope_for_buffer(access.buffer)
    if scope is None:
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` buffer scope "
                f"`{_buffer_scope(access.buffer)}` is not mapped"
            ),
        )
        return None
    kind = "addf" if _is_float_type(str(value.type)) else "addi"
    if returns_old_value:
        result_type = context.type_converter.map_dtype(dtype(expr))
        result = context.builder.view.rmw(
            kind=kind,
            value=value,
            view=access.view,
            indices=list(access.indices),
            ordering=ordering,
            scope=scope,
            results=[result_type],
            name=context.fresh_name("atomic_add"),
        )
        _map_call_result(expr, context, result, op_name)
        return result
    context.builder.view.reduce(
        kind=kind,
        value=value,
        view=access.view,
        indices=list(access.indices),
        ordering=ordering,
        scope=scope,
    )
    context.record_converted(node_text(expr), f"view.atomic.reduce<{kind}>")
    return None


def _decode_tilelang_access_ptr(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> AccessPtrValue | None:
    op_name = call_op_name(expr)
    if op_name != "tl.access_ptr":
        context.record_blocked(
            node_text(owner),
            f"atomic destination must be tl.access_ptr, got `{op_name}`",
        )
        return None
    args = _args(expr)
    if len(args) != 3:
        context.record_blocked(
            node_text(owner),
            "tl.access_ptr expects base load, extent, and rw mask",
        )
        return None
    extent = integer_value(args[1])
    if extent != 1:
        context.record_blocked(
            node_text(owner),
            f"tl.access_ptr extent `{extent}` is not a scalar element access",
        )
        return None
    rw_mask = integer_value(args[2])
    if rw_mask is None:
        context.record_blocked(node_text(owner), "tl.access_ptr rw mask is not static")
        return None
    base_load = args[0]
    if node_kind(base_load) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            "tl.access_ptr base is not a BufferLoad",
        )
        return None
    buffer = getattr(base_load, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(
            node_text(owner),
            "tl.access_ptr buffer is not mapped",
        )
        return None
    indices: list[int | ValueRef] = []
    for source_index in tuple(getattr(base_load, "indices", ())):
        index = converter.convert_expr(source_index, context, index_like=True)
        if index is None:
            context.record_blocked(
                node_text(owner),
                "tl.access_ptr index is not mapped",
            )
            return None
        indices.append(index)
    return AccessPtrValue(
        view=view,
        indices=tuple(indices),
        buffer=buffer,
        rw_mask=rw_mask,
    )


def _atomic_ordering(
    args: tuple[object, ...],
    owner: object,
    context: TileLangConversionContext,
    op_name: str,
) -> str | None:
    if len(args) == 2:
        return "relaxed"
    ordering_id = integer_value(args[2])
    if ordering_id is None:
        context.record_blocked(
            node_text(owner), f"call `{op_name}` ordering is not static"
        )
        return None
    ordering = _ATOMIC_ORDERING_BY_TILELANG_ID.get(ordering_id)
    if ordering is None:
        context.record_blocked(
            node_text(owner),
            f"call `{op_name}` memory order id {ordering_id} is not represented",
        )
        return None
    return ordering


def _atomic_scope_for_buffer(buffer: object) -> str | None:
    scope = _buffer_scope(buffer)
    if scope in ("shared", "shared.dyn"):
        return "workgroup"
    if scope in ("", "global"):
        return "device"
    return None


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""


def _convert_storage_sync_call(
    expr: object,
    context: TileLangConversionContext,
    op_name: str,
) -> None:
    args = _args(expr)
    if len(args) != 1:
        context.record_blocked(
            node_text(expr),
            (
                f"call `{op_name}` expects one storage scope; "
                "TileLang barrier_id/arrive_count forms are not imported"
            ),
        )
        return
    storage_scope = _string_value(args[0])
    if storage_scope not in _WORKGROUP_STORAGE_SCOPES:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` storage scope `{storage_scope}` is not mapped",
        )
        return
    context.builder.kernel.barrier(
        memory_space="workgroup",
        ordering="acq_rel",
        scope="workgroup",
    )
    context.record_converted(node_text(expr), "kernel.barrier")


def _call_result_type(
    expr: object,
    context: TileLangConversionContext,
    result_kind: str,
) -> Type:
    if result_kind == "bool":
        return I1
    return context.type_converter.map_dtype(dtype(expr))


def _map_call_result(
    expr: object,
    context: TileLangConversionContext,
    result: ValueRef,
    op_name: str,
    *,
    value_type: str | None = None,
) -> None:
    context.map_value(expr, result, value_type or str(result.type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = {op_name}")


def _record_unsupported_call(
    expr: object,
    context: TileLangConversionContext,
    op_name: str,
) -> None:
    row = _coverage_row(op_name)
    if row is None:
        context.record_blocked(
            node_text(expr),
            f"no TileLang/TIR call converter for `{op_name}`",
        )
        return
    context.record_blocked(
        node_text(expr),
        f"call `{op_name}` coverage state is {row.state.value}: {row.note}",
    )


def _coverage_row(op_name: str) -> OpCoverage | None:
    rows = coverage_by_name()
    exact = rows.get(op_name)
    if exact is not None:
        return exact
    for row in rows.values():
        if row.name.endswith(".*") and op_name.startswith(row.name[:-1]):
            return row
    return None


def _scalar_integer_builder(builder_name: str, source_dtype: object) -> str:
    if builder_name == "div":
        return "divui" if _is_unsigned_dtype(source_dtype) else "divsi"
    if builder_name == "floordiv":
        return "divui" if _is_unsigned_dtype(source_dtype) else "floordivsi"
    if builder_name == "rem":
        return "remui" if _is_unsigned_dtype(source_dtype) else "remsi"
    if builder_name == "shr":
        return "shrui" if _is_unsigned_dtype(source_dtype) else "shrsi"
    return builder_name


def _ensure_float_one(
    context: TileLangConversionContext,
    value_type: str,
) -> ValueRef:
    existing = context.constants.get((value_type, "1.0"))
    if existing is not None:
        return existing
    result = context.build_constant(1.0, value_type, context.reserve_name("one"))
    context.remember_constant("1.0", value_type, result)
    return result


def _ensure_integer_all_ones(
    context: TileLangConversionContext,
    value_type: str,
) -> ValueRef:
    existing = context.constants.get((value_type, "-1"))
    if existing is not None:
        return existing
    result = context.build_constant(-1, value_type, context.reserve_name("all_ones"))
    context.remember_constant("-1", value_type, result)
    return result


def _source_is_index_like(
    source: object,
    context: TileLangConversionContext,
) -> bool:
    mapped = context.mapped(source)
    return mapped is not None and str(mapped.type) in ("index", "offset")


def _is_vector_type(value_type: object) -> bool:
    return (
        isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VECTOR
    )


def _total_bit_width(value_type: Type) -> int | None:
    if isinstance(value_type, ScalarType):
        return value_type.bitwidth
    if not isinstance(value_type, ShapedType):
        return None
    total = value_type.element_type.bitwidth
    for dim in value_type.dims:
        if not isinstance(dim, StaticDim):
            return None
        total *= dim.size
    return total


def _is_float_type(value_type: str) -> bool:
    return value_type in ("f16", "bf16", "f32", "f64")


def _is_integer_type(value_type: str) -> bool:
    return value_type != "i1" and value_type.startswith("i")


def _is_unsigned_dtype(source_dtype: object) -> bool:
    return str(source_dtype).startswith("uint")


def _args(call: object) -> tuple[object, ...]:
    return tuple(getattr(call, "args", ()))


def _annotations(call: object) -> tuple[tuple[object, object], ...]:
    return mapping_items(getattr(call, "annotations", {}))


def _string_value(value: object) -> str:
    payload = getattr(value, "value", value)
    return str(payload)


_UNARY_FLOAT_CALLS = {
    "tir.acos": UnaryCallSpec("acosf"),
    "tir.acosh": UnaryCallSpec("acoshf"),
    "tir.asin": UnaryCallSpec("asinf"),
    "tir.asinh": UnaryCallSpec("asinhf"),
    "tir.atan": UnaryCallSpec("atanf"),
    "tir.atanh": UnaryCallSpec("atanhf"),
    "tir.cbrt": UnaryCallSpec("cbrtf"),
    "tir.ceil": UnaryCallSpec("ceilf"),
    "tir.cos": UnaryCallSpec("cosf"),
    "tir.cosh": UnaryCallSpec("coshf"),
    "tir.erf": UnaryCallSpec("erff"),
    "tir.erfc": UnaryCallSpec("erfcf"),
    "tir.exp": UnaryCallSpec("expf"),
    "tir.exp2": UnaryCallSpec("exp2f"),
    "tir.expm1": UnaryCallSpec("expm1f"),
    "tir.floor": UnaryCallSpec("floorf"),
    "tir.log": UnaryCallSpec("logf"),
    "tir.log10": UnaryCallSpec("log10f"),
    "tir.log1p": UnaryCallSpec("log1pf"),
    "tir.log2": UnaryCallSpec("log2f"),
    "tir.nearbyint": UnaryCallSpec("roundevenf"),
    "tir.round": UnaryCallSpec("roundf"),
    "tir.rsqrt": UnaryCallSpec("rsqrtf"),
    "tir.sin": UnaryCallSpec("sinf"),
    "tir.sinh": UnaryCallSpec("sinhf"),
    "tir.sqrt": UnaryCallSpec("sqrtf"),
    "tir.tan": UnaryCallSpec("tanf"),
    "tir.tanh": UnaryCallSpec("tanhf"),
    "tir.trunc": UnaryCallSpec("truncf"),
}

_BINARY_FLOAT_CALLS = {
    "tir.atan2": BinaryCallSpec("atan2f"),
    "tir.copysign": BinaryCallSpec("copysignf"),
    "tir.fmod": BinaryCallSpec("remf"),
    "tir.pow": BinaryCallSpec("powf"),
    "tir.power": BinaryCallSpec("powf"),
}

_TERNARY_FLOAT_CALLS = {
    "tir.fma": TernaryCallSpec("fmaf"),
}

_FLOAT_PREDICATE_CALLS = {
    "tir.isfinite": "isfinitef",
    "tir.isinf": "isinff",
    "tir.isnan": "isnanf",
}

_ABS_CALLS = {
    "tir.abs",
}

_UNARY_INTEGER_CALLS = {
    "tir.clz": "ctlzi",
    "tir.popcount": "ctpopi",
}

_BITWISE_NOT_CALLS = {
    "tir.bitwise_not",
}

_BINARY_INTEGER_CALLS = {
    "tir.bitwise_and": "andi",
    "tir.bitwise_or": "ori",
    "tir.bitwise_xor": "xori",
    "tir.div": "div",
    "tir.floordiv": "floordiv",
    "tir.floormod": "rem",
    "tir.indexdiv": "div",
    "tir.indexmod": "rem",
    "tir.shift_left": "shli",
    "tir.shift_right": "shr",
    "tir.truncdiv": "div",
    "tir.truncmod": "rem",
}

_FORCED_INDEX_CALLS = {
    "tir.indexdiv",
    "tir.indexmod",
}

_CEILDIV_CALLS = {
    "tir.ceildiv",
}

_REINTERPRET_CALLS = {
    "tir.reinterpret",
}

_ATOMIC_ADD_EFFECT_CALLS = {
    "tl.atomic_add_elem_op",
}

_ATOMIC_ADD_RETURNING_CALLS = {
    "tl.atomic_add_ret_elem_op",
}

_ATOMIC_ADD_CALLS = _ATOMIC_ADD_EFFECT_CALLS | _ATOMIC_ADD_RETURNING_CALLS

_ATOMIC_ORDERING_BY_TILELANG_ID = {
    0: "relaxed",
    2: "acquire",
    3: "release",
    4: "acq_rel",
    5: "seq_cst",
}

_STORAGE_SYNC_CALLS = {
    "tir.tvm_storage_sync",
}

_ASSUME_CALLS = {
    "tir.assume",
}

_EFFECT_CALLS = _STORAGE_SYNC_CALLS | _ASSUME_CALLS

_WORKGROUP_STORAGE_SCOPES = {
    "shared",
    "shared.dyn",
}

_SIGMOID_CALLS = {
    "tir.sigmoid",
}

_IDENTITY_CALLS = {
    "tir.likely",
}
